# Flat WAL durability and startup recovery

## Scope and ownership

This mode supports Linux, one cooperating process, one serialized writer, FlatIndex insertion
and one operation per commit. `DurableFlatIndex` owns an ordinary FlatIndex, an exclusive
`flock` on `writer.lock`, WAL state and the persistent retry table. VectorStore supplies the
exclusive write/checkpoint lock and shared search lock. No background checkpoint thread or
second write path is introduced. Low-level direct users must serialize every access.

The internal components in `src/durable_flat_index.cpp` have separate responsibilities:

- `WalWriter`: append an encoded record and `fdatasync` it.
- `WalReader`: bound lengths, validate checksums/LSNs and decode records sequentially.
- `RecoveryManager`: validate a recovery manifest and checkpoint before replay.
- `CheckpointManager`: publish an immutable snapshot and then its recovery manifest.
- `DurableFlatIndex::Impl`: validate, deduplicate, coordinate commit and fail closed.

This first version deliberately retains the entire WAL and all historical operation IDs.
Checkpointed operations are validated and used to reconstruct retry state; only the suffix
after the checkpoint LSN is applied to the index. A checkpoint therefore reduces index replay
work, but does not eliminate the WAL scan or bound the retry table. There is no log rotation,
garbage collection, group commit, durable deletion, transactional batch insertion, HNSW recovery,
writable replication, writable snapshot reload or automatic repair of corrupt files.

## Meaning of success

```text
VectorStore exclusive lock
  -> validate request and reserve retry metadata
  -> encode original and normalized float bits
  -> append WAL record
  -> fdatasync(WAL) succeeds
  -> apply normalized values to FlatIndex
  -> return LSN and original result size
```

An AddVector success in WAL mode means both synchronization and in-memory application completed.
It is conditional on the OS, filesystem, virtual disk and hardware honoring synchronization.
It does not imply that other processes or replicas received the operation. Default non-WAL
insertion remains volatile and returns LSN zero.

Validation/encoding/allocation failures before append admission leave the writer usable. Once
append admission begins, any failure conservatively enters `recovery_required`, including an
uncertain write/sync outcome and a post-sync allocation failure. No later commit, even a retry,
or checkpoint is accepted until restart. Queries can still read the last successfully applied
in-memory state. A fully written operation whose response failed may appear after recovery;
an incomplete trailing operation may disappear. Clients must retry using the same operation ID.

Recovery validates the entire accepted prefix before truncating an incomplete final record.
It then synchronizes the accepted WAL prefix, including any complete unacknowledged record,
before returning. Otherwise a retry might acknowledge page-cache-only data without a new sync.
The server does not bind a port until this recovery finishes. Startup corruption fails closed;
it never falls back to a new empty writable index.

## Retry contract

`client_operation_id` is required in WAL mode and is limited to 256 bytes. The retry identity
includes the VectorId, dimension and original float32 bits in order. The same ID and exact
content returns the original LSN and index size with `replayed=true`, without another append.
Reuse with different content is INVALID_ARGUMENT. Different operation IDs targeting an
existing VectorId are also rejected. In particular, `+0.0` and `-0.0` are different request
bits, and differently scaled vectors are different requests even if their directions agree.
Non-finite and zero vectors are rejected.

The WAL stores both original input bits and the once-normalized bits. Original bits reconstruct
the retry table; normalized bits are applied directly on recovery after finite/norm validation.
They are not normalized a second time. Checkpoint vectors are checked bit for bit against the
WAL prefix. Successful retries retain the original `index_size` result, which can be less than
the current size. `inserted_vectors` counts new applications in this process, not recovery or
retries. This is scoped persistent deduplication, not end-to-end exactly-once delivery.

An operation ID is rejected in volatile mode so that a caller cannot mistake volatile insertion
for durable deduplication. The plain `VectorIndex::add` API on a DurableFlatIndex is rejected;
callers cannot bypass operation IDs. BatchAdd is rejected in WAL mode before any batch element
is applied. No delete or mutable underlying FlatIndex accessor is exposed.

## Version 1 binary format

All integers are unsigned little-endian. Floats are IEEE binary32 bits, not native struct dumps.
Checksums use the project's FNV-1a-64 byte checksum (offset 14695981039346656037, prime
1099511628211). This detects accidental corruption; it is not authentication or a defense
against deliberate checksum recomputation. A private, trusted directory is required.

`operations.wal` has a 40-byte checksummed header:

| Offset | Width | Field |
| --- | --- | --- |
| 0 | 8 | `FVWAL01\0` magic |
| 8 | 4 | version = 1 |
| 12 | 4 | header length = 40 |
| 16 | 8 | fixed dimension |
| 24 | 8 | maximum encoded record length: `48 + 256 + 8 * dimension` |
| 32 | 8 | checksum of bytes 0..31 |

Each record has this layout:

| Offset | Width | Field |
| --- | --- | --- |
| 0 | 8 | total record length, including checksum |
| 8 | 8 | contiguous LSN starting at 1 |
| 16 | 4 | operation = 1 (insert) |
| 20 | 4 | operation-ID byte count |
| 24 | 8 | external VectorId |
| 32 | 8 | dimension |
| 40 | variable | operation-ID bytes |
| following | `4 * dimension` | original float bits |
| following | `4 * dimension` | normalized float bits |
| last | 8 | checksum of all preceding record bytes |

Dimension is limited to 1,000,000. Record lengths must match the configured dimension and
bounded ID length exactly. Files are bounded by `maximum_wal_bytes` (default 1 GiB), checked
against platform offsets before opening. Only a bounded fixed prefix is read before allocating
a record buffer; each available fixed field is checked even for partial records. The reader
rejects unsupported operations, duplicate/skipped/backward LSNs, duplicate operation/vector IDs,
non-finite or non-normalized payloads and complete-record checksum failures.

A final short prefix or a valid bounded record header followed by insufficient payload bytes
is an unfinished append. Only that final suffix is truncated and synchronized. An invalid
length, an already available invalid fixed field or a complete checksum failure is corruption,
including in the last record. The format cannot authenticate a plausible partial tail against
maliciously rewritten bytes; no automatic salvage of arbitrary corruption is attempted.

## Checkpoint publication

While holding the exclusive VectorStore lock:

1. Choose the applied LSN. No writer can move this boundary during the checkpoint.
2. Save a same-directory exclusive temporary Flat snapshot, reload/validate it and fsync it.
3. Publish immutable `checkpoint-LSN.fv` with a no-overwrite hard link and fsync the directory.
   A same-LSN retry may reuse an existing file only if its length and checksum match.
4. Write and fsync a same-directory temporary recovery manifest.
5. Rename it over `recovery.manifest`, then fsync the directory.

This follows the existing durable snapshot publication sequence without changing Flat format 1.
Failures before manifest replacement preserve the old recovery chain and permit more writes.
If replacement occurred but the directory sync or subsequent step fails, the coordinator marks
recovery required: it cannot claim whether the new name is durable. WAL files are never removed,
even after successful publication. Old checkpoint files are retained; temporary files from
normal exceptions are removed by RAII, while SIGKILL may leave harmless unreferenced temp files.

The fixed 72-byte `recovery.manifest` contains:

| Offset | Width | Field |
| --- | --- | --- |
| 0 | 8 | `FVREC01\0` magic |
| 8 | 4 | version = 1 |
| 12 | 4 | manifest length = 72 |
| 16 | 8 | dimension |
| 24 | 8 | checkpoint LSN |
| 32 | 8 | snapshot byte length |
| 40 | 8 | whole-snapshot checksum |
| 48 | 8 | byte offset immediately after checkpoint's WAL prefix |
| 56 | 8 | checksum of that WAL prefix, including its header |
| 64 | 8 | checksum of manifest bytes 0..63 |

The snapshot filename is derived from the checked LSN; no file path is trusted from a manifest.
Recovery checks manifest bounds, snapshot content and header shape before loading, validates the
WAL boundary checksum, checks every checkpointed vector against the original log, rebuilds
retry state, and applies only later records. Missing or inconsistent artifacts reject startup.
Copy/backup the WAL, manifest and referenced checkpoint together while writes are stopped;
copying just the snapshot loses persistent retry state.

## Running

Use an existing private directory on a filesystem that supports Linux flock, fsync, hard links
and atomic same-filesystem rename. The process syncs its directory and immediate parent during
startup; callers must provision durable ancestor directories and storage. Do not remove or
replace `writer.lock` while a process is running. The lock coordinates cooperating local
processes; it is not a distributed lease or protection against a privileged directory writer.

```bash
mkdir -m 700 /tmp/fast-vector-wal-demo
./build-clang-grpc/fast_vector_server \
  --address 127.0.0.1:50051 --index flat --dimension 3 \
  --wal-dir /tmp/fast-vector-wal-demo --max-wal-bytes 1073741824

python scripts/generate_grpc_python.py
python scripts/grpc_client.py add --id 10 --vector 1,2,3 --operation-id demo-10
python scripts/grpc_client.py checkpoint
```

Generate protobuf modules again after changing the contract. Restart with the same dimension
and WAL directory. This mode rejects read-only operation, `--load-index`, managed publication,
SIGHUP reload and HNSW. It does not change the process-local 11B read-only generation fence.
The reference container/Kubernetes deployment has no provisioned WAL volume; persistence there
requires explicitly mounted durable writable storage and matching ownership/permissions.

GetStats reports enabled/recovery-required, applied/checkpoint LSN, suffix replay count, discarded
tail bytes, recovery time, and last write/sync/checkpoint times. The standard health service still
describes RPC availability; inspect durability status for write availability. Checkpoint errors
use the ordinary RPC error handling. A subsequent write after a poisoned writer returns
FAILED_PRECONDITION. A transport deadline or disconnect after commit may hide a successful write;
the handler does not undo it or pretend it was never executed.

## Verification and measurement

```bash
cmake --build build-clang-grpc -j 4
ctest --test-dir build-clang-grpc --output-on-failure -j 4
FAST_VECTOR_SERVER=build-clang-grpc/fast_vector_server \
  build/python-grpc-venv/bin/python -m unittest -v tests/wal_recovery_test.py
./build-clang-grpc/fast_vector_wal_benchmark /tmp/fast-vector-wal-benchmark 1000 128
```

The GoogleTests exercise WAL-only and checkpoint recovery, every truncation point of the final
record, full corruption and invalid fields, failed checkpoint chains, exact retry bits, the
directory lock, resource bounds, serialized service ownership, post-sync allocation failures,
and SIGKILL at append/sync/apply/response boundaries. The dedicated allocation executable sweeps
actual `operator new` failures during Flat application, rather than only throwing at a hook.
The Python test starts real services, confirms writes, kills them, restarts, checks exact ordered
IDs/score bits, retries every acknowledged operation, and sends concurrent duplicate requests.

The Release benchmark measures commit latency (mean/P50/P95/P99), the fdatasync syscall portion,
checkpoint duration, recovery wall time, suffix records applied per total recovery time, full
validated records per total recovery time and WAL size. Sync durations use integer microseconds;
zero means below that timing resolution. Input generation is outside commit timing. There is no
warmup of durable operations or comparison with unsynchronized writes; every new operation syncs.
No production throughput claim follows from these small, warm-cache, single-writer measurements.

SIGKILL is a process-termination test with the kernel/page cache still alive. It does not simulate
power loss, a kernel crash, disk-cache loss, torn sectors or a dishonest virtual block device.
Those require a separate storage/power-failure test environment and are not demonstrated here.

See [local measurements](wal_recovery_measurements.md) for actual results, reproduction commands,
verification counts and the complete changed-file list.
