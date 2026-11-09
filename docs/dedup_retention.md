# Persistent deduplication epochs and retention

## Contract

Finite retention is opt-in. The deduplication key is `(dedup_epoch, client_operation_id)`.
An accepted new operation is still synchronized to WAL before index application and success.
An exact retry returns the original LSN/index size without appending. Original float32 bits
are still checked against disk after the SHA-256 filter. Retention does not weaken equality.

| Request | Behavior |
| --- | --- |
| Current epoch, existing identical key/content | Return original result |
| Current epoch, new key | Validate quotas and commit |
| Retained closed epoch, existing key | Compare content and return original result |
| Retained closed epoch, missing key | FAILED_PRECONDITION; never create a new operation |
| Older than oldest retained epoch | FAILED_PRECONDITION, even if the VectorId exists |
| Future epoch or positive epoch in legacy mode | FAILED_PRECONDITION |
| Same retained key with different content | INVALID_ARGUMENT |
| Current epoch count/byte quota full | RESOURCE_EXHAUSTED; no WAL append or LSN advance |

The same operation-ID string can identify different operations in different epochs, but an
existing VectorId remains reserved by the index. Expiry removes retry state, not vectors.
Search results, normalization and index data remain unchanged. This is window-scoped persistent
deduplication, not end-to-end exactly-once delivery.

## Epoch lifecycle

Enabling retention publishes an initial epoch-1 recovery root before serving writes. Legacy
operations are assigned epoch 0. Explicit Checkpoint closes the current epoch and increments
it, even when no writes occurred. With current epoch E and retention R:

```text
oldest_epoch = max(0, E - R + 1)
accepted retry window = [oldest_epoch, E]
```

The default R=3 includes the active epoch and up to two preceding epochs. Migration therefore
retains legacy epoch 0 at E=1 and E=2, and expires it at E=3. Choosing R=1 explicitly expires
legacy retries during migration. Epochs are not a time TTL: repeated empty checkpoints can
expire requests quickly, and an epoch may remain open indefinitely without checkpoints.

Automatic WAL rotation publishes a checkpoint **without** advancing the epoch. Otherwise a
record that filled a segment could unexpectedly change its own request namespace. Epoch quotas
do not auto-evict or auto-advance: operators/clients must explicitly checkpoint to open the next
epoch. New data and quotas remain serialized by VectorStore's exclusive lock.

The next table and policy are prepared before publication, but become effective only after the
new manifest is durable. Pre-publication failure leaves old retries valid. An uncertain post-rename
failure stops writes until recovery. Restart reads the selected root; it never reconstructs an
expired entry by scanning old files or interpreting it as a new request. Cleanup failure may leave
expired bytes on disk but cannot make them accessible through the retry API. This is logical
retention, not secure erasure of backups, filesystem blocks or device caches.

## Configuration and compatibility

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--dedup-retention` | false | Enable migration to finite retention; requires writable Flat WAL |
| `--dedup-retained-epochs` | 3 | Window including active epoch, allowed 1..64 |
| `--dedup-max-operations` | 100000 | Maximum operations in each retained epoch |
| `--dedup-max-bytes` | 268435456 | Maximum encoded dedup record bytes in each epoch |

The per-entry byte charge is `80 + 8 + operation_id_bytes + 4*dimension`: record metadata,
epoch-prefixed key, original float bits and checksum. File headers are additional. Subject to
the index and checkpoint caps, retained entries are at most `R * max_operations`, and the dedup
file is at most `64 + R * max_bytes`. Policy products are checked for overflow. Temporary table
copies, buffers, allocator overhead, the normalized index and transport memory are separate.
The existing writable index estimate remains conservative; expiring requests does not relax
its row count or index-byte limits.

Counts and bytes apply to every retained epoch, including legacy epoch 0 during migration.
Migration rejects a history that exceeds the requested legacy-epoch quota rather than silently
discarding records inside the advertised window. The old recovery chain remains usable after
that rejection. Choose quotas and checkpoint/workspace budgets large enough for migration, or
explicitly choose a policy with no legacy retry window. No package/data conversion command is
silently executed outside startup's existing exclusive recovery path.

Once version 4 is published, its policy is authoritative across restart even if the enable flag
is omitted/false. If the flag is explicitly true, supplied/default policy values must match the
persisted policy; a mismatch rejects startup. There is no live policy-change or downgrade API.
Without enabling retention, legacy request epoch 0 and formats 1..3 retain their prior behavior.

```bash
mkdir -m 700 /tmp/fast-vector-retained
./build-clang-grpc/fast_vector_server --index flat --dimension 3 \
  --wal-dir /tmp/fast-vector-retained --dedup-retention true \
  --dedup-retained-epochs 3 --dedup-max-operations 100000 \
  --dedup-max-bytes 268435456
python scripts/generate_grpc_python.py
python scripts/grpc_client.py add --id 10 --vector 1,2,3 \
  --operation-id example-10 --dedup-epoch 1
python scripts/grpc_client.py checkpoint --expected-epoch 1
```

Clients discover the current window through GetStats. Checkpoint returns LSN/current/oldest under
one store lock; a later concurrent checkpoint can still close that epoch before the next request.
The Checkpoint RPC requires `expected_epoch` equal to the current epoch (zero in legacy mode),
checked under that same lock. A duplicate/stale checkpoint cannot advance the window twice after
a lost response: it returns FAILED_PRECONDITION. Query the window to resolve that outcome instead
of automatically refreshing the precondition. The CLI uses `checkpoint --expected-epoch E`;
the Python method is `checkpoint(expected_epoch=E)` and does not fetch an epoch implicitly.
The Python client accepts an explicit `dedup_epoch` argument and does not automatically replace
it. Keep the original epoch and ID on retry. Do not refresh the epoch and resubmit an operation
with an uncertain prior outcome: that changes its deduplication identity. A rejected, not-executed
new request can be deliberately reissued as a new operation after resolving its outcome.

Restoring an older backup rewinds data and epoch state. The service does not provide anti-rollback
fencing across arbitrary backup restoration, cloned directories or multiple writers. Such recovery
requires operational coordination/new client identity fencing beyond this single-writer protocol.

## Binary format extensions

Index format remains Flat version 1. Existing recovery versions 1..3 and dedup version 1 remain
readable. Bounded mode always rotates the WAL and uses these new versions; older binaries reject
them rather than reading a partial policy. All integer fields remain unsigned little-endian.

**Recovery manifest version 4:** 208 bytes. Offsets 0..159 retain the version-3 layout, with version
and length changed to 4 and 208. The extension is:

| Offset | Width | Field |
| --- | --- | --- |
| 160 | 8 | Current epoch E, positive |
| 168 | 8 | Oldest epoch, must equal the policy formula |
| 176 | 8 | Retained epoch count R, 1..64 |
| 184 | 8 | Operations per epoch limit |
| 192 | 8 | Encoded dedup bytes per epoch limit |
| 200 | 8 | Manifest FNV checksum over bytes 0..199 |

**Segment header version 3:** the same 72-byte layout as version 2, but its maximum record length
is `48 + 264 + 8*dimension`. The root still binds segment identity/start LSN/header checksum.
Records use operation type 2 and keep the 40-byte fixed prefix. Their key length counts eight
epoch bytes plus 1..256 operation-ID bytes, beginning at record offset 40. Original and normalized
vectors follow that key. Total length is `48 + encoded_key_bytes + 8*dimension`. A record's epoch
must equal the root's active epoch. An available invalid epoch prefix is rejected even if the
final vector payload is incomplete; only a plausible unfinished final append can be truncated.
Legacy operation type 1 has unprefixed IDs and is accepted only in legacy segments.

**Dedup checkpoint version 2:** a 64-byte header. Offsets 0..47 retain the old magic, version/size,
dimension, entry count and previous WAL boundary/checksum. Version/size become 2/64. Offset 48
stores the index checkpoint LSN C; offset 56 stores the header checksum. Entry count can be less
than C. Record layout remains the old 72-byte prefix + key + raw vector + checksum, but the key
includes the eight epoch bytes and has length 9..264. LSNs must be strictly increasing, positive
and at most C; gaps represent expired operations. IDs and keys must be unique; VectorIds must
exist in the checkpoint index. Keys outside the root's retained window and per-epoch quota
violations reject the file, rather than being silently pruned during recovery.

All retained-file integrity checks remain active. SHA-256 is still only a request filter followed
by exact disk-byte comparison; file FNV checksums are corruption guards, not authentication.
The immutable dedup file's generation-qualified name allows the same index LSN to be checkpointed
again with a different epoch window. Reference-based cleanup deletes old artifacts only after
the new recovery root is durable. See [rotation publication rules](wal_rotation.md).

## Verification and measurements

Tests exercise window boundaries, closed/future/expired requests, count and byte caps, same-ID
reuse across epochs, migration and incompatible policy, automatic rotation keeping the epoch,
sparse retained LSNs, corrupt policies/WAL/sidecar epochs with recomputed checksums, and invalid
partial-tail epoch prefixes. Twenty epochs demonstrate stable retained metadata/file size while
all 40 index vectors remain. Fault tests cover pre-publication failure, actual allocation failure
and SIGKILL around expiry publication; old roots do not expire early and durable expiry does not
roll back on restart. Real gRPC tests verify status codes and query/result preservation.

The WAL benchmark accepts `NEW_DIRECTORY COUNT DIMENSION SEGMENT_BYTES EPOCH_OPERATIONS`.
In bounded mode it explicitly checkpoints when the epoch count fills, stores each fixture's
epoch, retries retained requests and verifies refusal of expired requests without index mutation.
Those quota checkpoints are measured separately from AddVector-style commit latency. Full pipeline
time includes generation, quota checkpoints and the halfway checkpoint. Startup/RSS/metadata/disk
measurements retain their previous definitions. See [actual local results](dedup_retention_measurements.md).
