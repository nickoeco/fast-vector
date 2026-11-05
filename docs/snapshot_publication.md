# Phase 11B: read-only multi-replica publication

An operator prepares immutable snapshot bytes and a JSON manifest, then uses an explicit
replica list to roll the release. The tool reads local files or HTTP(S)/file URLs separately
for each replica and relays bounded chunks over gRPC. Each replica independently verifies,
loads and atomically activates its own candidate. There is no leader, quorum or consensus.

## Identity and atomic ownership

| Field | Scope |
| --- | --- |
| `format_version` | Binary parser version: Flat 1, HNSW 1/2/3 |
| `snapshot_id` | `sha256:` plus the lowercase SHA-256 of the exact binary file |
| `generation` | Positive operator-assigned release sequence, independent of data identity |
| `embedding_revision` | Optional opaque vector-semantics compatibility label |
| `snapshot_version` | Retained process-local successful-publication counter from 11A |

Generation belongs to the deployment command, not to the immutable file manifest. Rollback
publishes a higher generation pointing at the old snapshot ID and bytes. A `ServingSnapshot`
owns the index, process-local counter and immutable identity together. The atomic shared
pointer never publishes data and identity separately. Queries acquire one publication only
when their SearchExecutor task starts and return the identity copied from that publication.

The manifest requires `format_version`, `snapshot_id`, `index_type`, `dimension`, live
`vector_count`, `file_size` and `sha256`; `embedding_revision` is optional. HNSW vector count
excludes tombstones. SHA-256 binds bytes; the existing binary checksum and full structural
loader validation are also required. SHA-256 does not authenticate the publisher or establish
that an embedding label is truthful. Two byte-identical files have the same snapshot ID even
if an operator changes the embedding label in a later generation.
JSON manifests and protobuf SnapshotManifest messages are each limited to 16 KiB, including unknown
protobuf fields. Embedding labels are limited to 256 UTF-8 bytes.

## Bootstrap and opt-in management

The gRPC build additionally links OpenSSL Crypto (`libssl-dev` on Ubuntu). Core-only builds
retain their standard-library-only runtime. The JSON helper uses protobuf's existing JSON
parser. The Python manifest creator needs only the standard library; the publisher uses the
existing Python gRPC requirements.

Create a Flat snapshot and immutable descriptor:

```bash
./build-grpc/fast_vector_build_index --input tests/data/vectors.jsonl \
  --output build/release-a.fv --dimension 3 --expected-count 3
python scripts/create_snapshot_manifest.py --index build/release-a.fv --index-type flat \
  --dimension 3 --vector-count 3 --embedding-revision model-at-fixed-revision \
  --output build/release-a.manifest.json
```

The creator hashes bytes and checks magic/version, but does not replace complete replica-side
validation. Supply actual live dimensions/counts. Output is created exclusively; existing
manifests are not overwritten. Keep both file and manifest immutable after creation.

Start each replica with its own address and explicit bootstrap generation:

```bash
./build-grpc/fast_vector_server --address 127.0.0.1:50061 --index flat \
  --load-index build/release-a.fv --snapshot-manifest build/release-a.manifest.json \
  --generation 1 --read-only true --enable-snapshot-publication true \
  --snapshot-staging-dir /tmp --max-snapshot-bytes 1073741824
```

Repeat for ports 50062 and 50063. For HNSW use its binary file, `--index hnsw`, and a descriptor
with the actual live count. Startup caches one manifest, copies the bootstrap file to a private
temporary file, verifies and loads it before listening, and removes the temporary copy.
The staging directory must already exist and be writable by the service user. Containers
with read-only filesystems need an explicit writable staging volume.

Management is disabled by default. Enabling it requires read-only mode, a load path, manifest
and positive generation. It cannot coexist with SIGHUP reload. Legacy unversioned SIGHUP
continues to work without these management options; a managed store rejects legacy replacement.
`--max-snapshot-bytes` defaults to 1 GiB and bounds incoming file bytes, not the loaded graph's RSS.

## Prepare and activate

`PrepareSnapshot` is client-streaming. The first frame contains only a manifest and generation.
Following frames contain at most 256 KiB of data each. The publisher sends an abort frame on
download failure. Repeated headers, missing/oversized/extra bytes, aborts and truncated streams
fail preparation. The transport message bound remains independent of the total file bound.
Private staging files use exclusive creation with owner-only permissions and are removed on
normal completion and exceptions; abrupt process death can leave temporary files to clean up.

Only one admitted preparation may run per replica. A second receives `RESOURCE_EXHAUSTED`.
When READY, a newer accepted preparation drops the old candidate before loading another.
Queries do not acquire the manager mutex and continue on the current serving index. Transfer,
hash verification and loading execute in the management handler, separately from SearchExecutor.
Preparation still competes for CPU/memory with live traffic; it is not free background work.

State progression is `SERVING -> PREPARING -> READY_TO_ACTIVATE -> SERVING`. An admitted
failure clears the candidate, reports `FAILED`, and continues serving the previous publication.
Rejected commands do not erase a valid PREPARING/READY candidate. Internal preparation tokens
prevent a late failure from erasing another attempt's candidate. Rejected metadata is bounded
and sanitized before it enters status records.

`ActivateSnapshot` requires the exact READY snapshot ID/generation. The index and metadata
object are allocated during prepare, so activation performs the guarded atomic pointer store.
A repeat activation of the already-current identity is idempotent, unless a higher generation
has been accepted. The old shared pointer is retained across the measured store interval;
final-reference reclamation may add time before the RPC returns.

Accepted generations form a process-local high-water mark, including failed preparations.
Older commands are rejected with `FAILED_PRECONDITION`. A failed generation can be retried
with the same manifest; changing its identity/shape/revision requires a newer generation.
Concurrent publishers can still cause partial releases; there is no global lock or election.

## Publisher and observed convergence

Create `build/replicas.json` with individually addressable replicas, not three aliases of a
load balancer:

```json
[
  {"name": "replica-a", "address": "127.0.0.1:50061"},
  {"name": "replica-b", "address": "127.0.0.1:50062"},
  {"name": "replica-c", "address": "127.0.0.1:50063"}
]
```

Generate Python protobuf modules, create a second immutable artifact/manifest, then publish:

```bash
python scripts/generate_grpc_python.py --output-dir build/python-grpc
python scripts/publish_snapshots.py --manifest build/release-b.manifest.json \
  --snapshot build/release-b.fv --generation 2 --replicas build/replicas.json \
  --generated-dir build/python-grpc --report build/release-2.json
```

Both source arguments also support HTTP(S) and file URLs. A replica entry can override
`snapshot_url` for that target. Each target has its own download/transfer, validation and
activation attempt; one failure does not prevent attempts on the remaining replicas. The
tool reuses matching READY candidates and skips already-current targets. It then re-observes
every replica. Convergence requires matching snapshot ID, generation and embedding revision,
no observation error, enabled management and no higher accepted generation. Even when an older
version is still serving everywhere, a stale command rejected by a higher fence is not success.

Exit 0 means observed convergence; exit 1 means failure/unconverged or invalid release input.
JSON contains per-replica phase, error codes/download error class, current/candidate identity,
fence, last result and metrics. No raw source URL or exception text is included in reports.
An unavailable replica produces an observation error rather than an assumed version. Failed
convergence has `convergence_us: null`; `elapsed_us` still describes the attempted rollout.

To roll back, use the same command with release A's original manifest/file and generation 3.
Do not decrease generation. A lost response does not prove prepare/activation failed: inspect
status or rerun the same target. Activation and client timeout cannot form a distributed
transaction; the final observation is what confirms the release outcome.

## Query and status contracts

`SearchRequest.expected_snapshot_id` is optional and owned by the queued task. The store
compares it against the actual acquired snapshot before retrieval. A mismatch returns
`FAILED_PRECONDITION`, including k=0. It detects mismatch but neither routes to another
replica nor pins a generation. Successful responses return `snapshot_id`, `generation` and
`embedding_revision` alongside existing results/timing fields. Legacy/mutable identities
remain empty/generation 0. Clients must select an appropriate embedding model themselves.

`GetStats.publication` reports state, current/candidate identity, high-water generation,
last attempted identity/result, preparation/activation times and peak process RSS. Current
identity and index size/dimension are observed together under the manager's activation lock;
queries remain independent of that lock. Health stays SERVING while a valid old index is
available, including after preparation failure. Health alone does not confirm target generation.

## Measurements and limits

`prepare_us` spans admission of the manifest through transfer completion, digest/format/shape
validation and full load/metadata allocation. `activation_us` measures the guarded pointer store,
excluding old-index reclamation, response serialization and transport. Client wall times for
both RPCs are also reported. A reused/already-serving target has zero new preparation time.
`convergence_us` spans this invocation through final observations, including transfers and
all sequential targets. It starts after input manifest/list parsing and excludes initial
descriptor retrieval. Linux `peak_rss_kib` is whole-process lifetime high-water RSS, not
candidate-only memory or an isolated per-release delta. Old readers may retain old indexes.
Separate manual retries start separate timers; the tool does not persist a cross-invocation job.

This is rolling, observed eventual convergence. Replicas may serve different versions during
rollout or indefinitely after partial failure. There is no cross-replica atomic switch,
writable replication, automatic leader, fixed-version gateway routing, source authentication
or management TLS/authentication in this stage. Management must be explicitly enabled and
restricted by the operator's deployment boundary.
Management RPCs use synchronous gRPC draining and the publisher's configured deadlines;
there is no new hard server-side control-RPC timeout or interruption inside binary loading.

Control state and active publications are in memory. No release journal or automatic restart
recovery is provided: restart must use the current immutable artifact/manifest/generation in
the operator's configuration. Restarting with an older bootstrap resets the fence and can
accept commands the former process would have rejected. This is not a durable control plane.

## Verification

`tests/test_snapshot_publication.cpp` verifies digest/format/shape validation, full loader checks,
candidate/fence behavior, retry, rollback, stale-token protection, disabled management and staging
cleanup. `tests/snapshot_publication_test.py` runs three actual processes for Flat, then three for
HNSW, while querying throughout. It covers corruption, truncation, wrong format, HTTP 404,
partial convergence, same-generation retry, READY recovery, rollback, stale activation,
expected-ID rejection, CLI reports, health, result/identity pairing and process/file cleanup.

```bash
FAST_VECTOR_SERVER=build-clang-grpc/fast_vector_server \
FAST_VECTOR_BUILDER=build-clang-grpc/fast_vector_build_index \
FAST_VECTOR_HNSW_FIXTURE=build-clang-grpc/fast_vector_hnsw_snapshot_fixture \
FAST_VECTOR_PUBLICATION_REPORT_DIR=build/publication-11b \
  build/python-grpc-venv/bin/python -m unittest -v tests/snapshot_publication_test.py
```

The CI gRPC job runs the same test and uploads JSON reports. Local measurements and their
scope are recorded in [publication measurements](snapshot_publication_measurements.md).
