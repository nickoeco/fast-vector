# Deletion: Flat swap-and-pop and HNSW tombstones

## Phase 10A contract

`FlatIndex::erase(VectorId)` removes one live vector and returns `true`. An absent ID,
including a repeated deletion, returns `false` without changing the index. A removed ID can
be added again. Dimension and dot-product kernel are unchanged. The operation requires
exclusive external access, just like insertion. Deletion remains a concrete-index operation;
`VectorIndex`, `VectorStore`, and protobuf do not expose it.

## Storage decision

The Flat index owns a float array, an ID array, and an `unordered_map<VectorId, size_t>`.
After every successful operation:

- `vectors_.size() == ids_.size() * dimension_`.
- `id_to_row_.size() == ids_.size()`.
- For every live row `i`, `id_to_row_.at(ids_[i]) == i`.
- There are no holes or tombstones in the live arrays.

Stable deletion would shift all later rows and update their map entries, costing
`O((N - i)d)` work for a row at position `i`. Swap-and-pop copies at most the last `d`
floats into the deleted row, copies its ID, updates that existing map entry, then removes
the old tail and deleted map entry. Average work is `O(d)` with average constant-time hash
lookup; pathological hash collisions can cost `O(N + d)`. No scratch vector is allocated.
Deleting the last row needs no copy. Empty and missing-ID cases are handled before computing
the final row. The map replaces, rather than supplements, the old duplicate-ID set.

Deletion does not insert map entries or rehash: all lookups happen before changing arrays,
and the moved entry is updated through its iterator. Float/ID assignments and shrinking the
arrays require no allocations. Insertion retains rollback if an array append fails.
Search sorts by score and then external ID, so internal row reordering cannot change the
documented result order. Container capacity may stay allocated; deletion neither guarantees
lower RSS nor securely wipes removed bytes.

## Persistence and validation

Flat version 1 already stores the current row count, IDs, and vector components, not a hash
table. The writer needs no format change: only live rows are saved. Loading validates the
existing format and reconstructs ID-to-row entries while rejecting duplicate IDs. An existing
snapshot is not mutated by an in-memory deletion. Callers must explicitly save/publish again.
Different deletion histories may produce different row order and file bytes while returning
the same exact search results.

Tests cover first/middle/last/only-row removal, deletion of a moved ID, empty and absent IDs,
ID zero and uint64 maximum, ID reuse, tie ordering, negative scores, and 300 seeded mixed
insertion/deletion steps checked against a freshly built index of surviving vectors. Snapshot
tests check compact file sizes, empty snapshots, query equivalence, post-load deletion and
reinsertion, and preservation of an older file before explicit resaving.

## HNSW source findings

The [HNSW paper](https://arxiv.org/abs/1603.09320) describes layered graph navigation and
neighbor selection. The upstream [hnswlib API](https://github.com/nmslib/hnswlib#api-description)
supports marking an item deleted so it is omitted from results. Its
[implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h), inspected on
2026-10-22, keeps the graph when marking deletion. In `searchBaseLayerST`, deleted nodes can
enter the exploration queue while only eligible nodes enter the result queue. These sources
inform the proposal below; this project does not use or copy hnswlib as a search dependency.

## Phase 10B: HNSW tombstones

`HnswIndex::erase(id)` keeps NodeIndex values, vectors, entry point, and edges fixed, marks
`Node::deleted`, and increments the deletion counter. Absent/already-deleted IDs return
`false`; successful first deletion returns `true`. The operation needs exclusive external
access and performs no allocation, RNG draws, or reconnection. ID mappings are retained, so
insertion rejects reuse of tombstoned IDs. `size()` is the live count. `stats().node_count`
is the physical count and equals `live_count + deleted_count`. New insertion and visited
arrays use physical counts, not live counts, including after deletion of every old vector.

"Skip deleted nodes" must mean exclusion from returned results, not unconditional exclusion
from traversal. A deleted node can be a bridge or the entry point. Upper-layer navigation
continues to use it, while `search_layer(..., live_only=true)` admits only live nodes to its
best-result queue. Frontier admission and termination use that live queue. Underfilled live
queues allow traversal through tombstones, rather than counting them as results and filtering
after Top-K selection. Search targets at most `min(k, live_count)` results and returns empty
for an all-deleted index after normal query validation. Construction uses the same layer
search with `live_only=false`, so insertion can connect through existing tombstones and
uses normal pruning. Approximate search still cannot promise exact recall.

Flat swap-and-pop cannot be applied directly to the HNSW graph: every inbound edge referencing
the moved/deleted NodeIndex would need repair. Tombstones retain these references but do not
reclaim vectors or adjacency memory. HNSW v3 keeps the 112-byte header with a physical node
count. After the existing graph payload it stores a little-endian uint64 live count and one
uint8 flag per physical node, then the v2 length-prefixed RNG block. FNV-1a covers the new
fields too. Only flags 0/1 are valid; observed live count must match the stored count. A
deleted entry point is valid because its physical node remains navigable. HNSW v1/v2 files
load as all-live; v1 reseeds RNG while v2 restores its state. Saving preserves all physical
rows and tombstones, including an all-deleted graph. Snapshotting requires mutation exclusion.

Tests cover missing/repeated deletions, reserved ID reuse, tie and negative-score ordering,
all-deleted queries, insertion after deletion, both neighbor selection policies, fixed-query
round trips, byte-identical graph/RNG preservation on deletion, invalid flags/counts even
with a recomputed checksum, and legacy v1/v2 loading. A controlled chain with a deleted entry
and bridge must still reach its sole live node. A connected 48-node graph with 75% tombstones
is compared with exact survivor-only Flat results; this is a correctness fixture, not a
general recall guarantee. The gRPC startup fixture also contains a tombstone to check that
loaded searches and vector counts exclude it without adding a deletion RPC.

## Insertion exception guarantee

Single-vector HNSW insertion has a strong logical exception guarantee, including allocation
failure. On exception, IDs, vector bits, graph topology, deletion state, entry/max level, and RNG
state match the pre-call state. Capacity, hash bucket layout, and RSS need not match. Configuration
validation uses the same uint32-based `M` limit as the snapshot format. Batch insertion has no
cross-vector transaction: earlier successful calls remain committed if a later call fails.

An `InsertionTransaction` saves RNG and original array sizes and journals an adjacency list
before its first write. This includes both owner-list insertion and reverse-link removal during
pruning. On failure, vector swaps restore original lists, appended storage is removed, a newly
inserted map entry is erased, and RNG/entry state is restored. Rollback does not allocate. The
journal's temporary memory depends on affected adjacency lists, not a full copy of the index.

The dedicated failure-test process overrides ordinary C++ new/delete only in that executable.
It rejects the nth allocation and keeps rejecting until the guarded operation unwinds, exercising
rollback under persistent allocation failure. Each covered allocation point is checked until
the operation first succeeds. Tests compare serialized bytes (including RNG), live IDs, queries,
and retry results against an uninterrupted control. Scenarios include first insertion, pruning
with both neighbor policies and tombstones, and source preservation when rebuild fails. This
does not simulate process termination, disk errors, or arbitrary operating-system failures.

## Offline compact/rebuild

`source.rebuild(seed)` returns a new index by inserting surviving normalized vectors in
ascending external-ID order. It preserves vector bits without renormalization, retains runtime
configuration/kernel except for the explicit new seed, and leaves the source unchanged on success
or failure. Its new graph has dense NodeIndex values, no tombstones, and no reservations for
deleted IDs. Empty/all-deleted inputs produce empty indexes. A fresh level sequence creates new
topology, so approximate results need not match the old graph. Repeating the same rebuild with
the same configuration and seed is deterministic on the supported execution setup.

On Linux, `fast_vector_rebuild_hnsw --input SOURCE --output DESTINATION [--seed UINT64]`
loads and validates the snapshot, rebuilds, checks exact live-ID/count equality, and publishes
through `publish_hnsw_index`. The output must differ from the source, including existing aliases;
default publication refuses overwriting. `--overwrite true` explicitly replaces a different
destination. Source files remain unchanged. Publication retains the existing caveat: failure of
directory fsync after replacement cannot roll back the already-published directory entry.

The CLI reports load/rebuild/validated-publication times, input/output file sizes, process RSS
after loading, with both indexes, and after publication validation, plus peak process RSS.
Both indexes coexist during rebuild, and validation loads a third index temporarily. Allocator
retention can keep RSS high even after temporary indexes are destroyed; these readings must not
be described as the isolated memory size of the compacted index. The snapshot and payload byte
measurements quantify storage reduction separately. Unit tests preserve exact vector bytes,
verify deterministic snapshots and post-load queries, and check source/output file protection.

Read-only serving can load compacted HNSW snapshots through its SIGHUP reload path. Publish to
the configured snapshot path, then signal the server to acquire and atomically install the
replacement. Writable serving still needs a write-exclusion or replay design to avoid losing
writes. This offline tool itself does not signal or install an index in a running server.

## Deletion-ratio experiments

The Release HNSW benchmark accepts `--delete-percent 0..99` (default 0). It shuffles IDs with
a separate RNG seeded from the reported data seed and deletes a prefix from both indexes.
This keeps datasets, queries, graph construction, and nested deletion sets comparable across
ratios. Ground truth is computed after Flat deletion. The benchmark reports live/deleted counts,
HNSW marking time, RSS after deletion, and short-result query count alongside Recall@K and
latency/QPS. Deleted or duplicate output IDs cause failure. Integrity checks run outside the
timed search region. At most 99% can be deleted in this benchmark to keep recall defined;
unit tests cover the all-deleted case. RSS includes data, queries, both indexes, and benchmark
bookkeeping; it is not tombstone memory alone.

```bash
for ratio in 0 25 50 75; do
  ./build-release/fast_vector_hnsw_benchmark \
    --vectors 1000 --dimension 32 --queries 100 --k 10 --warmup 20 \
    --m 16 --ef-construction 100 --ef-search 10,50 \
    --seed 20250908 --hnsw-seed 42 --delete-percent "$ratio"
done
```

Add `--rebuild true` to compare the rebuilt graph on the same queries, live IDs, and exact ground
truth. Output includes rebuilt counts, snapshot bytes, vector/ID payload bytes, rebuild time,
process RSS while both indexes coexist, and a second Recall/latency/QPS table. Structured deletion
and larger workload studies remain future work.

Use fixed vectors, queries, build seeds, and deletion sets. At ratios 0%, 10%, 25%, 50%, and
75%, compare tombstone search and a rebuilt graph over exactly the same live IDs. Include
uniform deletion and structured deletion of a cluster or high-degree/upper-layer nodes.
For each ratio, build exact ground truth from surviving vectors using FlatIndex deletion.
Measure several `efSearch` values with the same `M`, dimension, kernel, query count, and machine.

Report Recall@K against the live ground truth, fraction of queries returning fewer than
`min(k, live_count)` results, deleted-ID leakage (must be zero), latency percentiles, QPS,
distance evaluations, rebuild time, file size, and RSS. Use `min(k, live_count)` as the recall
denominator; for an all-deleted index, assert empty results and report recall as not applicable.
Do not compare recall against pre-deletion neighbors. Deleted nodes can remain useful routes
but increase work without yielding results; the net latency and recall effects depend on the
data and search budget. No universal rebuild threshold is claimed without these measurements.
