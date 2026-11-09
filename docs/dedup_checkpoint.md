# Disk-backed deduplication and paired checkpoints

This is the first stage of the approved deduplication retention design. It preserves unlimited
historical retry semantics. It does not implement epochs, expiry, WAL rotation or file deletion.
The next stage adds opt-in [WAL rotation](wal_rotation.md) with a version-3 recovery manifest;
the version-2 path described below remains available when rotation is disabled.

## Ownership and exact retries

The in-memory retry entry stores LSN, VectorId, SHA-256 of canonical original float32 bytes,
an offset and a source selector. The operation ID is the map key; dimension is fixed per index.
There is no retained vector in the entry. Source is either the current WAL descriptor or the
current dedup checkpoint descriptor. New records point into WAL; successful checkpoint adoption
switches all entries to the newly verified dedup file. Old files remain on disk in this stage.

A retry first compares VectorId and the request fingerprint. A match then reads the referenced
raw bytes, validates their fingerprint, and compares every byte. SHA-256 is a filter, not the
final equality decision. Signed zero and differently scaled equivalent directions remain distinct
requests. Read/corruption failures stop writes until recovery; a transient C++ allocation failure
while reading a retry does not poison an otherwise unchanged writer. Raw bits are canonical
little-endian IEEE binary32, preserving the existing contract across host byte orders.

Core CMake now requires OpenSSL Crypto for SHA-256, including non-gRPC builds. No index algorithm
is delegated to OpenSSL. Existing file/record FNV-1a-64 checksums still detect accidental corruption;
the manifest and local directory are not authenticated against malicious rewrite.

## Current recovery format

WAL format remains version 1 and Flat snapshots remain version 1. New recovery manifests use
version 2; old 72-byte/version-1 manifests remain readable. The version-2 manifest is 88 bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 8 | Existing `FVREC01\0` magic |
| 8 | 4 | Version 2 |
| 12 | 4 | Length 88 |
| 16 | 8 | Dimension |
| 24 | 8 | Checkpoint LSN C |
| 32 | 8 | Index file bytes |
| 40 | 8 | Whole-index FNV checksum |
| 48 | 8 | WAL byte boundary B after C |
| 56 | 8 | Rolling WAL checksum at B |
| 64 | 8 | Dedup file bytes |
| 72 | 8 | Whole-dedup FNV checksum |
| 80 | 8 | Manifest checksum over bytes 0..79 |

Referenced names are derived as `checkpoint-C.fv` and `dedup-C.fvd`. No arbitrary path is read
from disk metadata. Both files must exist and match their lengths/checksums. Missing/bad dedup
data does not trigger a silent fallback or a partial retry table.

The dedup file starts with a 56-byte header:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 8 | `FVDEDUP\0` |
| 8 | 4 | Version 1 |
| 12 | 4 | Header length 56 |
| 16 | 8 | Dimension |
| 24 | 8 | Entry count, equal to C in insert-only mode |
| 32 | 8 | WAL boundary B |
| 40 | 8 | WAL rolling checksum at B |
| 48 | 8 | Header checksum |

Records are ordered by contiguous LSN 1..C:

| Offset within record | Width | Meaning |
| --- | --- | --- |
| 0 | 8 | Record length: `80 + key_bytes + 4*dimension` |
| 8 | 8 | LSN |
| 16 | 8 | VectorId |
| 24 | 4 | Operation-ID byte count, 1..256 |
| 28 | 4 | Reserved zero |
| 32 | 8 | Dimension |
| 40 | 32 | SHA-256 of original vector bytes |
| 72 | variable | Operation-ID bytes |
| following | `4*dimension` | Original vector bits |
| last | 8 | Record checksum |

All integer fields are unsigned little-endian. The reader validates fixed fields and remaining
file bytes before allocating a record buffer. It verifies checksums/digests, finite nonzero raw
vectors, contiguous LSNs, unique operation IDs and unique vector IDs present in the checkpoint
index. Index count C and the uniqueness checks establish matching ID sets. Trailing bytes are
rejected. The whole dedup file is capped at `max(56, 2*maximum_wal_bytes)`; entry count, dimension
and index budget are also checked. This cap covers the higher fixed overhead at small dimensions.

The original result index size equals the saved LSN because this mode starts empty and only
allows single inserts. Future deletion/import support would need an explicit persisted result.

## Publication and exceptions

Writes remain paused under the coordinator/store lock:

1. Save, reload-verify and fsync the index snapshot; publish its immutable name.
2. Stream original requests from their current disk references into a temporary dedup file.
   Only one record buffer is needed; sorting entry pointers costs O(N) temporary metadata.
3. Fsync and publish the dedup file, then fully validate it and prepare the next in-memory table
   and open descriptor. These allocations occur before manifest publication.
4. Fsync the directory, write/fsync the version-2 manifest, rename it into place and fsync the directory.
5. Adopt the already prepared table/descriptor with swaps. There is no post-publication table build.

Publication uses same-filesystem temporaries and no-overwrite hard links. Same-LSN retries reuse
existing immutable files only when their content matches. Before manifest publication, any error
preserves the old table and recovery root. After rename but before confirmed directory sync, failure
marks recovery required. Normal temporary paths are removed by RAII. SIGKILL may leave unreferenced
files, which are not adopted automatically and are not deleted in this stage.

Temporary verification/index-copy and metadata workspace reservations use the existing loading
budget when configured. Durable index admission now uses `N*(4*d+1024)` units: vector storage plus
retry metadata. This is still an estimate, not exact RSS. Per-operation buffers remain O(d).

## Recovery and compatibility

Version-2 recovery validates/loads the index and dedup files, restores retry metadata, checks that
the WAL is at least B bytes long, then reads records starting exactly at B with expected LSN C+1.
The saved rolling checksum is continued over the suffix for the next checkpoint. Complete suffix
corruption and LSN anomalies remain fatal; only an incomplete final append may be truncated.
The accepted WAL is synchronized before opening service. The WAL header is always validated.

Checkpoint-covered WAL payload is no longer read or verified. Corruption confined to that old
payload does not invalidate a valid paired checkpoint. This is intentional: those bytes are no
longer a recovery source, and historical retries read the dedup file. Truncating the WAL below B
is still rejected. Rotation will replace this historical offset dependency in the next stage.

Without a manifest, recovery scans the entire WAL and creates disk references to its records.
Version-1 manifests use the existing full-scan/prefix verification path, also without retaining
vectors in memory. The next successful checkpoint upgrades the manifest to version 2. Older
binaries reject version 2; downgrade is not automatic. Backups must include the manifest, both
referenced checkpoint files and WAL from a quiescent writer.

## Counters and verification

GetStats durability fields add:

- `dedup_entries`: restored/current operation count.
- `dedup_metadata_bytes`: entry-structure bytes plus operation-ID sizes; excludes hash buckets,
  map nodes, string capacity, allocator overhead and temporary buffers.
- `dedup_file_bytes`: the current dedup checkpoint size; suffix raw requests remain in WAL.
- `wal_scan_bytes`: complete record bytes read during startup, excluding the 40-byte WAL header
  and incomplete tail. Legacy recovery counts all records; version 2 counts only the suffix.

Metadata size no longer depends on vector dimension, but the number of historical entries is
still unbounded until the later retention stage. Disk checkpoint size remains O(Nd). Startup
still validates the entire index and dedup file; reduced WAL scanning does not imply constant
recovery time or guaranteed faster recovery.

Tests cover legacy upgrade, suffix-only recovery, checkpoint-covered WAL corruption, exact
disk-backed retries, dedup field corruption with recomputed outer checksums, dimension-independent
metadata, failed paired publication, disk corruption after load and SIGKILL before/after manifest
replacement. Existing service tests verify recovered retry state through gRPC.

The Release WAL benchmark also reports retry latency, dedup metadata/file bytes, WAL scan bytes
and current Linux RSS immediately after recovery. Its `old_original_vector_payload_bytes` is a
calculated baseline for the removed vector payload, not a measured old RSS or complete old table
size. RSS includes the entire warm process and allocator history. See the actual measurement
report [here](dedup_checkpoint_measurements.md) for commands and results.
