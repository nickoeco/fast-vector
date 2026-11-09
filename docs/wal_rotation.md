# Checkpoint-bound WAL rotation

This stage adds segmented WAL publication and reference-based cleanup to disk-backed
deduplication. Without [finite retention](dedup_retention.md), all historical operation IDs remain valid.
The optional retention stage adds recovery version 4 and epoch-aware segments/checkpoints.
This document describes version-3 rotation. There is no
group commit, HNSW recovery or background checkpoint worker.

## Enable and migrate

```bash
mkdir -m 700 /tmp/fast-vector-rotating
./build-clang-grpc/fast_vector_server --index flat --dimension 128 \
  --wal-dir /tmp/fast-vector-rotating --wal-rotation true \
  --max-wal-bytes 262144 --max-checkpoint-bytes 2147483648
```

Rotation is opt-in for legacy directories. Existing version-1/version-2 roots remain readable.
The next checkpoint with rotation enabled publishes version 3. Once version 3 is selected, the
persisted segmented mode remains authoritative even if the CLI flag is omitted or false. There
is no automatic downgrade to a legacy full-history WAL.

`maximum_wal_bytes` becomes the per-active-segment cap, including its header. A commit which
does not fit first checkpoints/rotates under the existing exclusive store lock, then appends
the original operation with the same next LSN. An operation larger than an empty segment is
rejected before checkpoint/append. Exact retries never trigger rotation or advance LSN.
Explicit Checkpoint also rotates, including repeated checkpoints without intervening writes.

If an existing legacy WAL is larger than the desired segment cap, first open it with a cap
large enough for that file, enable rotation and checkpoint, then restart with a smaller cap
that still fits the current active segment. Lowering a cap does not truncate files to fit.

`maximum_checkpoint_bytes` (CLI `--max-checkpoint-bytes`, default 2 GiB) bounds the combined
logical index and dedup checkpoint bytes, independently of segment size. The insert-only
pair's exact encoded size is checked before either checkpoint is written. Resource limits and
loading-workspace reservations still apply. This is not a total filesystem quota or free-space
reservation: old and new artifacts must coexist until publication. ENOSPC and sync errors retain
the existing fail-closed rules. A full segment whose checkpoint cannot be published refuses the
new operation; it never deletes required history to make space.

## Publication order

With writes paused at LSN C:

1. Create a unique `wal-G-XXXXXX` file, write/fsync its header, and fsync the directory.
2. Save/verify/fsync and publish `checkpoint-C.fv` and `dedup-C-G.fvd`.
3. Prepare the replacement retry table, dedup descriptor, writer descriptor and root metadata.
4. Write/fsync a version-3 manifest; rename it over `recovery.manifest`; fsync the directory.
5. Swap in the prepared retry table and writer, then release old descriptors.
6. Remove unreferenced engine artifacts and fsync the directory. Resume writes.

New segments start at C+1; LSN never resets. G is a persisted WAL-generation counter and increments
even when C is unchanged. It is unrelated to read-only deployment generation or a future dedup
epoch. A segment filename includes a mkstemp suffix, and its header binds that suffix; abandoned
attempts at the same next generation cannot be mistaken for the published segment.

All post-publication adoption is prepared in advance. An exception after manifest rename, or at
the writer-switch boundaries, stops writes until recovery rather than appending to the old log.
The original operation has not been appended if automatic checkpoint fails before its append.
Cleanup failure is different: the new root/writer is already valid, so Checkpoint may succeed
with `cleanup_pending=true` and subsequent writes can continue. Startup or the next successful
checkpoint retries cleanup.

## Formats

Version-3 recovery manifests are 168 bytes. Fields 0..79 match version 2: magic, version/size,
dimension, checkpoint LSN, index size/checksum, previous WAL boundary/checksum, dedup size/checksum.
Version and size are now 3 and 168.

| Offset | Width | Additional meaning |
| --- | --- | --- |
| 80 | 8 | WAL generation G, positive |
| 88 | 8 | Whole active-segment header FNV checksum |
| 96 | 64 | Active basename, NUL-terminated with zero padding |
| 160 | 8 | Manifest checksum over bytes 0..159 |

The basename must exactly match `wal-<G>-<six ASCII alphanumeric characters>`. No path separators
or arbitrary external filenames are accepted. Index name remains `checkpoint-C.fv`; dedup name
is now `dedup-C-G.fvd`. The dedup file format remains version 1. Its saved boundary/checksum
describe the previous segment, and must agree with the manifest; recovery does not require that
old segment to remain present.

The new segment header is version 2, 72 bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 8 | Existing `FVWAL01\0` magic |
| 8 | 4 | Version 2 |
| 12 | 4 | Header length 72 |
| 16 | 8 | Fixed dimension |
| 24 | 8 | Maximum encoded record length |
| 32 | 8 | Start LSN C+1 |
| 40 | 8 | Generation G |
| 48 | 8 | Parent checkpoint LSN C |
| 56 | 8 | Filename suffix's six ASCII bytes packed little-endian; upper two bytes zero |
| 64 | 8 | Header checksum over bytes 0..63 |

Records retain version-1 encoding. Integer and float-bit fields remain little-endian. Recovery
checks header checksum, root-bound header hash, dimension, start LSN, generation and suffix identity
before replay. A renamed/substituted or incompatible segment is rejected. Checksums/identity are
corruption guards, not source authentication. Only the incomplete final append may be truncated.

## Recovery and retention

The authoritative root is the single `recovery.manifest`. Recovery restores the selected pair,
opens that root's segment, expects C+1, replays and synchronizes the accepted prefix. It does not
search for a newer generation or fall back to a previous checkpoint. Missing/corrupt referenced
segments or an invalid manifest fail startup.

Without a manifest, legacy WAL-only recovery is allowed only when it cannot hide segmented
writes: a nonempty segment, an unsafe segment entry, or segment artifacts without a legacy WAL
cause rejection. Header-only abandoned first-publication candidates may coexist with the intact
legacy WAL, allowing the original recovery chain to work after a failed initial migration.
Do not delete/replace the manifest or mix files from different backups.

Retention keeps exactly the current root's index, dedup file and active WAL among published engine
artifacts. The lock file and manifest remain. Old recovery roots are not retained as automatic
rollback targets: choosing one could lose later acknowledged writes. Historical dedup records
are still present in the current dedup file, so log reclamation does not expire old operation IDs.

Cleanup is restricted to direct entries matching the reserved engine namespace:

- `operations.wal`
- `wal-<digits>-<six alphanumeric characters>`
- `checkpoint-<digits>.fv`
- `dedup-<digits>.fvd` or `dedup-<digits>-<digits>.fvd`
- `.recovery-<six alphanumeric characters>` temporary paths

Files outside these names are left alone. The active reference set is computed from the validated
root; cleanup does not use modification times. No recursive deletion is performed. Matching
symlinks/directories are not followed or removed; they leave cleanup pending for operator action.
The directory must be owned by the effective user and not group/world writable. This is checked
on rotation startup and before cleanup. Use a private Linux filesystem directory; shared `/tmp`
itself is unsuitable. Some Windows mounts do not expose enforceable POSIX permissions; use the
WSL distribution filesystem when necessary. Never place user files/backups in reserved names.

After unlink, directory fsync makes the removal durable. A crash before that sync can leave
obsolete files present again; they are harmless and the next cleanup retries. A valid selected
version-3 root is recovered/synchronized before startup cleanup. Partial old-file deletion cannot
break it. Fault injection covers segment creation, manifest rename, both writer-switch boundaries
and both cleanup boundaries, including SIGKILL.

## Observability and limitations

GetStats adds `wal_generation`, `active_wal_bytes`, `cleaned_files` and `cleanup_pending`.
Generation persists in the root; cleaned count is process-local. Active bytes describe the accepted
writer prefix, and need not include an uncertain failed-append tail when recovery is required.
Checkpoint and recovery timing includes their cleanup attempt. Startup logs report effective WAL
generation and cleanup state, while configuration logs report the requested rotation flag and caps.
Paths, original vectors and operation IDs remain excluded from logs.

Checkpoint/rotation is synchronous: index/dedup validation, fsync and cleanup can increase write
and query tail latency. Historical dedup checkpoint and metadata size still grow with operations;
finite retention must be enabled separately. There is no cluster replication, background compaction or
power-loss test. SIGKILL tests leave kernel/device caches alive.

The benchmark accepts optional segment bytes:

```bash
./build-clang-grpc/fast_vector_wal_benchmark NEW_DIRECTORY 1000 128 65536
```

Rotation mode reports WAL-only comparison fields as null, since prefix history has been removed.
It reports final active bytes, generation, cleanup counts and directory bytes, plus before/after
the explicit halfway checkpoint. Directory bytes are logical file lengths, not physical block
usage. See [actual measurements](wal_rotation_measurements.md) for validation and results.
