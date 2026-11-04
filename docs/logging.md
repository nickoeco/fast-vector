# Structured service logging

`fast_vector_server` writes one application JSON object per line to stderr. There is no
external logging dependency. Redirect stderr to a file or let the container runtime collect it:

```bash
./build-grpc/fast_vector_server --index flat --dimension 128 2>server.jsonl
```

The sink must be consumed continuously. Do not leave a subprocess stderr pipe unread. gRPC
and other dependencies may emit their own diagnostics on stderr; their format is outside this
writer's control. stdout is unused by the server.

## Common fields

Every application record contains these fields. Unavailable values are JSON `null`.

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | string | UTC wall-clock time, ISO 8601 with milliseconds and `Z` |
| `severity` | string | `INFO`, `WARN`, or `ERROR` |
| `event` | string | Stable event name below |
| `request_id` | string or null | Server-generated decimal ID, unique within one service process |
| `method` | string or null | Application RPC method name |
| `status` | string or null | Handler gRPC status name, or lifecycle outcome |
| `latency_us` | integer or null | Elapsed microseconds measured with a monotonic clock |
| `index_type` | string or null | `flat` or `hnsw` |
| `vector_count` | integer or null | Live vector count observed when the record is produced |

Request IDs restart with the process and do not use client headers. Concurrent completion
records may appear in a different order from their IDs. Wall-clock timestamps are for
correlation; elapsed times remain valid across wall-clock adjustments. Vector counts are
observations, not a transactional view tied to the individual request.
RPC logging only attempts to observe a writable index's vector count; it emits `null` when a
writer owns the store lock, so logging metadata does not wait behind insertion. The sink itself
can still block.

## Events

| Event | Additional fields and behavior |
| --- | --- |
| `startup_config` | Effective dimension, kernel, read-only flag, configuration source, reload flag, batch and message limits; HNSW also records M, efConstruction, efSearch, seed, neighbor selection |
| `index_load_started` | `operation`: `startup` or `reload`; status `IN_PROGRESS` |
| `index_load_completed` | Operation, dimension, vector count, total load/validation time; status `OK` |
| `index_load_failed` | Operation, elapsed time, sanitized reason, `kept_previous_index`; status `FAILED_PRECONDITION` |
| `index_replaced` | Live replacement count and dimension after successful Flat or HNSW publication |
| `index_replace_failed` | Publication failure after successful loading; previous snapshot retained |
| `server_started` | Selected listening port after gRPC starts |
| `rpc_completed` | Method, request ID, gRPC status, numeric `status_code`, handler latency |
| `shutdown_requested` | Reason: `SIGINT`, `SIGTERM`, or `internal_stop` |
| `rpc_error_summary` | One record per nonzero error-status count: status, numeric code, `count` |
| `rpc_summary` | `completed_rpcs`, `successful_rpcs`, `failed_rpcs`, `dropped_log_records` |
| `search_executor_summary` | Admission, rejection, terminal counts, queue/running high-water marks, and total queue/execution microseconds |
| `shutdown_completed` | Shutdown reason and final vector count after requests drain |
| `startup_failed`, `server_failed` | Sanitized phase/reason and static command usage |

When loading a snapshot, load events precede `startup_config`: the snapshot supplies the
effective dimension and HNSW parameters. File paths are omitted. Flat and HNSW share the
read-only reload events. `index_load_completed` means loading and validation succeeded;
`index_replaced` confirms atomic publication. A failed load leaves the current index in place.

RPC latency includes application validation, index access, and response construction, ending
before logging. It excludes network transport, protobuf transport serialization, and the log
write itself. It is not client-observed latency. Snapshot latency includes file loading and
validation, but excludes publication into the store and the completion log write.

Only the four application handlers (`AddVector`, `BatchAdd`, `Search`, `GetStats`) contribute
to the counters. The health service and transport failures before handler invocation do not.
The status is the handler's return value; the client may observe another outcome after a
network failure or deadline. Totals cover the process lifetime and are emitted once after
requests drain during orderly shutdown. SIGKILL, a crash, or power loss cannot produce a
reliable shutdown summary. These counters are separate from `VectorStore` statistics.

## Privacy and operational limits

Call sites use an explicit metadata allowlist. Logs exclude vector values, query text,
document chunks, external vector IDs, client metadata/authorization, peer addresses, snapshot
paths, and raw exception messages. Errors use fixed status names and sanitized reason codes.
Do not add arbitrary request fields or `exception.what()` to future records.

One shared writer serializes complete records with a mutex and flushes each write. Control
characters, quotes, and backslashes are escaped. Allocation and stream errors are contained
and increment a dropped-record counter; they do not change the RPC's return status. If the
sink itself is failing, its summary may also be lost, and partial I/O cannot be rolled back.

Logging is synchronous: a slow sink can block handlers and reduce throughput. There is no
async queue, sampling, rotation, remote collector, or cross-process tracing in this stage.
Benchmark the deployed logging sink when measuring service performance. A summary's drop
count is sampled before its own write, so it cannot include failure of that same record.

## Verification

GoogleTest covers JSON escaping, concurrent record integrity, stream failures, unchanged RPC
outcomes on logging failure, and suppression of sensitive internal exception messages. The
Python gRPC suite parses actual server JSON, exercises concurrent success/error responses,
checks exact summary totals, verifies effective snapshot configuration and failed reloads,
and checks SIGINT/SIGTERM shutdown reasons. Neither suite treats synthetic benchmark numbers
as production performance evidence.
