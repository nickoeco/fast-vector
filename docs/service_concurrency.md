# Phase 11A: bounded Search execution

The synchronous gRPC Search handler validates the request, copies its query into an owned
task and attempts admission into a bounded FIFO. Fixed computation threads execute through
`VectorStore`. Insertion keeps its existing direct path. `BatchSearcher` remains an index-bound
batch utility, not a service scheduler.

```bash
./build-grpc/fast_vector_server --index flat --dimension 128 \
  --search-workers 2 --search-queue-capacity 64 --search-check-interval 64
```

All limits must be positive. Capacity bounds waiting tasks; at most `workers` tasks are active,
including store-lock acquisition and completion. Each task owns its query, context, cancellation
state, enqueue time and result promise. Workers retain no protobuf memory, caller span, or
`ServerContext*`. A handler polls its future every 2 ms and forwards transport cancellation.

## Admission and completion

| Condition | Application status | Behavior |
| --- | --- | --- |
| Invalid dimension, non-finite or zero query | `INVALID_ARGUMENT` | Reject before admission; k=0 retains the validation bypass |
| Queue at capacity | `RESOURCE_EXHAUSTED` | Refuse without waiting for capacity |
| Executor stopping | `UNAVAILABLE` | Refuse new searches |
| Deadline while waiting | `DEADLINE_EXCEEDED` | Remove on handler notification, or skip on dequeue before store access |
| Client cancellation | `CANCELLED` | Remove a waiting task or notify an active task |
| Shutdown | `UNAVAILABLE` | Cancel waiting tasks and notify active tasks |
| Task exception | Mapped status | Fulfill that task's outcome and keep the worker alive |

Transport outcomes can differ from handler outcomes. Cancellation reasons are first-writer-wins;
a context checks that shared reason before its monotonic deadline. Every admitted task receives
one terminal outcome. Standard and non-standard C++ exceptions remain task-local. Pre-admission
validation is repeated by the index at execution: reload can change dimensions while a task
waits. Invalid/refused requests never enter `VectorStore` search counters.

## Cooperative checks and snapshots

`SearchContext` owns a steady-clock deadline, cancellation state and check interval. The handler
converts the transport's wall-clock deadline once. Flat checks batches of scanned vectors and
result extraction. HNSW checks upper-level neighbors, layer-zero candidate/neighbor visits and
result extraction. Operation boundaries also check. Contexts without cancellation/deadline
avoid clock reads. The interval counts work items, not milliseconds. Dot products, normalization,
allocations, visited-array initialization and sorting are not interrupted in their middle.
Writable-store search uses timed shared-lock attempts, checking every 2 ms while awaiting a
writer. Insertions already modifying the index retain their completion semantics. Custom indexes
must override the context overload for checks during computation; the compatibility default
checks only before/after the legacy call.

A worker checks cancellation after dequeue and then acquires its snapshot inside `VectorStore`.
Waiting tasks retain the store, not an old snapshot. An immutable publication holds both index
and version. Read-only process-local versions start at 1 and increment only on successful
publication; mutable stores return 0. `SearchResponse.snapshot_version` identifies the actual
snapshot searched. `GetStats` observes its own current version. These are not file-format versions,
content hashes, durable identities or multi-process dataset versions. Managed releases now add
content snapshot IDs, deployment generations and embedding labels in the same immutable object.
The optional expected-ID guard belongs to the queued context and is checked against the acquired
publication. See [replica publication](snapshot_publication.md) for the 11B protocol.

## Shutdown

SIGINT/SIGTERM first stops Search admission, settles waiting promises and notifies active tasks.
The server then shuts down gRPC, drains handlers and in-progress insertions, and joins computation
threads. Stop/join are idempotent; destruction also joins. Partial construction joins any threads
already created before propagating its error. An owner must call shutdown, not a computation
worker. This is cooperative shutdown, not a hard time bound for uncooperative custom indexes,
stalled I/O, abrupt process termination, or OS scheduling delays.

## Metrics

`GetStats.search_executor` reports limits, current queued/running counts, lifetime high-water
marks, admitted/rejected/started/completed counts, succeeded/failed/cancelled/expired outcomes,
and summed queue/execution microseconds. Counters are observed together under the queue mutex:

- `admitted = queued + running + completed`.
- `completed = succeeded + failed + cancelled + expired`.
- Started means the task passed its first dequeue check; it may still stop before store access.
- Rejected includes full and stopping. Cancelled includes shutdown. Skipped tasks have zero
  execution time; removed waiting tasks record their time until removal.

Successful responses contain `queue_us` (enqueue through dispatch) and `execution_us` (including
store-lock wait, snapshot acquisition, normalization and retrieval). They exclude handler input
validation/copy, polling, response serialization, logging and network transport. Client latency
includes those additional costs. Error responses do not carry timings, but cumulative scheduler
totals include all admitted terminal tasks. `search_executor_summary` preserves terminal totals
and high-water marks at orderly shutdown without payloads.

Pre-admission dimension reads use the index's fixed-dimension contract. RPC log counts are
best-effort and may be null while a writer holds the store lock; collecting them does not delay
an admission refusal behind insertion.

## Limits

This bounds application computation, not all transport resources. Waiting synchronous handlers
still occupy gRPC threads. Decoding, validation, copies before admission, response buffers and
retained old snapshots are outside the queue bound. Message limits remain separate. Immediate
client retries can amplify overload. Logging is synchronous. Increasing queue capacity can
reduce rejection while raising queue time and tail latency; it does not add compute workers.
No async gRPC, priorities, adaptive pool size, tenant fairness or rate limiter is implemented.

In opt-in Flat WAL mode, AddVector and Checkpoint hold the existing exclusive store lock through
file synchronization/publication. Searches wait on the shared lock, so checkpoint duration and
per-operation sync directly affect query tail latency. Search cancellation cannot interrupt a
kernel filesystem sync. BatchAdd is disabled. Write availability is reported separately in
GetStats durability state; successful read RPCs do not imply that the writer is healthy.

## Verification and reproduction

Controlled tests verify limits, owned input, skipped cancelled/expired tasks, running deadlines,
exception recovery, complete shutdown and dequeue-time snapshot acquisition. Real gRPC tests
verify status mapping, cancellation, recovery and shutdown. Both built-in indexes reject stopped
contexts and remain usable. Allocation tests cover both control-block and publication allocation.

The existing Ubuntu gRPC 1.30/ASan client deadline ABI limitation applies to new C++ test helpers:
they omit client deadlines only in that known sanitizer configuration, while all new cancellation
and shutdown assertions still execute. Release helpers retain deadlines, and core running/dequeue
deadline tests execute under both configurations.

```bash
python scripts/benchmark_service_governance.py \
  --server build-clang-grpc/fast_vector_server --output-dir build/service-governance-11a
./build-clang-grpc/fast_vector_cancellation_benchmark
```

Use a Release build and the documented Python gRPC environment. Each closed-loop matrix case
owns a fresh server. JSON reports preserve commands, build mode/compiler, seed, platform,
successful/attempt QPS, successful-request P50/P95/P99, rejection rate, queue/execution times
and Linux RSS/high-water RSS. Warmup and population are outside timed queries; TLS and snapshots
are not used. Success percentiles exclude refusals; all-request P99 is separate.

The cancellation benchmark uses seed 42, dimension 64, Top-10 and 20 timed queries after three
warmups. Flat has 30,000 vectors. HNSW has 3,000, M=8, efConstruction=32 and deliberately broad
efSearch=3,000. Interval 0 is the ordinary no-context baseline; 1/16/64/256 enable cancellation
and a far-future deadline. For ten trials, a worker-entry signal precedes a 100-us main-thread
sleep and cancellation request. Cancel-to-completion includes wakeup/future overhead, not just
checking instructions. Trials completing without interruption are reported separately.

See [measured development results](service_concurrency_benchmark.md) for the actual Release
matrix, check-frequency observations, environment and changed-file list.
