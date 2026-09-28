# Project showcase guide

## Resume description

Use claims you can explain and support with the repository. A concise version is:

> Built a C++20 vector-search engine with exact cosine Top-K and a from-scratch HNSW index;
> designed contiguous vector storage, deterministic ranking, versioned binary persistence,
> concurrent query serving, gRPC APIs, independent NumPy recall validation, and reproducible
> Docker/Kubernetes deployment.

Add measured numbers only after recording the command, commit, machine, and result. For
example, replace placeholders rather than copying them literally:

> Measured `<QPS>` at `<P95 latency>` on `<CPU/workload>` while maintaining `<Recall@10>` with
> HNSW parameters `<M, efConstruction, efSearch>`.

## Suggested demonstration

1. Explain why normalized cosine search becomes a dot product.
2. Show contiguous row-major storage and deterministic heap-based Top-K.
3. Run the unit tests and NumPy exact validation.
4. Compare Flat and HNSW recall/latency using the same generated dataset.
5. Start the gRPC server, ingest vectors, and perform semantic search.
6. Explain the Docker image boundary and Kubernetes single-replica limitation.

## Interview questions

1. Why does the Top-K heap keep the worst retained candidate at its root?
2. Why is a final sort still required after removing candidates from a priority queue?
3. What locality advantages does one contiguous float buffer provide over nested vectors?
4. Why are vector IDs separate from HNSW node indexes?
5. How do `M`, `efConstruction`, and `efSearch` affect cost and recall?
6. Why must approximate-search performance always be reported with Recall@K?
7. What can an FNV-1a checksum detect, and what security property does it not provide?
8. Why must a loader validate dimensions and payload sizes before allocation?
9. Why can searches share a lock while insertion requires exclusive ownership?
10. Why is batch insertion currently non-transactional?
11. What does service-level QPS include that an in-process benchmark excludes?
12. Why can RSS differ substantially from vector and adjacency payload bytes?
13. Why does the Kubernetes Deployment use one replica and `Recreate`?
14. What is the difference between startup, readiness, and liveness probes?
15. What architectural work is required before safe multi-replica serving?

## Honest scope statement

This project demonstrates an understandable vector-search stack, not a production replacement
for a mature distributed vector database. Its strongest evidence is the connection between
algorithm implementation, independent correctness checks, measured tradeoffs, explicit failure
handling, and documented operational limits.
