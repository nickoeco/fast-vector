# syntax=docker/dockerfile:1

FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libgrpc++-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
        protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /source
COPY CMakeLists.txt LICENSE ./
COPY apps ./apps
COPY cmake ./cmake
COPY include ./include
COPY proto ./proto
COPY service ./service
COPY src ./src
COPY tools ./tools

RUN cmake -S . -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAST_VECTOR_BUILD_GRPC=ON \
        -DFAST_VECTOR_BUILD_TESTS=OFF \
        -DFAST_VECTOR_BUILD_BENCHMARKS=OFF \
        -DFAST_VECTOR_BUILD_VALIDATION=OFF \
    && cmake --build /build --parallel

FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgrpc++1 \
        libprotobuf23 \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin fast-vector

COPY --from=builder /build/fast_vector_server /usr/local/bin/fast_vector_server

USER 10001:10001
EXPOSE 50051
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["nc", "-z", "127.0.0.1", "50051"]

ENTRYPOINT ["/usr/local/bin/fast_vector_server"]
CMD ["--address", "0.0.0.0:50051", "--index", "hnsw", "--dimension", "384", "--max-batch-size", "1000", "--kernel", "auto", "--m", "16", "--ef-construction", "200", "--ef-search", "100", "--hnsw-seed", "42", "--neighbor-selection", "heuristic"]
