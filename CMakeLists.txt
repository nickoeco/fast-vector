cmake_minimum_required(VERSION 3.20)

project(fast-vector VERSION 0.1.0 LANGUAGES CXX)

option(FAST_VECTOR_BUILD_TESTS "Build unit tests" ON)
option(FAST_VECTOR_BUILD_BENCHMARKS "Build benchmarks" ON)
option(FAST_VECTOR_BUILD_VALIDATION "Build the cross-language validation runner" ON)
option(FAST_VECTOR_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FAST_VECTOR_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(FAST_VECTOR_ENABLE_AVX2 "Build the optional x86 AVX2 distance kernel" ON)

if(FAST_VECTOR_ENABLE_SANITIZERS AND FAST_VECTOR_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "Address/Undefined sanitizers and ThreadSanitizer cannot be combined")
endif()

find_package(Threads REQUIRED)

add_library(fast_vector
    src/batch_searcher.cpp
    src/distance.cpp
    src/distance_auto.cpp
    src/flat_index.cpp
    src/flat_index_io.cpp
    src/hnsw_index.cpp
)
add_library(fast_vector::fast_vector ALIAS fast_vector)

target_compile_features(fast_vector PUBLIC cxx_std_20)
target_include_directories(fast_vector
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(fast_vector PRIVATE Threads::Threads)

if(MSVC)
    set_source_files_properties(src/distance_auto.cpp PROPERTIES COMPILE_OPTIONS /fp:fast)
else()
    set_source_files_properties(src/distance_auto.cpp PROPERTIES COMPILE_OPTIONS -ffast-math)
endif()

if(FAST_VECTOR_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_sources(fast_vector PRIVATE src/distance_avx2.cpp)
    target_compile_definitions(fast_vector PRIVATE FAST_VECTOR_HAS_AVX2_IMPL=1)
    if(MSVC)
        set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
    else()
        set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS -mavx2)
    endif()
endif()

function(fast_vector_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(fast_vector_enable_sanitizers target)
    if(FAST_VECTOR_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        else()
            message(FATAL_ERROR "Sanitizers are supported only with GCC or Clang on non-Windows platforms")
        endif()
    endif()
    if(FAST_VECTOR_ENABLE_THREAD_SANITIZER)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
            target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=thread)
        else()
            message(FATAL_ERROR "ThreadSanitizer is supported only with GCC or Clang on non-Windows platforms")
        endif()
    endif()
endfunction()

fast_vector_set_warnings(fast_vector)
fast_vector_enable_sanitizers(fast_vector)

add_executable(fast_vector_demo apps/demo.cpp)
target_link_libraries(fast_vector_demo PRIVATE fast_vector::fast_vector)
fast_vector_set_warnings(fast_vector_demo)
fast_vector_enable_sanitizers(fast_vector_demo)

if(FAST_VECTOR_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(fast_vector_tests
        tests/test_distance.cpp
        tests/test_batch_searcher.cpp
        tests/test_flat_index.cpp
        tests/test_flat_index_io.cpp
        tests/test_hnsw_index.cpp
    )
    target_link_libraries(fast_vector_tests PRIVATE fast_vector::fast_vector GTest::gtest_main)
    fast_vector_set_warnings(fast_vector_tests)
    fast_vector_enable_sanitizers(fast_vector_tests)

    include(GoogleTest)
    gtest_discover_tests(fast_vector_tests)
endif()

if(FAST_VECTOR_BUILD_BENCHMARKS)
    add_executable(fast_vector_benchmark benchmarks/benchmark_flat_index.cpp)
    target_link_libraries(fast_vector_benchmark PRIVATE fast_vector::fast_vector)
    fast_vector_set_warnings(fast_vector_benchmark)
    fast_vector_enable_sanitizers(fast_vector_benchmark)

    add_executable(fast_vector_hnsw_benchmark benchmarks/benchmark_hnsw.cpp)
    target_link_libraries(fast_vector_hnsw_benchmark PRIVATE fast_vector::fast_vector)
    fast_vector_set_warnings(fast_vector_hnsw_benchmark)
    fast_vector_enable_sanitizers(fast_vector_hnsw_benchmark)
endif()

if(FAST_VECTOR_BUILD_VALIDATION)
    add_executable(fast_vector_validate tools/validate_recall.cpp)
    target_link_libraries(fast_vector_validate PRIVATE fast_vector::fast_vector)
    fast_vector_set_warnings(fast_vector_validate)
    fast_vector_enable_sanitizers(fast_vector_validate)
endif()
