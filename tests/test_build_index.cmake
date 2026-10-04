if(NOT DEFINED BUILDER OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "BUILDER, INPUT, and OUTPUT are required")
endif()

file(REMOVE "${OUTPUT}")
file(GLOB stale_temporary_files "${OUTPUT}.tmp-*")
if(stale_temporary_files)
    file(REMOVE ${stale_temporary_files})
endif()

execute_process(
    COMMAND "${BUILDER}"
        --input "${INPUT}"
        --output "${OUTPUT}"
        --dimension 3
        --expected-count 3
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0 OR NOT EXISTS "${OUTPUT}")
    message(FATAL_ERROR "index build failed: ${build_output}${build_error}")
endif()
file(SHA256 "${OUTPUT}" original_hash)

execute_process(
    COMMAND "${BUILDER}"
        --input "${INPUT}"
        --output "${OUTPUT}"
        --dimension 3
    RESULT_VARIABLE overwrite_result
)
if(overwrite_result EQUAL 0)
    message(FATAL_ERROR "builder unexpectedly overwrote an existing output")
endif()

execute_process(
    COMMAND "${BUILDER}"
        --input "${INPUT}"
        --output "${OUTPUT}"
        --dimension 3
        --overwrite yes
    RESULT_VARIABLE invalid_overwrite_result
)
if(invalid_overwrite_result EQUAL 0)
    message(FATAL_ERROR "builder accepted an invalid --overwrite value")
endif()

set(replacement_input "${OUTPUT}.replacement.jsonl")
file(WRITE "${replacement_input}" "{\"id\": 999, \"values\": [0.0, 0.0, 1.0]}\n")
execute_process(
    COMMAND "${BUILDER}"
        --input "${replacement_input}"
        --output "${OUTPUT}"
        --dimension 3
        --expected-count 1
        --overwrite true
    RESULT_VARIABLE replacement_result
    OUTPUT_VARIABLE replacement_output
    ERROR_VARIABLE replacement_error
)
if(NOT replacement_result EQUAL 0 OR NOT EXISTS "${OUTPUT}")
    message(FATAL_ERROR "atomic replacement failed: ${replacement_output}${replacement_error}")
endif()
file(SHA256 "${OUTPUT}" replacement_hash)
if(replacement_hash STREQUAL original_hash)
    message(FATAL_ERROR "replacement did not change the snapshot")
endif()

execute_process(
    COMMAND "${BUILDER}"
        --input "${INPUT}"
        --output "${OUTPUT}"
        --dimension 3
        --expected-count 4
        --overwrite true
    RESULT_VARIABLE failed_replacement_result
)
if(failed_replacement_result EQUAL 0)
    message(FATAL_ERROR "invalid replacement unexpectedly succeeded")
endif()
file(SHA256 "${OUTPUT}" preserved_hash)
if(NOT preserved_hash STREQUAL replacement_hash)
    message(FATAL_ERROR "failed replacement modified the existing snapshot")
endif()

file(REMOVE "${OUTPUT}")
file(REMOVE "${replacement_input}")
execute_process(
    COMMAND "${BUILDER}"
        --input "${INPUT}"
        --output "${OUTPUT}"
        --dimension 3
        --expected-count 4
    RESULT_VARIABLE count_result
)
if(count_result EQUAL 0 OR EXISTS "${OUTPUT}")
    message(FATAL_ERROR "expected-count mismatch did not fail cleanly")
endif()
file(GLOB temporary_files "${OUTPUT}.tmp-*")
if(temporary_files)
    message(FATAL_ERROR "builder left temporary files after failure: ${temporary_files}")
endif()
