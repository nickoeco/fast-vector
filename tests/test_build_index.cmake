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

file(REMOVE "${OUTPUT}")
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
