file(MAKE_DIRECTORY "${WORKSPACE}")
set(input "${WORKSPACE}/input.fvh")
set(output "${WORKSPACE}/output.fvh")
file(REMOVE "${output}" "${WORKSPACE}/empty-output.fvh")

execute_process(COMMAND "${FIXTURE}" "${input}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "snapshot fixture failed")
endif()
file(SHA256 "${input}" input_hash)
execute_process(COMMAND "${REBUILDER}" --input "${input}" --output "${output}" --seed 123
    RESULT_VARIABLE result OUTPUT_VARIABLE report ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT report MATCHES "Output physical nodes: 3" OR
   NOT report MATCHES "Output tombstones: 0" OR NOT report MATCHES "Live-ID verification: match")
    message(FATAL_ERROR "rebuild failed: ${error}\n${report}")
endif()
file(SHA256 "${output}" output_hash)
execute_process(COMMAND "${REBUILDER}" --input "${input}" --output "${output}" --seed 456
    RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
file(SHA256 "${output}" unchanged_hash)
if(result EQUAL 0 OR NOT unchanged_hash STREQUAL output_hash)
    message(FATAL_ERROR "default publication replaced an existing output")
endif()
execute_process(COMMAND "${REBUILDER}" --input "${input}" --output "${output}"
    --seed 456 --overwrite true RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "explicit overwrite failed: ${error}")
endif()
file(SHA256 "${output}" replacement_hash)
if(replacement_hash STREQUAL output_hash)
    message(FATAL_ERROR "explicit overwrite did not publish the changed seed")
endif()
execute_process(COMMAND "${REBUILDER}" --input "${input}" --output "${input}" --overwrite true
    RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
file(SHA256 "${input}" unchanged_hash)
if(result EQUAL 0 OR NOT unchanged_hash STREQUAL input_hash)
    message(FATAL_ERROR "in-place rebuild was not safely rejected")
endif()
file(WRITE "${WORKSPACE}/corrupt.fvh" "corrupt snapshot")
execute_process(COMMAND "${REBUILDER}" --input "${WORKSPACE}/corrupt.fvh" --output "${output}"
    --overwrite true RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
file(SHA256 "${output}" unchanged_hash)
if(result EQUAL 0 OR NOT unchanged_hash STREQUAL replacement_hash)
    message(FATAL_ERROR "corrupt source changed the valid output")
endif()
execute_process(COMMAND "${FIXTURE}" "${WORKSPACE}/empty-input.fvh" --all-deleted
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "all-deleted fixture failed")
endif()
execute_process(COMMAND "${REBUILDER}" --input "${WORKSPACE}/empty-input.fvh"
    --output "${WORKSPACE}/empty-output.fvh" RESULT_VARIABLE result
    OUTPUT_VARIABLE report ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT report MATCHES "Output physical nodes: 0" OR
   NOT report MATCHES "Output live vectors: 0")
    message(FATAL_ERROR "all-deleted rebuild failed: ${error}\n${report}")
endif()
