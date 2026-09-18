find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

if(TARGET gRPC::grpc++)
    set(FAST_VECTOR_GRPC_TARGET gRPC::grpc++)
    set(FAST_VECTOR_GRPC_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
    find_program(FAST_VECTOR_GRPC_PLUGIN grpc_cpp_plugin REQUIRED)
    set(FAST_VECTOR_GRPC_TARGET PkgConfig::GRPCPP)
endif()

if(TARGET protobuf::protoc)
    set(FAST_VECTOR_PROTOC $<TARGET_FILE:protobuf::protoc>)
else()
    set(FAST_VECTOR_PROTOC ${Protobuf_PROTOC_EXECUTABLE})
endif()

set(FAST_VECTOR_PROTO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(FAST_VECTOR_PROTO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(FAST_VECTOR_PROTO_FILES
    fast_vector/v1/vector_search.proto
)

set(FAST_VECTOR_PROTO_SOURCES)
set(FAST_VECTOR_PROTO_HEADERS)
foreach(proto_file IN LISTS FAST_VECTOR_PROTO_FILES)
    get_filename_component(proto_directory ${proto_file} DIRECTORY)
    get_filename_component(proto_name ${proto_file} NAME_WE)
    set(output_directory ${FAST_VECTOR_PROTO_OUTPUT}/${proto_directory})
    list(APPEND FAST_VECTOR_PROTO_SOURCES
        ${output_directory}/${proto_name}.pb.cc
        ${output_directory}/${proto_name}.grpc.pb.cc
    )
    list(APPEND FAST_VECTOR_PROTO_HEADERS
        ${output_directory}/${proto_name}.pb.h
        ${output_directory}/${proto_name}.grpc.pb.h
    )
    add_custom_command(
        OUTPUT
            ${output_directory}/${proto_name}.pb.cc
            ${output_directory}/${proto_name}.pb.h
            ${output_directory}/${proto_name}.grpc.pb.cc
            ${output_directory}/${proto_name}.grpc.pb.h
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_directory}
        COMMAND ${FAST_VECTOR_PROTOC}
        ARGS
            --proto_path=${FAST_VECTOR_PROTO_ROOT}
            --cpp_out=${FAST_VECTOR_PROTO_OUTPUT}
            --grpc_out=${FAST_VECTOR_PROTO_OUTPUT}
            --plugin=protoc-gen-grpc=${FAST_VECTOR_GRPC_PLUGIN}
            ${FAST_VECTOR_PROTO_ROOT}/${proto_file}
        DEPENDS ${FAST_VECTOR_PROTO_ROOT}/${proto_file}
        COMMENT "Generating C++ protobuf and gRPC sources for ${proto_file}"
        VERBATIM
    )
endforeach()

add_library(fast_vector_grpc_proto ${FAST_VECTOR_PROTO_SOURCES} ${FAST_VECTOR_PROTO_HEADERS})
target_include_directories(fast_vector_grpc_proto PUBLIC ${FAST_VECTOR_PROTO_OUTPUT})
target_link_libraries(fast_vector_grpc_proto PUBLIC protobuf::libprotobuf ${FAST_VECTOR_GRPC_TARGET})
target_compile_features(fast_vector_grpc_proto PUBLIC cxx_std_20)
