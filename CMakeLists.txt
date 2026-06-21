cmake_minimum_required(VERSION 3.15)
project(FraudInfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR")
endif()

include_directories("${ONNXRUNTIME_DIR}/build/native/include")

add_executable(fraud_infer src/main.cpp)

target_link_libraries(
    fraud_infer
    "${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.lib"
)

set_target_properties(fraud_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET fraud_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.dll"
    $<TARGET_FILE_DIR:fraud_infer>
)