cmake_minimum_required(VERSION 3.16)
project(flm_open_embed CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native")

set(FLM_REPO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(FLM_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")

add_subdirectory(${FLM_THIRD_PARTY}/tokenizers-cpp ${CMAKE_BINARY_DIR}/tokenizers-cpp)

add_executable(flm_open_embed
    main.cpp
    engine.cpp
)

target_include_directories(flm_open_embed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${FLM_THIRD_PARTY}/tokenizers-cpp/include
    ${FLM_THIRD_PARTY}/tokenizers-cpp/msgpack/include
)

target_link_libraries(flm_open_embed PRIVATE tokenizers_cpp tokenizers_c)

target_compile_options(flm_open_embed PRIVATE -O3 -march=native -Wall -Wextra)