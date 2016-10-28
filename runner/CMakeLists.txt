cmake_minimum_required(VERSION 3.5.1)
project(runner)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/neural-networks
    lib
)

link_directories(
    ${CMAKE_SOURCE_DIR}/build/lib
)

set(SOURCE_FILES main.cpp data-util.h data-util.cpp)
add_executable(runner ${SOURCE_FILES})

target_link_libraries(
    runner
    neural-networks
)

add_dependencies(runner neural-networks)
