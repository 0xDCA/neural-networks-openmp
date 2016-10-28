cmake_minimum_required(VERSION 3.4)

project(neural-network-test)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/neural-networks
    lib
)

link_directories(
    ${CMAKE_SOURCE_DIR}/build/lib
)

add_executable(
    neural-network-test
    ApproxMatrixMatcher.h
    ApproxMatrixMatcher.cpp
    main.cpp
)

target_link_libraries(
    neural-network-test
    neural-networks
)

add_dependencies(neural-network-test neural-networks)
