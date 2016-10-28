cmake_minimum_required(VERSION 3.5.1)
project(neural-networks)

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
)

set(SOURCE_FILES FeedforwardNeuralNetwork.cpp FeedforwardNeuralNetwork.h
  common/util.h common/util.cpp common/TrainSettings.h)
add_library(neural-networks ${SOURCE_FILES})
