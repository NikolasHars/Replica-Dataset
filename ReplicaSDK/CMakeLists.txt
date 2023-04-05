# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
cmake_minimum_required(VERSION 2.6.0)

project(ReplicaSDK)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../cmake")

set(PTEX_SHADER_DIR "${CMAKE_CURRENT_LIST_DIR}/shaders" CACHE PATH "Where the shaders live")

file(GLOB srcs_ptex ptex/*.cpp)
file(GLOB hdrs include/*.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2 -msse3 -Wall -std=c++11 -fopenmp -DSHADER_DIR=${PTEX_SHADER_DIR}")

list(APPEND CMAKE_PREFIX_PATH "/cluster/project/infk/courses/252-0579-00L/group04/yaml-cpp/build")
list(APPEND CMAKE_PREFIX_PATH "/cluster/project/infk/courses/252-0579-00L/group04")
find_package(yaml-cpp REQUIRED)

add_library(ptex SHARED
            ${srcs_ptex}
            ${hdrs}
)

target_link_libraries(ptex
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${dl_LIBRARIES}
                      stdc++fs
)
target_include_directories(ptex PUBLIC
            ${Pangolin_INCLUDE_DIRS}
            ${Eigen_INCLUDE_DIRS}
            ${dl_INCLUDE_DIRS}
            ${CMAKE_CURRENT_LIST_DIR}
)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${dl_INCLUDE_DIRS})
include_directories("./include")
include_directories("/cluster/project/infk/courses/252-0579-00L/group04/yaml-cpp/include")
include_directories("/cluster/project/infk/courses/252-0579-00L/group04/yaml-cpp")

add_executable(ReplicaViewer src/viewer.cpp src/MirrorSurface.cpp)

target_link_libraries(ReplicaViewer
                      ${Pangolin_LIBRARIES}
                      ${dl_LIBRARIES}
                      GL
                      GLEW
                      ptex
                      stdc++fs
)

add_executable(ReplicaRenderer src/render.cpp src/MirrorSurface.cpp)

target_link_libraries(ReplicaRenderer
                      yaml-cpp
                      ${Pangolin_LIBRARIES}
                      ${dl_LIBRARIES}
                      GL
                      GLEW
                      ptex
                      stdc++fs
)
