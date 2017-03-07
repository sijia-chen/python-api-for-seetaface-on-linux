cmake_minimum_required (VERSION 2.6)

project (viplnet)

set (VIPLNET_VERSION_MAJOR 4)
set (VIPLNET_VERSION_MINOR 5)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")

set(VIPLNET_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(VIPLNET_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${VIPLNET_INCLUDE_DIR})
include_directories(${VIPLNET_SRC_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(./src SRC_LIST)
aux_source_directory(./tools TOOLS_LIST)

find_package(OpenCV)
set(facevrf_required_libs ${OpenCV_LIBS})

link_directories(${PROJECT_SOURCE_DIR}/build)
list(APPEND facevrf_required_libs seeta_facedet_lib)
list(APPEND facevrf_required_libs seeta_fa_lib)

add_library(viplnet SHARED ${SRC_LIST} ${TOOLS_LIST})
target_link_libraries(viplnet ${facevrf_required_libs})
set_target_properties(viplnet PROPERTIES 
  VERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR} 
  SOVERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR}) 
