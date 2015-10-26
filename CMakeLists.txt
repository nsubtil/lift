project(lift)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-local")

include(ExternalProject)

option(USE_EXTERNAL_THRUST
       "Build using Thrust from Github (do not use the CUDA toolkit version)"
       ON)

if (DEFINED ENV{LIFT_GPU_ARCHITECTURE})
    set(GPU_ARCHITECTURE $ENV{LIFT_GPU_ARCHITECTURE} CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52)")
else ()
    # allow setting the list of GPU architectures on the command line
    set(GPU_ARCHITECTURE "sm_35,sm_52" CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52)")
endif ()

# build dependencies
if (USE_EXTERNAL_THRUST)
    include("cmake/thrust.cmake")
endif()

include("cmake/cub.cmake")
include("cmake/tbb.cmake")
set(SYSTEM_LINK_LIBRARIES ${SYSTEM_LINK_LIBRARIES} ${tbb_LIB})

# locate required packages
# CUDA 7.0 is required to build
find_package(CUDA 7.5 REQUIRED)
# enable C++11 support
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# enable device lambda support
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

# add nvcc flags to generate code for all enabled architectures
string(REPLACE "," ";" GPU_ARCHITECTURE ${GPU_ARCHITECTURE})
foreach(ARCH IN LISTS GPU_ARCHITECTURE)
    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        --generate-code arch=${COMPUTE},code=${ARCH})
endforeach(ARCH)

include_directories(${CMAKE_SOURCE_DIR})

if (USE_EXTERNAL_THRUST)
    # inject our Thrust include path to override the CUDA toolkit include path
    # (this is a bit of a hack)
    set(CUDA_INCLUDE_DIRS ${thrust_INCLUDE} -I${CUDA_INCLUDE_DIRS})
endif()

find_library(RT_LIBRARY rt)
if (RT_LIBRARY)
    set(SYSTEM_LINK_LIBRARIES ${SYSTEM_LINK_LIBRARIES} ${RT_LIBRARY})
endif()

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()


