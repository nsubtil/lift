project(lift)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-local")
set(CMAKE_CXX_STANDARD 11)

include(ExternalProject)

option(USE_EXTERNAL_THRUST
       "Build using Thrust from Github (do not use the CUDA toolkit version)"
       ON)

# parent project might define GPU architecture
if (NOT DEFINED GPU_ARCHITECTURE)
    if (DEFINED ENV{LIFT_GPU_ARCHITECTURE})
        set(GPU_ARCHITECTURE $ENV{LIFT_GPU_ARCHITECTURE} CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52, sm_61)")
    else ()
        # allow setting the list of GPU architectures on the command line
        set(GPU_ARCHITECTURE "sm_30,sm_35,sm_52,sm_61" CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_35, sm_52, sm_61)")
    endif ()
endif ()

# build dependencies
if (USE_EXTERNAL_THRUST)
    include("cmake/thrust.cmake")
endif()

include("cmake/cub.cmake")
include("cmake/tbb.cmake")

# locate required packages
# CUDA 7.0 is required to build
find_package(CUDA 7.0 REQUIRED)
# enable C++11 support
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# enable device lambda support
#list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

# add nvcc flags to generate code for all enabled architectures
string(REPLACE "," ";" GPU_ARCHITECTURE ${GPU_ARCHITECTURE})
foreach(ARCH IN LISTS GPU_ARCHITECTURE)
    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        --generate-code arch=${COMPUTE},code=${ARCH})
endforeach(ARCH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# xxxnsubtil: https://github.com/nsubtil/lift/issues/60
#if (USE_EXTERNAL_THRUST)
    # inject our Thrust include path to override the CUDA toolkit include path
    # (this is a bit of a hack)
#    set(CUDA_INCLUDE_DIRS ${thrust_INCLUDE} -I${CUDA_INCLUDE_DIRS})
#endif()

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()

include("gcc.cmake")

include_directories(${INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)

get_directory_property(has_parent_scope PARENT_DIRECTORY)
if (has_parent_scope)
    # propagate link and include flags to parent scope
    set(LIFT_LINK_LIBRARIES lift ${LINK_LIBS} PARENT_SCOPE)
    set(LIFT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS} PARENT_SCOPE)
else()
    # call into cuda_build_clean_target() only if there is no parent project
    # this is because cuda_build_clean_target() can only be called once per tree, so
    # it must always be called by the root-level CMakeLists
    cuda_build_clean_target()
endif()
