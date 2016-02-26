set(lift_lib_sources
    sys/cuda/compute_device_cuda.cpp
    sys/cuda/cuda_context.cpp
    sys/cuda/suballocator.cpp
    sys/host/x86_64/x86_64_cpuid.cpp)

cuda_add_library(lift ${lift_lib_sources})
add_dependencies(lift tbb)

set(lift_test_lib_sources
    test/command_line.cpp
    test/entrypoint.cpp
    test/random.cpp
    test/test_list.cpp)

cuda_add_library(liftest-lib ${lift_test_lib_sources} EXCLUDE_FROM_ALL)
