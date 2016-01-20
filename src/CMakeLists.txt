set(lift_lib_sources
    sys/cuda/compute_device_cuda.cu
    sys/host/x86_64/x86_64_cpuid.cu)

cuda_add_library(lift ${lift_lib_sources})
add_dependencies(lift tbb)

set(lift_test_lib_sources
    test/entrypoint.cu)

cuda_add_library(liftest-lib ${lift_test_lib_sources} EXCLUDE_FROM_ALL)
#add_dependencies(liftest-lib lift tbb)

