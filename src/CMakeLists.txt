set(lift_lib_sources
    sys/host/x86_64/x86_64_cpuid.cu)

cuda_add_library(lift ${lift_lib_sources})
add_dependencies(lift tbb)
