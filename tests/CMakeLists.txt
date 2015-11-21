cuda_add_executable(cpuid cpuid.cu)
target_link_libraries(cpuid lift ${tbb_LIB})
