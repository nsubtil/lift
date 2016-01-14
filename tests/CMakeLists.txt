cuda_add_executable(cpuid cpuid.cu EXCLUDE_FROM_ALL)
cuda_add_executable(parallel_tests parallel_tests.cu EXCLUDE_FROM_ALL)

target_link_libraries(cpuid lift ${LINK_LIBS})
target_link_libraries(parallel_tests lift ${LINK_LIBS})

add_custom_target(tests DEPENDS cpuid parallel_tests)
