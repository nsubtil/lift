cuda_add_executable(cpuid cpuid.cu EXCLUDE_FROM_ALL)
target_link_libraries(cpuid lift ${LINK_LIBS})

add_custom_target(tests DEPENDS cpuid)
