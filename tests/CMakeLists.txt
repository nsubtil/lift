cuda_add_executable(cpuid standalone/cpuid.cu EXCLUDE_FROM_ALL)
target_link_libraries(cpuid lift ${LINK_LIBS})

set(liftest_src
    harness/liftest.cu
    harness/test_harness.h
    harness/test_list.cu)

set(test_src
    sort.cu
    pointer.cu)

cuda_add_executable(liftest ${liftest_src} ${test_src} EXCLUDE_FROM_ALL)
target_link_libraries(liftest lift ${LINK_LIBS})

add_custom_target(lift-tests DEPENDS cpuid liftest)
add_custom_target(lift-run-tests
                  COMMAND "${CMAKE_BINARY_DIR}/tests/liftest"
                  DEPENDS lift-tests)
