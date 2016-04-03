cuda_add_executable(cpuid standalone/cpuid.cu EXCLUDE_FROM_ALL)
target_link_libraries(cpuid lift ${LINK_LIBS})

set(test_src
    pointer.cu
    fill.cu
    sort.cu
    inclusive_scan.cu
    timer.cu

    thrust/pointer_conversion.cu)

cuda_add_executable(liftest ${liftest_src} ${test_src} EXCLUDE_FROM_ALL)
target_link_libraries(liftest m liftest-lib lift ${LINK_LIBS})

add_custom_target(lift-tests DEPENDS cpuid liftest)
add_custom_target(lift-run-tests
                  COMMAND "${CMAKE_BINARY_DIR}/tests/liftest"
                  DEPENDS lift-tests)
