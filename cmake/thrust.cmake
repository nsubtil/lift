#set(thrust_PREFIX ${CMAKE_BINARY_DIR}/contrib/thrust-prefix)
#
#if (DEFINED THRUST_REPO_OVERRIDE)
#    set(thrust_repo ${THRUST_REPO_OVERRIDE})
#else()
#    set(thrust_repo "https://github.com/thrust/thrust")
#endif()
#
#ExternalProject_Add(thrust
#    PREFIX ${thrust_PREFIX}
#    GIT_REPOSITORY ${thrust_repo}
#    # we're tracking branch 1.8.0
#    GIT_TAG "1.8.2"
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND ""
#    INSTALL_COMMAND ""
#    LOG_DOWNLOAD 1
#    )
#
#set(thrust_INCLUDE ${thrust_PREFIX}/src/thrust)
#include_directories(${thrust_INCLUDE})

include_directories(${CMAKE_SOURCE_DIR}/contrib/thrust)
    
