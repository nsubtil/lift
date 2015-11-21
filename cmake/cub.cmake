#set(cub_PREFIX ${CMAKE_BINARY_DIR}/contrib/cub-prefix)
#
#if (DEFINED CUB_REPO_OVERRIDE)
#    set(cub_repo ${CUB_REPO_OVERRIDE})
#else()
#    set(cub_repo "https://github.com/nsubtil/cub")
#endif()
#
#ExternalProject_Add(cub
#    PREFIX ${cub_PREFIX}
#    GIT_REPOSITORY ${cub_repo}
#    # 1.4.1 + warning fixes
#    GIT_TAG "fix-signedness-warning"
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND ""
#    INSTALL_COMMAND ""
#    LOG_DOWNLOAD 0
#    )

include_directories(${CMAKE_SOURCE_DIR}/contrib/cub)

