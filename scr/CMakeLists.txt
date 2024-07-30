include(FetchContent)

#############
#   Eigen   #
#############

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)

FetchContent_MakeAvailable(Eigen)

###################
#   thread-pool   #
###################

set(CPM_DOWNLOAD_LOCATION "${libml_BINARY_DIR}/cmake/CPM.cmake")
if(NOT(EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME BS_thread_pool
        GITHUB_REPOSITORY bshoshany/thread-pool
        VERSION 4.1.0)

add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool
        INTERFACE
            ${BS_thread_pool_SOURCE_DIR}/include
)

#####################
#   Library setup   #
#####################

set(NAMESPACE mLearn)
set(INCLUDE_DIR
        ${libml_SOURCE_DIR}/include
)

add_library(ml INTERFACE ${INCLUDE_DIR}/ml/svd_predictor.h)
add_library(${NAMESPACE}::ml ALIAS ml)

target_link_libraries(ml
        INTERFACE
        Eigen3::Eigen
        BS_thread_pool
)

target_include_directories(ml
        INTERFACE
            ${INCLUDE_DIR}
)
