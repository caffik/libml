###############
#   OPTIONS   #
###############

option(ENABLE_TESTING "Enable testing of the library." ON)
option(ENABLE_DOCS "Enable generation of documentation." ON)

###############
#   PROJECT   #
###############

cmake_minimum_required(VERSION 3.28)
project(libml
        VERSION 1.0.0
        DESCRIPTION "Small machine learning library"
        LANGUAGES CXX
)

message(STATUS "Configuring libml 1.0.0")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

########################
#   GET DEPENDENCIES   #
########################

include(FetchContent)

#############
#   Eigen   #
#############

FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS 3.4.0
)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen)

###################
#   thread-pool   #
###################

    ####################
    #   Download CPM   #
    ####################

set(CPM_DOWNLOAD_LOCATION
        "${libml_BINARY_DIR}/cmake/CPM.cmake"
)

if(NOT(EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD
            https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

####
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
    NAME BS_thread_pool
    GITHUB_REPOSITORY bshoshany/thread-pool
    VERSION 4.1.0
)

add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool
    INTERFACE
        ${BS_thread_pool_SOURCE_DIR}/include
)

#############
#   libml   #
#############

set(INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/include
)

#################################
#   libml::svd_classification   #
#################################

set(INCLUDE_DIR_SVD_CLASSIFIER
        ${INCLUDE_DIR}/libml/svd_classification
)

add_library(libml_svd_classification INTERFACE)
add_library(libml::svd_classification ALIAS libml_svd_classification)

target_link_libraries(libml_svd_classification
        INTERFACE
        Eigen3::Eigen
        BS_thread_pool
)

target_include_directories(
    libml_svd_classification
        INTERFACE
            $<BUILD_INTERFACE:${INCLUDE_DIR}>
)

######################
#   SUBDIRECTORIES   #
######################

if (ENABLE_TESTING)
    add_subdirectory(tests)
endif()

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()

message(STATUS "Configured libml 1.0.0")