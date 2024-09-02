message(STATUS "libml: Configuring tests...")

include(FetchContent)

##################
#   Google Test  #
##################

enable_testing()

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG v1.14.0
    FIND_PACKAGE_ARGS NAMES GTest
)

set(BUILD_GMOCK OFF)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

#################
#   Set Tests   #
#################

add_executable(Tests
    test.svd_classifier.cpp
    test.projection.cpp
)

target_link_libraries(Tests
    PRIVATE
        GTest::gtest_main
        libml::svd_classifier
)

gtest_add_tests(TARGET Tests)
