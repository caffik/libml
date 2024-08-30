include(FetchContent)

##################
#   Google Test  #
##################

enable_testing()

find_package(GTest 1.14.0
        QUIET
)

if (NOT GTest_FOUND)
    message(STATUS "Google Test not found. Fetching Google Test...")

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.14.0
    )

    set(BUILD_GMOCK OFF)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif ()

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
