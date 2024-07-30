include(FetchContent)

#################
#   googletest  #
#################

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.14.0
)

set(BUILD_GMOCK OFF)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(TESTS
        svd_predictor.test.cpp
)

add_executable(TESTS ${TESTS})

target_include_directories(TESTS
        PRIVATE
        ${libml_SOURCE_DIR}/include
)

target_link_libraries(TESTS
    PRIVATE
        GTest::gtest_main mLearn::ml
)

gtest_add_tests(TARGET TESTS)
