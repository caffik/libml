cmake_minimum_required(VERSION 3.28)
project(libml
        VERSION 1.0.0
        DESCRIPTION "Small machine learning library"
        LANGUAGES CXX
)

option(ML_ENABLE_TESTING "Enable testing of the mLearn library." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

add_subdirectory(scr)

if (ML_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
