
# libml

This is a simple library that implements a classifier based on singular value decomposition (SVD).

Features: 

* A header-only library based on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page). 
* Computations are fast executed thanks to [BS::thread_pool](https://github.com/bshoshany/thread-pool).

## Installation

libml is a header-only library. Simply add the header files to your project using:

```cpp
#include "ml/svd_predictor.h"
```

However, as mentioned above, the library depends on BS::thread_pool and Eigen. Therefore, these libraries need to be included as well.

### Proposed method of installation.

It is highly recommend to use this library in your project using [CMake](https://cmake.org).  

Add the following contents to `CMakeLists.txt`: 

```cmake
cmake_minimum_required(VERSION 3.28)
project(project)

# libml requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  libml
  GIT_REPOSITORY https://github.com/caffik/libml
  GIT_TAG v1.0.0      
)

FetchContent_MakeAvailable(libml)
```
and then link library by [`target_link_libraries`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) command.

## CMake Options

Available `CMake` options that can be used to configure:

```
cmake -DML_ENABLE_TESTING=OFF
```

## Examples

```c++
#include <helo.hpp>
```

