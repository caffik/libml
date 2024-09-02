
# libml

This is a simple library that implements some machine learning algorithms. 
Currently, it includes a classifier based on singular value decomposition (SVD).

Features: 

* A header-only library based on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
* Computations are fast executed thanks to [BS::thread_pool](https://github.com/bshoshany/thread-pool).

## Installation

`libml` is a header-only library. Simply add the header files to your project using:

```cpp
#include "libml/svd_classifier/svd_classifier.hpp"
```

However, the library depends on `BS::thread_pool` and `Eigen3`. Therefore, these libraries need to be included as well.

### Proposed method of installation.

It is highly recommend to use this library in your project using [CMake](https://cmake.org).  

Add the following contents to `CMakeLists.txt` (CMake 3.28 or later is required): 

```cmake
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
and then link library with [`target_link_libraries`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) command. 

## CMake Options

- `ENABLE_TESTING` - Build tests for the library. Default: `OFF`.

## SVD Classifier Class:

The `SVDClassifier` class was implemented with having in mind flexibility and ease of use. 

### Potential improvements:

* Current implementation is based on classical SVD. It would be beneficial to introduce a randomized SVD algorithm. 
Please see [this](https://epubs.siam.org/doi/10.1137/090771806) paper for more information. 
C++ implementation can be found [here](https://github.com/mp4096/rsvd).


