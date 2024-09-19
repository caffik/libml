# libml

`libml` is a simple, header-only library that implements machine learning algorithms, currently featuring a classifier
based on Singular Value Decomposition (SVD).

## Features

- **Header-only**: Easy to integrate into your project.
- **Fast Computations**: Utilizes [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix operations
  and [BS::thread_pool](https://github.com/bshoshany/thread-pool) for parallel processing.

## Description

- `projection.hpp`: contains functions for projecting the columns of one matrix onto the columns of another matrix.
- `svd_classifier.hpp`: defines the SVDClassifier class, which uses Singular Value Decomposition (SVD) for
  classification tasks. The class includes methods for fitting the model, predicting labels, and managing the data.

## Project Structure

- `include/libml/`: Header files
- `tests/`: Unit tests
- `docs/`: Documentation files

## Installation

To use `libml`, include the header files in your project:

```cpp
#include "libml/svd_classifier/svd_classifier.hpp"
```

### Dependencies

`libml` depends on the following libraries:

- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [BS::thread_pool](https://github.com/bshoshany/thread-pool)

### Using CMake

It is highly recommended to use CMake (version 3.28 or later) to manage your project. Add the following to your
`CMakeLists.txt`:

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

Then link the library using the `target_link_libraries` command.

## CMake Options

- `ENABLE_TESTING`: Build tests for the library. Default is `OFF`.

## SVD Classifier

The `SVDClassifier` class is designed for flexibility and ease of use.

## Documentation

To generate the documentation for `libml`, you need to have Doxygen and LaTeX installed on your system.
The CMake configuration will automatically detect these tools and generate the necessary documentation files.

### Requirements

- **Doxygen**: Used to generate HTML documentation.
- **LaTeX**: Specifically, the `pdflatex` compiler is used to generate PDF documentation.

## Examples

This example demonstrates how to use the `projection` function.

```cpp
#include <iostream>
#include <Eigen/Core>
#include "libml/utils/projection.hpp

int main() {
  // Example matrices
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  Eigen::MatrixXd onto(3, 2);
  onto << 1, 0,
          0, 1,
          0, 0;

  Eigen::MatrixXd result = ml::projection(from, onto);
  std::cout << "Projection result:\n" << result << std::endl;
``` 
This example demonstrates how to use the SVDClassifier class.

```cpp
#include <iostream>
#include <Eigen/Core>
#include "libml/svd_classification/svd_classifier.hpp"

int main() {
  // Example data [here each matrix represents a training set where each row is a sample]
  // [by default the data sets are labeled: 0, 1, 2, ...]
  std::vector<Eigen::MatrixXd> data = {
      Eigen::MatrixXd::Random(4, 4),    // label 0 
      Eigen::MatrixXd::Random(4, 4),    // label 1
      Eigen::MatrixXd::Random(4, 4)     // label 2
  }
  // Create SVDClassifier instance
  ml::SVDClassifier classifier(data);
  
  // Fit the model
  classifier.fit();
  
  // Predict labels for new data
  auto new_data{Eigen::MatrixXd::Random(4, 4)};
  auto labels{classifier.fit_predict(new_data)}; // each row of new_data represents a sample 
  // for which the label is predicted
  
  std::cout << "Predicted labels:\n" << labels << std::endl
  return 0;
}
```

### Potential Improvements

- **Randomized SVD**: The current implementation uses classical SVD. Introducing a randomized SVD algorithm could
  improve performance. For more information, see [this paper](https://epubs.siam.org/doi/10.1137/090771806) and a C++
  implementation [here](https://github.com/mp4096/rsvd).
