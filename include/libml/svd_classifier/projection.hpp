#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <Eigen/Eigen>
#include <vector>

#include "BS_thread_pool.hpp"
#include "../ml.hpp"

namespace ml {

/**
 * Projects the columns of the matrix `from` onto the columns of the matrix
 * `onto`.
 *
 * @tparam Derived1 is the derived type, e.g. a matrix type, or an
 * expression.
 * @tparam Derived2 is the derived type, e.g. a matrix type, or an
 * expression.
 * @param from The matrix whose columns are to be projected.
 * @param onto The matrix whose columns define the space onto which the columns
 * of `from` are projected.
 *
 * @note Above template types guarantees generality of accepted arguments.
 * Please @see: https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html
 *
 * @remark It is assumed that the columns of `onto` are normalized.
 *
 * @return A matrix containing the projections of the columns of `from` onto the
 * columns of `onto`.
 */
template <typename Derived1, typename Derived2>
auto projection(const Eigen::MatrixBase<Derived1> &from,
                const Eigen::MatrixBase<Derived2> &onto) {
  using common_type =
      std::common_type_t<typename Eigen::MatrixBase<Derived1>::Scalar,
                         typename Eigen::MatrixBase<Derived2>::Scalar>;
  using matrix_t = Eigen::Matrix<common_type, Eigen::Dynamic, Eigen::Dynamic>;

  matrix_t projections{matrix_t::Zero(onto.rows(), from.cols())};
  for (auto i{0}; i < from.cols(); ++i) {
    for (auto j{0}; j < onto.cols(); ++j) {
      projections.col(i) += from.col(i).dot(onto.col(j)) * onto.col(j);
    }
  }
  return projections;
}

// Forward declaration of the functions.

namespace utils {
template <typename Derived, template <typename...> typename Container,
          typename MatrixType1, typename MatrixType2>
auto projections_singlethread(const Eigen::MatrixBase<Derived> &,
                              const Container<MatrixType1> &,
                              std::vector<MatrixType2> &, std::size_t);

template <typename Derived, template <typename...> typename Container,
          typename MatrixType1, typename MatrixType2>
auto projections_multithreads(const Eigen::MatrixBase<Derived> &,
                              const Container<MatrixType1> &,
                              std::vector<MatrixType2> &, std::size_t);
} // namespace utils

/**
 * Returns the vector of matrices that containis the projections of the columns
 * of `from` onto the space spanned by columns of `onto[i]`. Where $0 \\leq i
 * < onto.size().$
 *
 * @tparam Derived is the derived type, e.g. a matrix type, or an
 * expression.
 * @tparam Container any container type that has the following member functions:
 * a) `size()`
 * b) `MatrixType operator[](IntegerType)`
 * @tparam MatrixType matrix type, or an expression.
 * @param from the matrix whose columns are to be projected.
 * @param onto container coposed by matrices or row vectors.
 * @param span_size number of columns into which the columns of `from`
 * should be projected. By default all columns are used (`span_size = 0`).
 *
 * @return `std::vector` containing matrices with projections.
 *
 * @note `span_size` must be between $0$ and
 * $\\min(onto[0].cols(), \\dots, onto[onto.size() - 1].cols()).$
 */
template <typename Derived, template <typename...> typename Container,
          typename MatrixType>
auto projections(const Eigen::MatrixBase<Derived> &from,
                 const Container<MatrixType> &onto,
                 const std::size_t span_size = 0) {
  static_assert(std::is_base_of_v<Eigen::MatrixBase<MatrixType>, MatrixType>,
                "MatrixType is not derived class of Eigen::MatrixBase.");

  using common_type =
      std::common_type_t<typename Eigen::MatrixBase<Derived>::Scalar,
                         typename MatrixType::Scalar>;
  using matrix_t = Eigen::Matrix<common_type, Eigen::Dynamic, Eigen::Dynamic>;

  std::vector<matrix_t> projections{};
  if (getNumThreads() == 1) {
    return utils::projections_singlethread(from, onto, projections, span_size);
  }
  return utils::projections_multithreads(from, onto, projections, span_size);
}

namespace utils {

inline bool isSpanSizeValid(const std::size_t dim,
                            const std::size_t span_size) {
  return span_size != 0 && span_size < dim;
}

inline std::size_t getAdjustedSpanSize(const std::size_t dim,
                                       const std::size_t span_size) {
  return isSpanSizeValid(dim, span_size) ? span_size : dim;
}

template <typename Derived, template <typename...> typename Container,
          typename MatrixType1, typename MatrixType2>
auto projections_singlethread(const Eigen::MatrixBase<Derived> &from,
                              const Container<MatrixType1> &onto,
                              std::vector<MatrixType2> &projections,
                              const std::size_t span_size) {
  projections.reserve(onto.size());
  for (decltype(onto.size()) i{0}; i < onto.size(); ++i) {
    auto adjusted_span_size{getAdjustedSpanSize(onto[i].cols(), span_size)};
    projections.emplace_back(
        projection(from, onto[i].leftCols(adjusted_span_size)));
  }
  return projections;
}

template <typename Derived, template <typename...> typename Container,
          typename MatrixType1, typename MatrixType2>
auto projections_multithreads(const Eigen::MatrixBase<Derived> &from,
                              const Container<MatrixType1> &onto,
                              std::vector<MatrixType2> &projections,
                              const std::size_t span_size) {
  projections.resize(onto.size());
  BS::thread_pool pool{getNumThreads()};
  pool.detach_loop<unsigned int>(
      0, onto.size(), [&projections, &from, &onto, span_size](auto i) {
        auto adjusted_span_size{getAdjustedSpanSize(onto[i].cols(), span_size)};
        projections[i] =
            std::move(projection(from, onto[i].leftCols(adjusted_span_size)));
      });
  pool.wait();
  return projections;
}

} // namespace utils
} // namespace ml

#endif // PROJECTION_HPP
