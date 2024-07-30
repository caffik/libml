#ifndef SVD_PREDICTOR_UTILS_H
#define SVD_PREDICTOR_UTILS_H

#include <Eigen/Eigen>

namespace ml {

template <typename Derived1, typename Derived2>
auto projection(const Eigen::MatrixBase<Derived1> &,
                const Eigen::MatrixBase<Derived2> &);

namespace utils {
template <typename Container>
using size_t = decltype(std::declval<Container>().size());

template <typename Container, typename = std::void_t<>>
struct has_size : std::false_type {};

template <typename Container>
struct has_size<Container, std::void_t<size_t<Container>>> : std::true_type {};

template <typename Container>
bool has_size_v = has_size<Container>::value;

template <typename Container>
using operator_brackets_t = decltype(std::declval<Container &>().operator[](
    std::declval<std::size_t>()));

template <typename Container, typename = std::void_t<>>
struct has_operator_brackets : std::false_type {};

template <typename Container>
struct has_operator_brackets<Container,
                             std::void_t<operator_brackets_t<Container>>>
    : std::true_type {};

template <typename Container>
bool has_operator_brackets_v = has_operator_brackets<Container>::value;

template <typename Derived, template <typename...> typename Container,
          typename MatrixType>
auto projections_singlethread(const Eigen::MatrixBase<Derived> &from,
                              const Container<MatrixType> &onto,
                              const std::size_t span_size = 0) {
  using common_type =
      std::common_type_t<typename Eigen::MatrixBase<Derived>::Scalar,
                         typename MatrixType::Scalar>;
  using matrix_t = Eigen::Matrix<common_type, Eigen::Dynamic, Eigen::Dynamic>;

  std::vector<matrix_t> projections{};
  projections.reserve(onto.size());
  for (decltype(onto.size()) i{0}; i < onto.size(); ++i) {
    auto cols{static_cast<std::size_t>(onto[i].cols())};
    if (span_size != 0 && span_size < cols) {
      cols = span_size;
    }
    projections.emplace_back(projection(from, onto[i].leftCols(cols)));
  }
  return projections;
}

template <typename Derived, template <typename...> typename Container,
          typename MatrixType>
auto projections_multithreads(const Eigen::MatrixBase<Derived> &from,
                              const Container<MatrixType> &onto,
                              const unsigned int threads,
                              const std::size_t span_size = 0) {
  using common_type =
      std::common_type_t<typename Eigen::MatrixBase<Derived>::Scalar,
                         typename MatrixType::Scalar>;
  using matrix_t = Eigen::Matrix<common_type, Eigen::Dynamic, Eigen::Dynamic>;

  std::vector<matrix_t> projections{};
  projections.resize(onto.size());

  BS::thread_pool pool{threads};
  pool.detach_loop<unsigned int>(
      0, onto.size(), [&projections, &from, &onto, span_size](auto i) {
        auto cols{static_cast<std::size_t>(onto[i].cols())};
        if (span_size != 0 && span_size < cols) {
          cols = span_size;
        }
        projections[i] = std::move(projection(from, onto[i].leftCols(cols)));
      });
  pool.wait();
  return projections;
}

} // namespace utils
} // namespace ml

#endif // SVD_PREDICTOR_UTILS_H
