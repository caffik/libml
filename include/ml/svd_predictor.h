#ifndef SVD_PREDICTOR_H
#define SVD_PREDICTOR_H

#include "BS_thread_pool.hpp"
#include "svd_predictor_utils.h"
#include <vector>

namespace ml {
/**
 * Returns the matrix that containing the projections of the columns of `from`
 * onto the space spanned by columns of `onto`.
 *
 * @tparam Derived1 is the derived type, e.g. a matrix type, or an
 * expression, etc.
 * @tparam Derived2 is the derived type, e.g. a matrix type, or an
 * expression, etc.
 *
 * @note Above template types guarantees generality of accepted arguments.
 * Please @see: https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html
 *
 * @param from Matrix or row type vector.
 * @param onto Matrix or row type vector.
 * @return Matrix containing projections of each column in (or row vector)
 * `from`.
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

/**
 * Returns the vector of matrices that containis the projections of the columns
 * of `from` onto the space spanned by columns of `onto[i]`. Where $0 \leq i
 * < onto.size().$
 *
 * @tparam Derived is the derived type, e.g. a matrix type, or an
 * expression, etc. *
 * @tparam Container Any container type that has the following member functions:
 * a) `size()`
 * b) `MatrixType operator[](IntegerType)`
 * @tparam MatrixType matrix type, or an expression, etc.
 * @param from Matrix or row type vector.
 * @param onto Container coposed by matrices or row vectors.
 * @param span_size Number of columns into which the columns of `from`
 * should be projected. By default all columns are used (`span_size = 0`).
 * @param num_threads number of threads to be used in process of projection.
 * By default main thread is used (`num_threads = 0`).
 * @return `std::vector` containing matrices with projections.
 *
 * @note `span_size` must be between $0$ and
 * $\min(onto[0].cols(), \dots, onto[onto.size() - 1].cols()).$
 */
template <typename Derived,
          template <typename...> typename Container, typename MatrixType>
auto projections(const Eigen::MatrixBase<Derived> &from,
                 const Container<MatrixType> &onto,
                 const std::size_t span_size = 0,
                 const std::size_t num_threads = 0) {
  static_assert(std::is_base_of_v<Eigen::MatrixBase<MatrixType>, MatrixType>,
                "MatrixType is not derived class of Eigen::MatrixBase.");
  if (!num_threads) {
    return ml::utils::projections_singlethread(from, onto, span_size);
  }
  return ml::utils::projections_multithreads(from, onto, num_threads, span_size);
}

/**
 * Classifier based on singular value decomposition.
 *
 * @tparam MatrixType e.g. a matrix type, or an expression, etc.
 * @see https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html
 * @tparam Allocator allocator used in std::vector containg matrices.
 *
 * @note For usage please see unit tests.
 */
template <typename MatrixType, typename Allocator> class SVDPredict {
  static_assert(std::is_base_of_v<Eigen::MatrixBase<MatrixType>, MatrixType>,
                "MatrixType is not derived class of Eigen::MatrixBase.");

public:
  using data_t = std::vector<MatrixType, Allocator>;
  using const_data_t = const data_t;
  using reference_data_t = data_t &;
  using const_reference_data_t = const data_t &;

  template <typename S>
  using matrix_t = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
  template <typename S> using matrices_t = std::vector<matrix_t<S>>;

  SVDPredict() = default;

  SVDPredict(const SVDPredict &) = delete;

  SVDPredict(SVDPredict &&) = delete;

  SVDPredict &operator=(const SVDPredict &) = delete;

  SVDPredict &operator=(SVDPredict &&) = delete;

  ~SVDPredict() = default;
  /**
   * Copy constructor.
   *
   * @param data std::vector with matrices, each matrix contains data tagged
   * with a different label.
   */
  explicit SVDPredict(const_reference_data_t data) : data_{data} {};

  /**
   * Move constructor.
   *
   * @param data std::vector with matrices, each matrix contains data tagged
   * with a different label.
   */
  explicit SVDPredict(data_t &&data) : data_{std::move(data)} {};

  /**
   * Calculates right singular matrix for each matrix in matrices.
   *
   * @param num_threads number of threads to be used in process of projection.
   * By default only main thread is used (`num_threads = 0`).
   */
  void fit(const unsigned int num_threads = 0) {
    const unsigned int n_threads = num_threads ? num_threads : 1;
    u_matrices_.resize(data_.size());
    BS::thread_pool pool{n_threads};
    pool.detach_loop<std::size_t>(0, data_.size(), [this](auto i) {
      Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU> svd(data_[i]);
      u_matrices_[i] = svd.matrixU();
    });
    pool.wait();
  }

  /**
   * 
   * @tparam Derived 
   * @param pred_data Matrix composed of row vectors data.
   * @param number_of_singulars Number of singular vectors to be used in
   * projection. Vectors with the greatest singular value are taken.
   * By default all vectors are used (`span_size = 0`).
   * @param num_threads Number of threads to be used. By default only main
   * thread is used (`num_threads = 0`).
   * @return Predicted labels.
   */
  template <typename Derived>
  const matrix_t<std::size_t> &
  fit_predict(const Eigen::MatrixBase<Derived> &pred_data,
              const std::size_t number_of_singulars = 0,
              const unsigned int num_threads = 0) {
    projections_ =
        projections(pred_data, u_matrices_, number_of_singulars, num_threads);
    generate_labels(pred_data);
    return pred_labels_;
  }

  reference_data_t getData() { return data_; }

  [[nodiscard]] reference_data_t getData() const { return data_; }

  void setData(const reference_data_t data) { data_ = data; }

  void setData(data_t &&data) { data_ = std::move(data); }

  matrices_t<double> &getUMatrices() { return u_matrices_; }

  [[nodiscard]] const matrices_t<double> &getUMatrices() const {
    return u_matrices_;
  }

  void setUMatrices(const matrices_t<double> &u_matrices) {
    u_matrices_ = u_matrices;
  }

  void setUMatrices(matrices_t<double> &&u_matrices) {
    u_matrices_ = std::move(u_matrices);
  }

  [[nodiscard]] const matrices_t<double> &getProjections() const {
    return projections_;
  }

private:
  data_t data_;
  matrices_t<double> u_matrices_;
  matrices_t<double> projections_;
  matrix_t<std::size_t> pred_labels_;

  template <typename Derived>
  void generate_labels(const Eigen::MatrixBase<Derived> &pred_data) {
    const decltype(pred_data.cols()) rows{1}, cols{pred_data.cols()};
    constexpr auto row{rows - 1};
    pred_labels_.resize(rows, cols);
    std::vector<double> minimum(cols, std::numeric_limits<double>::max());

    for (decltype(projections_.size()) i{0}; i < projections_.size(); ++i) {
      for (auto j{0}; j < pred_data.cols(); ++j) {
        if (auto potential_minium{
                (pred_data.col(j) - projections_[i].col(j)).squaredNorm()};
            potential_minium < minimum[j]) {
          minimum[j] = potential_minium;
          pred_labels_(row, j) = i;
        }
      }
    }
  }
};

} // namespace ml

#endif // SVD_PREDICTOR_H
