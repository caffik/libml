#ifndef SVD_PREDICTOR_H
#define SVD_PREDICTOR_H

#include "projection.hpp"

namespace ml {

/**
 * Classifier based on singular value decomposition.
 *
 * @tparam MatrixType e.g. a matrix type, or an expression, etc.
 * @see https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html
 * @tparam Allocator allocator used in std::vector containg matrices.
 *
 * @note For usage please see unit tests.
 */
template <typename MatrixType, typename Allocator> class SVDClassifier {
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

  /**
   * Default constructor for SVDClassifier.
   */
  SVDClassifier() = default;

  /**
   * Copy constructor is deleted.
   */
  SVDClassifier(const SVDClassifier &) = delete;

  /**
   * Move constructor for SVDClassifier.
   */
  SVDClassifier(SVDClassifier &&) = default;

  /**
   * Copy assignment operator is deleted.
   */
  SVDClassifier &operator=(const SVDClassifier &) = delete;

  /**
   * Move assignment operator for SVDClassifier.
   */
  SVDClassifier &operator=(SVDClassifier &&) = default;

  /**
   * Default destructor for SVDClassifier.
   */
  ~SVDClassifier() = default;

  /**
   * Copy constructor.
   *
   * @param data std::vector with matrices, each matrix contains data tagged
   * with a different label.
   */
  explicit SVDClassifier(const_reference_data_t data) : data_{data} {};

  /**
   * Move constructor.
   *
   * @param data std::vector with matrices, each matrix contains data tagged
   * with a different label.
   */
  explicit SVDClassifier(data_t &&data) : data_{std::move(data)} {};

  /**
   * Calculates right singular matrix for each matrix in matrices.
   */
  void fit() {
    u_matrices_.resize(data_.size());
    BS::thread_pool pool{getNumThreads()};
    pool.detach_loop<std::size_t>(0, data_.size(), [this](auto i) {
      Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU> svd(data_[i]);
      u_matrices_[i] = svd.matrixU();
    });
    pool.wait();
  }

  /**
   * Predicts labels for given data.
   *
   * @tparam Derived The type of the input matrix.
   * @param pred_data Matrix composed of row vectors data.
   * @param number_of_singulars Number of singular vectors to be used in
   * projection. Vectors with the greatest singular value are taken.
   * By default all vectors are used (`number_of_singulars = 0`).
   *
   * @return Predicted labels.
   */
  template <typename Derived>
  const matrix_t<std::size_t> &
  fit_predict(const Eigen::MatrixBase<Derived> &pred_data,
              const std::size_t number_of_singulars = 0) {
    projections_ = projections(pred_data, u_matrices_, number_of_singulars);
    generate_labels(pred_data);
    return pred_labels_;
  }

  /**
   * Gets the data.
   *
   * @return Reference to the data.
   */
  reference_data_t getData() { return data_; }

  /**
   * Gets the data (const version).
   *
   * @return Reference to the data.
   */
  [[nodiscard]] reference_data_t getData() const { return data_; }

  /**
   * Sets the data.
   *
   * @param data Reference to the data to be set.
   */
  void setData(const_reference_data_t data) { data_ = data; }

  /**
   * Sets the data (move version).
   *
   * @param data Rvalue reference to the data to be set.
   */
  void setData(data_t &&data) { data_ = std::move(data); }

  /**
   * Gets the U matrices.
   *
   * @return Reference to the U matrices.
   */
  matrices_t<double> &getUMatrices() { return u_matrices_; }

  /**
   * Gets the U matrices (const version).
   *
   * @return Reference to the U matrices.
   */
  [[nodiscard]] const matrices_t<double> &getUMatrices() const {
    return u_matrices_;
  }
  /**
   * Sets the U matrices.
   *
   * @param u_matrices Reference to the U matrices to be set.
   */
  void setUMatrices(const matrices_t<double> &u_matrices) {
    u_matrices_ = u_matrices;
  }

  /**
   * Sets the U matrices (move version).
   *
   * @param u_matrices Rvalue reference to the U matrices to be set.
   */
  void setUMatrices(matrices_t<double> &&u_matrices) {
    u_matrices_ = std::move(u_matrices);
  }

  /**
   * Gets the projections.
   *
   * @return Reference to the projections.
   */
  [[nodiscard]] const matrices_t<double> &getProjections() const {
    return projections_;
  }

private:
  data_t data_;
  matrices_t<double> u_matrices_;
  matrices_t<double> projections_;
  matrix_t<std::size_t> pred_labels_;

  // TODO: Split this function into smaller ones.
  template <typename Derived>
  void generate_labels(const Eigen::MatrixBase<Derived> &pred_data) {
    pred_labels_.resize(1, pred_data.cols());
    std::vector<double> minimal_distance(pred_data.cols(),
                                         std::numeric_limits<double>::max());

    // pick the ith projection matrix then calculate the distance
    // between the jth columns of pred_data and projection matrix
    // if the distance is smaller than the minimal_distance[j] save
    // it as minimal_distance[j] and assign the label i to the jth
    // column of pred_labels_
    for (std::size_t i{0}; i < projections_.size(); ++i) {
      for (int j{0}; j < pred_data.cols(); ++j) {
        if (auto potential_minimum{
                (pred_data.col(j) - projections_[i].col(j)).squaredNorm()};
            potential_minimum < minimal_distance[j]) {
          minimal_distance[j] = potential_minimum;
          pred_labels_(0, j) = i;
        }
      }
    }
  }
};

} // namespace ml

#endif // SVD_PREDICTOR_H
