#include <gtest/gtest.h>
#include <libml/svd_classifier/svd_classifier.hpp>

std::string address(const void *ptr) {
  return std::to_string(reinterpret_cast<uintptr_t>(ptr));
}

/*
 * Tests for the `SVDClassifier` class.
 */

/*
 * Tests for the constructors.
 */

TEST(SVDClassifier, DefaultConstructor) {
  ml::SVDClassifier<Eigen::MatrixXd, std::allocator<Eigen::MatrixXd>> svd_predict;
  EXPECT_TRUE(svd_predict.getData().empty());
}

// The following tests are intended to show flexibility in the data type.

TEST(SVDClassifier, CopyConstructorMatrix) {
  const std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4),
                                          Eigen::MatrixXd::Random(4, 4),
                                          Eigen::MatrixXd::Random(4, 4)};

  ml::SVDClassifier pred{data};
  ASSERT_TRUE(pred.getData() == data);
}

TEST(SVDClassifier, CopyConstructorMap) {
  std::vector<std::vector<double>> vec_data{{1, 2, 3, 4, 5, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 5, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 5, 6, 7, 8, 9}};

  std::vector<Eigen::Map<Eigen::MatrixXd>> data{};
  for (decltype(vec_data.size()) i{0}; i < vec_data.size(); ++i) {
    data.emplace_back(vec_data[i].data(), 3, 3);
  }

  ml::SVDClassifier pred{data};

  ASSERT_TRUE(pred.getData() == data);
}

TEST(SVDClassifier, CopyConstructorBlock) {
  std::vector<Eigen::MatrixXd> data_matrices{Eigen::MatrixXd::Random(4, 4),
                                             Eigen::MatrixXd::Random(4, 4),
                                             Eigen::MatrixXd::Random(4, 4)};

  std::vector<decltype(data_matrices[0].leftCols(2))> data{};
  for (decltype(data_matrices.size()) i{0}; i < data_matrices.size(); ++i) {
    data.emplace_back(data_matrices[i].leftCols(2));
  }

  ml::SVDClassifier pred{data};

  ASSERT_TRUE(pred.getData() == data);
}

// Same applies for moveConstructor

TEST(SVDClassifier, MoveConstructor) {
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(3, 3)};
  const auto data_address{address(data.data())};

  ml::SVDClassifier svd_predict(std::move(data));

  EXPECT_EQ(svd_predict.getData().size(), 1);
  EXPECT_TRUE(address(svd_predict.getData().data()) == data_address);
}

/*
 * Tests for the `fit` method.
 */

TEST(SVDClassifier, fit) {
  ml::SVDClassifier svd_predict{std::vector<Eigen::MatrixXd>{
      Eigen::MatrixXd::Random(4, 30), Eigen::MatrixXd::Random(5, 30),
      Eigen::MatrixXd::Random(6, 30)}};

  svd_predict.fit();
  using shape_t = std::pair<std::size_t, std::size_t>;

  constexpr auto expected_u_matrices_size{3};
  const std::vector<shape_t> expected_u_matrices_shape{
      std::make_pair(4, 4), std::make_pair(5, 5), std::make_pair(6, 6)};

  for (auto i{0}; i < svd_predict.getUMatrices().size(); ++i) {
    const auto &m{svd_predict.getUMatrices()[i]};
    EXPECT_EQ(shape_t(m.rows(), m.cols()), expected_u_matrices_shape[i]);
  }

  EXPECT_EQ(svd_predict.getUMatrices().size(), expected_u_matrices_size);
}

/*
 * Tests for the `setData` method.
 */

TEST(SVDClassifier, SetData) {
  ml::SVDClassifier<Eigen::MatrixXd, std::allocator<Eigen::MatrixXd>> svd_predict;
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(3, 3)};
  svd_predict.setData(data);
  EXPECT_TRUE(svd_predict.getData() == data);
}

/*
 * Tests for the `setData` method with move semantics.
 */

TEST(SVDClassifier, SetDataMove) {
  ml::SVDClassifier<Eigen::MatrixXd, std::allocator<Eigen::MatrixXd>> svd_predict;
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4), Eigen::MatrixXd::Random(4, 4),
                   Eigen::MatrixXd::Random(4, 4)};
  const auto expected_data_address{address(data.data())};

  svd_predict.setData(std::move(data));

  EXPECT_TRUE(address(svd_predict.getData().data()) == expected_data_address);
}

/*
 * Tests for the `setUMatrices` method.
 */

TEST(SVDClassifier, SetUMatrices) {
    ml::SVDClassifier<Eigen::MatrixXd, std::allocator<Eigen::MatrixXd>> svd_predict;
    std::vector<Eigen::MatrixXd> u_matrices{Eigen::MatrixXd::Random(3, 3)};
    svd_predict.setUMatrices(u_matrices);
    EXPECT_EQ(svd_predict.getUMatrices(), u_matrices);
}

/*
 * Tests for the `setUMatrices` method with move semantics.
 */

TEST(SVDClassifier, SetUMatricesMove) {
    ml::SVDClassifier<Eigen::MatrixXd, std::allocator<Eigen::MatrixXd>> svd_predict;
    std::vector<Eigen::MatrixXd> u_matrices{Eigen::MatrixXd::Random(3, 3)};
    const auto expected_u_matrices_address{address(u_matrices.data())};

    svd_predict.setUMatrices(std::move(u_matrices));
    EXPECT_TRUE(address(svd_predict.getUMatrices().data()) == expected_u_matrices_address);
}

/*
 * Tests for the `fit_predict` method.
 */

TEST(SVDClassifier, fit_predict) {
  ml::SVDClassifier svd_predict{std::vector<Eigen::MatrixXd>{
      Eigen::MatrixXd::Random(4, 30), Eigen::MatrixXd::Random(4, 20),
      Eigen::MatrixXd::Random(4, 10)}};

  svd_predict.fit();
  const Eigen::MatrixXd pred_data{Eigen::MatrixXd::Random(4, 30)};
  const auto result{svd_predict.fit_predict(pred_data)};

  EXPECT_EQ(result.size(), pred_data.cols());
}
