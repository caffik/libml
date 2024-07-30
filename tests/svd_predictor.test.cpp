#include <gtest/gtest.h>
#include <ml/svd_predictor.h>

std::string address(const void *ptr) {
  return std::to_string(reinterpret_cast<uintptr_t>(ptr));
}

TEST(SVDPredict, Constructor_copy_Matrix) {
  const std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4),
                                          Eigen::MatrixXd::Random(4, 4),
                                          Eigen::MatrixXd::Random(4, 4)};

  ml::SVDPredict pred{data};
  ASSERT_TRUE(pred.getData() == data);
}

TEST(SVDPredict, Constructor_copy_Map) {
  std::vector<std::vector<double>> vec_data{{1, 2, 3, 4, 5, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 5, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 5, 6, 7, 8, 9}};
  std::vector<Eigen::Map<Eigen::MatrixXd>> data{};
  for (decltype(vec_data.size()) i{0}; i < vec_data.size(); ++i) {
    data.emplace_back(vec_data[i].data(), 3, 3);
  }
  ml::SVDPredict pred{data};
  ASSERT_TRUE(pred.getData() == data);
}

TEST(SVDPredict, Constructor_copy_Block) {
  std::vector<Eigen::MatrixXd> data_matrices{Eigen::MatrixXd::Random(4, 4),
                                             Eigen::MatrixXd::Random(4, 4),
                                             Eigen::MatrixXd::Random(4, 4)};
  std::vector<decltype(data_matrices[0].leftCols(2))> data{};
  for (decltype(data_matrices.size()) i{0}; i < data_matrices.size(); ++i) {
    data.emplace_back(data_matrices[i].leftCols(2));
  }
  ml::SVDPredict pred{data};
  ASSERT_TRUE(pred.getData() == data);
}
// Same applies for moveConstructor

TEST(SVDPredict, Constructor_move) {
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4)};
  const auto data_address{address(data.data())};

  ml::SVDPredict pred{std::move(data)};
  ASSERT_TRUE(address(pred.getData().data()) == data_address);
}

TEST(SVDPredict, setData_copy) {
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4)};

  using value_t = std::vector<Eigen::MatrixXd>::value_type;
  using allocator_t = std::vector<Eigen::MatrixXd>::allocator_type;

  ml::SVDPredict<value_t, allocator_t> pred{};
  pred.setData(data);
  ASSERT_TRUE(pred.getData() == data);
}

TEST(SVDPredict, setData_move) {
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4),
                                    Eigen::MatrixXd::Random(4, 4)};

  using value_t = std::vector<Eigen::MatrixXd>::value_type;
  using allocator_t = std::vector<Eigen::MatrixXd>::allocator_type;

  const auto expected_data_address{address(data.data())};

  ml::SVDPredict<value_t, allocator_t> pred{};
  pred.setData(std::move(data));
  ASSERT_TRUE(address(pred.getData().data()) == expected_data_address);
}

TEST(SVDPredict, fit) {
  std::vector<Eigen::MatrixXd> data{Eigen::MatrixXd::Random(4, 30),
                                    Eigen::MatrixXd::Random(5, 30),
                                    Eigen::MatrixXd::Random(6, 30)};

  using value_t = std::vector<Eigen::MatrixXd>::value_type;
  using allocator_t = std::vector<Eigen::MatrixXd>::allocator_type;
  ml::SVDPredict<value_t, allocator_t> pred{};
  pred.setData(std::move(data));
  pred.fit();

  constexpr auto expected_u_matrices_size{3};
  const std::vector<std::size_t> expected_u_matrices_rows{4, 5, 6};
  const std::vector<std::size_t> expected_u_matrices_cols{4, 5, 6};

  std::vector<std::size_t> u_matrices_rows{};
  std::vector<std::size_t> u_matrices_cols{};

  for (const auto &m : pred.getUMatrices()) {
    u_matrices_rows.push_back(m.rows());
    u_matrices_cols.push_back(m.cols());
  }

  ASSERT_EQ(pred.getUMatrices().size(), expected_u_matrices_size);
  ASSERT_TRUE(expected_u_matrices_rows == u_matrices_rows);
  ASSERT_TRUE(expected_u_matrices_cols == u_matrices_cols);
}

TEST(SVDPredictor, fit_predict) {
  std::vector<Eigen::MatrixXd> data{};
  for (auto i{0}; i < 5; ++i) {
    data.emplace_back(Eigen::MatrixXd::Random(3, 2));
  }

  const Eigen::MatrixXd pred_data{{1, 1, 1}, {10, 35, 100}, {-10, 20, -90}};
  ml::SVDPredict svd_pred{data};
  svd_pred.fit(2);

  const auto &labels{svd_pred.fit_predict(pred_data, 1)};

  constexpr auto expected_size{3};
  ASSERT_TRUE(labels.size() == expected_size);
}

TEST(ml, projection) {
  const Eigen::MatrixXd from{Eigen::MatrixXd::Random(3, 3)};
  Eigen::MatrixXd onto{Eigen::MatrixXd::Random(3, 3)};
  onto.colwise().normalize();

  const auto projection{ml::projection(from, onto)};

  ASSERT_TRUE(projection.rows() == 3 && projection.cols() == 3);
}

TEST(ml, projections) {
  const Eigen::MatrixXd from{Eigen::MatrixXd::Random(1000, 300)};
  std::vector<Eigen::MatrixXd> onto{};
  constexpr auto number_of_matrices{10};
  for (auto i{0}; i < number_of_matrices; ++i) {
    onto.emplace_back(Eigen::MatrixXd::Random(1000, 10));
  }
  const auto projections{ml::projections(from, onto)};

  const auto expected_size{onto.size()};
  const std::vector<std::vector<Eigen::Index>> expected_shapes(
      expected_size, {from.rows(), from.cols()});

  const auto size{projections.size()};
  std::vector<std::vector<Eigen::Index>> shapes;
  for (const auto &p : projections) {
    shapes.push_back({p.rows(), p.cols()});
  }
  ASSERT_TRUE(size == expected_size);
  ASSERT_TRUE(shapes == expected_shapes);
}
