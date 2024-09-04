#include <gtest/gtest.h>
#include <libml/svd_classification/projection.hpp>

/*
 * The following tests are for the `projection` function in `svd_classifier.hpp`.
 */

TEST(ProjectionTests, ProjectsColumnsCorrectly) {
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  Eigen::MatrixXd onto(3, 2);
  onto << 1, 0,
          0, 1,
          0, 0;

  const auto result{ml::projection(from, onto)};

  Eigen::MatrixXd expected(3, 2);
  expected << 1, 2,
              3, 4,
              0, 0;

  ASSERT_TRUE(result.isApprox(expected));
}

TEST(ProjectionTests, HandlesEmptyMatrices) {
  const Eigen::MatrixXd from(0, 0);
  const Eigen::MatrixXd onto(0, 0);

  const auto result{ml::projection(from, onto)};

  const Eigen::MatrixXd expected(0, 0);

  ASSERT_TRUE(result.isApprox(expected));
}

TEST(ProjectionTests, HandlesDifferentDimensions) {
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  Eigen::MatrixXd onto(3, 1);
  onto << 1,
          0,
          0;

  const auto result{ml::projection(from, onto)};

  Eigen::MatrixXd expected(3, 2);
  expected << 1, 2,
              0, 0,
              0, 0;

  ASSERT_TRUE(result.isApprox(expected));
}

TEST(ProjectionTests, HandlesNonOrthogonalOnto) {
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  Eigen::MatrixXd onto(3, 2);
  onto << 1, 1,
          1, 1,
          1, 1;

  const auto result{ml::projection(from, onto)};

  Eigen::MatrixXd expected(3, 2);
  expected << 18, 24,
              18, 24,
              18, 24;

  ASSERT_TRUE(result.isApprox(expected));
}

TEST(ProjectionTests, HandlesZeroColumnsInOnto) {
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  const Eigen::MatrixXd onto(3, 0);

  const auto result{ml::projection(from, onto)};

  Eigen::MatrixXd expected(3, 2);
  expected.setZero();

  ASSERT_TRUE(result.isApprox(expected));
}

TEST(ProjectionTests, HandlesZeroColumnsInFrom) {
  const Eigen::MatrixXd from(3, 0);
  Eigen::MatrixXd onto(3, 2);
  onto << 1, 0,
          0, 1,
          0, 0;

  const auto result{ml::projection(from, onto)};

  const Eigen::MatrixXd expected(3, 0);

  ASSERT_TRUE(result.isApprox(expected));
}

/*
 * The following tests are for the `projections` function in `svd_classifier.hpp`.
 */

TEST(ProjectionsTests, ProjectsColumnsCorrectly) {
  Eigen::MatrixXd from(3, 2);
  from << 1, 2,
          3, 4,
          5, 6;

  const std::vector onto{
      (Eigen::MatrixXd(3, 2) << 1, 0,
                                       0, 1,
                                       0, 0).finished()};

  const auto result{ml::projections(from, onto)};

  const std::vector expected{
      (Eigen::MatrixXd(3, 2) << 1, 2,
                                       3, 4,
                                       0, 0).finished()};

  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_TRUE(result[i].isApprox(expected[i]));
  }
}

TEST(ProjectionsTests, HandlesEmptyMatrices) {
  const Eigen::MatrixXd from(0, 0);
  const std::vector onto{Eigen::MatrixXd(0, 0)};

  const auto result{ml::projections(from, onto)};

  const std::vector expected{Eigen::MatrixXd(0, 0)};

  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_TRUE(result[i].isApprox(expected[i]));
  }
}
