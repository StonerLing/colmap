// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>

namespace colmap {

struct PosePriorGlobalOrienterOptions {
  // Options for the robust baseline alignment RANSAC. `max_error` is the
  // angular inlier threshold (in radians) below which a baseline is considered
  // an inlier of a candidate rotation.
  RANSACOptions alignment_ransac_options;

  // Constrain the minimum number of views per track for the ray-consistency
  // refinement.
  int min_num_view_per_track = 3;

  // Scaling factor for the ray-consistency refinement loss function.
  // The residuals have length units (approximately the prior position error).
  double refinement_loss_scale = 1.0;

  // Minimum length of the prior baseline (in the prior coordinate units) for an
  // image pair to participate in the baseline alignment. Short baselines have
  // noisy directions due to prior position errors.
  double min_baseline_length = 0.1;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // The options for the solver.
  ceres::Solver::Options solver_options;

  PosePriorGlobalOrienterOptions() {
    // Default inlier threshold: 5 degrees.
    alignment_ransac_options.max_error = DegToRad(5.0);
    alignment_ransac_options.min_inlier_ratio = 0.5;
    alignment_ransac_options.confidence = 0.999;
    alignment_ransac_options.random_seed = -1;

    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction(double loss_scale) {
    return std::make_shared<ceres::HuberLoss>(loss_scale);
  }
};

// Robust estimator for the Wahba problem that aligns the baseline directions
// computed from the pose graph relative translations to the baseline directions
// computed from the prior positions. The model is the single rotation that maps
// the reconstruction frame to the prior frame, stored as a unit quaternion.
// The minimal sample is two non-parallel baseline direction pairs, which are
// aligned with a closed-form (SVD) solution.
struct OrientationAlignmentEstimator {
  using X_t = Eigen::Vector3d;
  using Y_t = Eigen::Vector3d;
  using M_t = Eigen::Quaterniond;

  // Number of baseline direction pairs required to determine a rotation.
  static const int kMinNumSamples = 2;

  // Estimate the rotation that best aligns the given reconstruction-frame
  // baseline directions `dirs_a` to the prior baseline directions `dirs_b`.
  static void Estimate(const std::vector<X_t>& dirs_a,
                       const std::vector<Y_t>& dirs_b,
                       std::vector<M_t>* models) {
    THROW_CHECK_EQ(dirs_a.size(), dirs_b.size());
    models->clear();

    if (dirs_a.size() < static_cast<size_t>(kMinNumSamples)) {
      return;
    }

    // Degenerate minimal sample: two nearly parallel baselines do not
    // determine a well-conditioned rotation, so skip this sample.
    if (std::abs(dirs_a[0].normalized().dot(dirs_a[1].normalized())) > 0.999) {
      return;
    }

    models->push_back(SolveWahbaCorrespondences(dirs_a, dirs_b));
  }

  // Squared chordal distance between the aligned baseline direction R * dirs_a
  // and the prior baseline direction dirs_b, which approximates the squared
  // angular error in radians.
  static void Residuals(const std::vector<X_t>& dirs_a,
                        const std::vector<Y_t>& dirs_b,
                        const M_t& model,
                        std::vector<double>* residuals) {
    THROW_CHECK_EQ(dirs_a.size(), dirs_b.size());
    const Eigen::Matrix3d R = model.toRotationMatrix();
    residuals->resize(dirs_a.size());
    for (size_t i = 0; i < dirs_a.size(); ++i) {
      (*residuals)[i] = (R * dirs_a[i] - dirs_b[i]).squaredNorm();
    }
  }

  // Closed-form (Kabsch) solution of the Wahba problem:
  //
  //     R = argmax_R sum_i dirs_b_i^T R dirs_a_i,
  //
  // computed via the SVD of B = sum_i dirs_b_i dirs_a_i^T.
  static Eigen::Quaterniond SolveWahbaCorrespondences(
      const std::vector<X_t>& dirs_a, const std::vector<Y_t>& dirs_b) {
    THROW_CHECK_EQ(dirs_a.size(), dirs_b.size());
    Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < dirs_a.size(); ++i) {
      B += dirs_b[i] * dirs_a[i].transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        B, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
      R = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) *
          svd.matrixV().transpose();
    }
    return Eigen::Quaterniond(R).normalized();
  }
};

class PosePriorGlobalOrienter {
 public:
  explicit PosePriorGlobalOrienter(
      const PosePriorGlobalOrienterOptions& options);

  bool Solve(const PoseGraph& pose_graph,
             const std::vector<PosePrior>& pose_priors,
             Reconstruction& reconstruction);

  PosePriorGlobalOrienterOptions& GetOptions() { return options_; }

 protected:
  // Solve a Wahba problem that aligns the baseline directions computed from
  // the pose graph relative translations to the prior baselines computed from
  // the prior positions.
  void AlignRotationsToPriorBaselines(
      const PoseGraph& pose_graph,
      const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior,
      Reconstruction& reconstruction);

  void SetupProblem();

  void AddRayConsistencyConstraints(
      Reconstruction& reconstruction,
      const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior);

  void ParameterizeVariables(Reconstruction& reconstruction);

  // Baseline direction correspondences between the reconstruction frame and the
  // prior frame, extracted from the pose graph relative translations.
  struct BaselineCorrespondences {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Reconstruction-frame baseline directions (unit vectors).
    std::vector<Eigen::Vector3d> baseline_dirs_in_world;
    // Prior baseline directions (unit vectors).
    std::vector<Eigen::Vector3d> baseline_dirs_prior;
  };

  // Extract the baseline direction correspondences from the pose graph
  // relative translations and the pose priors, filtered by the minimum prior
  // baseline length.
  BaselineCorrespondences ExtractBaselineCorrespondences(
      const PoseGraph& pose_graph,
      const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior,
      const Reconstruction& reconstruction) const;

  PosePriorGlobalOrienterOptions options_;
  std::unique_ptr<ceres::Problem> problem_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

// Solve global orienting using point-to-camera constraints.
bool RunPosePriorGlobalOrienting(const PosePriorGlobalOrienterOptions& options,
                                 const PoseGraph& pose_graph,
                                 const std::vector<PosePrior>& pose_priors,
                                 Reconstruction& reconstruction);

}  // namespace colmap
