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

#include "colmap/estimators/global_orienting.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"
#include <ceres/types.h>
#ifdef COLMAP_CUDA_ENABLED
#include "colmap/util/cuda.h"
#endif
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace colmap {
namespace {

NodeHashMap<image_t, PosePrior> ExtractImagePosePriors(
    const std::vector<PosePrior>& pose_priors) {
  NodeHashMap<image_t, PosePrior> image_id_to_pose_prior;
  for (const PosePrior& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    image_id_to_pose_prior.emplace(pose_prior.corr_data_id.id, pose_prior);
  }

  return image_id_to_pose_prior;
}

double PrintBaselineDirError(
    std::string_view header,
    const std::vector<Eigen::Vector3d>& baseline_dirs_in_world,
    const std::vector<Eigen::Vector3d>& baseline_dirs_prior,
    const Eigen::Quaterniond& prior_from_src) {
  THROW_CHECK_EQ(baseline_dirs_in_world.size(), baseline_dirs_prior.size());

  std::vector<double> angle_errors_rad;
  angle_errors_rad.reserve(baseline_dirs_in_world.size());

  for (size_t i = 0; i < baseline_dirs_in_world.size(); ++i) {
    const Eigen::Vector3d aligned_baseline =
        prior_from_src * baseline_dirs_in_world[i];
    const double cos_angle = aligned_baseline.dot(baseline_dirs_prior[i]);
    const double angle = std::acos(std::max(-1.0, std::min(1.0, cos_angle)));
    angle_errors_rad.push_back(angle);
  }

  if (!angle_errors_rad.empty()) {
    double sum_squared = std::accumulate(
        angle_errors_rad.cbegin(),
        angle_errors_rad.cend(),
        0.0,
        [](double res, double angle) { return res + angle * angle; });
    const double rmse_deg =
        RadToDeg(std::sqrt(sum_squared / angle_errors_rad.size()));
    const double median_deg = RadToDeg(Median(angle_errors_rad));

    VLOG(2) << header << "\n"
            << "  - rmse:   " << rmse_deg << " deg\n"
            << "  - median: " << median_deg << " deg\n";

    return median_deg;
  } else {
    VLOG(2) << "No valid baseline pairs for error evaluation.";
    return std::numeric_limits<double>::max();
  }
}

}  // namespace

PosePriorGlobalOrienter::PosePriorGlobalOrienter(
    const PosePriorGlobalOrienterOptions& options)
    : options_(options) {}

bool PosePriorGlobalOrienter::Solve(const PoseGraph& pose_graph,
                                    const std::vector<PosePrior>& pose_priors,
                                    Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    VLOG(1) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    VLOG(1) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  VLOG(1) << "Setting up the global orienter problem";

  NodeHashMap<image_t, PosePrior> image_id_to_pose_prior =
      ExtractImagePosePriors(pose_priors);

  // Align baselines to priors
  AlignRotationsToPriorBaselines(
      pose_graph, image_id_to_pose_prior, reconstruction);

  // Setup the problem.
  SetupProblem();

  // Add the point to camera constraints to the problem.
  AddRayConsistencyConstraints(reconstruction, image_id_to_pose_prior);

  // Parameterize the variables, set rotations to be constant if desired.
  ParameterizeVariables(reconstruction);

  VLOG(1) << "Solving the global orienter problem";

  ceres::Solver::Summary summary;
  ceres::Solver::Options solver_options = options_.solver_options;
  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
  solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    VLOG(2) << summary.FullReport();
  } else {
    VLOG(1) << summary.BriefReport();
  }

  return summary.IsSolutionUsable();
}

void PosePriorGlobalOrienter::SetupProblem() {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction(options_.refinement_loss_scale);
}

PosePriorGlobalOrienter::BaselineCorrespondences
PosePriorGlobalOrienter::ExtractBaselineCorrespondences(
    const PoseGraph& pose_graph,
    const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior,
    const Reconstruction& reconstruction) const {
  BaselineCorrespondences correspondences;

  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      continue;
    }

    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    if (!image1.HasPose() || !image2.HasPose()) {
      continue;
    }

    // A prior position for both images is required to form a baseline.
    const auto pose_prior1_it = image_id_to_pose_prior.find(image_id1);
    const auto pose_prior2_it = image_id_to_pose_prior.find(image_id2);
    if (pose_prior1_it == image_id_to_pose_prior.end() ||
        pose_prior2_it == image_id_to_pose_prior.end()) {
      continue;
    }
    const PosePrior& pose_prior1 = pose_prior1_it->second;
    const PosePrior& pose_prior2 = pose_prior2_it->second;
    if (!pose_prior1.HasPosition() || !pose_prior2.HasPosition()) {
      continue;
    }

    const Eigen::Vector3d tvec_in_cam1 =
        edge.cam2_from_cam1.rotation().inverse() *
        edge.cam2_from_cam1.translation();
    const Eigen::Vector3d dirs_a =
        (image1.CamFromWorld().rotation().inverse() * (-tvec_in_cam1))
            .normalized();

    const Eigen::Vector3d baseline_prior =
        pose_prior2.position - pose_prior1.position;
    if (baseline_prior.norm() < options_.min_baseline_length) {
      continue;
    }
    const Eigen::Vector3d dirs_b = baseline_prior.normalized();

    correspondences.baseline_dirs_in_world.push_back(dirs_a);
    correspondences.baseline_dirs_prior.push_back(dirs_b);
  }

  return correspondences;
}

void PosePriorGlobalOrienter::AlignRotationsToPriorBaselines(
    const PoseGraph& pose_graph,
    const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior,
    Reconstruction& reconstruction) {
  const BaselineCorrespondences correspondences =
      ExtractBaselineCorrespondences(
          pose_graph, image_id_to_pose_prior, reconstruction);

  if (correspondences.baseline_dirs_in_world.empty()) {
    return;
  }

  VLOG(1) << "Aligning " << correspondences.baseline_dirs_in_world.size()
          << " baseline directions to the prior baselines";
  PrintBaselineDirError("Baseline direction error before alignment:",
                        correspondences.baseline_dirs_in_world,
                        correspondences.baseline_dirs_prior,
                        Eigen::Quaterniond::Identity());

  RANSAC<OrientationAlignmentEstimator> ransac(
      options_.alignment_ransac_options);
  const RANSAC<OrientationAlignmentEstimator>::Report report =
      ransac.Estimate(correspondences.baseline_dirs_in_world,
                      correspondences.baseline_dirs_prior);

  Eigen::Quaterniond prior_from_src = Eigen::Quaterniond::Identity();
  if (report.success) {
    // Refine the closed-form solution on the consensus set of inliers.
    std::vector<Eigen::Vector3d> inlier_dirs;
    std::vector<Eigen::Vector3d> inlier_priors;
    inlier_dirs.reserve(correspondences.baseline_dirs_in_world.size());
    inlier_priors.reserve(correspondences.baseline_dirs_prior.size());
    for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
      if (report.inlier_mask[i]) {
        inlier_dirs.push_back(correspondences.baseline_dirs_in_world[i]);
        inlier_priors.push_back(correspondences.baseline_dirs_prior[i]);
      }
    }
    if (inlier_dirs.size() >=
        static_cast<size_t>(OrientationAlignmentEstimator::kMinNumSamples)) {
      prior_from_src = OrientationAlignmentEstimator::SolveWahbaCorrespondences(
          inlier_dirs, inlier_priors);
    } else {
      prior_from_src = report.model;
    }
  } else {
    // Degenerate data: fall back to the closed-form solution on all
    // correspondences.
    prior_from_src = OrientationAlignmentEstimator::SolveWahbaCorrespondences(
        correspondences.baseline_dirs_in_world,
        correspondences.baseline_dirs_prior);
  }

  PrintBaselineDirError("Baseline direction error after alignment:",
                        correspondences.baseline_dirs_in_world,
                        correspondences.baseline_dirs_prior,
                        prior_from_src);

  // Apply the rotation to the whole reconstruction (frames and points) so that
  // it maps the reconstruction frame to the prior frame.
  const Sim3d prior_from_src_tform(1, prior_from_src, Eigen::Vector3d::Zero());
  reconstruction.Transform(prior_from_src_tform);
}

void PosePriorGlobalOrienter::AddRayConsistencyConstraints(
    Reconstruction& reconstruction,
    const NodeHashMap<image_t, PosePrior>& image_id_to_pose_prior) {
  if (VLOG_IS_ON(2)) {
    VLOG(2) << reconstruction.NumPoints3D()
            << " point to camera constraints were added to the orientation "
               "estimation problem.";
  }

  for (auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    // For each view in the track add the ray-consistency constraint.
    for (const auto& observation : point3D.track.Elements()) {
      if (!reconstruction.ExistsImage(observation.image_id)) {
        continue;
      }

      Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) {
        continue;
      }

      // A prior position is required to build the ray-consistency residual.
      const auto pose_prior_it =
          image_id_to_pose_prior.find(observation.image_id);
      if (pose_prior_it == image_id_to_pose_prior.end()) {
        continue;
      }
      const Eigen::Vector3d prior_position = pose_prior_it->second.position;

      Camera& camera = reconstruction.Camera(image.CameraId());
      const std::optional<Eigen::Vector3d> cam_ray =
          camera.CamRayFromImg(image.Point2D(observation.point2D_idx).xy);
      if (!cam_ray.has_value()) {
        VLOG(2) << "Ignoring feature because it failed to project: point3D_id="
                << point3D_id << ", image_id=" << observation.image_id
                << ", feature_id=" << observation.point2D_idx;
        continue;
      }

      ceres::CostFunction* cost_function =
          PriorPositionRayConsistencyCostFunctor::Create(*cam_ray,
                                                         prior_position);

      double* rotation = image.FramePtr()->RigFromWorld().params.data();
      problem_->AddResidualBlock(cost_function,
                                 loss_function_.get(),
                                 rotation,
                                 reconstruction.Point3D(point3D_id).xyz.data());
    }
  }
}

void PosePriorGlobalOrienter::ParameterizeVariables(
    Reconstruction& reconstruction) {
  // Optimize the rotations on the unit quaternion manifold.
  for (image_t image_id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) {
      continue;
    }
    double* rotation = image.FramePtr()->RigFromWorld().params.data();
    if (problem_->HasParameterBlock(rotation)) {
      SetManifold(problem_.get(), rotation, CreateEigenQuaternionManifold());
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  constexpr int kMaxNumImagesDirectSolver = 1500;
  if (reconstruction.NumRegImages() > kMaxNumImagesDirectSolver) {
    options_.solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  }
}

bool RunPosePriorGlobalOrienting(const PosePriorGlobalOrienterOptions& options,
                                 const PoseGraph& pose_graph,
                                 const std::vector<PosePrior>& pose_priors,
                                 Reconstruction& reconstruction) {
  PosePriorGlobalOrienter orienter(options);
  return orienter.Solve(pose_graph, pose_priors, reconstruction);
}

}  // namespace colmap
