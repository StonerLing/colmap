#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <optional>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

// Computes the covariance of a BATA residual when prior positions are used.
// The angular measurement noise of a camera with a prior focal length is
// inversely proportional to the focal length, so the residual standard
// deviation is the reciprocal of the focal length. Returns std::nullopt if the
// camera has no prior focal length.
std::optional<Eigen::Matrix3d> PriorPositionBataCovariance(
    const Camera& camera) {
  if (!camera.has_prior_focal_length) {
    return std::nullopt;
  }
  const double stddev = 1.0 / camera.MeanFocalLength();
  return stddev * stddev * Eigen::Matrix3d::Identity();
}

// Returns the world position of the frame center implied by a pose prior on
// the sensor. For non-reference sensors, the fixed cam_from_rig offset is
// folded into the prior. Returns std::nullopt if the sensor position cannot be
// derived from the frame center alone (e.g., when cam_from_rig is estimated).
std::optional<Eigen::Vector3d> FrameCenterPriorPosition(
    const Reconstruction& reconstruction,
    const Image& image,
    const PosePrior& pose_prior) {
  Eigen::Vector3d prior_position = pose_prior.position;
  if (!image.IsRefInFrame()) {
    const Rigid3d& cam_from_rig =
        reconstruction.Rig(image.FramePtr()->RigId())
            .SensorFromRig(image.CameraPtr()->SensorId());
    if (cam_from_rig.translation().hasNaN()) {
      return std::nullopt;
    }
    const Eigen::Matrix3d world_from_rig_rotation =
        image.FramePtr()->RigFromWorld().rotation().toRotationMatrix().transpose();
    prior_position -= world_from_rig_rotation * cam_from_rig.translation();
  }
  return prior_position;
}

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction,
                             const std::vector<PosePrior>& pose_priors) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Whether prior positions are used to constrain and initialize positions.
  const bool use_prior_position =
      options_.use_prior_position && !pose_priors.empty();

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random, or from the pose priors if
  // prior positions are used. Also, convert the camera pose translation to be
  // the camera center.
  InitializeRandomPositions(
      pose_graph, reconstruction, use_prior_position, pose_priors);

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction, use_prior_position);

  // Add the prior position constraints to the problem.
  bool added_prior_position_constraints = false;
  if (use_prior_position) {
    added_prior_position_constraints =
        AddPriorPositionConstraints(reconstruction, pose_priors);
  }

  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction, added_prior_position_constraints);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertBackResults(reconstruction);
  if (VLOG_IS_ON(2)) {
    PrintPositionError(reconstruction, pose_priors);
  }
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid tracks, the actual number of residuals may be
  // smaller.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    bool use_prior_position,
    const std::vector<PosePrior>& pose_priors) {
  FlatHashSet<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    if (image1.HasPose()) {
      constrained_positions.insert(image1.FrameId());
    }
    if (image2.HasPose()) {
      constrained_positions.insert(image2.FrameId());
    }
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  // Map pose priors to frame center positions for initialization.
  FlatHashMap<frame_t, Eigen::Vector3d> prior_positions;
  if (use_prior_position) {
    for (const auto& pose_prior : pose_priors) {
      if (!pose_prior.HasPosition() ||
          pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
        continue;
      }
      const image_t image_id = pose_prior.corr_data_id.id;
      if (!reconstruction.ExistsImage(image_id)) {
        continue;
      }
      const Image& image = reconstruction.Image(image_id);
      if (!image.HasPose()) {
        continue;
      }
      const std::optional<Eigen::Vector3d> frame_center_prior_position =
          FrameCenterPriorPosition(reconstruction, image, pose_prior);
      if (frame_center_prior_position.has_value()) {
        prior_positions[image.FrameId()] = *frame_center_prior_position;
      }
    }
  }

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    const auto prior_itr = prior_positions.find(frame_id);
    if (prior_itr != prior_positions.end()) {
      frame_centers_[frame_id] = prior_itr->second;
      continue;
    }
    if (options_.generate_random_positions && options_.optimize_positions) {
      frame_centers_[frame_id] = 100.0 * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size()
          << ", prior positions: " << prior_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction, bool use_prior_position) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
      loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  loss_function_ptcam_calibrated_ = loss_function_;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction, use_prior_position);
  }
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction,
                                           bool use_prior_position) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = 100.0 * RandVector3d(-1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : point3D.track.Elements()) {
    if (!reconstruction.ExistsImage(observation.image_id)) continue;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) continue;

    const std::optional<Eigen::Vector3d> cam_ray =
        image.CameraPtr()->CamRayFromImg(
            image.Point2D(observation.point2D_idx).xy);
    if (!cam_ray.has_value()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to project: point3D_id="
          << point3D_id << ", image_id=" << observation.image_id
          << ", feature_id=" << observation.point2D_idx;
      continue;
    }

    const Eigen::Vector3d cam_from_point3D_dir =
        image.CamFromWorld().rotation().inverse() * (*cam_ray);

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    if (!options_.generate_scales && random_initialization) {
      const Eigen::Vector3d cam_from_point3D_translation =
          point3D.xyz - frame_centers_[image.FrameId()];
      scale = std::max(1e-5,
                       cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                           cam_from_point3D_translation.squaredNorm());
    }

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    // Down weight the uncalibrated cameras
    Camera& camera = reconstruction.Camera(image.CameraId());
    ceres::LossFunction* loss_function =
        (camera.has_prior_focal_length)
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // When prior positions are used, weight the BATA residuals with the
    // measurement covariance derived from the prior focal length.
    const std::optional<Eigen::Matrix3d> bata_cov =
        use_prior_position ? PriorPositionBataCovariance(camera) : std::nullopt;

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.IsRefInFrame()) {
      ceres::CostFunction* cost_function = nullptr;
      if (bata_cov.has_value()) {
        cost_function =
            CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>::
                Create(*bata_cov, cam_from_point3D_dir);
      } else {
        cost_function =
            BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);
      }

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 frame_centers_[image.FrameId()].data(),
                                 point3D.xyz.data(),
                                 &scale);
    } else {
      // If the image is part of a camera rig, use the RigBATA error.

      const rig_t rig_id = image.FramePtr()->RigId();
      Rig& rig = reconstruction.Rig(rig_id);
      Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

      if (!cam_from_rig.translation().hasNaN()) {
        const Eigen::Vector3d cam_from_rig_dir =
            image.CamFromWorld().rotation().inverse() *
            cam_from_rig.translation();

        ceres::CostFunction* cost_function = nullptr;
        if (bata_cov.has_value()) {
          cost_function =
              CovarianceWeightedCostFunctor<
                  RigBATAPairwiseDirectionConstantRigCostFunctor>::
                  Create(*bata_cov, cam_from_point3D_dir, cam_from_rig_dir);
        } else {
          cost_function =
              RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
                  cam_from_point3D_dir, cam_from_rig_dir);
        }

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   &scale);
      } else {
        // NaN translation means the sensor's cam_from_rig must be
        // re-estimated, which requires refine_sensor_from_rig=true.
        THROW_CHECK(options_.refine_sensor_from_rig)
            << "sensor_from_rig has NaN translation but "
               "refine_sensor_from_rig=false (image_id="
            << observation.image_id << ")";
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
          // Will be initialized to random values in ParameterizeVariables().
          cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
        }

        ceres::CostFunction* cost_function = nullptr;
        if (bata_cov.has_value()) {
          cost_function =
              CovarianceWeightedCostFunctor<
                  RigBATAPairwiseDirectionCostFunctor>::
                  Create(*bata_cov,
                         cam_from_point3D_dir,
                         image.FramePtr()->RigFromWorld().rotation());
        } else {
          cost_function = RigBATAPairwiseDirectionCostFunctor::Create(
              cam_from_point3D_dir,
              image.FramePtr()->RigFromWorld().rotation());
        }

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   cams_in_rig_[sensor_id].data(),
                                   &scale);
      }
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }
}

bool GlobalPositioner::AddPriorPositionConstraints(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) {
  loss_function_prior_position_ =
      std::make_shared<ceres::HuberLoss>(options_.pp_loss_scale);

  const Eigen::Matrix3d fallback_cov =
      options_.pp_fallback_stddev * options_.pp_fallback_stddev *
      Eigen::Matrix3d::Identity();

  size_t num_added_constraints = 0;
  for (const auto& pose_prior : pose_priors) {
    if (!pose_prior.HasPosition()) {
      continue;
    }
    if (pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    const image_t image_id = pose_prior.corr_data_id.id;
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) {
      continue;
    }

    const frame_t frame_id = image.FrameId();
    const auto frame_center_itr = frame_centers_.find(frame_id);
    if (frame_center_itr == frame_centers_.end()) {
      continue;
    }
    double* frame_center = frame_center_itr->second.data();
    if (!problem_->HasParameterBlock(frame_center)) {
      continue;
    }

    const std::optional<Eigen::Vector3d> frame_center_prior_position =
        FrameCenterPriorPosition(reconstruction, image, pose_prior);
    if (!frame_center_prior_position.has_value()) {
      continue;
    }

    const Eigen::Matrix3d position_cov =
        pose_prior.HasPositionCov() ? pose_prior.position_covariance
                                    : fallback_cov;
    ceres::CostFunction* cost_function =
        CovarianceWeightedCostFunctor<AbsolutePositionPriorCostFunctor>::Create(
            position_cov, *frame_center_prior_position);
    problem_->AddResidualBlock(
        cost_function, loss_function_prior_position_.get(), frame_center);
    num_added_constraints++;
  }

  VLOG(2) << num_added_constraints
          << " prior position constraints were added to the position "
             "estimation problem.";

  return num_added_constraints > 0;
}

void GlobalPositioner::PrintPositionError(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) const {
  std::vector<double> verr2_wrt_prior;
  verr2_wrt_prior.reserve(pose_priors.size());
  for (const auto& pose_prior : pose_priors) {
    if (!pose_prior.HasPosition() ||
        pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    const image_t image_id = pose_prior.corr_data_id.id;
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) {
      continue;
    }
    verr2_wrt_prior.push_back(
        (image.ProjectionCenter() - pose_prior.position).squaredNorm());
  }
  if (verr2_wrt_prior.empty()) {
    return;
  }
  LOG(INFO) << "Optimization error w.r.t. prior positions:"
            << "\n"
            << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << '\n'
            << "  - median: " << std::sqrt(Median(verr2_wrt_prior));
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (reconstruction.NumPoints3D() > 0) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(
    Reconstruction& reconstruction, bool added_prior_position_constraints) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // Initialize cams_in_rig_ with random values if optimizing positions.
  if (options_.optimize_positions) {
    for (auto& [sensor_id, center] : cams_in_rig_) {
      if (problem_->HasParameterBlock(center.data())) {
        center = RandVector3d(-1, 1);
      }
    }
  }

  // If not optimizing positions, set frame centers to be constant.
  if (!options_.optimize_positions) {
    for (auto& [frame_id, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
      }
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
      }
    }
  }
  // Set the first scale to be constant to remove the gauge ambiguity.
  // Prior position constraints provide the gauge, so the scale is not fixed
  // in that case.
  if (!added_prior_position_constraints) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
        break;
      }
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

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  // TODO: Investigate whether the direct solver should be chosen
  // adaptively based on problem scale.
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  // Convert optimized frame centers back to rig_from_world translations.
  for (const auto& [frame_id, center] : frame_centers_) {
    if (!reconstruction.Frame(frame_id).HasPose()) {
      continue;
    }
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
  }

  for (const auto& [sensor_id, center] : cams_in_rig_) {
    // Find the rig containing this sensor.
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      if (!rig.HasSensor(sensor_id)) {
        continue;
      }
      Rigid3d& sensor_from_rig =
          reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
      sensor_from_rig.translation() = sensor_from_rig.rotation() * -center;
      break;
    }
  }
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction, pose_priors);
}

}  // namespace colmap
