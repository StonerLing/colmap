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

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// 6-DoF error on the absolute sensor pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// sensor frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 7> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world) *
        world_from_sensor_prior_.rotation().cast<T>();
    AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
                                 residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(sensor_from_world + 4) +
        EigenQuaternionMap<T>(sensor_from_world) *
            world_from_sensor_prior_.translation().cast<T>();

    return true;
  }

 private:
  const Rigid3d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor, 3, 7> {
 public:
  explicit AbsolutePosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = position_in_world_prior_.cast<T>() +
                EigenQuaternionMap<T>(sensor_from_world).inverse() *
                    EigenVector3Map<T>(sensor_from_world + 4);
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on a position in the world coordinate frame, given as a plain
// 3-vector (e.g., a camera frame center in global positioning).
struct AbsolutePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePositionPriorCostFunctor, 3, 3> {
 public:
  explicit AbsolutePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const position_in_world, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position_in_world) -
                position_in_world_prior_.template cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
 public:
  explicit AbsoluteRigPosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
        EigenVector3Map<T>(sensor_from_rig + 4) +
        EigenQuaternionMap<T>(sensor_from_rig) *
            EigenVector3Map<T>(rig_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals =
        position_in_world_prior_.cast<T>() +
        sensor_from_world_rotation.inverse() * sensor_from_world_translation;
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 6-DoF error between two absolute camera poses based on a prior on their
// relative pose, with identical scale for the translation. The residual is
// computed in the frame of camera i. Its first and last three components
// correspond to the rotation and translation errors, respectively.
//
// Derivation:
//    i_T_w = ΔT_i·i_T_j·j_T_w
//    where ΔT_i = exp(η_i) is the resjdual in SE(3) and η_i in tangent space.
//    Thus η_i = log(i_T_w·j_T_w⁻¹·j_T_i)
//    Rotation term: ΔR = log(i_R_w·j_R_w⁻¹·j_R_i)
//    Translation term: Δt = i_t_w + i_R_w·j_R_w⁻¹·(j_t_i -j_t_w)
struct RelativePosePriorCostFunctor
    : public AutoDiffCostFunctor<RelativePosePriorCostFunctor, 6, 7, 7> {
 public:
  explicit RelativePosePriorCostFunctor(const Rigid3d& i_from_j_prior)
      : j_from_i_prior_(Inverse(i_from_j_prior)) {}

  template <typename T>
  bool operator()(const T* const i_from_world,
                  const T* const j_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> i_from_j_rotation =
        EigenQuaternionMap<T>(i_from_world) *
        EigenQuaternionMap<T>(j_from_world).inverse();
    const Eigen::Quaternion<T> param_from_prior_rotation =
        i_from_j_rotation * j_from_i_prior_.rotation().template cast<T>();
    AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
                                 residuals_ptr);

    const Eigen::Matrix<T, 3, 1> j_from_i_prior_translation =
        j_from_i_prior_.translation().cast<T>() -
        EigenVector3Map<T>(j_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(i_from_world + 4) +
        i_from_j_rotation * j_from_i_prior_translation;

    return true;
  }

 private:
  const Rigid3d j_from_i_prior_;
};

// Scalar coplanarity residual enforcing the epipolar constraint between the
// cam rays of a correspondence and the prior baseline direction. The residual
// is the dot product of the prior baseline with the cross product of the two
// cam rays rotated into the world frame, i.e.
// r = baseline * ((R1^-1 * cam_ray1) x (R2^-1 * cam_ray2)), which vanishes
// when the two cam rays and the baseline are coplanar in the world frame.
class PriorBaselineCoplanarityCostFunctor
    : public AutoDiffCostFunctor<PriorBaselineCoplanarityCostFunctor, 1, 4, 4> {
 public:
  explicit PriorBaselineCoplanarityCostFunctor(const Eigen::Vector3d& baseline,
                                               const Eigen::Vector3d& cam_ray1,
                                               const Eigen::Vector3d& cam_ray2)
      : baseline_(baseline.normalized()),
        cam_ray1_(cam_ray1),
        cam_ray2_(cam_ray2) {}

  template <typename T>
  bool operator()(const T* const rotation1_cam_from_world_ptr,
                  const T* const rotation2_cam_from_world_ptr,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> rotation1 =
        EigenQuaternionMap<T>(rotation1_cam_from_world_ptr);
    const Eigen::Quaternion<T> rotation2 =
        EigenQuaternionMap<T>(rotation2_cam_from_world_ptr);

    const Eigen::Matrix<T, 3, 1> ray1_world =
        rotation1.inverse() * cam_ray1_.cast<T>();
    const Eigen::Matrix<T, 3, 1> ray2_world =
        rotation2.inverse() * cam_ray2_.cast<T>();

    residuals_ptr[0] = baseline_.cast<T>().dot(ray1_world.cross(ray2_world));
    return true;
  }

 private:
  Eigen::Vector3d baseline_;
  Eigen::Vector3d cam_ray1_;
  Eigen::Vector3d cam_ray2_;
};

// Ray-consistency residual between a 3D point and a prior camera position.
// The residual is the cross product of the world-frame camera ray (obtained by
// rotating the observed camera ray by the inverse of the cam_from_world
// rotation) with the vector from the prior camera center to the 3D point, i.e.
// r = (R^-1 * cam_ray) x (point3D - prior_position), which vanishes when the
// 3D point lies on the line through the prior camera center along the
// observed camera ray. This couples the global rotations with the prior
// positions through the reconstructed 3D points.
class PriorPositionRayConsistencyCostFunctor
    : public AutoDiffCostFunctor<PriorPositionRayConsistencyCostFunctor, 3, 4, 3> {
 public:
  explicit PriorPositionRayConsistencyCostFunctor(
      const Eigen::Vector3d& cam_ray, const Eigen::Vector3d& prior_position)
      : cam_ray_(cam_ray), prior_position_(prior_position) {}

  template <typename T>
  bool operator()(const T* const rotation_cam_from_world_ptr,
                  const T* const point3D_ptr,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> rotation =
        EigenQuaternionMap<T>(rotation_cam_from_world_ptr);
    const Eigen::Matrix<T, 3, 1> ray_world =
        rotation.inverse() * cam_ray_.cast<T>();
    const Eigen::Matrix<T, 3, 1> point3D =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D_ptr);
    const Eigen::Matrix<T, 3, 1> residual =
        (point3D - prior_position_.cast<T>()).cross(ray_world);
    residuals_ptr[0] = residual(0);
    residuals_ptr[1] = residual(1);
    residuals_ptr[2] = residual(2);
    return true;
  }

 private:
  Eigen::Vector3d cam_ray_;
  Eigen::Vector3d prior_position_;
};

}  // namespace colmap
