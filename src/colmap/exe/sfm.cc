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

#include "colmap/exe/sfm.h"

#include "colmap/controllers/automatic_reconstruction.h"
#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/exe/gui.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {

std::pair<std::vector<image_t>, std::vector<Eigen::Vector3d>>
ExtractExistingImages(const Reconstruction& reconstruction) {
  std::vector<image_t> fixed_image_ids = reconstruction.RegImageIds();
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  orig_fixed_image_positions.reserve(fixed_image_ids.size());
  for (const image_t image_id : fixed_image_ids) {
    orig_fixed_image_positions.push_back(
        reconstruction.Image(image_id).ProjectionCenter());
  }
  return {std::move(fixed_image_ids), std::move(orig_fixed_image_positions)};
}

void UpdateDatabasePosePriorsCovariance(const std::string& database_path,
                                        const Eigen::Matrix3d& covariance) {
  Database database(database_path);
  DatabaseTransaction database_transaction(&database);

  LOG(INFO)
      << "Setting up database pose priors with the same covariance matrix: \n"
      << covariance << '\n';

  for (const auto& image : database.ReadAllImages()) {
    if (database.ExistsPosePrior(image.ImageId())) {
      PosePrior prior = database.ReadPosePrior(image.ImageId());
      prior.position_covariance = covariance;
      database.UpdatePosePrior(image.ImageId(), prior);
    }
  }
}

}  // namespace

int RunAutomaticReconstructor(int argc, char** argv) {
  AutomaticReconstructionController::Options reconstruction_options;
  std::string image_list_path;
  std::string data_type = "individual";
  std::string quality = "high";
  std::string mesher = "poisson";

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("mask_path", &reconstruction_options.mask_path);
  options.AddDefaultOption("vocab_tree_path",
                           &reconstruction_options.vocab_tree_path);
  options.AddDefaultOption(
      "data_type", &data_type, "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
  options.AddDefaultOption("single_camera_per_folder",
                           &reconstruction_options.single_camera_per_folder);
  options.AddDefaultOption("camera_params",
                           &reconstruction_options.camera_params);
  options.AddDefaultOption("extraction", &reconstruction_options.extraction);
  options.AddDefaultOption("matching", &reconstruction_options.matching);
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("mesher", &mesher, "{poisson, delaunay}");
  options.AddDefaultOption("num_threads", &reconstruction_options.num_threads);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("gpu_index", &reconstruction_options.gpu_index);
  options.Parse(argc, argv);

  if (!image_list_path.empty()) {
    reconstruction_options.image_names = ReadTextFileLines(image_list_path);
  }

  StringToUpper(&data_type);
  reconstruction_options.data_type =
      AutomaticReconstructionController::DataTypeFromString(data_type);

  StringToUpper(&quality);
  reconstruction_options.quality =
      AutomaticReconstructionController::QualityFromString(quality);

  StringToUpper(&mesher);
  reconstruction_options.mesher =
      AutomaticReconstructionController::MesherFromString(mesher);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  if (reconstruction_options.use_gpu && kUseOpenGL &&
      (reconstruction_options.extraction || reconstruction_options.matching)) {
    QApplication app(argc, argv);
    AutomaticReconstructionController controller(reconstruction_options,
                                                 reconstruction_manager);
    RunThreadWithOpenGLContext(&controller);
  } else {
    AutomaticReconstructionController controller(reconstruction_options,
                                                 reconstruction_manager);
    controller.Start();
    controller.Wait();
  }

  return EXIT_SUCCESS;
}

int RunBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  BundleAdjustmentController ba_controller(options, reconstruction);
  ba_controller.Run();

  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunColorExtractor(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.ExtractColorsForAllImages(*options.image_path);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunMapper(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    options.mapper->image_names = ReadTextFileLines(image_list_path);
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      LOG(ERROR) << "`input_path` is not a directory.";
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  // If fix_existing_frames is enabled, we store the initial positions of
  // existing images in order to transform them back to the original coordinate
  // frame, as the reconstruction is normalized multiple times for numerical
  // stability.
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  std::vector<image_t> fixed_image_ids;
  if (options.mapper->fix_existing_frames &&
      reconstruction_manager->Size() > 0) {
    std::tie(fixed_image_ids, orig_fixed_image_positions) =
        ExtractExistingImages(*reconstruction_manager->Get(0));
  }

  IncrementalPipeline mapper(options.mapper,
                             *options.image_path,
                             *options.database_path,
                             reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (input_path == "") {
    mapper.AddCallback(IncrementalPipeline::LAST_IMAGE_REG_CALLBACK, [&]() {
      // If the number of reconstructions has not changed, the last model
      // was discarded for some reason.
      if (reconstruction_manager->Size() > prev_num_reconstructions) {
        const std::string reconstruction_path =
            JoinPaths(output_path, std::to_string(prev_num_reconstructions));
        CreateDirIfNotExists(reconstruction_path);
        reconstruction_manager->Get(prev_num_reconstructions)
            ->Write(reconstruction_path);
        options.Write(JoinPaths(reconstruction_path, "project.ini"));
        prev_num_reconstructions = reconstruction_manager->Size();
      }
    });
  }

  mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (input_path != "") {
    const auto& reconstruction = reconstruction_manager->Get(0);

    // Transform the final reconstruction back to the original coordinate frame.
    if (options.mapper->fix_existing_frames) {
      if (fixed_image_ids.size() < 3) {
        LOG(WARNING) << "Too few images to transform the reconstruction.";
      } else {
        std::vector<Eigen::Vector3d> new_fixed_image_positions;
        new_fixed_image_positions.reserve(fixed_image_ids.size());
        for (const image_t image_id : fixed_image_ids) {
          new_fixed_image_positions.push_back(
              reconstruction->Image(image_id).ProjectionCenter());
        }
        Sim3d orig_from_new;
        if (EstimateSim3d(new_fixed_image_positions,
                          orig_fixed_image_positions,
                          orig_from_new)) {
          reconstruction->Transform(orig_from_new);
        } else {
          LOG(WARNING) << "Failed to transform the reconstruction back "
                          "to the input coordinate frame.";
        }
      }
    }

    reconstruction->Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalPipeline::Options mapper_options;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &mapper_options.database_path);
  options.AddRequiredOption("image_path", &mapper_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_workers", &mapper_options.num_workers);
  options.AddDefaultOption("image_overlap",
                           &mapper_options.clustering_options.image_overlap);
  options.AddDefaultOption(
      "leaf_max_num_images",
      &mapper_options.clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  mapper_options.incremental_options = *options.mapper;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipeline hierarchical_mapper(mapper_options,
                                           reconstruction_manager);
  hierarchical_mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  reconstruction_manager->Write(output_path);
  options.Write(JoinPaths(output_path, "project.ini"));

  return EXIT_SUCCESS;
}

int RunPosePriorMapper(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  bool overwrite_priors_covariance = false;
  double prior_position_std_x = 1.;
  double prior_position_std_y = 1.;
  double prior_position_std_z = 1.;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();

  options.mapper->use_prior_position = true;

  options.AddDefaultOption(
      "overwrite_priors_covariance",
      &overwrite_priors_covariance,
      "Priors covariance read from database. If true, overwrite the priors "
      "covariance using the follwoing prior_position_std_... options");
  options.AddDefaultOption("prior_position_std_x", &prior_position_std_x);
  options.AddDefaultOption("prior_position_std_y", &prior_position_std_y);
  options.AddDefaultOption("prior_position_std_z", &prior_position_std_z);
  options.AddDefaultOption("use_robust_loss_on_prior_position",
                           &options.mapper->use_robust_loss_on_prior_position);
  options.AddDefaultOption("prior_position_loss_scale",
                           &options.mapper->prior_position_loss_scale);
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  if (overwrite_priors_covariance) {
    const Eigen::Matrix3d covariance =
        Eigen::Vector3d(
            prior_position_std_x, prior_position_std_y, prior_position_std_z)
            .cwiseAbs2()
            .asDiagonal();
    UpdateDatabasePosePriorsCovariance(*options.database_path, covariance);
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      LOG(ERROR) << "`input_path` is not a directory.";
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  // If fix_existing_frames is enabled, we store the initial positions of
  // existing images in order to transform them back to the original coordinate
  // frame, as the reconstruction is normalized multiple times for numerical
  // stability.
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  std::vector<image_t> fixed_image_ids;
  if (options.mapper->fix_existing_frames &&
      reconstruction_manager->Size() > 0) {
    std::tie(fixed_image_ids, orig_fixed_image_positions) =
        ExtractExistingImages(*reconstruction_manager->Get(0));
  }

  IncrementalPipeline mapper(options.mapper,
                             *options.image_path,
                             *options.database_path,
                             reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (input_path == "") {
    mapper.AddCallback(IncrementalPipeline::LAST_IMAGE_REG_CALLBACK, [&]() {
      // If the number of reconstructions has not changed, the last model
      // was discarded for some reason.
      if (reconstruction_manager->Size() > prev_num_reconstructions) {
        const std::string reconstruction_path =
            JoinPaths(output_path, std::to_string(prev_num_reconstructions));
        CreateDirIfNotExists(reconstruction_path);
        reconstruction_manager->Get(prev_num_reconstructions)
            ->Write(reconstruction_path);
        options.Write(JoinPaths(reconstruction_path, "project.ini"));
        prev_num_reconstructions = reconstruction_manager->Size();
      }
    });
  }

  mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (input_path != "") {
    const auto& reconstruction = reconstruction_manager->Get(0);

    // Transform the final reconstruction back to the original coordinate frame.
    if (options.mapper->fix_existing_frames) {
      if (fixed_image_ids.size() < 3) {
        LOG(WARNING) << "Too few images to transform the reconstruction.";
      } else {
        std::vector<Eigen::Vector3d> new_fixed_image_positions;
        new_fixed_image_positions.reserve(fixed_image_ids.size());
        for (const image_t image_id : fixed_image_ids) {
          new_fixed_image_positions.push_back(
              reconstruction->Image(image_id).ProjectionCenter());
        }
        Sim3d orig_from_new;
        if (EstimateSim3d(new_fixed_image_positions,
                          orig_fixed_image_positions,
                          orig_from_new)) {
          reconstruction->Transform(orig_from_new);
        } else {
          LOG(WARNING) << "Failed to transform the reconstruction back "
                          "to the input coordinate frame.";
        }
      }
    }

    reconstruction->Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunPointFiltering(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  size_t min_track_len = 2;
  double max_reproj_error = 4.0;
  double min_tri_angle = 1.5;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_track_len", &min_track_len);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("min_tri_angle", &min_tri_angle);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  size_t num_filtered = ObservationManager(reconstruction)
                            .FilterAllPoints3D(max_reproj_error, min_tri_angle);

  for (const auto point3D_id : reconstruction.Point3DIds()) {
    const auto& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() < min_track_len) {
      num_filtered += point3D.track.Length();
      reconstruction.DeletePoint3D(point3D_id);
    }
  }

  LOG(INFO) << "Filtered observations: " << num_filtered;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunPointTriangulator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  bool clear_points = true;
  bool refine_intrinsics = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points",
      &clear_points,
      "Whether to clear all existing points and observations and recompute "
      "the image_ids based on matching filenames between the model and the "
      "database");
  options.AddDefaultOption("refine_intrinsics",
                           &refine_intrinsics,
                           "Whether to refine the intrinsics of the cameras "
                           "(fixing the principal point)");
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  PrintHeading1("Loading model");

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  RunPointTriangulatorImpl(reconstruction,
                           *options.database_path,
                           *options.image_path,
                           output_path,
                           *options.mapper,
                           clear_points,
                           refine_intrinsics);
  return EXIT_SUCCESS;
}

void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const IncrementalPipelineOptions& options,
    const bool clear_points,
    const bool refine_intrinsics) {
  THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
      << "Need at least two images for triangulation";
  if (clear_points) {
    const Database database(database_path);
    reconstruction->DeleteAllPoints2DAndPoints3D();
    reconstruction->TranscribeImageIdsToDatabase(database);
  }

  auto options_tmp = std::make_shared<IncrementalPipelineOptions>(options);
  options_tmp->fix_existing_frames = true;
  options_tmp->ba_refine_focal_length = refine_intrinsics;
  options_tmp->ba_refine_principal_point = false;
  options_tmp->ba_refine_extra_params = refine_intrinsics;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(
      options_tmp, image_path, database_path, reconstruction_manager);
  mapper.TriangulateReconstruction(reconstruction);
  reconstruction->Write(output_path);
}

// TODO: Remove once version 3.12 is released.
int RunRigBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string rig_config_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  LOG(WARNING)
      << "rig_bundle_adjuster is deprecated and will be removed in the next "
         "version, run rig_configurator and bundle_adjuster instead.";

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  Database database(Database::kInMemoryDatabasePath);
  for (const auto& [_, camera] : reconstruction.Cameras()) {
    database.WriteCamera(camera, /*use_camera_id=*/true);
  }
  for (const auto& [image_id, image] : reconstruction.Images()) {
    database.WriteImage(image, /*use_image_id=*/true);
    config.AddImage(image_id);
  }
  ApplyRigConfig(ReadRigConfig(rig_config_path), database, &reconstruction);

  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      *options.bundle_adjustment, std::move(config), reconstruction);
  if (bundle_adjuster->Solve().termination_type == ceres::FAILURE) {
    LOG(ERROR) << "Failed to solve rig bundle adjustment";
    return EXIT_FAILURE;
  }
  reconstruction.UpdatePoint3DErrors();
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
