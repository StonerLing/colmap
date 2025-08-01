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

#include "colmap/util/logging.h"

#include <memory>

#include <boost/program_options.hpp>

namespace colmap {

struct ImageReaderOptions;
struct FeatureExtractionOptions;
struct FeatureMatchingOptions;
struct SiftMatchingOptions;
struct TwoViewGeometryOptions;
struct ExhaustivePairingOptions;
struct SequentialPairingOptions;
struct VocabTreePairingOptions;
struct SpatialPairingOptions;
struct TransitivePairingOptions;
struct ImportedPairingOptions;
struct BundleAdjustmentOptions;
struct IncrementalPipelineOptions;
struct RenderOptions;

namespace mvs {
struct PatchMatchOptions;
struct StereoFusionOptions;
struct PoissonMeshingOptions;
struct DelaunayMeshingOptions;
}  // namespace mvs

class OptionManager {
 public:
  explicit OptionManager(bool add_project_options = true);

  // Create "optimal" set of options for different reconstruction scenarios.
  // Note that the existing options are modified, so if your parameters are
  // already low quality, they will be further modified.
  void ModifyForIndividualData();
  void ModifyForVideoData();
  void ModifyForInternetData();

  // Create "optimal" set of options for different quality settings.
  // Note that the existing options are modified, so if your parameters are
  // already low quality, they will be further degraded.
  void ModifyForLowQuality();
  void ModifyForMediumQuality();
  void ModifyForHighQuality();
  void ModifyForExtremeQuality();

  void AddAllOptions();
  void AddLogOptions();
  void AddRandomOptions();
  void AddDatabaseOptions();
  void AddImageOptions();
  void AddExtractionOptions();
  void AddMatchingOptions();
  void AddExhaustivePairingOptions();
  void AddSequentialPairingOptions();
  void AddVocabTreePairingOptions();
  void AddSpatialPairingOptions();
  void AddTransitivePairingOptions();
  void AddImportedPairingOptions();
  void AddBundleAdjustmentOptions();
  void AddMapperOptions();
  void AddPatchMatchStereoOptions();
  void AddStereoFusionOptions();
  void AddPoissonMeshingOptions();
  void AddDelaunayMeshingOptions();
  void AddRenderOptions();

  template <typename T>
  void AddRequiredOption(const std::string& name,
                         T* option,
                         const std::string& help_text = "");
  template <typename T>
  void AddDefaultOption(const std::string& name,
                        T* option,
                        const std::string& help_text = "");

  void Reset();
  void ResetOptions(bool reset_paths);

  bool Check();

  void Parse(int argc, char** argv);
  bool Read(const std::string& path);
  bool ReRead(const std::string& path);
  void Write(const std::string& path) const;

  std::shared_ptr<std::string> project_path;
  std::shared_ptr<std::string> database_path;
  std::shared_ptr<std::string> image_path;

  std::shared_ptr<ImageReaderOptions> image_reader;
  std::shared_ptr<FeatureExtractionOptions> feature_extraction;
  std::shared_ptr<FeatureMatchingOptions> feature_matching;
  std::shared_ptr<TwoViewGeometryOptions> two_view_geometry;
  std::shared_ptr<ExhaustivePairingOptions> exhaustive_pairing;
  std::shared_ptr<SequentialPairingOptions> sequential_pairing;
  std::shared_ptr<VocabTreePairingOptions> vocab_tree_pairing;
  std::shared_ptr<SpatialPairingOptions> spatial_pairing;
  std::shared_ptr<TransitivePairingOptions> transitive_pairing;
  std::shared_ptr<ImportedPairingOptions> imported_pairing;

  std::shared_ptr<BundleAdjustmentOptions> bundle_adjustment;
  std::shared_ptr<IncrementalPipelineOptions> mapper;

  std::shared_ptr<mvs::PatchMatchOptions> patch_match_stereo;
  std::shared_ptr<mvs::StereoFusionOptions> stereo_fusion;
  std::shared_ptr<mvs::PoissonMeshingOptions> poisson_meshing;
  std::shared_ptr<mvs::DelaunayMeshingOptions> delaunay_meshing;

  std::shared_ptr<RenderOptions> render;

 private:
  template <typename T>
  void AddAndRegisterRequiredOption(const std::string& name,
                                    T* option,
                                    const std::string& help_text = "");
  template <typename T>
  void AddAndRegisterDefaultOption(const std::string& name,
                                   T* option,
                                   const std::string& help_text = "");

  template <typename T>
  void AddAndRegisterDefaultEnumOption(const std::string& name,
                                       T* option,
                                       const std::string& help_text = "");

  template <typename T>
  void RegisterOption(const std::string& name, const T* option);

  std::shared_ptr<boost::program_options::options_description> desc_;

  std::vector<std::pair<std::string, const bool*>> options_bool_;
  std::vector<std::pair<std::string, const int*>> options_int_;
  std::vector<std::pair<std::string, const double*>> options_double_;
  std::vector<std::pair<std::string, const std::string*>> options_string_;

  std::string feature_extraction_type_;
  std::string feature_matching_type_;

  bool added_log_options_;
  bool added_random_options_;
  bool added_database_options_;
  bool added_image_options_;
  bool added_extraction_options_;
  bool added_match_options_;
  bool added_exhaustive_match_options_;
  bool added_sequential_match_options_;
  bool added_vocab_tree_match_options_;
  bool added_spatial_match_options_;
  bool added_transitive_match_options_;
  bool added_image_pairs_match_options_;
  bool added_ba_options_;
  bool added_mapper_options_;
  bool added_patch_match_stereo_options_;
  bool added_stereo_fusion_options_;
  bool added_poisson_meshing_options_;
  bool added_delaunay_meshing_options_;
  bool added_render_options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void OptionManager::AddRequiredOption(const std::string& name,
                                      T* option,
                                      const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
}

template <typename T>
void OptionManager::AddDefaultOption(const std::string& name,
                                     T* option,
                                     const std::string& help_text) {
  desc_->add_options()(
      name.c_str(),
      boost::program_options::value<T>(option)->default_value(*option),
      help_text.c_str());
}

template <typename T>
void OptionManager::AddAndRegisterRequiredOption(const std::string& name,
                                                 T* option,
                                                 const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
  RegisterOption(name, option);
}

template <typename T>
void OptionManager::AddAndRegisterDefaultOption(const std::string& name,
                                                T* option,
                                                const std::string& help_text) {
  desc_->add_options()(
      name.c_str(),
      boost::program_options::value<T>(option)->default_value(*option),
      help_text.c_str());
  RegisterOption(name, option);
}

template <typename T>
void OptionManager::RegisterOption(const std::string& name, const T* option) {
  if (std::is_same<T, bool>::value) {
    options_bool_.emplace_back(name, reinterpret_cast<const bool*>(option));
  } else if (std::is_same<T, int>::value) {
    options_int_.emplace_back(name, reinterpret_cast<const int*>(option));
  } else if (std::is_same<T, double>::value) {
    options_double_.emplace_back(name, reinterpret_cast<const double*>(option));
  } else if (std::is_same<T, std::string>::value) {
    options_string_.emplace_back(name,
                                 reinterpret_cast<const std::string*>(option));
  } else {
    LOG(FATAL_THROW) << "Unsupported option type";
  }
}

}  // namespace colmap
