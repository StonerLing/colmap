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

#include "colmap/controllers/base_option_manager.h"

#include "colmap/math/random.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/version.h"

#include <fstream>
#include <map>
#include <sstream>

namespace colmap {
namespace {

class CustomFormatter : public CLI::Formatter {
 public:
  CustomFormatter() { label("REQUIRED", "[REQUIRED]"); }

  std::string make_usage(const CLI::App* app, std::string name) const override {
    std::string result = "Usage: " + name + " [OPTIONS]";
    if (app->get_subcommands().size() > 0) {
      result += " [SUBCOMMAND]";
    }
    return result + "\n";
  }

  std::string make_option_opts(const CLI::Option* opt) const override {
    std::stringstream out;
    if (opt->get_type_size() != 0) {
      if (!opt->get_default_str().empty()) {
        out << " (=" << opt->get_default_str() << ")";
      }
      if (opt->get_required()) {
        out << " " << get_label("REQUIRED");
      }
    }
    return out.str();
  }
};

class CustomConfig : public CLI::ConfigINI {
  static std::string GetValue(const CLI::Option* opt, bool default_also) {
    // Use raw results (no ini-style quoting/escaping) for backward compat
    const auto& results = opt->results();
    if (!results.empty()) {
      // Multi-option values are joined with space (matching boost behavior)
      std::string out;
      for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) out += ' ';
        out += results[i];
      }
      return out;
    }
    if (default_also) {
      if (!opt->get_default_str().empty()) {
        return opt->get_default_str();
      }
      if (opt->get_expected_min() == 0) {
        return "false";
      }
      if (!opt->get_required()) {
        return {};
      }
      return "<REQUIRED>";
    }
    return {};
  }

  std::string to_config(const CLI::App* app,
                        bool default_also,
                        bool write_description,
                        std::string prefix) const override {
    std::stringstream out;

    // Collect options grouped by section (prefix before first dot).
    // Top-level options (no dot) go into the default section.
    std::map<std::string, std::vector<const CLI::Option*>> section_opts;
    for (const auto* opt : app->get_options({})) {
      if (!opt->get_configurable()) continue;
      std::string name = opt->get_single_name();
      if (name.empty() || name == "--help") {
        continue;
      }
      if (name.size() > 2 && name[0] == '-' && name[1] == '-') {
        name = name.substr(2);
      }
      auto dot = name.find(parentSeparatorChar);
      if (dot != std::string::npos) {
        section_opts[name.substr(0, dot)].push_back(opt);
      } else {
        section_opts[""].push_back(opt);
      }
    }

    for (auto& [section, opts] : section_opts) {
      if (!section.empty()) {
        out << '[' << section << "]\n";
      }
      for (const auto* opt : opts) {
        std::string single_name = opt->get_single_name();
        if (single_name.size() > 2 && single_name[0] == '-' &&
            single_name[1] == '-') {
          single_name = single_name.substr(2);
        }

        std::string value = GetValue(opt, default_also);
        if (value.empty()) {
          continue;
        }

        // For sectioned options, strip the section prefix from the key name
        std::string key = single_name;
        auto dot = key.find(parentSeparatorChar);
        if (dot != std::string::npos) {
          key = key.substr(dot + 1);
        }

        out << key << valueDelimiter << value << '\n';
      }
    }

    return out.str();
  }
};

}  // namespace

BaseOptionManager::BaseOptionManager(bool add_project_options) {
  project_path = std::make_shared<std::filesystem::path>();
  database_path = std::make_shared<std::filesystem::path>();
  image_path = std::make_shared<std::filesystem::path>();

  app_ = std::make_unique<CLI::App>("COLMAP");
  app_->set_version_flag("-v,--version", GetVersionInfo());
  app_->formatter(std::make_shared<CustomFormatter>());
  app_->config_formatter(std::make_shared<CustomConfig>());

  ResetImpl(/*reset_logging=*/true);

  if (add_project_options) {
    app_->set_config("--project_path");
  }

  AddRandomOptions();
  AddLogOptions();
}

void BaseOptionManager::AddRandomOptions() {
  if (added_random_options_) {
    return;
  }
  added_random_options_ = true;

  AddDefaultOption("default_random_seed", &kDefaultPRNGSeed);
}

void BaseOptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  AddDefaultOption(
      "log_target", &log_target_, "{stderr, stdout, file, stderr_and_file}");
  // Directory for log files. If empty, glog uses $GOOGLE_LOG_DIR, /tmp, or
  // %TEMP%.
  AddDefaultOption("log_path", &FLAGS_log_dir);
  AddDefaultOption("log_level", &FLAGS_v);
  AddDefaultOption("log_severity",
                   &FLAGS_minloglevel,
                   "0:INFO, 1:WARNING, 2:ERROR, 3:FATAL");
#if COLMAP_GLOG_HAS_COLOR_SUPPORT
  AddDefaultOption("log_color", &FLAGS_colorlogtostderr);
#endif
}

void BaseOptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddRequiredOption("database_path", database_path.get());
}

void BaseOptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddRequiredOption("image_path", image_path.get());
}

void BaseOptionManager::Reset(bool reset_logging) { ResetImpl(reset_logging); }

void BaseOptionManager::ResetOptions(const bool reset_paths) {
  auto saved_project_path = std::move(*project_path);
  auto saved_database_path = std::move(*database_path);
  auto saved_image_path = std::move(*image_path);

  // Re-register all options to update raw pointers, since subclass
  // ResetOptions() may reallocate internal sub-objects.
  Reset(/*reset_logging=*/false);
  AddAllOptions();

  if (!reset_paths) {
    *project_path = std::move(saved_project_path);
    *database_path = std::move(saved_database_path);
    *image_path = std::move(saved_image_path);
  }
}

void BaseOptionManager::ResetImpl(bool reset_logging) {
  if (reset_logging) {
    log_target_ = "stderr_and_file";
    FLAGS_log_dir = "";
    FLAGS_v = 0;
    FLAGS_minloglevel = 0;
#if COLMAP_GLOG_HAS_COLOR_SUPPORT
    FLAGS_colorlogtostderr = true;
#endif
    ApplyLogFlags();
  }

  const bool kResetPaths = true;
  ResetOptionsImpl(kResetPaths);

  app_->clear();

  added_random_options_ = false;
  added_log_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
}

void BaseOptionManager::ResetOptionsImpl(const bool reset_paths) {
  if (reset_paths) {
    *project_path = "";
    *database_path = "";
    *image_path = "";
  }
}

bool BaseOptionManager::Check() {
  bool success = true;

  if (added_database_options_) {
    const auto database_parent_path = GetParentDir(*database_path);
    success = success && CHECK_OPTION_IMPL(!ExistsDir(*database_path)) &&
              CHECK_OPTION_IMPL(database_parent_path.empty() ||
                                ExistsDir(database_parent_path));
  }

  if (added_image_options_) {
    success = success && CHECK_OPTION_IMPL(ExistsDir(*image_path));
  }

  return success;
}

void BaseOptionManager::PostParse() {
  // Default implementation does nothing. Subclasses can override.
}

void BaseOptionManager::ApplyEnumConversions() {
  for (const auto& info : enum_options_) {
    info->apply();
  }
}

void BaseOptionManager::ApplyLogFlags() {
  FLAGS_logtostderr = false;
#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
  FLAGS_logtostdout = false;
#endif
  FLAGS_alsologtostderr = false;

  if (log_target_ == "stderr") {
    FLAGS_logtostderr = true;
  } else if (log_target_ == "stdout") {
#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
    FLAGS_logtostdout = true;
#else
    LOG(WARNING) << "log_target=stdout requires glog >= 0.6. "
                    "Falling back to stderr.";
    FLAGS_logtostderr = true;
#endif
  } else if (log_target_ == "file") {
  } else if (log_target_ == "stderr_and_file") {
    FLAGS_alsologtostderr = true;
  } else {
    LOG(ERROR) << "Invalid log_target: " << log_target_
               << ". Falling back to stderr_and_file.";
    FLAGS_alsologtostderr = true;
  }

#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
  FLAGS_colorlogtostdout = FLAGS_colorlogtostderr;
#endif

  if (!FLAGS_log_dir.empty() &&
      (log_target_ == "file" || log_target_ == "stderr_and_file")) {
    CreateDirIfNotExists(FLAGS_log_dir);
  }
}

void BaseOptionManager::PrintHelp() const {
  LOG(INFO) << "Options can either be specified via command-line or by "
               "defining them in a .ini project file.\n"
            << app_->help();
}

void BaseOptionManager::AddAllOptions() {
  AddRandomOptions();
  AddLogOptions();
  AddDatabaseOptions();
  AddImageOptions();
}

bool BaseOptionManager::Parse(const int argc, char** argv) {
  try {
    app_->parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    app_->exit(e);
    return false;
  }

  ApplyEnumConversions();
  ApplyLogFlags();
  PostParse();

  if (!Check()) {
    LOG(ERROR) << "Invalid options provided.";
    return false;
  }

  return true;
}

bool BaseOptionManager::Read(const std::filesystem::path& path,
                             bool allow_unregistered) {
  if (!ExistsFile(path)) {
    LOG(ERROR) << "Configuration file does not exist.";
    return false;
  }

  app_->allow_config_extras(allow_unregistered);

  try {
    app_->parse("--project_path " + path.string());
  } catch (const CLI::ParseError& e) {
    app_->exit(e);
    return false;
  }

  return true;
}

bool BaseOptionManager::ReRead(const std::filesystem::path& path,
                               bool reset_logging,
                               bool allow_unregistered) {
  Reset(reset_logging);
  AddAllOptions();
  return Read(path, allow_unregistered);
}

void BaseOptionManager::Write(const std::filesystem::path& path) const {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  file << app_->config_to_str(/*default_values=*/true,
                              /*write_description=*/false);
  file.close();
}

}  // namespace colmap
