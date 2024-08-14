#include "doc_color_decomposer/doc_color_decomposer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <regex>

#include <filesystem>

#include <ranges>

#include <string>
#include <vector>

#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);

  if (args.size() >= 2) {
    std::filesystem::path src_path = args[0];
    std::filesystem::path dst_path = args[1];

    int tolerance = 35;
    bool nopreprocess = false;
    bool visualize = false;

    for (const auto& arg : args | std::views::drop(2)) {
      if (std::regex_match(arg, std::regex("^--tolerance=[0-9]*[13579]$"))) {
        tolerance = std::stoi(arg.substr(std::string("--tolerance=").size()));

      } else if (arg == "--nopreprocess") {
        nopreprocess = true;
      } else if (arg == "--visualize") {
        visualize = true;

      } else {
        std::cerr << "Error: invalid arguments\n";
        std::cerr << "Checkout `./doc-color-decomposer-app --help`";
        return EXIT_FAILURE;
      }
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {
      if (!std::filesystem::exists(dst_path) || !std::filesystem::is_directory(dst_path)) {
        std::filesystem::create_directory(dst_path);
      }

    } catch (...) {
      std::cerr << "Error: invalid arguments\n";
      std::cerr << "Checkout `./doc-color-decomposer-app --help`";
      return EXIT_FAILURE;
    }

    DocColorDecomposer dcd;
    try {
      cv::Mat src = cv::imread(src_path.string(), cv::IMREAD_COLOR);
      dcd = DocColorDecomposer(src, tolerance, !nopreprocess);

    } catch (...) {
      std::cerr << "Error: invalid image";
      return EXIT_FAILURE;
    }

    for (const auto& [i, layer] : dcd.GetLayers() | std::views::enumerate) {
      cv::imwrite((dst_path / (src_path.stem().string() + "-layer-")).string() + std::to_string(i + 1) + ".png", layer);
    }

    if (visualize) {
      cv::imwrite((dst_path / (src_path.stem().string() + "-merged-layers.png")).string(), dcd.MergeLayers());
      cv::imwrite((dst_path / (src_path.stem().string() + "-plot-2d-lab.png")).string(), dcd.Plot2dLab());

      std::ofstream(dst_path / (src_path.stem().string() + "-plot-3d-rgb.tex")) << dcd.Plot3dRgb();
      std::ofstream(dst_path / (src_path.stem().string() + "-plot-1d-phi.tex")) << dcd.Plot1dPhi();
      std::ofstream(dst_path / (src_path.stem().string() + "-plot-1d-clusters.tex")) << dcd.Plot1dClusters();
    }

    std::cout << "Success: files saved";

  } else if (args.empty() || args.size() == 1 && (args[0] == "--help" || args[0] == "-h")) {
    std::cout << "DESCRIPTION\n";
    std::cout << "  App of the `Doc Color Decomposer` library for documents decomposition by color clustering\n";
    std::cout << "  More info: https://github.com/Sh1kar1/doc-color-decomposer\n\n";

    std::cout << "SYNOPSIS\n";
    std::cout << "  ./doc-color-decomposer-app <path-to-image> <path-to-output-directory> [options]\n\n";

    std::cout << "OPTIONS\n";
    std::cout << "  --tolerance=<odd-positive-value>  Set tolerance of decomposition (default: 35)\n";
    std::cout << "  --nopreprocess                    Disable image preprocessing\n";
    std::cout << "  --visualize                       Save visualizations";

  } else {
    std::cerr << "Error: invalid arguments\n";
    std::cerr << "Checkout `./doc-color-decomposer-app --help`";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
