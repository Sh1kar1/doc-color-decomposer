#include "doc_color_decomposer/doc_color_decomposer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <filesystem>

#include <ranges>

#include <string>
#include <vector>

#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);

  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  if (args.size() == 3 || args.size() == 4 && (args[3] == "--visualize" || args[3] == "-v")) {
    std::filesystem::path src_path = args[1];
    std::filesystem::path dst_path = args[2];
    bool visualize = args.size() == 4;

    DocColorDecomposer dcd;
    try {
      cv::Mat src = cv::imread(src_path.string(), cv::IMREAD_COLOR);
      dcd = DocColorDecomposer(src);
    } catch (...) {
      std::cerr << "Error: invalid image";
      return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(dst_path) || !std::filesystem::is_directory(dst_path)) {
      std::filesystem::create_directory(dst_path);
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

  } else if (args.size() == 1 || args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) {
    std::cout << "DESCRIPTION\n";
    std::cout << "  App of the `Doc Color Decomposer` library for documents decomposition by color clustering\n";
    std::cout << "  More info: https://github.com/Sh1kar1/doc-color-decomposer\n";

    std::cout << "SYNOPSIS\n";
    std::cout << "  ./doc_color_decomposer_app <path-to-image> <path-to-output-directory> [options]\n";

    std::cout << "OPTIONS\n";
    std::cout << "  -v, --visualize  Save visualizations\n";
    std::cout << "  -h, --help       Print help message";

  } else {
    std::cerr << "Error: invalid arguments";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
