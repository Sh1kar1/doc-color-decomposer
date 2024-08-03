#include "doc_color_decomposer/doc_color_decomposer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <filesystem>

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

    std::vector<cv::Mat> layers = dcd.GetLayers();
    for (int i = 0; i < layers.size(); ++i) {
      cv::imwrite((dst_path / (src_path.stem().string() + "-layer-")).string() + std::to_string(i + 1) + ".png", layers[i]);
    }

    if (visualize) {
      cv::Mat merged_layers = dcd.MergeLayers();
      cv::imwrite((dst_path / (src_path.stem().string() + "-merged-layers.png")).string(), merged_layers);

      cv::Mat plot_2d_lab = dcd.Plot2dLab();
      cv::imwrite((dst_path / (src_path.stem().string() + "-plot-2d-lab.png")).string(), plot_2d_lab);

      std::ofstream plot_3d_rgb(dst_path / (src_path.stem().string() + "-plot-3d-rgb.tex"));
      std::ofstream plot_1d_phi(dst_path / (src_path.stem().string() + "-plot-1d-phi.tex"));
      std::ofstream plot_1d_clusters(dst_path / (src_path.stem().string() + "-plot-1d-clusters.tex"));

      plot_3d_rgb << dcd.Plot3dRgb();
      plot_1d_phi << dcd.Plot1dPhi();
      plot_1d_clusters << dcd.Plot1dClusters();
    }

    std::cout << "Success: files saved";

  } else if (args.size() == 1 || args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) {
    std::cout << "USAGE\n";
    std::cout << "    ./doc_color_decomposer_app <path-to-image> <path-to-output-directory> [options]\n";

    std::cout << "OPTIONS\n";
    std::cout << "    -v, --visualize    Save visualizations";

  } else {
    std::cerr << "Error: invalid arguments";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
