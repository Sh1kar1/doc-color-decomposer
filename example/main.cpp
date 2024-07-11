#include <doc_color_clustering/doc_color_clustering.h>

#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>

#include <fstream>

int main() {
  cv::Mat img = cv::imread(R"(..\example\data\1.png)", cv::IMREAD_COLOR);
  DocColorClustering dcc = DocColorClustering(img);

  std::vector<cv::Mat> layers = dcc.GetLayers();
  for (int i = 0; i < layers.size(); ++i) {
    cv::imwrite("layer-" + std::to_string(i + 1) + ".png", layers[i]);
  }

  cv::Mat plot_2d_lab = dcc.Plot2dLab();
  cv::imwrite(".\\plot-2d-lab.png", plot_2d_lab);

  std::ofstream plot_3d_rgb(".\\plot-3d-rgb.tex"), plot_1d_phi(".\\plot-1d-phi.tex"), plot_1d_clusters(".\\plot-1d-clusters.tex");
  plot_3d_rgb << dcc.Plot3dRgb();
  plot_1d_phi << dcc.Plot1dPhi();
  plot_1d_clusters << dcc.Plot1dClusters();
  plot_3d_rgb.close();
  plot_1d_phi.close();
  plot_1d_clusters.close();

  return EXIT_SUCCESS;
}
