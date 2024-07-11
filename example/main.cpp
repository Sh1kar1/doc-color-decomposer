#include <doc_color_clustering/doc_color_clustering.h>


int main() {
  cv::Mat img = cv::imread("..\\example\\data\\1.png", cv::IMREAD_COLOR);
  DocColorClustering dcc = DocColorClustering(img);
  dcc.Plot3dRgb();
  dcc.Plot2dLab();
  dcc.Plot1dPhi();
  dcc.Plot1dClusters();
  std::vector<cv::Mat> layers = dcc.GetLayers();
  for (int i = 0; i < layers.size(); ++i) {
    cv::imwrite("1-" + std::to_string(i + 1) + ".png", layers[i]);
  }
  return EXIT_SUCCESS;
}
