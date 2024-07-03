#include <doc_color_clustering/doc_color_clustering.h>


int main() {
  cv::Mat img = cv::imread("..\\example\\data\\doc-1.png", cv::IMREAD_COLOR);
  DocColorClustering dcc = DocColorClustering(img);
  dcc.Plot3dRgb();
  dcc.Plot2dLab();
  dcc.Plot1dPhi(".\\plot-1d-phi.tex", false);
  dcc.Plot1dPhi(".\\plot-1d-phi-smooth.tex", true);
  return EXIT_SUCCESS;
}
