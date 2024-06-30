#include <doc_color_clustering/doc_color_clustering.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {
  cv::Mat img = cv::imread("..\\example\\data\\doc-1.png", cv::IMREAD_COLOR);
  DocColorClustering dcc = DocColorClustering(img);
  dcc.Plot3dRgb();
  return EXIT_SUCCESS;
}
