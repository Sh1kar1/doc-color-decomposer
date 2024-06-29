#include <doc_color_clustering/doc_color_clustering.h>

void dcc::ShowImage(const std::string &path) {
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  cv::imshow(path, img);
  cv::waitKey(0);
}
