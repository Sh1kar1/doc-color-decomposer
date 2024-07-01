#ifndef DOC_COLOR_CLUSTERING_H
#define DOC_COLOR_CLUSTERING_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>

class DocColorClustering {
public:
  DocColorClustering(const cv::Mat rhs);
  void Plot3dRgb(const std::string&lhs = ".\\plot-3d-rgb.tex", int rhs = 115, int pitch = 15);
  void Plot2dLab(const std::string& output_path = ".\\plot-2d-lab.png");

private:
  void CalcUniqueColorsAndColorToN();
  void CalcColorToPhi();
  cv::Mat CentralProjOnLab(const cv::Mat& rgb_point);

  cv::Mat src_;
  std::set<std::tuple<int, int, int>> unique_colors;
  std::map<std::tuple<int, int, int>, int> color_to_n;
  std::map<std::tuple<int, int, int>, int> color_to_phi;
};

#endif // DOC_COLOR_CLUSTERING_H
