#ifndef DOC_COLOR_CLUSTERING_H
#define DOC_COLOR_CLUSTERING_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <fstream>
#include <string>
#include <map>


class DocColorClustering {
public:
  DocColorClustering(const cv::Mat& rhs);

  void Plot3dRgb(const std::string& output_path = ".\\plot-3d-rgb.tex", int yaw = 115, int pitch = 15);
  void Plot2dLab(const std::string& output_path = ".\\plot-2d-lab.png");
  void Plot1dPhi(const std::string& output_path = ".\\plot-1d-phi.tex", bool smooth = false);


private:
  cv::Mat SRgbToLinRgb(cv::Mat src);
  cv::Mat LinRgbToSRgb(cv::Mat src);
  cv::Mat CentralProjOnLab(const cv::Mat& rgb_point);
  void CalcColorToN();
  void CalcPhiHist();

  cv::Mat src_;
  std::map<std::tuple<double, double, double>, int> color_to_n_;
  std::map<std::tuple<double, double, double>, int> color_to_phi_;
  cv::Mat phi_hist_;
};


#endif // DOC_COLOR_CLUSTERING_H
