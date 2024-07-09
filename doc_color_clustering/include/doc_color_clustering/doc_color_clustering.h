#ifndef DOC_COLOR_CLUSTERING_H
#define DOC_COLOR_CLUSTERING_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <random>

#include <string>
#include <vector>
#include <map>

#include <iomanip>
#include <fstream>


class DocColorClustering {
public:
  DocColorClustering(const cv::Mat& src);

  void Plot3dRgb(const std::string& output_path = ".\\plot-3d-rgb.tex", int yaw = 115, int pitch = 15);
  void Plot2dLab(const std::string& output_path = ".\\plot-2d-lab.png");
  void Plot1dPhi(const std::string& output_path = ".\\plot-1d-phi.tex");
  void Plot1dClusters(const std::string& output_path = ".\\plot-1d-clusters.tex");


private:
  cv::Mat SRgbToLinRgb(cv::Mat src);
  cv::Mat LinRgbToSRgb(cv::Mat src);
  std::map<std::tuple<float, float, float>, long long> LutColorToN(const cv::Mat& src);
  cv::Mat CentralProjOnLab(const cv::Mat& rgb_point);
  std::vector<int> HistPeaks(const cv::Mat& hist);
  std::vector<int> FilterMaximums(const std::vector<int>& peaks, const cv::Mat& hist, int min_h, int min_w);

  void CalcPhiHist();
  void CalcPhiClusters();

  cv::Mat src_;
  cv::Mat phi_hist_;
  cv::Mat smooth_phi_hist_;
  std::vector<int> phi_clusters_;
  std::map<std::tuple<float, float, float>, long long> color_to_n_;
  std::map<std::tuple<float, float, float>, int> color_to_phi_;
  std::vector<int> phi_to_cluster_;
};


#endif // DOC_COLOR_CLUSTERING_H
