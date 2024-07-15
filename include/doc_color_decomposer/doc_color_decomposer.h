#ifndef DOC_COLOR_DECOMPOSER_H
#define DOC_COLOR_DECOMPOSER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <random>
#include <ranges>

#include <map>
#include <string>
#include <vector>

#include <iomanip>
#include <sstream>

class DocColorDecomposer {
public:
  explicit DocColorDecomposer(const cv::Mat& src);

  std::vector<cv::Mat> GetLayers();

  std::string Plot3dRgb(int yaw = 115, int pitch = 15);
  cv::Mat Plot2dLab();
  std::string Plot1dPhi();
  std::string Plot1dClusters();

private:
  static cv::Mat ThreshSaturationAndVibrance(cv::Mat src, double s_thresh = 25.0, double v_thresh = 64.0);
  static cv::Mat SmoothHue(cv::Mat src, int ker_size = 15);
  static cv::Mat ConvertSRgbToLinRgb(cv::Mat src);
  static cv::Mat ConvertLinRgbToSRgb(cv::Mat src);
  static std::map<std::tuple<float, float, float>, long long> LutColorToN(const cv::Mat& src);
  static cv::Mat ProjOnLab(const cv::Mat& rgb_point);
  static std::vector<int> FindHistPeaks(const cv::Mat& hist, int min_h = 0);

  void ComputePhiHist(int ker_size = 35);
  void ComputePhiClusters();
  void ComputeLayers();

  cv::Mat src_;
  cv::Mat phi_hist_;
  cv::Mat smooth_phi_hist_;
  std::vector<int> phi_clusters_;
  std::map<std::tuple<float, float, float>, long long> color_to_n_;
  std::map<std::tuple<float, float, float>, int> color_to_phi_;
  std::vector<int> phi_to_cluster_;
  std::vector<cv::Mat> layers_;
};

#endif // DOC_COLOR_DECOMPOSER_H
