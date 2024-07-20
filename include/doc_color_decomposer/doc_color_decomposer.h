#ifndef DOC_COLOR_DECOMPOSER_H
#define DOC_COLOR_DECOMPOSER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <random>
#include <ranges>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <iomanip>
#include <sstream>

class DocColorDecomposer {
public:
  explicit DocColorDecomposer(const cv::Mat& src);

  [[nodiscard]] std::vector<cv::Mat> GetLayers();

  [[nodiscard]] std::string Plot3dRgb(int yaw = 115, int pitch = 15);
  [[nodiscard]] cv::Mat Plot2dLab();
  [[nodiscard]] std::string Plot1dPhi();
  [[nodiscard]] std::string Plot1dClusters();

private:
  [[nodiscard]] static cv::Mat ThreshSaturation(cv::Mat src, double s_thresh = 25.0, double v_thresh = 64.0);
  [[nodiscard]] static cv::Mat SmoothHue(cv::Mat src, int ker_size = 15);
  [[nodiscard]] static cv::Mat CvtSRgbToLinRgb(cv::Mat src);
  [[nodiscard]] static cv::Mat CvtLinRgbToSRgb(cv::Mat src);
  [[nodiscard]] static std::map<std::tuple<float, float, float>, long long> LutColorToN(const cv::Mat& src);
  [[nodiscard]] static cv::Mat ProjOnLab(const cv::Mat& rgb_point);
  [[nodiscard]] static std::vector<int> FindHistPeaks(const cv::Mat& hist, int min_h = 0);

  void ComputePhiHist(int ker_size = 35);
  void ComputePhiClusters();

  cv::Mat src_;
  cv::Mat phi_hist_;
  cv::Mat smooth_phi_hist_;
  std::vector<int> phi_clusters_;
  std::map<std::tuple<float, float, float>, long long> color_to_n_;
  std::map<std::tuple<float, float, float>, int> color_to_phi_;
  std::vector<int> phi_to_cluster_;
};

#endif // DOC_COLOR_DECOMPOSER_H
