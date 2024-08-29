#ifndef UTILS_H_
#define UTILS_H_

#include <array>
#include <map>
#include <vector>

#include <opencv2/core/core.hpp>

namespace doc_color_decomposer {

[[nodiscard]] cv::Mat SmoothHue(cv::Mat src, int ker_size = 5);
[[nodiscard]] cv::Mat ThreshSaturation(cv::Mat src, double thresh = 10.0);
[[nodiscard]] cv::Mat ThreshLightness(cv::Mat src, double thresh = 50.0);
[[nodiscard]] std::map<std::array<int, 3>, int> ColorToN(const cv::Mat& src);
[[nodiscard]] cv::Mat ProjOnPlane(const cv::Mat& point, const cv::Mat& center, const cv::Mat& norm, const cv::Mat& transform);
[[nodiscard]] cv::Mat ProjOnLab(cv::Mat rgb);
[[nodiscard]] int RadToDeg(double rad);
[[nodiscard]] std::vector<int> FindExtremes(const cv::Mat& hist);
[[nodiscard]] std::vector<int> FindPeaks(const cv::Mat& hist, int min_h = 0);
[[nodiscard]] double ComputeIou(const cv::Mat& predicted_mask, const cv::Mat& truth_mask);
[[nodiscard]] double ComputePq(const std::vector<cv::Mat>& predicted_masks, const std::vector<cv::Mat>& truth_masks);

}  // namespace doc_color_decomposer

#endif  // UTILS_H_
