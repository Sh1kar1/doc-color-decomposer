#ifndef DOC_COLOR_DECOMPOSER_H
#define DOC_COLOR_DECOMPOSER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <random>

#include <ranges>

#include <array>
#include <map>
#include <string>
#include <vector>

#include <iomanip>
#include <iostream>
#include <sstream>

/**
 * @brief Interface of the [Doc Color Decomposer](https://github.com/Sh1kar1/doc-color-decomposer) library for documents decomposition by color clustering
 */
class [[nodiscard]] DocColorDecomposer final {
public:
  /**
   * @brief Constructs an empty instance
   */
  explicit DocColorDecomposer() = default;

  /**
   * @brief Constructs an instance from the given document and precomputes its layers
   *
   * @param[in] src source image of the document in the sRGB format
   * @param[in] tolerance odd positive value with an increase of which the number of layers decreases
   * @param[in] preprocessing true if the source image needs to be processed by aberration reduction
   */
  explicit DocColorDecomposer(const cv::Mat& src, int tolerance = 35, bool preprocessing = true);

  /**
   * @brief Retrieves the precomputed layers
   *
   * @return list of the decomposed document layers in the sRGB format with a white background
   */
  [[nodiscard]] std::vector<cv::Mat> GetLayers() const & noexcept;

  /**
   * @brief Retrieves the precomputed masks of the layers
   *
   * @return list of the binary masks of the layers in the sRGB format with a black background
   */
  [[nodiscard]] std::vector<cv::Mat> GetMasks() const & noexcept;

  /**
   * @brief Generates a 3D scatter plot of the document colors in the linRGB space
   *
   * @param[in] yaw yaw-rotation angle of the view in degrees
   * @param[in] pitch pitch-rotation angle of the view in degrees
   *
   * @return LaTeX code of the plot that can be saved in the .tex format and compiled
   */
  [[nodiscard]] std::string Plot3dRgb(double yaw = 135.0, double pitch = 35.25) &;

  /**
   * @brief Generates a 2D scatter plot of the document colors projections on the \f$\alpha\beta\f$ plane
   *
   * @return image of the plot in the sRGB format
   */
  [[nodiscard]] cv::Mat Plot2dLab() &;

  /**
   * @brief Generates a 1D histogram plot with respect to the angle \f$\phi\f$ in polar coordinates
   *
   * @return LaTeX code of the plot that can be saved in the .tex format and compiled
   */
  [[nodiscard]] std::string Plot1dPhi() &;

  /**
   * @brief Generates a smoothed and separated by clusters 1D histogram plot
   *
   * @return LaTeX code of the plot that can be saved in the .tex format and compiled
   */
  [[nodiscard]] std::string Plot1dClusters() &;

private:
  void ComputePhiHist();
  void ComputePhiClusters();
  void ComputeLayers();

  [[nodiscard]] static cv::Mat ThreshSaturation(cv::Mat src, double thresh = 10.0);
  [[nodiscard]] static cv::Mat ThreshLightness(cv::Mat src, double thresh = 50.0);
  [[nodiscard]] static cv::Mat CvtSRgbToLinRgb(cv::Mat src);
  [[nodiscard]] static cv::Mat CvtLinRgbToSRgb(cv::Mat src);
  [[nodiscard]] static std::map<std::array<float, 3>, long long> LutColorToN(const cv::Mat& src);
  [[nodiscard]] static cv::Mat ProjOnLab(const cv::Mat& rgb);
  [[nodiscard]] static std::vector<int> FindHistPeaks(const cv::Mat& hist, int min_h = 0);

  cv::Mat src_;
  cv::Mat processed_src_;
  int tolerance_;
  cv::Mat phi_hist_;
  cv::Mat smoothed_phi_hist_;
  std::vector<int> phi_clusters_;
  std::map<std::array<float, 3>, long long> color_to_n_;
  std::map<std::array<float, 3>, int> color_to_phi_;
  std::vector<int> phi_to_cluster_;
  std::vector<cv::Mat> masks_;
  std::vector<cv::Mat> layers_;
};

#endif // DOC_COLOR_DECOMPOSER_H
