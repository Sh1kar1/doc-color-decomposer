#ifndef DOC_COLOR_DECOMPOSER_H_
#define DOC_COLOR_DECOMPOSER_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

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
   * @return list of the decomposed document layers with a white background in the sRGB format
   */
  [[nodiscard]] std::vector<cv::Mat> GetLayers() const & noexcept;

  /**
   * @brief Retrieves the precomputed masks of the layers
   *
   * @return list of the binary masks of the layers in the grayscale format
   */
  [[nodiscard]] std::vector<cv::Mat> GetMasks() const & noexcept;

  /**
   * @brief Computes a Panoptic Quality (PQ) of the document decomposition (segmentation)
   *
   * @param[in] truth_masks list of the ground-truth binary masks in the grayscale format
   *
   * @return value between 0 and 1 that represents a quality
   */
  [[nodiscard]] double ComputeQuality(const std::vector<cv::Mat>& truth_masks) const &;

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
  void ComputeClusters();
  void ComputeLayers();

  [[nodiscard]] std::vector<std::array<int, 3>> ComputePhiToMeanRgb();
  [[nodiscard]] std::vector<std::array<int, 3>> ComputeClusterToMeanRgb();

  [[nodiscard]] static cv::Mat ThreshS(cv::Mat src, double thresh = 10.0);
  [[nodiscard]] static cv::Mat ThreshL(cv::Mat src, double thresh = 50.0);
  [[nodiscard]] static std::map<std::array<int, 3>, int> ComputeColorToN(const cv::Mat& src);
  [[nodiscard]] static cv::Mat ProjOnPlane(const cv::Mat& point, const cv::Mat& center, const cv::Mat& norm, const cv::Mat& transformation);
  [[nodiscard]] static cv::Mat ProjOnLab(cv::Mat rgb);
  [[nodiscard]] static int RadToDeg(double rad);
  [[nodiscard]] static std::vector<int> FindExtremes(const cv::Mat& hist);
  [[nodiscard]] static std::vector<int> FindPeaks(const cv::Mat& hist, int min_h = 0);
  [[nodiscard]] static double ComputeIou(const cv::Mat& predicted_mask, const cv::Mat& truth_mask);
  [[nodiscard]] static double ComputePq(const std::vector<cv::Mat>& predicted_masks, const std::vector<cv::Mat>& truth_masks);

  cv::Mat src_;
  cv::Mat processed_src_;
  int tolerance_;
  cv::Mat phi_hist_;
  cv::Mat smoothed_phi_hist_;
  std::vector<int> clusters_;
  std::map<std::array<int, 3>, int> rgb_to_n_;
  std::map<std::array<int, 3>, std::array<int, 3>> rgb_to_lab_;
  std::map<std::array<int, 3>, int> lab_to_phi_;
  std::vector<int> phi_to_cluster_;
  std::vector<cv::Mat> masks_;
  std::vector<cv::Mat> layers_;
};

#endif  // DOC_COLOR_DECOMPOSER_H_
