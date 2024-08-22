#include "doc_color_decomposer/doc_color_decomposer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <ranges>
#include <sstream>
#include <utility>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../src/plot_2d_lab_image.h"

DocColorDecomposer::DocColorDecomposer(const cv::Mat& src, int tolerance, bool preprocessing) {
  src_ = src;
  processed_src_ = preprocessing ? ThreshL(ThreshS(src)) : src_;
  tolerance_ = tolerance;

  color_to_n_ = ComputeColorToN(processed_src_);

  ComputePhiHist();
  ComputeClusters();
  ComputeLayers();
}

std::vector<cv::Mat> DocColorDecomposer::GetLayers() const & noexcept {
  return layers_;
}

std::vector<cv::Mat> DocColorDecomposer::GetMasks() const & noexcept {
  return masks_;
}

double DocColorDecomposer::ComputeQuality(const std::vector<cv::Mat>& truth_masks) const & {
  return ComputePq(masks_, truth_masks);
}

std::string DocColorDecomposer::Plot3dRgb(double yaw, double pitch) & {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

  std::vector<std::pair<std::array<int, 3>, int>> shuffled_color_to_n(color_to_n_.begin(), color_to_n_.end());
  std::ranges::shuffle(shuffled_color_to_n, std::mt19937(std::random_device()()));
  shuffled_color_to_n.resize(std::min(shuffled_color_to_n.size(), static_cast<std::size_t>(5000)));

  plot << "\\documentclass[tikz, border=1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n\n";

  plot << "\\pagecolor{black}\n";
  plot << "\\color{white}\n\n";

  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "  view={" << yaw << "}{" << pitch << "},\n";
  plot << "  height=10cm,\n";
  plot << "  width=10cm,\n";
  plot << "  scale only axis,\n";
  plot << "  xmin=0, xmax=1,\n";
  plot << "  ymin=0, ymax=1,\n";
  plot << "  zmin=0, zmax=1,\n";
  plot << "  tick style={white},\n";
  plot << "  xlabel={$R$},\n";
  plot << "  ylabel={$G$},\n";
  plot << "  zlabel={$B$}\n";
  plot << "]\n\n";

  plot << "\\addplot3[\n";
  plot << "  scatter,\n";
  plot << "  scatter/@pre marker code/.code={\n";
  plot << "    \\edef\\temp{\\noexpand\\definecolor{mycolor}{rgb}{\\pgfplotspointmeta}}\n";
  plot << "    \\temp\n";
  plot << "    \\scope[color=mycolor]\n";
  plot << "  },\n";
  plot << "  scatter/@post marker code/.code={\n";
  plot << "    \\endscope\n";
  plot << "  },\n";
  plot << "  only marks,\n";
  plot << "  mark size=0.01cm,\n";
  plot << "  point meta={TeX code symbolic={\\edef\\pgfplotspointmeta{\\thisrow{R}, \\thisrow{G}, \\thisrow{B}}}}\n";
  plot << "]\n";
  plot << "table[] {\n";
  plot << "R G B\n";

  for (const auto& color : shuffled_color_to_n | std::views::keys) {
    plot << color[0] / 255.0 << ' ' << color[1] / 255.0 << ' ' << color[2] / 255.0 << '\n';
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

cv::Mat DocColorDecomposer::Plot2dLab() & {
  cv::Mat plot = cv::imdecode(cv::Mat(1, doc_color_decomposer::plot_2d_lab_len, CV_8U, doc_color_decomposer::plot_2d_lab_data), cv::IMREAD_UNCHANGED);

  for (const auto& color : color_to_n_ | std::views::keys) {
    cv::Mat rgb = (cv::Mat_<int>(1, 3) << color[0], color[1], color[2]);
    cv::Mat lab = ProjOnLab(rgb);

    int y = std::lround(lab.at<int>(0, 1) + 752);
    int x = std::lround(lab.at<int>(0, 0) + 752);

    plot.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);
  }

  return plot;
}

std::string DocColorDecomposer::Plot1dPhi() & {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

  std::vector<std::array<int, 3>> phi_to_sum_color(360);
  std::vector<int> phi_to_n(360);

  for (const auto& [color, n] : color_to_n_) {
    int phi = color_to_phi_[color];

    if (phi != -1) {
      std::ranges::transform(phi_to_sum_color[phi], (color | std::views::transform([&n](int c) { return c * n; })), phi_to_sum_color[phi].begin(), std::plus{});
      phi_to_n[phi] += n;
    }
  }

  std::vector<std::array<int, 3>> phi_to_mean_color(360);

  for (const auto& [phi, mean_color] : phi_to_mean_color | std::views::enumerate) {
    std::array<int, 3> sum_color = phi_to_sum_color[phi];
    int n = phi_to_n[phi];

    if (n != 0) {
      std::ranges::transform(sum_color, mean_color.begin(), [&n](int c) { return c / n; });
    }
  }

  double max_n;
  cv::minMaxLoc(phi_hist_, nullptr, &max_n, nullptr, nullptr);

  plot << "\\documentclass[tikz, border=1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n\n";

  plot << "\\pagecolor{black}\n";
  plot << "\\color{white}\n\n";

  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "  height=10cm,\n";
  plot << "  width=30cm,\n";
  plot << "  xmin=0, xmax=360,\n";
  plot << "  ymin=0, ymax=" << std::lround(max_n) << ",\n";
  plot << "  tick style={white},\n";
  plot << "  xtick style={draw=none},\n";
  plot << "  xlabel={$\\phi$},\n";
  plot << "  ylabel={$n$}\n";
  plot << "]\n\n";

  for (int phi = 0; phi < 359; ++phi) {
    std::array<int, 3> mean_color = phi_to_mean_color[phi];

    plot << "\\addplot[\n";
    plot << "  ybar interval,\n";
    plot << "  color={rgb,1: red," << mean_color[0] / 255.0 << "; green," << mean_color[1] / 255.0 << "; blue," << mean_color[2] / 255.0 << "},\n";
    plot << "  fill={rgb,1: red," << mean_color[0] / 255.0 << "; green," << mean_color[1] / 255.0 << "; blue," << mean_color[2] / 255.0 << "}\n";
    plot << "]\n";
    plot << "table[] {\n";
    plot << "X Y\n";
    plot << phi << ' ' << std::lround(phi_hist_.at<double>(phi)) << '\n';
    plot << phi + 1 << ' ' << std::lround(phi_hist_.at<double>(phi + 1)) << '\n';
    plot << "};\n\n";
  }

  plot << "\\draw (axis cs:0,0) -- (axis cs:360,0);\n";
  plot << "\\draw (axis cs:0," << std::lround(max_n) << ") -- (axis cs:360," << std::lround(max_n) << ");\n\n";

  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

std::string DocColorDecomposer::Plot1dClusters() & {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

  std::vector<std::array<int, 3>> cluster_to_sum_color(clusters_.size() + 1);
  std::vector<int> cluster_to_n(clusters_.size() + 1);

  for (const auto& [color, n] : color_to_n_) {
    int phi = color_to_phi_[color];
    int cluster = (phi != -1) ? phi_to_cluster_[phi] : -1;

    if (cluster != -1) {
      std::ranges::transform(cluster_to_sum_color[cluster], (color | std::views::transform([&n](int c) { return c * n; })), cluster_to_sum_color[cluster].begin(), std::plus{});
      cluster_to_n[cluster] += n;
    }
  }

  std::vector<std::array<int, 3>> cluster_to_mean_color(clusters_.size() + 1);

  for (const auto& [cluster, mean_color] : cluster_to_mean_color | std::views::enumerate) {
    std::array<int, 3> sum_color = cluster_to_sum_color[cluster];
    int n = cluster_to_n[cluster];

    if (n != 0) {
      std::ranges::transform(sum_color, mean_color.begin(), [&n](int c) { return c / n; });
    }
  }

  double max_n;
  cv::minMaxLoc(smoothed_phi_hist_, nullptr, &max_n, nullptr, nullptr);

  plot << "\\documentclass[tikz, border=1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n\n";

  plot << "\\pagecolor{black}\n";
  plot << "\\color{white}\n\n";

  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "  height=10cm,\n";
  plot << "  width=30cm,\n";
  plot << "  xmin=0, xmax=360,\n";
  plot << "  ymin=0, ymax=" << std::lround(max_n) << ",\n";
  plot << "  tick style={white},\n";
  plot << "  xtick style={draw=none},\n";
  plot << "  xlabel={$\\phi$},\n";
  plot << "  ylabel={$n$}\n";
  plot << "]\n\n";

  for (int phi = 0; phi < 359; ++phi) {
    int cluster = phi_to_cluster_[phi];
    std::array<int, 3> mean_color = cluster_to_mean_color[cluster];

    plot << "\\addplot[\n";
    plot << "  ybar interval,\n";
    plot << "  color={rgb,1: red," << mean_color[0] / 255.0 << "; green," << mean_color[1] / 255.0 << "; blue," << mean_color[2] / 255.0 << "},\n";
    plot << "  fill={rgb,1: red," << mean_color[0] / 255.0 << "; green," << mean_color[1] / 255.0 << "; blue," << mean_color[2] / 255.0 << "}\n";
    plot << "]\n";
    plot << "table[] {\n";
    plot << "X Y\n";
    plot << phi << ' ' << smoothed_phi_hist_.at<int>(phi) << '\n';
    plot << phi + 1 << ' ' << smoothed_phi_hist_.at<int>(phi + 1) << '\n';
    plot << "};\n\n";
  }

  for (const auto& cluster : clusters_) {
    plot << "\\draw (axis cs:" << cluster << ",0) -- (axis cs:" << cluster << ',' << std::lround(max_n) << ");\n";
  }

  plot << "\n";
  plot << "\\draw (axis cs:0,0) -- (axis cs:360,0);\n";
  plot << "\\draw (axis cs:0," << std::lround(max_n) << ") -- (axis cs:360," << std::lround(max_n) << ");\n\n";

  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

void DocColorDecomposer::ComputePhiHist() {
  phi_hist_ = cv::Mat::zeros(1, 360, CV_64FC1);

  for (const auto& [color, n] : color_to_n_) {
    bool is_gray = (color[0] == color[1]) && (color[1] == color[2]) && (color[0] == color[2]);
    if (!is_gray) {
      cv::Mat rgb = (cv::Mat_<int>(1, 3) << color[0], color[1], color[2]);
      cv::Mat lab = ProjOnLab(rgb);

      double phi_rad = std::atan2(-lab.at<int>(0, 1), lab.at<int>(0, 0));
      int phi = std::lround((phi_rad * 180.0 / CV_PI) + 360.0) % 360;

      phi_hist_.at<double>(phi) += n;

      color_to_phi_[color] = phi;
    } else {
      color_to_phi_[color] = -1;
    }
  }

  cv::GaussianBlur(phi_hist_, smoothed_phi_hist_, cv::Size(tolerance_, tolerance_), 0.0);
  smoothed_phi_hist_.convertTo(smoothed_phi_hist_, CV_32SC1);
}

void DocColorDecomposer::ComputeClusters() {
  phi_to_cluster_ = std::vector<int>(360, 1);

  double max_h;
  cv::minMaxLoc(smoothed_phi_hist_, nullptr, &max_h, nullptr, nullptr);

  std::vector<int> peaks = FindHistPeaks(smoothed_phi_hist_, std::lround(0.01 * max_h));
  peaks.push_back(peaks[0] + 360);

  for (std::size_t i = 0; i < peaks.size() - 1; ++i) {
    int mid = ((peaks[i + 1] + peaks[i]) / 2) % 360;
    clusters_.push_back(mid);
  }
  std::ranges::sort(clusters_);

  for (int i = 0; i < clusters_.size() - 1; ++i) {
    for (std::size_t j = clusters_[i]; j < clusters_[i + 1]; ++j) {
      phi_to_cluster_[j] = i + 2;
    }
  }
}

void DocColorDecomposer::ComputeLayers() {
  layers_ = std::vector<cv::Mat>(clusters_.size() + 1);
  for (auto& layer : layers_) {
    layer = cv::Mat(processed_src_.rows, processed_src_.cols, CV_8UC3, cv::Vec3b(255, 255, 255));
  }

  masks_ = std::vector<cv::Mat>(clusters_.size() + 1);
  for (auto& mask : masks_) {
    mask = cv::Mat::zeros(processed_src_.rows, processed_src_.cols, CV_8UC1);
  }

  for (int y = 0; y < processed_src_.rows; ++y) {
    for (int x = 0; x < processed_src_.cols; ++x) {
      auto& pixel = processed_src_.at<cv::Vec3b>(y, x);
      std::array<int, 3> color = {pixel[2], pixel[1], pixel[0]};

      int phi = color_to_phi_[color];
      int cluster = (phi != -1) ? phi_to_cluster_[phi] : 0;

      layers_[cluster].at<cv::Vec3b>(y, x) = src_.at<cv::Vec3b>(y, x);
      masks_[cluster].at<uchar>(y, x) = 255;
    }
  }
}

cv::Mat DocColorDecomposer::ThreshS(cv::Mat src, double thresh) {
  cv::cvtColor(src, src, cv::COLOR_BGR2HSV_FULL);

  std::vector<cv::Mat> hsv_channels;
  cv::split(src, hsv_channels);

  cv::threshold(hsv_channels[1], hsv_channels[1], thresh, 0.0, cv::THRESH_TOZERO);

  cv::merge(hsv_channels, src);

  cv::cvtColor(src, src, cv::COLOR_HSV2BGR_FULL);

  return src;
}

cv::Mat DocColorDecomposer::ThreshL(cv::Mat src, double thresh) {
  cv::cvtColor(src, src, cv::COLOR_BGR2HLS_FULL);

  std::vector<cv::Mat> hls_channels;
  cv::split(src, hls_channels);

  cv::threshold(hls_channels[1], hls_channels[1], thresh, 0.0, cv::THRESH_TOZERO);

  cv::merge(hls_channels, src);

  cv::cvtColor(src, src, cv::COLOR_HLS2BGR_FULL);

  return src;
}

std::map<std::array<int, 3>, int> DocColorDecomposer::ComputeColorToN(const cv::Mat& src) {
  std::map<std::array<int, 3>, int> color_to_n;

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
      std::array<int, 3> color = {pixel[2], pixel[1], pixel[0]};

      ++color_to_n[color];
    }
  }

  return color_to_n;
}

cv::Mat DocColorDecomposer::ProjOnLab(const cv::Mat& rgb) {
  const cv::Mat kWhite = (cv::Mat_<double>(1, 3) << 1.0, 1.0, 1.0);
  const cv::Mat kNorm = (cv::Mat_<double>(1, 3) << 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0));
  const cv::Mat kRgbToLab = (cv::Mat_<double>(3, 3) <<
      -1.0 / std::sqrt(2.0), +1.0 / std::sqrt(2.0), -0.0,
      +1.0 / std::sqrt(6.0), +1.0 / std::sqrt(6.0), -2.0 / std::sqrt(6.0),
      +1.0 / std::sqrt(3.0), +1.0 / std::sqrt(3.0), +1.0 / std::sqrt(3.0)
  );

  cv::Mat norm_rgb;
  rgb.convertTo(norm_rgb, CV_64FC3, 1.0 / 255.0);

  bool is_white = (norm_rgb.at<double>(0) == 1.0) && (norm_rgb.at<double>(1) == 1.0) && (norm_rgb.at<double>(2) == 1.0);
  if (!is_white) {
    cv::Mat proj_in_rgb = kWhite - (kNorm.dot(kWhite) / kNorm.dot(norm_rgb - kWhite)) * (norm_rgb - kWhite);
    cv::Mat proj_in_lab = proj_in_rgb * kRgbToLab.t();

    cv::Mat full_proj_in_lab;
    proj_in_lab.convertTo(full_proj_in_lab, CV_32SC1, 255.0);

    return full_proj_in_lab;
  } else {
    return (cv::Mat_<int>(1, 3) << 0, 0, 0);
  }
}

std::vector<int> DocColorDecomposer::FindHistPeaks(const cv::Mat& hist, int min_h) {
  std::vector<int> extremes, peaks;

  int curr_delta = 0;
  int prev_delta = hist.at<int>(0) - hist.at<int>(hist.cols - 1);
  for (int i = 0; i < hist.cols; ++i) {
    curr_delta = hist.at<int>((i + 1) % hist.cols) - hist.at<int>(i);

    if (prev_delta != 0 && curr_delta == 0) {
      int j = i;
      while (hist.at<int>(++j % hist.cols) == hist.at<int>(i)) {}

      int next_delta = hist.at<int>(j % hist.cols) - hist.at<int>(i);
      if (prev_delta * next_delta < 0) {
        int mid = ((i + j) / 2) % hist.cols;
        extremes.push_back(mid);
      }

    } else if (prev_delta * curr_delta < 0) {
      extremes.push_back(i);
    }

    prev_delta = curr_delta;
  }
  std::ranges::sort(extremes);

  if (hist.at<int>(extremes[0]) > hist.at<int>(extremes[1])) {
    std::ranges::rotate(extremes, extremes.begin() + 1);
  }

  for (std::size_t i = 1; i < extremes.size(); i += 2) {
    int lh = hist.at<int>(extremes[i]) - hist.at<int>(extremes[(i - 1) % extremes.size()]);
    int rh = hist.at<int>(extremes[i]) - hist.at<int>(extremes[(i + 1) % extremes.size()]);

    if (std::min(lh, rh) >= min_h) {
      peaks.push_back(extremes[i]);
    }
  }
  std::ranges::sort(peaks);

  return peaks;
}

double DocColorDecomposer::ComputeIou(const cv::Mat& predicted_mask, const cv::Mat& truth_mask) {
  double intersection_area = cv::countNonZero(predicted_mask & truth_mask);
  double union_area = cv::countNonZero(predicted_mask | truth_mask);

  return intersection_area / union_area;
}

double DocColorDecomposer::ComputePq(const std::vector<cv::Mat>& predicted_masks, const std::vector<cv::Mat>& truth_masks) {
  double sum_iou = 0.0;
  double tp = 0.0;

  std::vector<bool> matched_predicted_masks(predicted_masks.size(), false);
  std::vector<bool> matched_truth_masks(truth_masks.size(), false);

  for (const auto& [predicted_mask_idx, predicted_mask] : predicted_masks | std::views::enumerate) {
    double max_iou = 0.0;
    std::size_t max_iou_idx = -1;

    for (const auto& [truth_mask_idx, truth_mask] : truth_masks | std::views::enumerate) {
      double iou = ComputeIou(predicted_mask, truth_mask);

      if (iou >= max_iou) {
        max_iou = iou;
        max_iou_idx = truth_mask_idx;
      }
    }

    if (max_iou >= 0.5) {
      sum_iou += max_iou;
      ++tp;

      matched_predicted_masks[predicted_mask_idx] = true;
      matched_truth_masks[max_iou_idx] = true;
    }
  }

  double fp = static_cast<double>(std::ranges::count(matched_predicted_masks, false));
  double fn = static_cast<double>(std::ranges::count(matched_truth_masks, false));

  return sum_iou / (tp + 0.5 * (fp + fn));
}
