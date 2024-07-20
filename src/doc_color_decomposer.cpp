#include "doc_color_decomposer/doc_color_decomposer.h"

DocColorDecomposer::DocColorDecomposer(const cv::Mat& src) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  src_ = CvtSRgbToLinRgb(SmoothHue(ThreshSaturation(src)));
  color_to_n_ = LutColorToN(src_);

  ComputePhiHist();
  ComputePhiClusters();
}

std::vector<cv::Mat> DocColorDecomposer::GetLayers() {
  std::vector<cv::Mat> layers(phi_clusters_.size() + 1);

  for (auto& layer : layers) {
    layer = cv::Mat(src_.rows, src_.cols, CV_32FC3, cv::Vec3f(1.0F, 1.0F, 1.0F));
  }

  for (int y = 0; y < src_.rows; ++y) {
    for (int x = 0; x < src_.cols; ++x) {
      auto& pixel = src_.at<cv::Vec3f>(y, x);
      std::tuple<float, float, float> color = std::make_tuple(pixel[2], pixel[1], pixel[0]);
      int phi = color_to_phi_[color];
      int cluster = (phi != -1) ? phi_to_cluster_[phi] : 0;
      layers[cluster].at<cv::Vec3f>(y, x) = pixel;
    }
  }

  std::ranges::for_each(layers, [](auto& layer) { layer = CvtLinRgbToSRgb(layer); });

  return layers;
}

cv::Mat DocColorDecomposer::ThreshSaturation(cv::Mat src, double s_thresh, double v_thresh) {
  cv::cvtColor(src, src, cv::COLOR_BGR2HSV_FULL);

  std::vector<cv::Mat> hsv_channels;
  cv::split(src, hsv_channels);

  cv::threshold(hsv_channels[1], hsv_channels[1], s_thresh, 0.0, cv::THRESH_TOZERO);
  cv::threshold(hsv_channels[2], hsv_channels[2], v_thresh, 0.0, cv::THRESH_TOZERO);

  cv::merge(hsv_channels, src);

  cv::cvtColor(src, src, cv::COLOR_HSV2BGR_FULL);

  return src;
}

cv::Mat DocColorDecomposer::SmoothHue(cv::Mat src, int ker_size) {
  cv::Mat smooth_src;
  cv::GaussianBlur(src, smooth_src, cv::Size(ker_size, ker_size), ker_size);

  cv::cvtColor(src, src, cv::COLOR_BGR2HSV_FULL);
  cv::cvtColor(smooth_src, smooth_src, cv::COLOR_BGR2HSV_FULL);

  const int kFromTo[] = {0, 0};
  cv::mixChannels(&smooth_src, 1, &src, 1, kFromTo, 1);

  cv::cvtColor(src, src, cv::COLOR_HSV2BGR_FULL);

  return src;
}

cv::Mat DocColorDecomposer::CvtSRgbToLinRgb(cv::Mat src) {
  src.convertTo(src, CV_32FC3, 1.0 / 255.0);

  src.forEach<cv::Vec3f>([](cv::Vec3f& pixel, const int*) -> void {
    for (int i = 0; i < 3; ++i) {
      pixel[i] = (pixel[i] <= 0.04045F) ? (pixel[i] / 12.92F) : std::pow((pixel[i] + 0.055F) / 1.055F, 2.4F);
    }
  });

  return src;
}

cv::Mat DocColorDecomposer::CvtLinRgbToSRgb(cv::Mat src) {
  src.forEach<cv::Vec3f>([](cv::Vec3f& pixel, const int*) -> void {
    for (int i = 0; i < 3; ++i) {
      pixel[i] = (pixel[i] <= 0.0031308F) ? (pixel[i] * 12.92F) : (1.055F * std::pow(pixel[i], 1.0F / 2.4F) - 0.055F);
    }
  });

  src.convertTo(src, CV_8UC3, 255.0);

  return src;
}

std::map<std::tuple<float, float, float>, long long> DocColorDecomposer::LutColorToN(const cv::Mat& src) {
  std::map<std::tuple<float, float, float>, long long> color_to_n;

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3f pixel = src.at<cv::Vec3f>(y, x);
      std::tuple<float, float, float> color = std::make_tuple(pixel[2], pixel[1], pixel[0]);
      color_to_n[color] += 1LL;
    }
  }

  return color_to_n;
}

cv::Mat DocColorDecomposer::ProjOnLab(const cv::Mat& rgb_point) {
  const cv::Mat kWhite = (cv::Mat_<float>(1, 3) << 1.0F, 1.0F, 1.0F);
  const cv::Mat kNorm = (cv::Mat_<float>(1, 3) << 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F));
  const cv::Mat kRgbToLab = (cv::Mat_<float>(3, 3) <<
      +1.0F / std::sqrt(2.0F), -1.0F / std::sqrt(2.0F), +0.0F,
      -1.0F / std::sqrt(6.0F), -1.0F / std::sqrt(6.0F), +2.0F / std::sqrt(6.0F),
      +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F)
  );

  cv::Mat proj_in_lab;
  if (rgb_point.at<float>(0) != 1.0F || rgb_point.at<float>(1) != 1.0F || rgb_point.at<float>(2) != 1.0F) {
    cv::Mat proj_in_rgb = kWhite - (kNorm.dot(kWhite) / kNorm.dot(rgb_point - kWhite)) * (rgb_point - kWhite);
    proj_in_lab = proj_in_rgb * kRgbToLab.t();
  } else {
    proj_in_lab = (cv::Mat_<float>(1, 3) << 0.0F, 0.0F, 0.0F);
  }

  return proj_in_lab;
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

  for (int i = 1; i < extremes.size(); i += 2) {
    int lh = hist.at<int>(extremes[i]) - hist.at<int>(extremes[(i - 1) % extremes.size()]);
    int rh = hist.at<int>(extremes[i]) - hist.at<int>(extremes[(i + 1) % extremes.size()]);
    if (std::max(lh, rh) >= min_h) {
      peaks.push_back(extremes[i]);
    }
  }

  return peaks;
}

void DocColorDecomposer::ComputePhiHist(int ker_size) {
  phi_hist_ = cv::Mat::zeros(1, 360, CV_64F);

  for (const auto& [color, n] : color_to_n_) {
    if (std::get<0>(color) != std::get<1>(color) || std::get<1>(color) != std::get<2>(color) || std::get<0>(color) != std::get<2>(color)) {
      cv::Mat rgb_point = (cv::Mat_<float>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
      cv::Mat lab_point = ProjOnLab(rgb_point);
      float phi_rad = std::atan2(lab_point.at<float>(0, 1), lab_point.at<float>(0, 0));
      int phi = std::lround((phi_rad * 180.0F / CV_PI) + 360.0F) % 360;
      phi_hist_.at<double>(phi) += static_cast<double>(n);
      color_to_phi_[color] = phi;
    } else {
      color_to_phi_[color] = -1;
    }
  }

  cv::GaussianBlur(phi_hist_, smooth_phi_hist_, cv::Size(ker_size, ker_size), 0.0);
  smooth_phi_hist_.convertTo(smooth_phi_hist_, CV_32S);
}

void DocColorDecomposer::ComputePhiClusters() {
  phi_to_cluster_ = std::vector<int>(360, 1);

  double max_h;
  cv::minMaxLoc(smooth_phi_hist_, nullptr, &max_h, nullptr, nullptr);
  std::vector<int> peaks = FindHistPeaks(smooth_phi_hist_, std::lround(0.01 * max_h));
  peaks.push_back(peaks[0] + 360);

  for (int i = 0; i < peaks.size() - 1; ++i) {
    phi_clusters_.push_back(((peaks[i + 1] + peaks[i]) / 2) % 360);
  }
  std::ranges::sort(phi_clusters_);

  for (int i = 0; i < phi_clusters_.size() - 1; ++i) {
    for (int j = phi_clusters_[i]; j < phi_clusters_[i + 1]; ++j) {
      phi_to_cluster_[j] = i + 2;
    }
  }
}

std::string DocColorDecomposer::Plot3dRgb(int yaw, int pitch) {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

  std::vector<std::pair<std::tuple<float, float, float>, long long>> sorted_color_to_n(color_to_n_.begin(), color_to_n_.end());
  std::ranges::shuffle(sorted_color_to_n, std::mt19937(std::random_device()()));
  sorted_color_to_n.resize(std::min(sorted_color_to_n.size(), static_cast<size_t>(7000)));

  plot << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n";
  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "view={" << yaw << "}{" << pitch << "},\n";
  plot << "height=10cm, width=10cm,\n";
  plot << "axis lines=center,\n";
  plot << "axis equal,\n";
  plot << "scale only axis,\n";
  plot << "enlargelimits=true,\n";
  plot << "xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1,\n";
  plot << "xtick={0}, ytick={0}, ztick={0},\n";
  plot << "xlabel={$R$}, ylabel={$G$}, zlabel={$B$}]\n\n";

  plot << "\\draw[lightgray] (axis cs:1,0,0) -- (axis cs:1,1,0) -- (axis cs:0,1,0);\n";
  plot << "\\draw[lightgray] (axis cs:1,1,1) -- (axis cs:0,1,1) -- (axis cs:0,0,1) -- (axis cs:1,0,1) -- (axis cs:1,1,1);\n";
  plot << "\\draw[lightgray] (axis cs:1,0,0) -- (axis cs:1,0,1);\n";
  plot << "\\draw[lightgray] (axis cs:1,1,0) -- (axis cs:1,1,1);\n";
  plot << "\\draw[lightgray] (axis cs:0,1,0) -- (axis cs:0,1,1);\n\n";

  plot << "\\addplot3[\n";
  plot << "only marks,\n";
  plot << "mark=*,\n";
  plot << "mark size=0.1,\n";
  plot << "color=purple!75]\n";
  plot << "table[]{\n";

  for (const auto& color : sorted_color_to_n | std::views::keys) {
    plot << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << '\n';
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

cv::Mat DocColorDecomposer::Plot2dLab() {
  cv::Mat plot(1275, 1275, CV_32FC3, cv::Vec3f(0.0F, 0.0F, 0.0F));

  for (const auto& color : color_to_n_ | std::views::keys) {
    cv::Mat rgb_point = (cv::Mat_<float>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
    cv::Mat lab_point = ProjOnLab(rgb_point);
    int y = std::lround(255.0F * (lab_point.at<float>(0, 1) + 3.0F));
    int x = std::lround(255.0F * (lab_point.at<float>(0, 0) + 2.5F));
    plot.at<cv::Vec3f>(y, x) = cv::Vec3f(std::get<2>(color), std::get<1>(color), std::get<0>(color));
  }

  return CvtLinRgbToSRgb(plot);
}

std::string DocColorDecomposer::Plot1dPhi() {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(1);

  double max_n;
  cv::minMaxLoc(phi_hist_, nullptr, &max_n, nullptr, nullptr);

  plot << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n";
  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "height=10cm, width=30cm,\n";
  plot << "xmin=0, xmax=360, ymin=0, ymax=" << std::lround(max_n) << ",\n";
  plot << "tick align=outside,\n";
  plot << "grid=both,\n";
  plot << "yminorgrids=true,\n";
  plot << "xlabel={$\\phi$}, ylabel={$n$}]\n\n";

  plot << "\\addplot[\n";
  plot << "ybar interval,\n";
  plot << "mark=none,\n";
  plot << "fill=purple!25,\n";
  plot << "draw=purple]\n";
  plot << "coordinates{\n";

  for (int phi = 0; phi < 360; ++phi) {
    plot << '(' << phi << ',' << std::lround(phi_hist_.at<double>(phi)) << ")\n";
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

std::string DocColorDecomposer::Plot1dClusters() {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(1);

  double max_n;
  cv::minMaxLoc(smooth_phi_hist_, nullptr, &max_n, nullptr, nullptr);

  plot << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot << "\\usepackage{pgfplots}\n";
  plot << "\\pgfplotsset{compat=newest}\n";
  plot << "\\begin{document}\n";
  plot << "\\begin{tikzpicture}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "height=10cm, width=30cm,\n";
  plot << "xmin=0, xmax=360, ymin=0, ymax=" << std::lround(max_n) << ",\n";
  plot << "tick align=outside,\n";
  plot << "grid=both,\n";
  plot << "yminorgrids=true,\n";
  plot << "xlabel={$\\phi$}, ylabel={$n$}]\n\n";

  plot << "\\addplot[\n";
  plot << "ybar interval,\n";
  plot << "mark=none,\n";
  plot << "fill=purple!25,\n";
  plot << "draw=purple]\n";
  plot << "coordinates{\n";

  for (int phi = 0; phi < 360; ++phi) {
    plot << '(' << phi << ',' << smooth_phi_hist_.at<int>(phi) << ")\n";
  }
  plot << "};\n\n";

  for (const auto& cluster : phi_clusters_) {
    plot << "\\draw[cyan!75, thick] (axis cs:" << cluster << ",0) -- (axis cs:" << cluster << ',' << std::lround(max_n) << ");\n";
  }
  plot << '\n';

  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}
