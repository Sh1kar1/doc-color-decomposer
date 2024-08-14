#include "doc_color_decomposer/doc_color_decomposer.h"

#include "../src/plot_2d_lab_image.h"

DocColorDecomposer::DocColorDecomposer(const cv::Mat& src, int tolerance, bool preprocessing) {
  src_ = CvtSRgbToLinRgb(src);
  processed_src_ = preprocessing ? CvtSRgbToLinRgb(ThreshLightness(ThreshSaturation(src))) : src_;
  tolerance_ = tolerance;

  color_to_n_ = LutColorToN(processed_src_);

  ComputePhiHist();
  ComputePhiClusters();
  ComputeLayers();
}

std::vector<cv::Mat> DocColorDecomposer::GetLayers() const {
  return layers_;
}

cv::Mat DocColorDecomposer::MergeLayers() {
  cv::Mat merged_layers(processed_src_.rows, processed_src_.cols, CV_8UC3, cv::Vec3b(255, 255, 255));

  std::ranges::for_each(layers_, [&merged_layers](auto& layer) { layer.copyTo(merged_layers, (layer != cv::Vec3b(255, 255, 255))); });

  return merged_layers;
}

std::string DocColorDecomposer::Plot3dRgb(double yaw, double pitch) {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

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
  plot << "  axis lines=center,\n";
  plot << "  axis equal,\n";
  plot << "  scale only axis,\n";
  plot << "  enlargelimits,\n";
  plot << "  xmin=0, xmax=1,\n";
  plot << "  ymin=0, ymax=1,\n";
  plot << "  zmin=0, zmax=1,\n";
  plot << "  xtick={0},\n";
  plot << "  ytick={0},\n";
  plot << "  ztick={0},\n";
  plot << "  xlabel={$R$},\n";
  plot << "  ylabel={$G$},\n";
  plot << "  zlabel={$B$},\n";
  plot << "  zlabel style={anchor=south}\n";
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

  std::vector<std::pair<std::array<float, 3>, long long>> shuffled_color_to_n(color_to_n_.begin(), color_to_n_.end());
  std::ranges::shuffle(shuffled_color_to_n, std::mt19937(std::random_device()()));
  shuffled_color_to_n.resize(std::min(shuffled_color_to_n.size(), static_cast<size_t>(5000)));

  for (const auto& color : shuffled_color_to_n | std::views::keys) {
    plot << color[0] << ' ' << color[1] << ' ' << color[2] << '\n';
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n\n";

  plot << "\\begin{axis}[\n";
  plot << "  view={" << yaw << "}{" << pitch << "},\n";
  plot << "  height=10cm,\n";
  plot << "  width=10cm,\n";
  plot << "  axis lines=none,\n";
  plot << "  axis equal,\n";
  plot << "  scale only axis,\n";
  plot << "  enlargelimits,\n";
  plot << "  xmin=0, xmax=1,\n";
  plot << "  ymin=0, ymax=1,\n";
  plot << "  zmin=0, zmax=1\n";
  plot << "]\n\n";

  plot << "\\draw (axis cs:1,0,0) -- (axis cs:1,1,0) -- (axis cs:0,1,0);\n";
  plot << "\\draw (axis cs:1,1,1) -- (axis cs:0,1,1) -- (axis cs:0,0,1) -- (axis cs:1,0,1) -- (axis cs:1,1,1);\n";
  plot << "\\draw (axis cs:1,0,0) -- (axis cs:1,0,1);\n";
  plot << "\\draw (axis cs:1,1,0) -- (axis cs:1,1,1);\n";
  plot << "\\draw (axis cs:0,1,0) -- (axis cs:0,1,1);\n\n";

  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  return plot.str();
}

cv::Mat DocColorDecomposer::Plot2dLab() {
  cv::Mat plot = cv::imdecode(cv::Mat(1, plot_2d_lab_data_len, CV_8U, plot_2d_lab_data), cv::IMREAD_UNCHANGED);
  plot = CvtSRgbToLinRgb(plot);

  for (const auto& color : color_to_n_ | std::views::keys) {
    cv::Mat rgb = (cv::Mat_<float>(1, 3) << color[0], color[1], color[2]);
    cv::Mat lab = ProjOnLab(rgb);
    int y = std::lround(255.0F * (lab.at<float>(0, 1) + 2.95F));
    int x = std::lround(255.0F * (lab.at<float>(0, 0) + 2.95F));

    plot.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
  }

  return CvtLinRgbToSRgb(plot);
}

std::string DocColorDecomposer::Plot1dPhi() {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

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
  plot << "  tick style={draw=none},\n";
  plot << "  xlabel={$\\phi$},\n";
  plot << "  ylabel={$n$}\n";
  plot << "]\n\n";

  for (int phi = 0; phi < 359; ++phi) {
    std::array<float, 3> mean_color = phi_to_mean_color_[phi];

    plot << "\\addplot[\n";
    plot << "  ybar interval,\n";
    plot << "  color={rgb,1: red," << mean_color[0] << "; green," << mean_color[1] << "; blue," << mean_color[2] << "},\n";
    plot << "  fill={rgb,1: red," << mean_color[0] << "; green," << mean_color[1] << "; blue," << mean_color[2] << "}\n";
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

std::string DocColorDecomposer::Plot1dClusters() {
  std::stringstream plot;
  plot << std::fixed << std::setprecision(4);

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
  plot << "  tick style={draw=none},\n";
  plot << "  xlabel={$\\phi$},\n";
  plot << "  ylabel={$n$}\n";
  plot << "]\n\n";

  for (int phi = 0; phi < 359; ++phi) {
    int cluster = phi_to_cluster_[phi];
    std::array<float, 3> mean_color = cluster_to_mean_color_[cluster];

    plot << "\\addplot[\n";
    plot << "  ybar interval,\n";
    plot << "  color={rgb,1: red," << mean_color[0] << "; green," << mean_color[1] << "; blue," << mean_color[2] << "},\n";
    plot << "  fill={rgb,1: red," << mean_color[0] << "; green," << mean_color[1] << "; blue," << mean_color[2] << "}\n";
    plot << "]\n";
    plot << "table[] {\n";
    plot << "X Y\n";
    plot << phi << ' ' << std::lround(smoothed_phi_hist_.at<int>(phi)) << '\n';
    plot << phi + 1 << ' ' << std::lround(smoothed_phi_hist_.at<int>(phi + 1)) << '\n';
    plot << "};\n\n";
  }

  for (const auto& cluster : phi_clusters_) {
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
  phi_hist_ = cv::Mat::zeros(1, 360, CV_64F);

  for (const auto& [color, n] : color_to_n_) {
    if (color[0] != color[1] || color[1] != color[2] || color[0] != color[2]) {
      cv::Mat rgb = (cv::Mat_<float>(1, 3) << color[0], color[1], color[2]);
      cv::Mat lab = ProjOnLab(rgb);
      float phi_rad = std::atan2(-lab.at<float>(0, 1), lab.at<float>(0, 0));
      int phi = std::lround((phi_rad * 180.0F / CV_PI) + 360.0F) % 360;

      phi_hist_.at<double>(phi) += static_cast<double>(n);

      color_to_lab_[color] = lab;
      color_to_phi_[color] = phi;

    } else {
      color_to_lab_[color] = (cv::Mat_<float>(1, 3) << 0.0F, 0.0F, 0.0F);
      color_to_phi_[color] = -1;
    }
  }

  cv::GaussianBlur(phi_hist_, smoothed_phi_hist_, cv::Size(tolerance_, tolerance_), 0.0);
  smoothed_phi_hist_.convertTo(smoothed_phi_hist_, CV_32S);
}

void DocColorDecomposer::ComputePhiClusters() {
  phi_to_cluster_ = std::vector<int>(360, 1);

  double max_h;
  cv::minMaxLoc(smoothed_phi_hist_, nullptr, &max_h, nullptr, nullptr);
  std::vector<int> peaks = FindHistPeaks(smoothed_phi_hist_, std::lround(0.01 * max_h));
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

void DocColorDecomposer::ComputeLayers() {
  layers_ = std::vector<cv::Mat>(phi_clusters_.size() + 1);
  for (auto& layer : layers_) {
    layer = cv::Mat(processed_src_.rows, processed_src_.cols, CV_32FC3, cv::Vec3f(1.0F, 1.0F, 1.0F));
  }

  phi_to_n_ = std::vector<long long>(360);
  cluster_to_n_ = std::vector<long long>(phi_clusters_.size() + 1);
  phi_to_mean_color_ = std::vector<std::array<float, 3>>(360);
  cluster_to_mean_color_ = std::vector<std::array<float, 3>>(phi_clusters_.size() + 1);

  for (int y = 0; y < processed_src_.rows; ++y) {
    for (int x = 0; x < processed_src_.cols; ++x) {
      auto& pixel = processed_src_.at<cv::Vec3f>(y, x);
      std::array<float, 3> color = {pixel[2], pixel[1], pixel[0]};
      int phi = color_to_phi_[color];
      int cluster = (phi != -1) ? phi_to_cluster_[phi] : 0;

      layers_[cluster].at<cv::Vec3f>(y, x) = src_.at<cv::Vec3f>(y, x);

      if (phi != -1) {
        auto n = static_cast<float>(phi_to_n_[phi]);
        std::array<float, 3> mean_color = phi_to_mean_color_[phi];
        float r = (color[0] + n * mean_color[0]) / (n + 1);
        float g = (color[1] + n * mean_color[1]) / (n + 1);
        float b = (color[2] + n * mean_color[2]) / (n + 1);

        phi_to_mean_color_[phi] = {r, g, b};
        ++phi_to_n_[phi];
      }

      if (cluster != 0) {
        auto n = static_cast<float>(cluster_to_n_[cluster]);
        std::array<float, 3> mean_color = cluster_to_mean_color_[cluster];
        float r = (color[0] + n * mean_color[0]) / (n + 1);
        float g = (color[1] + n * mean_color[1]) / (n + 1);
        float b = (color[2] + n * mean_color[2]) / (n + 1);

        cluster_to_mean_color_[cluster] = {r, g, b};
        ++cluster_to_n_[cluster];
      }
    }
  }

  std::ranges::for_each(layers_, [](auto& layer) { layer = CvtLinRgbToSRgb(layer); });
}

cv::Mat DocColorDecomposer::ThreshSaturation(cv::Mat src, double thresh) {
  cv::cvtColor(src, src, cv::COLOR_BGR2HSV_FULL);

  std::vector<cv::Mat> hsv_channels;
  cv::split(src, hsv_channels);

  cv::threshold(hsv_channels[1], hsv_channels[1], thresh, 0.0, cv::THRESH_TOZERO);

  cv::merge(hsv_channels, src);

  cv::cvtColor(src, src, cv::COLOR_HSV2BGR_FULL);

  return src;
}

cv::Mat DocColorDecomposer::ThreshLightness(cv::Mat src, double thresh) {
  cv::cvtColor(src, src, cv::COLOR_BGR2HLS_FULL);

  std::vector<cv::Mat> hls_channels;
  cv::split(src, hls_channels);

  cv::threshold(hls_channels[1], hls_channels[1], thresh, 0.0, cv::THRESH_TOZERO);

  cv::merge(hls_channels, src);

  cv::cvtColor(src, src, cv::COLOR_HLS2BGR_FULL);

  return src;
}

cv::Mat DocColorDecomposer::CvtSRgbToLinRgb(cv::Mat src) {
  src.convertTo(src, CV_32FC3, 1.0 / 255.0);

  // src.forEach<cv::Vec3f>([](cv::Vec3f& pixel, const int*) -> void {
  //   for (int i = 0; i < 3; ++i) {
  //     pixel[i] = (pixel[i] <= 0.04045F) ? (pixel[i] / 12.92F) : std::pow((pixel[i] + 0.055F) / 1.055F, 2.4F);
  //   }
  // });

  return src;
}

cv::Mat DocColorDecomposer::CvtLinRgbToSRgb(cv::Mat src) {
  // src.forEach<cv::Vec3f>([](cv::Vec3f& pixel, const int*) -> void {
  //   for (int i = 0; i < 3; ++i) {
  //     pixel[i] = (pixel[i] <= 0.0031308F) ? (pixel[i] * 12.92F) : (1.055F * std::pow(pixel[i], 1.0F / 2.4F) - 0.055F);
  //   }
  // });

  src.convertTo(src, CV_8UC3, 255.0);

  return src;
}

std::map<std::array<float, 3>, long long> DocColorDecomposer::LutColorToN(const cv::Mat& src) {
  std::map<std::array<float, 3>, long long> color_to_n;

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3f pixel = src.at<cv::Vec3f>(y, x);
      std::array<float, 3> color = {pixel[2], pixel[1], pixel[0]};

      ++color_to_n[color];
    }
  }

  return color_to_n;
}

cv::Mat DocColorDecomposer::ProjOnLab(const cv::Mat& rgb) {
  const cv::Mat kWhite = (cv::Mat_<float>(1, 3) << 1.0F, 1.0F, 1.0F);
  const cv::Mat kNorm = (cv::Mat_<float>(1, 3) << 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F));
  const cv::Mat kRgbToLab = (cv::Mat_<float>(3, 3) <<
      -1.0F / std::sqrt(2.0F), +1.0F / std::sqrt(2.0F), -0.0F,
      +1.0F / std::sqrt(6.0F), +1.0F / std::sqrt(6.0F), -2.0F / std::sqrt(6.0F),
      +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F)
  );

  cv::Mat proj_in_lab;
  if (rgb.at<float>(0) != 1.0F || rgb.at<float>(1) != 1.0F || rgb.at<float>(2) != 1.0F) {
    cv::Mat proj_in_rgb = kWhite - (kNorm.dot(kWhite) / kNorm.dot(rgb - kWhite)) * (rgb - kWhite);
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

    if (std::min(lh, rh) >= min_h) {
      peaks.push_back(extremes[i]);
    }
  }
  std::ranges::sort(peaks);

  return peaks;
}
