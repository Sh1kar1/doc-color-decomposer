#include <doc_color_clustering/doc_color_clustering.h>


DocColorClustering::DocColorClustering(const cv::Mat& src) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  std::ios::sync_with_stdio(false);

  this->src_ = this->SRgbToLinRgb(src);
  this->color_to_n_ = this->LutColorToN(this->src_);

  this->CalcPhiHist();
  this->CalcPhiClusters();
}


cv::Mat DocColorClustering::SRgbToLinRgb(cv::Mat src) {
  src.convertTo(src, CV_32FC3, 1.0 / 255.0);

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3f& pixel = src.at<cv::Vec3f>(y, x);
      for (int k = 0; k < 3; ++k) {
        if (pixel[k] <= 0.04045F) {
          pixel[k] /= 12.92F;
        } else {
          pixel[k] = std::pow((pixel[k] + 0.055F) / 1.055F, 2.4F);
        }
      }
    }
  }

  return src;
}


cv::Mat DocColorClustering::LinRgbToSRgb(cv::Mat src) {
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3f& pixel = src.at<cv::Vec3f>(y, x);
      for (int k = 0; k < 3; ++k) {
        if (pixel[k] <= 0.0031308F) {
          pixel[k] *= 12.92F;
        } else {
          pixel[k] = 1.055F * std::pow(pixel[k], 1.0F / 2.4F) - 0.055F;
        }
      }
    }
  }

  src.convertTo(src, CV_8UC3, 255.0);

  return src;
}


std::map<std::tuple<float, float, float>, long long> DocColorClustering::LutColorToN(const cv::Mat& src) {
  std::map<std::tuple<float, float, float>, long long> color_to_n;

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      cv::Vec3f pixel = src.at<cv::Vec3f>(y, x);
      float r = std::round(pixel[2] * 10000.0F) / 10000.0F;
      float g = std::round(pixel[1] * 10000.0F) / 10000.0F;
      float b = std::round(pixel[0] * 10000.0F) / 10000.0F;
      std::tuple<float, float, float> color = std::make_tuple(r, g, b);
      color_to_n[color] += 1LL;
    }
  }

  return color_to_n;
}


cv::Mat DocColorClustering::CentralProjOnLab(const cv::Mat& rgb_point) {
  const cv::Mat white = (cv::Mat_<float>(1, 3) << 1.0F, 1.0F, 1.0F);
  const cv::Mat norm = (cv::Mat_<float>(1, 3) << 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F), 1.0F / std::sqrt(3.0F));
  const cv::Mat rgb_to_lab = (cv::Mat_<float>(3, 3) <<
      +1.0F / std::sqrt(2.0F), -1.0F / std::sqrt(2.0F), +0.0F,
      -1.0F / std::sqrt(6.0F), -1.0F / std::sqrt(6.0F), +2.0F / std::sqrt(6.0F),
      +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F), +1.0F / std::sqrt(3.0F)
  );

  if (rgb_point.at<float>(0) == 1.0F && rgb_point.at<float>(1) == 1.0F && rgb_point.at<float>(2) == 1.0F) {
    return cv::Mat_<float>(1, 3) << 0.0F, 0.0F, 0.0F;
  }

  cv::Mat proj_in_rgb = white - (norm.dot(white) / norm.dot(rgb_point - white)) * (rgb_point - white);
  cv::Mat proj_in_lab = proj_in_rgb * rgb_to_lab.t();

  return proj_in_lab;
}


std::vector<int> DocColorClustering::HistPeaks(const cv::Mat& hist) {
  std::vector<int> peaks;

  int delta = 0;
  int prev_delta = hist.at<int>(0) - hist.at<int>(hist.cols - 1);
  for (int i = 0; i < hist.cols; ++i) {
    delta = hist.at<int>((i + 1) % hist.cols) - hist.at<int>(i);
    if (prev_delta != 0 && delta == 0) {
      int j = i;
      while (hist.at<int>(++j % hist.cols) == hist.at<int>(i));
      int next_delta = hist.at<int>(j % hist.cols) - hist.at<int>(i);
      if (prev_delta * next_delta < 0) {
        peaks.push_back(((i + j) / 2) % hist.cols);
      }
    } else if (prev_delta * delta < 0) {
      peaks.push_back(i);
    }
    prev_delta = delta;
  }

  std::sort(peaks.begin(), peaks.end());
  if (hist.at<int>(peaks[0]) > hist.at<int>(peaks[1])) {
    std::rotate(peaks.begin(), peaks.begin() + 1, peaks.end());
  }

  return peaks;
}


std::vector<int> DocColorClustering::FilterMaximums(const std::vector<int>& peaks, const cv::Mat& hist, int min_h, int min_w) {
  std::vector<int> high_maximums;
  for (int i = 1; i < static_cast<int>(peaks.size()); i += 2) {
    int lh = hist.at<int>(peaks[i]) - hist.at<int>(peaks[(i - 1) % static_cast<int>(peaks.size())]);
    int rh = hist.at<int>(peaks[i]) - hist.at<int>(peaks[(i + 1) % static_cast<int>(peaks.size())]);
    if (std::max(lh, rh) >= min_h) {
      high_maximums.push_back(peaks[i]);
    }
  }

  std::vector<int> high_long_maximums;
  for (int i = 0; i < static_cast<int>(high_maximums.size()); ++i) {
    if (std::abs(high_maximums[(i + 1) % static_cast<int>(high_maximums.size())] - high_maximums[i]) < min_w) {
      high_long_maximums.push_back((high_maximums[i] + high_maximums[(i + 1) % static_cast<int>(high_maximums.size())]) / 2);
      ++i;
    } else {
      high_long_maximums.push_back(high_maximums[i]);
    }
  }
  std::sort(high_long_maximums.begin(), high_long_maximums.end());

  return high_long_maximums;
}


void DocColorClustering::CalcPhiHist() {
  this->phi_hist_ = cv::Mat::zeros(1, 360, CV_64F);

  for (const auto& [color, n] : this->color_to_n_) {
    if (std::get<0>(color) != std::get<1>(color) || std::get<1>(color) != std::get<2>(color) || std::get<0>(color) != std::get<2>(color)) {
      cv::Mat rgb_point = (cv::Mat_<float>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
      cv::Mat lab_point = this->CentralProjOnLab(rgb_point);
      int phi = std::lround(std::atan2(lab_point.at<float>(0, 1), lab_point.at<float>(0, 0)) * 180.0F / CV_PI + 360.0F) % 360;
      phi_hist_.at<double>(phi) += static_cast<double>(n);
      this->color_to_phi_[color] = phi;
    } else {
      this->color_to_phi_[color] = -1;
    }
  }

  cv::GaussianBlur(this->phi_hist_, this->smooth_phi_hist_, cv::Size(15, 1), 0.0);
  this->smooth_phi_hist_.convertTo(this->smooth_phi_hist_, CV_32S);
}


void DocColorClustering::CalcPhiClusters() {
  this->phi_to_cluster_ = std::vector<int>(360, 0);

  double max_h;
  cv::minMaxLoc(this->smooth_phi_hist_, nullptr, &max_h, nullptr, nullptr);
  std::vector<int> peaks = this->FilterMaximums(this->HistPeaks(this->smooth_phi_hist_), this->smooth_phi_hist_, std::lround(0.05 * max_h), 10);
  peaks.push_back(peaks[0] + 360);

  for (int i = 0; i < static_cast<int>(peaks.size()) - 1; ++i) {
    this->phi_clusters_.push_back(((peaks[i + 1] + peaks[i]) / 2) % 360);
  }
  std::sort(this->phi_clusters_.begin(), this->phi_clusters_.end());

  for (int i = 0; i < static_cast<int>(this->phi_clusters_.size()) - 1; ++i) {
    for (int j = this->phi_clusters_[i]; j < this->phi_clusters_[i + 1]; ++j) {
      this->phi_to_cluster_[j] = i + 1;
    }
  }
}


void DocColorClustering::Plot3dRgb(const std::string& output_path, int yaw, int pitch) {
  std::ofstream plot(output_path);
  plot << std::fixed << std::setprecision(4);

  std::vector<std::pair<std::tuple<float, float, float>, long long>> sorted_color_to_n(this->color_to_n_.begin(), this->color_to_n_.end());
  std::shuffle(sorted_color_to_n.begin(), sorted_color_to_n.end(), std::mt19937(std::random_device()()));
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

  for (const auto& [color, _] : sorted_color_to_n) {
    plot << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << '\n';
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  plot.close();
}


void DocColorClustering::Plot2dLab(const std::string& output_path) {
  cv::Mat plot(1275, 1275, CV_32FC3, cv::Vec3f(0.0F, 0.0F, 0.0F));

  for (const auto& [color, _] : this->color_to_n_) {
    cv::Mat rgb_point = (cv::Mat_<float>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
    cv::Mat lab_point = this->CentralProjOnLab(rgb_point);
    int y = std::lround(255.0F * (lab_point.at<float>(0, 1) + 3.0F));
    int x = std::lround(255.0F * (lab_point.at<float>(0, 0) + 2.5F));
    plot.at<cv::Vec3f>(y, x) = cv::Vec3f(std::get<2>(color), std::get<1>(color), std::get<0>(color));
  }

  cv::imwrite(output_path, this->LinRgbToSRgb(plot));
}


void DocColorClustering::Plot1dPhi(const std::string& output_path) {
  std::ofstream plot(output_path);
  plot << std::fixed << std::setprecision(1);

  double max_n;
  cv::minMaxLoc(this->phi_hist_, nullptr, &max_n, nullptr, nullptr);

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
    plot << '(' << phi << ',' << std::lround(this->phi_hist_.at<double>(phi)) << ")\n";
  }

  plot << "};\n\n";
  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  plot.close();
}


void DocColorClustering::Plot1dClusters(const std::string& output_path) {
  std::ofstream plot(output_path);
  plot << std::fixed << std::setprecision(1);

  double max_n;
  cv::minMaxLoc(this->smooth_phi_hist_, nullptr, &max_n, nullptr, nullptr);

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
    plot << '(' << phi << ',' << this->smooth_phi_hist_.at<int>(phi) << ")\n";
  }
  plot << "};\n\n";

  for (const auto& cluster : phi_clusters_) {
    plot << "\\draw[cyan!75, thick] (axis cs:" << cluster << ",0) -- (axis cs:" << cluster << ',' << std::lround(max_n) << ");\n";
  }
  plot << '\n';

  plot << "\\end{axis}\n";
  plot << "\\end{tikzpicture}\n";
  plot << "\\end{document}\n";

  plot.close();
}
