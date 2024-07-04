#include <doc_color_clustering/doc_color_clustering.h>


DocColorClustering::DocColorClustering(const cv::Mat& rhs) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  std::ios::sync_with_stdio(false);

  this->src_ = this->SRgbToLinRgb(rhs);

  this->CalcColorToN();
  this->CalcPhiHist();
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


void DocColorClustering::CalcColorToN() {
  for (int y = 0; y < this->src_.rows; ++y) {
    for (int x = 0; x < this->src_.cols; ++x) {
      cv::Vec3f pixel = this->src_.at<cv::Vec3f>(y, x);
      std::tuple<float, float, float> color = std::make_tuple(std::round(pixel[2] * 10000.0F) / 10000.0F, std::round(pixel[1] * 10000.0F) / 10000.0F, std::round(pixel[0] * 10000.0F) / 10000.0F);
      this->color_to_n_[color] += 1LL;
    }
  }
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
}


void DocColorClustering::Plot3dRgb(const std::string& output_path, int yaw, int pitch) {
  std::ofstream plot_3d(output_path);
  plot_3d << std::fixed << std::setprecision(4);

  std::vector<std::pair<std::tuple<float, float, float>, long long>> sorted_color_to_n(this->color_to_n_.begin(), this->color_to_n_.end());
  std::shuffle(sorted_color_to_n.begin(), sorted_color_to_n.end(), std::mt19937(std::random_device()()));
  sorted_color_to_n.resize(std::min(sorted_color_to_n.size(), static_cast<size_t>(7000)));

  plot_3d << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot_3d << "\\usepackage{pgfplots}\n";
  plot_3d << "\\pgfplotsset{compat=newest}\n";
  plot_3d << "\\begin{document}\n";
  plot_3d << "\\begin{tikzpicture}\n\n";

  plot_3d << "\\begin{axis}[\n";
  plot_3d << "view={" << yaw << "}{" << pitch << "},\n";
  plot_3d << "height=10cm, width=10cm,\n";
  plot_3d << "axis lines=center,\n";
  plot_3d << "axis equal,\n";
  plot_3d << "scale only axis,\n";
  plot_3d << "enlargelimits=true,\n";
  plot_3d << "xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1,\n";
  plot_3d << "xtick={0}, ytick={0}, ztick={0},\n";
  plot_3d << "xlabel={$R$}, ylabel={$G$}, zlabel={$B$}]\n\n";

  plot_3d << "\\draw[lightgray] (axis cs:1,0,0) -- (axis cs:1,1,0) -- (axis cs:0,1,0);\n";
  plot_3d << "\\draw[lightgray] (axis cs:1,1,1) -- (axis cs:0,1,1) -- (axis cs:0,0,1) -- (axis cs:1,0,1) -- (axis cs:1,1,1);\n";
  plot_3d << "\\draw[lightgray] (axis cs:1,0,0) -- (axis cs:1,0,1);\n";
  plot_3d << "\\draw[lightgray] (axis cs:1,1,0) -- (axis cs:1,1,1);\n";
  plot_3d << "\\draw[lightgray] (axis cs:0,1,0) -- (axis cs:0,1,1);\n\n";

  plot_3d << "\\addplot3[\n";
  plot_3d << "only marks,\n";
  plot_3d << "mark=*,\n";
  plot_3d << "mark size=0.1,\n";
  plot_3d << "color=purple!75]\n";
  plot_3d << "table[]{\n";

  for (const auto& [color, _] : sorted_color_to_n) {
    plot_3d << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << '\n';
  }

  plot_3d << "};\n\n";
  plot_3d << "\\end{axis}\n";
  plot_3d << "\\end{tikzpicture}\n";
  plot_3d << "\\end{document}\n";

  plot_3d.close();
}


void DocColorClustering::Plot2dLab(const std::string& output_path) {
  cv::Mat plot_2d(1275, 1275, CV_32FC3, cv::Vec3f(0.0F, 0.0F, 0.0F));

  for (const auto& [color, _] : this->color_to_n_) {
    cv::Mat rgb_point = (cv::Mat_<float>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
    cv::Mat lab_point = this->CentralProjOnLab(rgb_point);
    plot_2d.at<cv::Vec3f>(std::lround(255.0F * (lab_point.at<float>(0, 1) + 3.0F)), std::lround(255.0F * (lab_point.at<float>(0, 0) + 2.5F))) = cv::Vec3f(std::get<2>(color), std::get<1>(color), std::get<0>(color));
  }

  cv::imwrite(output_path, this->LinRgbToSRgb(plot_2d));
}


void DocColorClustering::Plot1dPhi(const std::string& output_path, bool smooth) {
  std::ofstream plot_1d(output_path);
  plot_1d << std::fixed << std::setprecision(1);

  cv::Mat phi_hist = cv::Mat(this->phi_hist_);
  if (smooth) {
    cv::GaussianBlur(phi_hist, phi_hist, cv::Size(15, 1), 0.0);
  }

  double max_n;
  cv::minMaxLoc(phi_hist, nullptr, &max_n, nullptr, nullptr);

  plot_1d << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot_1d << "\\usepackage{pgfplots}\n";
  plot_1d << "\\pgfplotsset{compat=newest}\n";
  plot_1d << "\\begin{document}\n";
  plot_1d << "\\begin{tikzpicture}\n\n";

  plot_1d << "\\begin{axis}[\n";
  plot_1d << "height=10cm, width=30cm,\n";
  plot_1d << "xmin=0, xmax=360, ymin=0, ymax=" << std::lround(max_n) << ",\n";
  plot_1d << "tick align=outside,\n";
  plot_1d << "grid=both,\n";
  plot_1d << "yminorgrids=true,\n";
  plot_1d << "xlabel={$\\phi$}, ylabel={$n$}]\n\n";

  plot_1d << "\\addplot[\n";
  plot_1d << "ybar interval,\n";
  plot_1d << "mark=none,\n";
  plot_1d << "fill=purple!25,\n";
  plot_1d << "draw=purple]\n";
  plot_1d << "coordinates{\n";

  for (int phi = 0; phi < 360; ++phi) {
    plot_1d << '(' << phi << ',' << std::lround(phi_hist.at<double>(phi)) << ")\n";
  }

  plot_1d << "};\n\n";
  plot_1d << "\\end{axis}\n";
  plot_1d << "\\end{tikzpicture}\n";
  plot_1d << "\\end{document}\n";

  plot_1d.close();
}
