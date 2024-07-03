#include <doc_color_clustering/doc_color_clustering.h>

DocColorClustering::DocColorClustering(const cv::Mat& rhs) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  this->src_ = this->SRgbToLinRgb(rhs);
  this->CalcUniqueColorsAndColorToN();
}

cv::Mat DocColorClustering::SRgbToLinRgb(cv::Mat src) {
  src.convertTo(src, CV_32FC3, 1.0 / 255.0);

  for (ptrdiff_t y = 0; y < src.rows; ++y) {
    for (ptrdiff_t x = 0; x < src.cols; ++x) {
      cv::Vec3f& pixel = src.at<cv::Vec3f>(y, x);
      for (ptrdiff_t k = 0; k < 3; ++k) {
        if (pixel[k] <= 0.04045) {
          pixel[k] /= 12.92;
        } else {
          pixel[k] = std::pow((pixel[k] + 0.055) / 1.055, 2.4);
        }
      }
    }
  }

  return src;
}

cv::Mat DocColorClustering::LinRgbToSRgb(cv::Mat src) {
  for (ptrdiff_t y = 0; y < src.rows; ++y) {
    for (ptrdiff_t x = 0; x < src.cols; ++x) {
      cv::Vec3f& pixel = src.at<cv::Vec3f>(y, x);
      for (int k = 0; k < 3; ++k) {
        if (pixel[k] <= 0.0031308) {
          pixel[k] *= 12.92;
        } else {
          pixel[k] = 1.055 * std::pow(pixel[k], 1.0 / 2.4) - 0.055;
        }
      }
    }
  }

  src.convertTo(src, CV_8UC3, 255.0);

  return src;
}

cv::Mat DocColorClustering::CentralProjOnLab(const cv::Mat& rgb_point) {
  const cv::Mat white = (cv::Mat_<double>(1, 3) << 1.0, 1.0, 1.0);
  const cv::Mat norm = (cv::Mat_<double>(1, 3) << 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0));
  const cv::Mat rgb_to_lab = (cv::Mat_<double>(3, 3) <<
      +1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0), +0.0,
      -1.0 / std::sqrt(6.0), -1.0 / std::sqrt(6.0), +2.0 / std::sqrt(6.0),
      +1.0 / std::sqrt(3.0), +1.0 / std::sqrt(3.0), +1.0 / std::sqrt(3.0)
  );

  cv::Mat proj_in_rgb = white - (norm.dot(white) / norm.dot(rgb_point - white)) * (rgb_point - white);
  cv::Mat proj_in_lab = proj_in_rgb * rgb_to_lab.t();

  return proj_in_lab;
}

void DocColorClustering::CalcUniqueColorsAndColorToN() {
  for (ptrdiff_t y = 0; y < this->src_.rows; ++y) {
    for (ptrdiff_t x = 0; x < this->src_.cols; ++x) {
      cv::Vec3f pixel = this->src_.at<cv::Vec3f>(y, x);
      std::tuple<double, double, double> color = std::make_tuple(pixel[2], pixel[1], pixel[0]);
      this->unique_colors.insert(color);
      this->color_to_n[color] += 1;
    }
  }
}

void DocColorClustering::Plot3dRgb(const std::string& output_path, int yaw, int pitch) {
  std::ofstream plot_3d(output_path);

  std::vector<std::pair<std::tuple<double, double, double>, int>> sorted_n_colors(this->color_to_n.begin(), this->color_to_n.end());
  std::sort(sorted_n_colors.begin(), sorted_n_colors.end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
  sorted_n_colors.resize(std::min(sorted_n_colors.size(), (size_t) 5000));

  plot_3d << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot_3d << "\\usepackage{pgfplots}\n";
  plot_3d << "\\pgfplotsset{compat=newest}\n";
  plot_3d << "\\begin{document}\n";
  plot_3d << "\\begin{tikzpicture}\n\n";

  plot_3d << "\\begin{axis}[\n";
  plot_3d << "view={" << yaw << "}{" << pitch << "},\n";
  plot_3d << "axis lines=center,\n";
  plot_3d << "axis equal,\n";
  plot_3d << "scale only axis,\n";
  plot_3d << "enlargelimits=true,\n";
  plot_3d << "xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1,\n";
  plot_3d << "xtick={0}, ytick={0}, ztick={0},\n";
  plot_3d << "xlabel={$R$}, ylabel={$G$}, zlabel={$B$}]\n\n";

  plot_3d << "\\draw[] (axis cs:1,0,0) -- (axis cs:1,1,0) -- (axis cs:0,1,0);\n";
  plot_3d << "\\draw[] (axis cs:1,1,1) -- (axis cs:0,1,1) -- (axis cs:0,0,1) -- (axis cs:1,0,1) -- (axis cs:1,1,1);\n";
  plot_3d << "\\draw[] (axis cs:1,0,0) -- (axis cs:1,0,1);\n";
  plot_3d << "\\draw[] (axis cs:1,1,0) -- (axis cs:1,1,1);\n";
  plot_3d << "\\draw[] (axis cs:0,1,0) -- (axis cs:0,1,1);\n\n";

  plot_3d << "\\addplot3[\n";
  plot_3d << "only marks,\n";
  plot_3d << "mark=*,\n";
  plot_3d << "mark size=0.05,\n";
  plot_3d << "color=purple]\n";
  plot_3d << "table[]{\n";

  for (const auto& [color, _] : sorted_n_colors) {
    plot_3d << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << '\n';
  }

  plot_3d << "};\n\n";
  plot_3d << "\\end{axis}\n";
  plot_3d << "\\end{tikzpicture}\n";
  plot_3d << "\\end{document}\n";

  plot_3d.close();
}

void DocColorClustering::Plot2dLab(const std::string& output_path) {
  cv::Mat plot_2d(255 * 5, 255 * 5, CV_32FC3, cv::Vec3f(0.25, 0.25, 0.25));

  for (const auto& color : this->unique_colors) {
    if (color != std::make_tuple(1.0, 1.0, 1.0)) {
      cv::Mat rgb_point = (cv::Mat_<double>(1, 3) << std::get<0>(color), std::get<1>(color), std::get<2>(color));
      cv::Mat lab_point = this->CentralProjOnLab(rgb_point);
      plot_2d.at<cv::Vec3f>(255.0 * (lab_point.at<double>(0, 1) + 3.0), 255.0 * (lab_point.at<double>(0, 0) + 2.5)) = cv::Vec3f(std::get<2>(color), std::get<1>(color), std::get<0>(color));
    }
  }

  cv::imwrite(output_path, this->LinRgbToSRgb(plot_2d));
}
