#include <doc_color_clustering/doc_color_clustering.h>

DocColorClustering::DocColorClustering(const cv::Mat rhs) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  this->src_ = rhs;
  this->CalcUniqueColorsAndColorToN();
}

void DocColorClustering::CalcUniqueColorsAndColorToN() {
  for (ptrdiff_t y = 0; y < this->src_.rows; ++y) {
    for (ptrdiff_t x = 0; x < this->src_.cols; ++x) {
      cv::Vec3b pixel = this->src_.at<cv::Vec3b>(y, x);
      std::tuple<int, int, int> color = std::make_tuple(pixel[2], pixel[1], pixel[0]);
      this->unique_colors.insert(color);
      this->color_to_n[color] += 1;
    }
  }
}

cv::Mat DocColorClustering::CentralProjOnLab(const cv::Mat& rgb_point) {
  const cv::Mat white = (cv::Mat_<double>(1, 3) << 1.0, 1.0, 1.0);
  const cv::Mat norm = (cv::Mat_<double>(1, 3) << 1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0));
  const cv::Mat delta = (cv::Mat_<double>(1, 3) << 2.35, 2.95, 0);
  const cv::Mat rgb_to_lab = (cv::Mat_<double>(3, 3) <<
      +1.0 / sqrt(2.0), -1.0 / sqrt(2.0), +0.0 / sqrt(2.0),
      -1.0 / sqrt(6.0), -1.0 / sqrt(6.0), +2.0 / sqrt(6.0),
      +1.0 / sqrt(3.0), +1.0 / sqrt(3.0), +1.0 / sqrt(3.0)
  );

  cv::Mat proj_in_rgb = white - (norm.dot(white) / norm.dot(rgb_point - white)) * (rgb_point - white);
  cv::Mat proj_in_lab = proj_in_rgb * rgb_to_lab.t() + delta;

  return proj_in_lab;
}

void DocColorClustering::Plot3dRgb(const std::string& output_path, int yaw, int pitch) {
  std::ofstream plot_3d(output_path);

  std::vector<std::pair<std::tuple<int, int, int>, int>> sorted_n_colors(this->color_to_n.begin(), this->color_to_n.end());
  std::sort(sorted_n_colors.begin(), sorted_n_colors.end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
  sorted_n_colors.resize(std::min(sorted_n_colors.size(), (size_t) 7000));
  double max_n = std::max_element(this->color_to_n.begin(), this->color_to_n.end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->second;

  plot_3d << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  plot_3d << "\\usepackage{pgfplots}\n";
  plot_3d << "\\pgfplotsset{compat=newest}\n";
  plot_3d << "\\begin{document}\n";
  plot_3d << "\\begin{tikzpicture}\n";

  plot_3d << "\\begin{axis}[\n";
  plot_3d << "view={" << yaw << "}{" << pitch << "},\n";
  plot_3d << "axis lines=center,\n";
  plot_3d << "axis equal,\n";
  plot_3d << "scale only axis,\n";
  plot_3d << "enlargelimits=true,\n";
  plot_3d << "xmin=0, xmax=255, ymin=0, ymax=255, zmin=0, zmax=255,\n";
  plot_3d << "xtick={0}, ytick={0}, ztick={0},\n";
  plot_3d << "xlabel={$R$}, ylabel={$G$}, zlabel={$B$}]\n";

  plot_3d << "\\draw[lightgray] (axis cs:255,0,0) -- (axis cs:255,255,0) -- (axis cs:0,255,0);\n";
  plot_3d << "\\draw[lightgray] (axis cs:255,255,255) -- (axis cs:0,255,255) -- (axis cs:0,0,255) -- (axis cs:255,0,255) -- (axis cs:255,255,255);\n";
  plot_3d << "\\draw[lightgray] (axis cs:255,0,0) -- (axis cs:255,0,255);\n";
  plot_3d << "\\draw[lightgray] (axis cs:255,255,0) -- (axis cs:255,255,255);\n";
  plot_3d << "\\draw[lightgray] (axis cs:0,255,0) -- (axis cs:0,255,255);\n";

  plot_3d << "\\addplot3[\n";
  plot_3d << "only marks,\n";
  plot_3d << "mark=*,\n";
  plot_3d << "color=purple,\n";
  plot_3d << "scatter=true,\n";
  plot_3d << "point meta=explicit symbolic,\n";
  plot_3d << "scatter/@pre marker code/.style={/tikz/mark size=\\pgfplotspointmeta},\n";
  plot_3d << "scatter/@post marker code/.style={}]\n";
  plot_3d << "table[meta index=3]{\n";

  for (const auto& [color, _] : sorted_n_colors) {
    double t = this->color_to_n[color] / max_n;
    double point_size = std::lerp(0.1, 1.0, t);
    plot_3d << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << ' ' << point_size << '\n';
  }

  plot_3d << "};\n";
  plot_3d << "\\end{axis}\n";
  plot_3d << "\\end{tikzpicture}\n";
  plot_3d << "\\end{document}\n";

  plot_3d.close();
}

void DocColorClustering::Plot2dLab(const std::string& output_path) {
  cv::Mat plot_2d(1200, 1200, CV_8UC3, cv::Vec3b(127, 127, 127));

  for (ptrdiff_t y = 0; y < this->src_.rows; ++y) {
    for (ptrdiff_t x = 0; x < this->src_.cols; ++x) {
      cv::Vec3b pixel = this->src_.at<cv::Vec3b>(y, x);
      if (pixel != cv::Vec3b(255, 255, 255)) {
        cv::Mat rgb_point = (cv::Mat_<double>(1, 3) << pixel[2] / 255.0, pixel[1] / 255.0, pixel[0] / 255.0);
        cv::Mat lab_point = this->CentralProjOnLab(rgb_point);
        plot_2d.at<cv::Vec3b>(255 * lab_point.at<double>(0, 1), 255 * lab_point.at<double>(0, 0)) = pixel;
      }
    }
  }

  cv::imwrite(output_path, plot_2d);
}
