#include <doc_color_clustering/doc_color_clustering.h>

DocColorClustering::DocColorClustering(const cv::Mat rhs) {
  this->src_ = rhs;
}

void DocColorClustering::Plot3dRgb(const std::string& output_path, int yaw, int pitch) const {
  std::ofstream file(output_path);

  std::set<std::tuple<int, int, int>> unique_colors;
  std::map<std::tuple<int, int, int>, int> n_colors;

  for (ptrdiff_t y = 0; y < this->src_.rows; ++y) {
    for (ptrdiff_t x = 0; x < this->src_.cols; ++x) {
      cv::Vec3b pixel = this->src_.at<cv::Vec3b>(y, x);
      std::tuple<int, int, int> color = std::make_tuple(pixel[2], pixel[1], pixel[0]);
      unique_colors.insert(color);
      n_colors[color] += 1;
    }
  }
  double max_n = std::max_element(n_colors.begin(), n_colors.end(), [](const auto& p1, const auto& p2) { return p1.second < p2.second; })->second;

  file << "\\documentclass[tikz, border=0.1cm]{standalone}\n";
  file << "\\usepackage{pgfplots}\n";
  file << "\\pgfplotsset{compat=newest}\n";
  file << "\\begin{document}\n";
  file << "\\begin{tikzpicture}\n";

  file << "\\begin{axis}[\n";
  file << "view={" << yaw << "}{" << pitch << "},\n";
  file << "axis lines=center,\n";
  file << "axis equal,\n";
  file << "scale only axis,\n";
  file << "enlargelimits=true,\n";
  file << "xmin=0, xmax=255, ymin=0, ymax=255, zmin=0, zmax=255,\n";
  file << "xtick={0}, ytick={0}, ztick={0},\n";
  file << "xlabel={$R$}, ylabel={$G$}, zlabel={$B$}]\n";

  file << "\\draw[lightgray] (axis cs:255,0,0) -- (axis cs:255,255,0) -- (axis cs:0,255,0);\n";
  file << "\\draw[lightgray] (axis cs:255,255,255) -- (axis cs:0,255,255) -- (axis cs:0,0,255) -- (axis cs:255,0,255) -- (axis cs:255,255,255);\n";
  file << "\\draw[lightgray] (axis cs:255,0,0) -- (axis cs:255,0,255);\n";
  file << "\\draw[lightgray] (axis cs:255,255,0) -- (axis cs:255,255,255);\n";
  file << "\\draw[lightgray] (axis cs:0,255,0) -- (axis cs:0,255,255);\n";

  file << "\\addplot3[\n";
  file << "only marks,\n";
  file << "mark=*,\n";
  file << "color=purple,\n";
  file << "scatter=true,\n";
  file << "point meta=explicit symbolic,\n";
  file << "scatter/@pre marker code/.style={/tikz/mark size=\\pgfplotspointmeta},\n";
  file << "scatter/@post marker code/.style={}]\n";
  file << "table[meta index=3]{\n";

  for (const auto& color : unique_colors) {
    double point_size = std::lerp(0.01, 1.0, sqrt(n_colors[color] / max_n));
    file << std::get<2>(color) << ' ' << std::get<1>(color) << ' ' << std::get<0>(color) << ' ' << point_size << '\n';
  }

  file << "};\n";
  file << "\\end{axis}\n";
  file << "\\end{tikzpicture}\n";
  file << "\\end{document}\n";

  file.close();
}
