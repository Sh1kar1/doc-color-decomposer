#ifndef DOC_COLOR_CLUSTERING_H
#define DOC_COLOR_CLUSTERING_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <string>
#include <set>
#include <map>

class DocColorClustering {
public:
  DocColorClustering(const cv::Mat rhs);

  void Plot3dRgb(const std::string& output_path = ".\\plot-3d-rgb.tex", int yaw = 115, int pitch = 15) const;

private:
  cv::Mat src_;
};

#endif // DOC_COLOR_CLUSTERING_H

