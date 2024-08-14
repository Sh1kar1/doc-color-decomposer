# Doc Color Decomposer

**Library for documents decomposition by color clustering created using C++ & OpenCV**

## Overview

### Technologies

- C++ & CMake
- LaTeX
- [OpenCV](https://opencv.org/)

### Features

- API
- CLI app
- Optional visualizations
- Test data

### Description

Implementation of the following research article:
[documents-decomposition-by-color-clustering.pdf](./assets/documents-decomposition-by-color-clustering.pdf)

1. Colors projection onto the $\alpha\beta$ plane
2. Histogram calculation with respect to the angle $\phi$ in polar coordinates
3. Clustering by peak detection in the histogram
4. Decomposition into layers via clusters

### Demonstration

#### 1

| ![](./data/input/doc-2/up.png) | ![](./data/output/doc-2/up/layers/1.png) | ![](./data/output/doc-2/up/layers/2.png) | ![](./data/output/doc-2/up/layers/3.png) | ![](./data/output/doc-2/up/layers/4.png) |
|--------------------------------|:----------------------------------------:|:----------------------------------------:|:----------------------------------------:|:----------------------------------------:|

| ![](./data/output/doc-2/up/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-2/up/visualizations/plot-2d-lab.png) | ![](./data/output/doc-2/up/visualizations/plot-1d-phi.png) | ![](./data/output/doc-2/up/visualizations/plot-1d-clusters.png) |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:---------------------------------------------------------------:|

#### 2

| ![](./data/input/doc-2/front.png) | ![](./data/output/doc-2/front/layers/1.png) | ![](./data/output/doc-2/front/layers/2.png) | ![](./data/output/doc-2/front/layers/3.png) | ![](./data/output/doc-2/front/layers/4.png) |
|-----------------------------------|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|

| ![](./data/output/doc-2/front/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-2/front/visualizations/plot-2d-lab.png) | ![](./data/output/doc-2/front/visualizations/plot-1d-phi.png) | ![](./data/output/doc-2/front/visualizations/plot-1d-clusters.png) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|

#### 3

| ![](./data/input/doc-2/right.png) | ![](./data/output/doc-2/right/layers/1.png) | ![](./data/output/doc-2/right/layers/2.png) | ![](./data/output/doc-2/right/layers/3.png) |
|-----------------------------------|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|

| ![](./data/output/doc-2/right/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-2/right/visualizations/plot-2d-lab.png) | ![](./data/output/doc-2/right/visualizations/plot-1d-phi.png) | ![](./data/output/doc-2/right/visualizations/plot-1d-clusters.png) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|

#### 4

| ![](./data/input/doc-1/up.png) | ![](./data/output/doc-1/up/layers/1.png) | ![](./data/output/doc-1/up/layers/2.png) | ![](./data/output/doc-1/up/layers/3.png) |
|--------------------------------|:----------------------------------------:|:----------------------------------------:|:----------------------------------------:|

| ![](./data/output/doc-1/up/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-1/up/visualizations/plot-2d-lab.png) | ![](./data/output/doc-1/up/visualizations/plot-1d-phi.png) | ![](./data/output/doc-1/up/visualizations/plot-1d-clusters.png) |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:---------------------------------------------------------------:|

#### 5

| ![](./data/input/doc-1/front.png) | ![](./data/output/doc-1/front/layers/1.png) | ![](./data/output/doc-1/front/layers/2.png) | ![](./data/output/doc-1/front/layers/3.png) |
|-----------------------------------|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|

| ![](./data/output/doc-1/front/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-1/front/visualizations/plot-2d-lab.png) | ![](./data/output/doc-1/front/visualizations/plot-1d-phi.png) | ![](./data/output/doc-1/front/visualizations/plot-1d-clusters.png) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|

#### 6

| ![](./data/input/doc-1/right.png) | ![](./data/output/doc-1/right/layers/1.png) | ![](./data/output/doc-1/right/layers/2.png) | ![](./data/output/doc-1/right/layers/3.png) |
|-----------------------------------|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|

| ![](./data/output/doc-1/right/visualizations/plot-3d-rgb.png) | ![](./data/output/doc-1/right/visualizations/plot-2d-lab.png) | ![](./data/output/doc-1/right/visualizations/plot-1d-phi.png) | ![](./data/output/doc-1/right/visualizations/plot-1d-clusters.png) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|

## Usage

### App

```
./doc-color-decomposer-app <path-to-image> <path-to-output-directory> [options]
```

|               Option               | Usage                                        |
|:----------------------------------:|----------------------------------------------|
| `--tolerance=<odd-positive-value>` | set tolerance of decomposition (default: 35) |
|          `--nopreprocess`          | disable image preprocessing                  |
|           `--visualize`            | save visualizations                          |

### Interface

- ```c++
  explicit DocColorDecomposer() = default
  ```

  Constructs an empty instance

<br>

- ```c++
  explicit DocColorDecomposer(const cv::Mat& src, int tolerance = 35, bool preprocessing = true)
  ```

  Constructs an instance from the given document and precomputes its layers

  |    Parameter    | Description                                                                 |
  |:---------------:|-----------------------------------------------------------------------------|
  |      `src`      | source image of the document in the sRGB format                             |
  |   `tolerance`   | odd positive value with an increase of which the number of layers decreases |
  | `preprocessing` | true if the source image needs to be processed by aberration reduction      |

<br>

- ```c++
  [[nodiscard]] std::vector<cv::Mat> DocColorDecomposer::GetLayers() const
  ```

  Retrieves the precomputed layers

  Returns the list of the decomposed document layers in the sRGB format with a white background

<br>

- ```c++
  [[nodiscard]] cv::Mat DocColorDecomposer::MergeLayers()
  ```

  Merges the precomputed layers for testing

  Returns the image of the merged layers in the sRGB format that must be the same as the source document

<br>

- ```c++
  [[nodiscard]] std::string DocColorDecomposer::Plot3dRgb(double yaw = 135.0, double pitch = 35.25)
  ```

  Generates a 3D scatter plot of the document colors in the linRGB space

  Returns the LaTeX code of the plot that can be saved in the .tex format and compiled

  | Parameter | Description                                 |
  |:---------:|---------------------------------------------|
  |   `yaw`   | yaw-rotation angle of the view in degrees   |
  |  `pitch`  | pitch-rotation angle of the view in degrees |

<br>

- ```c++
  [[nodiscard]] cv::Mat DocColorDecomposer::Plot2dLab()
  ```

  Generates a 2D scatter plot of the document colors projections on the $\alpha\beta$ plane

  Returns the image of the plot in the sRGB format

<br>

- ```c++
  [[nodiscard]] std::string DocColorDecomposer::Plot1dPhi()
  ```

  Generates a 1D histogram plot with respect to the angle $\phi$ in polar coordinates

  Returns the LaTeX code of the plot that can be saved in the .tex format and compiled

<br>

- ```c++
  [[nodiscard]] std::string DocColorDecomposer::Plot1dClusters()
  ```

  Generates a smoothed and separated by clusters 1D histogram plot

  Returns the LaTeX code of the plot that can be saved in the .tex format and compiled

### Example

```c++
#include "doc_color_decomposer/doc_color_decomposer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <fstream>
#include <ranges>

int main() {
  cv::Mat src = cv::imread("/PATH/TO/IMAGE/", cv::IMREAD_COLOR);
  DocColorDecomposer dcd = DocColorDecomposer(src);

  for (const auto& [i, layer] : dcd.GetLayers() | std::views::enumerate) {
    cv::imwrite("layer-" + std::to_string(i + 1) + ".png", layer);
  }

  cv::imwrite("merged-layers.png", dcd.MergeLayers());
  cv::imwrite("plot-2d-lab.png", dcd.Plot2dLab());

  std::ofstream("plot-3d-rgb.tex") << dcd.Plot3dRgb();
  std::ofstream("plot-1d-phi.tex") << dcd.Plot1dPhi();
  std::ofstream("plot-1d-clusters.tex") << dcd.Plot1dClusters();

  return EXIT_SUCCESS;
}
```

## License

Distributed under the Unlicense license - see [LICENSE](./LICENSE) for more information

_This project was completed as part of studies at [NUST MISIS](https://en.misis.ru/) in the Applied Mathematics program_
