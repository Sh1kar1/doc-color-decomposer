# Doc Color Decomposer

**Library for documents decomposition by color clustering created using C++ & OpenCV**

<details>

<summary>Table of Contents</summary>

- [Overview](#overview)
    - [Technologies](#technologies)
    - [Description](#description)
    - [Algorithm](#algorithm)
    - [Features](#features)
    - [Demonstration](#demonstration)
- [Usage](#usage)
    - [App](#app)
    - [Interface](#interface)
    - [Example](#example)
- [License](#license)

</details>

---

## Overview

### Technologies

- C++ & CMake
- LaTeX
- [OpenCV](https://opencv.org/)

### Description

Implementation of the following research article:
[documents-decomposition-by-color-clustering.pdf](./assets/documents-decomposition-by-color-clustering.pdf)

### Algorithm

1. Preprocessing by aberration reduction
2. Conversion to the linRGB color model
3. Colors projection onto the $\alpha\beta$ plane
4. Histogram calculation with respect to the rotation angle $\phi$
5. Clustering by peak detection in the histogram
6. Decomposition into layers via clusters

### Features

- API
- CLI app
- LaTeX visualizations
- Test data

### Demonstration



## Usage

### App

```
./doc_color_decomposer_app <path-to-image> <path-to-output-directory> [options]
```

|       Option       | Usage               |
|:------------------:|---------------------|
| `-v` `--visualize` | save visualizations |
|   `-h` `--help`    | print help message  |

### Interface

- ```c++
  explicit DocColorDecomposer::DocColorDecomposer() = default
  ```

  Constructs an empty instance

<br>

- ```c++
  explicit DocColorDecomposer::DocColorDecomposer(const cv::Mat& src)
  ```

  Constructs an instance from the given document and pre-computes its layers

  | Parameter | Description                                     |
  |:---------:|-------------------------------------------------|
  |   `src`   | source image of the document in the sRGB format |

<br>

- ```c++
  [[nodiscard]] std::vector<cv::Mat> DocColorDecomposer::GetLayers() const
  ```

  Retrieves the pre-computed layers

  Returns the list of the decomposed document layers in the sRGB format with a white background

<br>

- ```c++
  [[nodiscard]] cv::Mat DocColorDecomposer::MergeLayers()
  ```

  Merges the pre-computed layers for testing

  Returns the image of the merged layers in the sRGB format that must be the same as the source document

<br>

- ```c++
  [[nodiscard]] std::string DocColorDecomposer::Plot3dRgb(int yaw = 115, int pitch = 15)
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

  Generates a 1D histogram plot with respect to the rotation angle $\phi$

  Returns the LaTeX code of the plot that can be saved in the .tex format and compiled

<br>

- ```c++
  [[nodiscard]] std::string DocColorDecomposer::Plot1dClusters()
  ```

  Generates a smoothed and separated by clusters 1D histogram plot with respect to the rotation angle $\phi$

  Returns the LaTeX code of the plot that can be saved in the .tex format and compiled

### Example

```c++
#include "doc_color_decomposer/doc_color_decomposer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <ranges>

int main() {
  cv::Mat src = cv::imread("/path/to/image/", cv::IMREAD_COLOR);
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
