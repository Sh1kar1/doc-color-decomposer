# [Doc Color Decomposer](https://github.com/Sh1kar1/doc-color-decomposer)

**Library for documents decomposition by color clustering created using C++ & OpenCV**

## Overview

### Demonstration

| ![](data/1.png) | ![](assets/1-layer-1.png) | ![](assets/1-layer-2.png) | ![](assets/1-layer-3.png) | ![](assets/1-layer-4.png) |
|:---------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|

| ![](assets/1-plot-3d-rgb.png) | ![](assets/1-plot-2d-lab.png) | ![](assets/1-plot-1d-phi.png) | ![](assets/1-plot-1d-clusters.png) |
|:-----------------------------:|:-----------------------------:|:-----------------------------:|:----------------------------------:|

### Features

- Customizable document color decomposition
- LaTeX visualizations generation
- Quality computation

### Details

Implementation of the following research article:
<br>
[documents-decomposition-by-color-clustering.pdf](assets/documents-decomposition-by-color-clustering.pdf)

1. Document colors projection onto the $\alpha\beta$ plane
2. Histogram calculation with respect to the angle $\phi$ in polar coordinates
3. Histogram smoothing using _Gaussian filter_
4. Clustering by peak detection in the smoothed histogram
5. Decomposition into layers via clusters

### Technologies

- C++ & CMake
- LaTeX
- [OpenCV](https://opencv.org/)
- [Doxygen](https://www.doxygen.nl/)

## Installation

> [!NOTE]
> You can download pre-built binaries from the [releases](https://github.com/Sh1kar1/doc-color-decomposer/releases) instead of building from source

1. Ensure all requirements are installed:
   ```shell
   git --version
   cmake --version
   doxygen --version
   <opencv>
   <toolchain>
   ```
2. Clone the repository:
   ```shell
   git clone https://github.com/Sh1kar1/doc-color-decomposer.git
   cd doc-color-decomposer
   ```
3. Build the library:
   ```shell
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build .
   ```
4. Install the library:
   ```shell
   cmake --install . [--prefix <installation-path>]
   ```
5. Link to your project:
   ```cmake
   find_package(DocColorDecomposer REQUIRED)
   target_link_libraries(<your-target> PRIVATE DocColorDecomposer::DocColorDecomposer)
   ```

## Usage

### Interface

Generated API documentation is available at `<installation-path>/share/doc_color_decomposer/docs`

API usage example can be found in [main.cc](app/main.cc) (source of the demo CLI app)

### App

The demo CLI app is available at `<installation-path>/bin/doc-color-decomposer`

Run the following command to see how to use it:
```shell
./doc-color-decomposer --help
```

## License

Distributed under the Unlicense license - see [LICENSE](LICENSE) for more information
