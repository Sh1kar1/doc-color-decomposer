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
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

</details>

---

## Overview

### Technologies

#### Languages

- C++ & CMake
- LaTeX

#### Dependencies

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
- Demo CLI app
- Optional LaTeX visualizations

### Demonstration



## Installation



## Usage



## License

Distributed under the Unlicense license - see [LICENSE](./LICENSE) for more information

_This project was completed as part of studies at [NUST MISIS](https://en.misis.ru/) in the Applied Mathematics program_
