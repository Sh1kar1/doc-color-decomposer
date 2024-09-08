cmake_minimum_required(VERSION 3.20)

project(
    DocColorDecomposer
    VERSION 1.0.0
    DESCRIPTION "Library for documents decomposition by color clustering created using C++ & OpenCV"
    HOMEPAGE_URL https://github.com/Sh1kar1/doc-color-decomposer
    LANGUAGES CXX
)

set(PROJECT_NAME_SPACE "Doc Color Decomposer")
set(PROJECT_NAME_SCREAM DOC_COLOR_DECOMPOSER)
set(PROJECT_NAME_SNAKE doc_color_decomposer)
set(PROJECT_NAME_KEBAB doc-color-decomposer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(${PROJECT_NAME_SCREAM}_BUILD_LIBRARY "Build `${PROJECT_NAME_SPACE}` library" ON)
option(${PROJECT_NAME_SCREAM}_BUILD_APP "Build `${PROJECT_NAME_SPACE}` app" ON)
option(${PROJECT_NAME_SCREAM}_BUILD_DOCUMENTATION "Build `${PROJECT_NAME_SPACE}` documentation" ON)
option(${PROJECT_NAME_SCREAM}_BUILD_PACKAGE "Build `${PROJECT_NAME_SPACE}` package" ON)

if(${PROJECT_NAME_SCREAM}_BUILD_LIBRARY)
  add_subdirectory(src)
endif()
if(${PROJECT_NAME_SCREAM}_BUILD_APP)
  add_subdirectory(app)
endif()
if(${PROJECT_NAME_SCREAM}_BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()
if(${PROJECT_NAME_SCREAM}_BUILD_PACKAGE)
  add_subdirectory(cmake)
endif()

install(
    FILES LICENSE README.md
    DESTINATION share/${PROJECT_NAME_SNAKE}
)
install(
    DIRECTORY assets data
    DESTINATION share/${PROJECT_NAME_SNAKE}
)
