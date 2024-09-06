cmake_minimum_required(VERSION 3.20)

project(doc-color-decomposer)

set(CMAKE_PROJECT_NAME_SNAKE doc_color_decomposer)
set(CMAKE_PROJECT_NAME_CAMEL DocColorDecomposer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_package(OpenCV REQUIRED)

install(
    FILES LICENSE README.md
    DESTINATION share/${CMAKE_PROJECT_NAME_SNAKE}
)
install(
    DIRECTORY assets data
    DESTINATION share/${CMAKE_PROJECT_NAME_SNAKE}
)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(docs)
