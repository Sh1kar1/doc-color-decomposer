cmake_minimum_required(VERSION 3.20)

project(doc-color-decomposer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_package(OpenCV REQUIRED)

install(
    FILES README.md LICENSE
    DESTINATION .
)
install(
    DIRECTORY assets data include
    DESTINATION .
)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(docs)
