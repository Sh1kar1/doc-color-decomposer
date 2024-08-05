cmake_minimum_required(VERSION 3.12)

project(doc_color_decomposer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_package(OpenCV REQUIRED)

install(DIRECTORY include DESTINATION .)
install(DIRECTORY data DESTINATION .)

add_subdirectory(src)
add_subdirectory(app)
