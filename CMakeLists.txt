cmake_minimum_required(VERSION 3.20)

project(doc-color-decomposer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_package(OpenCV REQUIRED)

install(FILES README.md LICENSE DESTINATION .)
install(DIRECTORY include data assets DESTINATION .)

add_subdirectory(src)
add_subdirectory(app)
