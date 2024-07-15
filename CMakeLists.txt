cmake_minimum_required(VERSION 3.12)

project(doc_color_decomposer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_package(OpenCV REQUIRED)

add_library(${CMAKE_PROJECT_NAME} STATIC src/doc_color_decomposer.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include DESTINATION .)

add_subdirectory(app)
