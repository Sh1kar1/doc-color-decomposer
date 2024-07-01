add_library(${PROJECT_NAME} STATIC src/doc_color_clustering.cpp include/doc_color_clustering/doc_color_clustering.h)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
