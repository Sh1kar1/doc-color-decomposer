project(${CMAKE_PROJECT_NAME}_library LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC doc_color_decomposer.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
