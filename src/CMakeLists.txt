project(${CMAKE_PROJECT_NAME}-library)

add_library(${PROJECT_NAME} STATIC doc_color_decomposer.cc utils.cc data.cc)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

install(
    TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(
    DIRECTORY ../include
    DESTINATION .
)
