find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME_SNAKE}_library STATIC doc_color_decomposer.cc utils.cc data.cc)

set_target_properties(${PROJECT_NAME_SNAKE}_library PROPERTIES OUTPUT_NAME ${PROJECT_NAME_KEBAB})

target_include_directories(
    ${PROJECT_NAME_SNAKE}_library PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME_SNAKE}_library PUBLIC ${OpenCV_LIBS})

install(
    TARGETS ${PROJECT_NAME_SNAKE}_library
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(
    DIRECTORY ../include
    DESTINATION .
)
