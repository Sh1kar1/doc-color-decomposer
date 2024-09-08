add_executable(${PROJECT_NAME_SNAKE}_app main.cc)

set_target_properties(${PROJECT_NAME_SNAKE}_app PROPERTIES OUTPUT_NAME ${PROJECT_NAME_KEBAB})

target_link_libraries(${PROJECT_NAME_SNAKE}_app PRIVATE ${PROJECT_NAME_SNAKE}_library)

install(
    TARGETS ${PROJECT_NAME_SNAKE}_app
    RUNTIME DESTINATION bin
)
