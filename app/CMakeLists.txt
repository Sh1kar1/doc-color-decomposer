project(${CMAKE_PROJECT_NAME}-app)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_PROJECT_NAME}-library)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
