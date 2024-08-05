project(${CMAKE_PROJECT_NAME}_app LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_PROJECT_NAME}_library)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
