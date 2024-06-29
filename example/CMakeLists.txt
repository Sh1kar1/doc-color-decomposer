add_executable(example main.cpp)
set_property(TARGET example PROPERTY CXX_STANDARD 20)

target_link_libraries(example ${PROJECT_NAME})
