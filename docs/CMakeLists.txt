find_package(Doxygen)

if(DOXYGEN_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      doxygen_awesome_css
      GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
      GIT_TAG main
  )
  FetchContent_MakeAvailable(doxygen_awesome_css)

  configure_file(Doxyfile.in Doxyfile)

  add_custom_target(
      ${PROJECT_NAME_SNAKE}_documentation ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      VERBATIM
  )

  add_dependencies(${PROJECT_NAME_SNAKE}_documentation ${PROJECT_NAME_SNAKE}_library)

  install(
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
      DESTINATION share/${PROJECT_NAME_SNAKE}
  )

else()
  message(WARNING "Doxygen not found")
endif()
