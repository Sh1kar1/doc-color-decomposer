project(${CMAKE_PROJECT_NAME}-documentation)

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
      ${PROJECT_NAME} ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      VERBATIM
  )
  add_dependencies(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}-library)

  install(
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
      DESTINATION .
  )
endif()
