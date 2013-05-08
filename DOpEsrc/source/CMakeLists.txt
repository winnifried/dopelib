INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/basic
  ${CMAKE_SOURCE_DIR}/container
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/interfaces
  ${CMAKE_SOURCE_DIR}/opt_algorithms
  ${CMAKE_SOURCE_DIR}/problemdata
  ${CMAKE_SOURCE_DIR}/reducedproblems
  ${CMAKE_SOURCE_DIR}/templates
  ${CMAKE_SOURCE_DIR}/tsschemes
  ${CMAKE_SOURCE_DIR}/wrapper

  ${DOPE_INCLUDE_DIRS}
  )

FILE(GLOB _src *.cc)

FOREACH(_deal_ii_dimension 1 2 3) #TODO
  FOREACH(_dope_dimension 0 ${_deal_ii_dimension})

    DOPE_ADD_LIBRARY(dope_${_dope_dimension}${_deal_ii_dimension} OBJECT ${_src})

    DOPE_SET_PROPERTIES(TARGET dope_${_dope_dimension}${_deal_ii_dimension} APPEND PROPERTY
      COMPILE_DEFINITIONS "deal_II_dimension=${_deal_ii_dimension};dope_dimension=${_dope_dimension};DEAL_II_MAJOR_VERSION=${DEAL_II_VERSION_MAJOR};DEAL_II_MINOR_VERSION=${DEAL_II_VERSION_MINOR}"
      )

    LIST(APPEND _dope_libs
      "$<TARGET_OBJECTS:dope_${_dope_dimension}${_deal_ii_dimension}>"
      )

  ENDFOREACH()
ENDFOREACH()


FOREACH(build ${DOPE_BUILD_TYPES})
  STRING(TOLOWER ${build} build_lowercase)

  #
  # Combine all ${build} OBJECT targets to a ${build} library:
  #
  FILE(STRINGS
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dope_objects_${build_lowercase}
    dope_objects_${build_lowercase}
    )
  ADD_LIBRARY(dope${DOPE_${build_lowercase}_SUFFIX}
    ${dope_objects_${build_lowercase}}
    )

  SET_TARGET_PROPERTIES(dope.${build_lowercase}
    PROPERTIES
    VERSION "${DOPE_PACKAGE_VERSION}"
    SOVERSION "${DOPE_PACKAGE_VERSION}"
    LINK_FLAGS "${DOPE_SHARED_LINKER_FLAGS_${build}}"
    COMPILE_DEFINITIONS "${DOPE_DEFINITIONS};${DOPE_DEFINITIONS_${build}}"
    COMPILE_FLAGS "${DOPE_CXX_FLAGS_${build}}"
    )

  TARGET_LINK_LIBRARIES(dope.${build_lowercase}
    ${DOPE_EXTERNAL_LIBRARIES}
    ${DOPE_EXTERNAL_LIBRARIES_${build}}
    )
ENDFOREACH()

