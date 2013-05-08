#####
##
## Copyright (C) 2012, 2013 by Matthias Maier
##
## This file is dual licensed under QPL 1.0 and LGPL 2.1 or any later
## version of the LGPL license.
##
#####

#
# A small wrapper around ADD_LIBRARY that will define a target for each
# build type specified in DOPE_BUILD_TYPES
#
# It is assumed that the desired compilation configuration is set via
#   DOPE_SHARED_LINKER_FLAGS_${build}
#   DOPE_CXX_FLAGS_${build}
#   DOPE_DEFINITIONS_${build}
#
# as well as the global (for all build types)
#   CMAKE_SHARED_LINKER_FLAGS
#   CMAKE_CXX_FLAGS
#   DOPE_DEFINITIONS
#

MACRO(DOPE_ADD_LIBRARY _library)

  FOREACH(_build ${DOPE_BUILD_TYPES})
    STRING(TOLOWER ${_build} _build_lowercase)

    ADD_LIBRARY(${_library}.${_build_lowercase}
      ${ARGN}
      )

    SET_TARGET_PROPERTIES(${_library}.${_build_lowercase} PROPERTIES
      LINK_FLAGS "${DOPE_SHARED_LINKER_FLAGS_${_build}}"
      COMPILE_DEFINITIONS "${DOPE_DEFINITIONS};${DOPE_DEFINITIONS_${_build}}"
      COMPILE_FLAGS "${DOPE_CXX_FLAGS_${_build}}"
      LINKER_LANGUAGE "CXX"
      )

    FILE(APPEND
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dope_objects_${_build_lowercase}
      "$<TARGET_OBJECTS:${_library}.${_build_lowercase}>\n"
      )
  ENDFOREACH()

ENDMACRO()
