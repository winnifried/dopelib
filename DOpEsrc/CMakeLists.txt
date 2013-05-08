CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

LIST(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/
  )

FILE(GLOB _macro_files "cmake/macro_*.cmake")
FOREACH(_file ${_macro_files})
  INCLUDE(${_file})
ENDFOREACH()


#
# Configure DOpe:
#

#
#     DEAL_II_PROJECT_CONFIG_NAME     *)
#     DEAL_II_BASE_NAME               *)
#     DEAL_II_DEBUG_SUFFIX            *)
#     DEAL_II_RELEASE_SUFFIX          *)
#
#     DEAL_II_BUILD_TYPES
#     DEAL_II_WITH_BUNDLED_DIRECTORY
#     DEAL_II_WITH_DOC_DIRECTORY
#
# *)  Can be overwritten by the command line via -D<...>
#


SET_IF_EMPTY(DOPE_PACKAGE_NAME "DOpE")

SET_IF_EMPTY(DOPE_PACKAGE_VERSION "0.1.pre")

STRING(REGEX REPLACE
  "^([0-9]+)\\..*" "\\1" DOPE_VERSION_MAJOR "${DOPE_PACKAGE_VERSION}"
  )
STRING(REGEX REPLACE
  "^[0-9]+\\.([0-9]+).*" "\\1" DOPE_VERSION_MINOR "${DOPE_PACKAGE_VERSION}"
  )
SET(DOPE_VERSION ${DOPE_VERSION_MAJOR}.${DOPE_VERSION_MINOR})

SET(DOPE_PROJECT_CONFIG_NAME "${DOPE_PACKAGE_NAME}")

STRING(REPLACE "." "_" _base_name "${DOPE_PACKAGE_NAME}")
SET_IF_EMPTY(DOPE_BASE_NAME "${_base_name}")
SET_IF_EMPTY(DOPE_DEBUG_SUFFIX ".g")
SET_IF_EMPTY(DOPE_RELEASE_SUFFIX "")

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  LIST(APPEND DOPE_BUILD_TYPES "DEBUG")
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES "Release")
  LIST(APPEND DOPE_BUILD_TYPES "RELEASE")
ENDIF()

#
# Cleanup some files used for storing the names of all object targets that
# will be bundled to the deal.II library.
# (Right now, i.e. cmake 2.8.8, this is the only reliable way to get
# information into a global scope...)
#
FOREACH(_build ${DOPE_BUILD_TYPES})
  STRING(TOLOWER "${_build}" _build_lowercase)
  FILE(REMOVE
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dope_objects_${_build_lowercase}
    )
ENDFOREACH()

PROJECT(DOpE CXX)


#
# Find and setup deal.II:
#

FIND_PACKAGE(deal.II 8.0 REQUIRED HINTS $ENV{DEAL_II_DIR})


SET(CMAKE_CXX_FLAGS "-std=gnu++0x -Wall -fvisibility-inlines-hidden")

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

ADD_SUBDIRECTORY(source)
