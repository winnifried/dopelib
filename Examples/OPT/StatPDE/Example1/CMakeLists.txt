# Set the name of the project and target:
SET(TARGET "DOpE-OPT-StatPDE-Example1")

# Declare all source files the target consists of:
SET(TARGET_SRC
  main.cc
  # You can specify additional files here!
  )

#Define the path to DOpE
SET(DOpE ../../../../)

#Set dimensions
SET(dope_dimension 2)
SET(deal_dimension 2)

# Usually, you will not need to modify anything beyond this point...
GET_FILENAME_COMPONENT(DOpE "${CMAKE_SOURCE_DIR}/${DOpE}" ABSOLUTE)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

INCLUDE_DIRECTORIES(
  ${DOpE}/DOpEsrc/basic
  ${DOpE}/DOpEsrc/container
  ${DOpE}/DOpEsrc/include
  ${DOpE}/DOpEsrc/interfaces
  ${DOpE}/DOpEsrc/opt_algorithms
  ${DOpE}/DOpEsrc/problemdata
  ${DOpE}/DOpEsrc/reducedproblems
  ${DOpE}/DOpEsrc/templates
  ${DOpE}/DOpEsrc/tsschemes
  ${DOpE}/DOpEsrc/wrapper
  )

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(run COMMAND ${TARGET} 
  COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
  )

LINK_DIRECTORIES(${DOpE}/DOpEsrc/lib/${dope_dimension}d/${deal_dimension}d/)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${DEAL_II_DIR} $ENV{HOME}/deal.II $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})

DEAL_II_SETUP_TARGET(${TARGET})
   
STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
IF("${_cmake_build_type}" STREQUAL "debug")
  SET(_suffix ".g")
ENDIF()

TARGET_LINK_LIBRARIES(${TARGET} dope${_suffix})

SET_PROPERTY(TARGET ${TARGET} APPEND PROPERTY
      COMPILE_DEFINITIONS "deal_II_dimension=${deal_dimension};dope_dimension=${dope_dimension};DEAL_II_MAJOR_VERSION=${DEAL_II_VERSION_MAJOR};DEAL_II_MINOR_VERSION=${DEAL_II_VERSION_MINOR}"
      )
SET_PROPERTY(TARGET ${TARGET} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})