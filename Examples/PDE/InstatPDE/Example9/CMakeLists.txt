CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
# Set the name of the project and target:
SET(TARGET "DOpE-PDE-InstatPDE-Example9")

# Declare all source files the target consists of:
SET(TARGET_SRC
  main.cc
  # You can specify additional files here!
  )

#Set dimensions
SET(dope_dimension 2)
SET(deal_dimension 2)

#Find the DOpE library
#The ../../../../ is included first to make shure we always use 
# the dope shipped with the examples - unless we specifically move the 
# directory
FIND_PACKAGE(DOpElib QUIET
  HINTS ${CMAKE_SOURCE_DIR}/../../../../ ${DOPE_DIR} $ENV{DOPE_DIR} $ENV{HOME}/DOpE
  )
IF(NOT ${DOpElib_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate DOpElib. ***\n\n"
    "You may want to either pass a flag -DDOPE_DIR=/path/to/DOpE to cmake\n"
    "or set an environment variable \"DOPE_DIR\" that contains this path.")
ELSE()
  MESSAGE(STATUS "Found DOpElib at ${DOpE}.")
ENDIF()

Project(${TARGET} CXX)

#Load default example rules
INCLUDE(${DOpE}/Examples/CMakeExamples.txt)
