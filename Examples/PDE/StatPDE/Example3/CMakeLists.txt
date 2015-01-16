CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# Set the name of the project and target:
SET(TARGET "DOpE-PDE-StatPDE-Example3")

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

#Load default example rules
INCLUDE(${DOpE}/Examples/CMakeExamples.txt)