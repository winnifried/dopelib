CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{HOME}/deal.II 
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ELSE()
  MESSAGE(STATUS "Found deal.II at ${DEAL_II_PATH}.")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(STATUS "The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_TRILINOS = ON. You may not use trilinos related solvers within your code."
    )
ELSE()
	MESSAGE(STATUS "Configured for use of Trilinos")
	ADD_DEFINITIONS(-DDOPELIB_WITH_TRILINOS)	
ENDIF()

IF(NOT DEAL_II_WITH_MPI)
  MESSAGE(STATUS "The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON. You may not use MPI functionality in your code"
    )
ELSE()
	MESSAGE(STATUS "Configured for use of MPI")
	ADD_DEFINITIONS(-DDOPELIB_WITH_MPI)	
ENDIF()

IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(STATUS "The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_P4EST = ON. You may not use trilinos related solvers within your code."
    )
ELSE()
	MESSAGE(STATUS "Configured for use of P4EST")
	ADD_DEFINITIONS(-DDOPELIB_WITH_P4EST)	
ENDIF()


IF(NOT DEAL_II_WITH_UMFPACK)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_UMFPACK = ON
One or all of these are OFF in your installation but are required for most Examples."
    )
ENDIF()

PROJECT(DOpElib)

ADD_SUBDIRECTORY(source)

EXPORT(PACKAGE DOpElib)

GET_FILENAME_COMPONENT(DOPELIB_ROOT_DIR "${CMAKE_SOURCE_DIR}/../" ABSOLUTE)

CONFIGURE_FILE(DOpElibConfig.cmake.in ${DOPELIB_ROOT_DIR}/lib/cmake/DOpElib/DOpElibConfig.cmake )

