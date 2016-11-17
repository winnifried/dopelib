INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

FILE(GLOB _src *.cc)

FOREACH(_deal_ii_dimension 1 2 3) 
  FOREACH(_dope_dimension 0 ${_deal_ii_dimension})
    ADD_LIBRARY(dope_${_dope_dimension}${_deal_ii_dimension} OBJECT ${_src})
    
    DEAL_II_SETUP_TARGET(dope_${_dope_dimension}${_deal_ii_dimension})
    
    SET_PROPERTY(TARGET dope_${_dope_dimension}${_deal_ii_dimension} APPEND PROPERTY
      COMPILE_DEFINITIONS "deal_II_dimension=${_deal_ii_dimension};dope_dimension=${_dope_dimension};DEAL_II_MAJOR_VERSION=${DEAL_II_VERSION_MAJOR};DEAL_II_MINOR_VERSION=${DEAL_II_VERSION_MINOR}"
      )

#    FOREACH(_this_type SHARED STATIC)
    FOREACH(_this_type STATIC)

      ADD_LIBRARY(dope_${_dope_dimension}${_deal_ii_dimension}${_this_type} ${_this_type} 
	$<TARGET_OBJECTS:dope_${_dope_dimension}${_deal_ii_dimension}>)
      
      DEAL_II_SETUP_TARGET(dope_${_dope_dimension}${_deal_ii_dimension}${_this_type})

      STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
      IF("${_cmake_build_type}" STREQUAL "debug")
	SET(_suffix ".g")
      ENDIF()

      SET_PROPERTY(TARGET dope_${_dope_dimension}${_deal_ii_dimension}${_this_type} PROPERTY OUTPUT_NAME dope${_suffix})
      
      SET_PROPERTY(TARGET dope_${_dope_dimension}${_deal_ii_dimension}${_this_type} PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib/${_dope_dimension}d/${_deal_ii_dimension}d/)
      SET_PROPERTY(TARGET dope_${_dope_dimension}${_deal_ii_dimension}${_this_type} PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib/${_dope_dimension}d/${_deal_ii_dimension}d/)

    ENDFOREACH()
  ENDFOREACH()
ENDFOREACH()




#SET_TARGET_PROPERTIES(dope
#  PROPERTIES
#  VERSION "${DOPE_PACKAGE_VERSION}"
#  SOVERSION "${DOPE_PACKAGE_VERSION}"
#  )


