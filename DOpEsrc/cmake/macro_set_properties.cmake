#####
##
## Copyright (C) 2012, 2013 by Matthias Maier
##
## <TODO: Full License information>
## This file is dual licensed under QPL 1.0 and LGPL 2.1 or any later
## version of the LGPL license.
##
#####

#
# A small wrapper around
# SET_TARGET_PROPERTY(... PROPERTIES COMPILE_DEFINITIONS ...)
# to _add_ compile definitions to every target we have specified.
#

MACRO(DOPE_SET_PROPERTIES _keyword _name)

  FOREACH(_build ${DOPE_BUILD_TYPES})
    STRING(TOLOWER ${_build} _build_lowercase)

    SET_PROPERTY(${_keyword} ${_name}.${_build_lowercase}
      ${ARGN}
      )
  ENDFOREACH()

ENDMACRO()

