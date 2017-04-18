/**
*
* Copyright (C) 2012-2014 by the DOpElib authors
*
* This file is part of DOpElib
*
* DOpElib is free software: you can redistribute it
* and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later
* version.
*
* DOpElib is distributed in the hope that it will be
* useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the GNU General Public License for more
* details.
*
* Please refer to the file LICENSE.TXT included in this distribution
* for further information on this license.
*
**/

#ifndef DOPE_EXCEPTION_HANDLER_H_
#define DOPE_EXCEPTION_HANDLER_H_

#include <include/dopeexception.h>
//#include <include/outputhandler.h>
#include <string>

namespace DOpE
{
//Predeclaration necessary
  template<typename VECTOR> class DOpEOutputHandler;
/////////////////////////////
  template<typename VECTOR>
  class DOpEExceptionHandler
  {
  public:
    DOpEExceptionHandler(DOpEOutputHandler<VECTOR> *OutputHandler);
    ~DOpEExceptionHandler();

    /**
     * This function handles non critical exceptions, e.g.,  if an iterative
     * solver reached its maximal iteration count.
     * This function writes a warning message, but doesn't stop the computation.
     */
    void HandleException(DOpEException &e, std::string reporter = "undefined");
    /**
     * This function handles critical exceptions, e.g., the equation couln't be solved.
     * This function writes a warning message, and then terminates the program because
     * it is impossible to continue after a severe problem.
     */
    void HandleCriticalException(DOpEException &e, std::string reporter = "undefined");
  protected:
    DOpEOutputHandler<VECTOR> *GetOutputHandler()
    {
      return OutputHandler_;
    }
  private:
    DOpEOutputHandler<VECTOR> *OutputHandler_;
  };

}
#endif
