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


#include <include/dopeexceptionhandler.h>
#include <include/outputhandler.h>

#include <iostream>
#include <string>
#include <cstdlib>


namespace DOpE
{

  /*******************************************************/

  template<typename VECTOR>
  DOpEExceptionHandler<VECTOR>::DOpEExceptionHandler(DOpEOutputHandler<VECTOR> *OutputHandler)
  {
    OutputHandler_ = OutputHandler;
  }

  /*******************************************************/

  template<typename VECTOR>
  DOpEExceptionHandler<VECTOR>::~DOpEExceptionHandler()
  {

  }

  /*******************************************************/

  template<typename VECTOR>
  void DOpEExceptionHandler<VECTOR>::HandleException(DOpEException &e,std::string reporter)
  {
    GetOutputHandler()->WriteError("The following `" + e.GetName() + "` reports!");
    GetOutputHandler()->WriteError("Warning: During execution of `" + e.GetThrowingInstance()
                                   + "` the following Problem occurred!");
    GetOutputHandler()->WriteError(e.GetErrorMessage());
    GetOutputHandler()->WriteError("Reported by `"+reporter+"`");
  }

  /*******************************************************/

  template<typename VECTOR>
  void DOpEExceptionHandler<VECTOR>::HandleCriticalException(DOpEException &e,std::string reporter)
  {
    GetOutputHandler()->WriteError("The following `" + e.GetName() + "` reports!");
    GetOutputHandler()->WriteError("Error: During execution of `" + e.GetThrowingInstance()
                                   + "` the following fatal Problem occurred!");
    GetOutputHandler()->WriteError(e.GetErrorMessage());
    GetOutputHandler()->WriteError("Reported by `"+reporter+"`");
    abort();
  }

}//Endof namespace
/******************************************************/

/******************************************************/
/******************************************************/
template class DOpE::DOpEExceptionHandler<dealii::Vector<double> >;
template class DOpE::DOpEExceptionHandler<dealii::BlockVector<double> >;

