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

#ifndef SNOPT_WRAPPER_
#define SNOPT_WRAPPER_

#ifdef DOPELIB_WITH_SNOPT
//BEGIN DONOT_TOUCH
//The following code is sensitive to ordering
//due to some bugs in f2c and snopt
//please do not touch this unless you are very
//sure of what you are doing! And even then---don't.
//If you do something wrong here your code may compile
//and then fail if you include some other headers you
//may additionally need for no apparent reason with no
//reasonable compiler warning whatsoever.
//         You have been warned: HERE BE DRAGONS!
#include "snopt.hh"
#include "snfilewrapper.hh"
#include "snoptProblem.hh"
///NEEDED DUE TO CONFLICTING defines in f2c.h
#undef abs
#undef dabs
#undef min
#undef max
#undef dmin
#undef dmax
#undef bit_test
#undef bit_clear
#undef bit_set
//END OF f2c BUGFIX
//END DONOT_TOUCH

#include <boost/function.hpp>

namespace DOpEWrapper
{
  /**
   * @struct SNOPT_FUNC_DATA
   *
   * A struct defined to contain all the data needed by the
   * SNOPT interface.
   *
   * Since the number of arguments required by the SNOPT userinterface
   * is larger than usually initialized number N of arguments
   * accepted by boost::functionN we use this struct to allow use
   * of the SNOPT interface with any boost functionality as it
   * uses only one argument now.
   */
  struct SNOPT_FUNC_DATA
  {
    integer    *Status;
    integer *n;
    doublereal *x;
    integer    *needF;
    integer *neF;
    doublereal *F;
    integer    *needG;
    integer *neG;
    doublereal *G;
    char       *cu;
    integer *lencu;
    integer    *iu;
    integer *leniu;
    doublereal *ru;
    integer *lenru;
  };

  /**
   * @fn SNOPT_A_userfunc_
   *
   * This function is required by the SNOPT userinterface.
   * here it bundles the arguments into the struct SNOPT_FUNC_DATA
   * which is then passed to the function
   * SNOPT_A_userfunc_interface
   * which we will load at runtime using boost with the user defined data.
   */
  boost::function1<int, SNOPT_FUNC_DATA &> SNOPT_A_userfunc_interface;
  int SNOPT_A_userfunc_(integer    *Status, integer *n,    doublereal x[],
                        integer    *needF,  integer *neF,  doublereal F[],
                        integer    *needG,  integer *neG,  doublereal G[],
                        char       *cu,     integer *lencu,
                        integer    iu[],    integer *leniu,
                        doublereal ru[],    integer *lenru )
  {
    SNOPT_FUNC_DATA data;
    data.Status = Status;
    data.n      = n;
    data.x      = x;
    data.needF  = needF;
    data.neF    = neF;
    data.F      = F;
    data.needG  = needG;
    data.neG    = neG;
    data.G      = G;
    data.cu     = cu;
    data.lencu  = lencu;
    data.iu     = iu;
    data.leniu  = leniu;
    data.ru     = ru;
    data.lenru  = lenru;

    if (SNOPT_A_userfunc_interface)
      return SNOPT_A_userfunc_interface(data);
    else
      throw DOpE::DOpEException("The boost::function SNOPT_userfunc_interface has not been declared","DOpEWrapper::SNOPT::dope_snopt_userfunc_");
  }

  /**
   * @class SNOPT_Problem
   *
   * A class that wrapps the snoptProblem interface for us.
   */
  class SNOPT_Problem : public snoptProblem
  {
  public:
    int GetReturnStatus()
    {
      return snoptProblem::inform;
    }
  };
}
#endif //Endof DOPELIB_WITH_SNOPT
#endif //Endof File
