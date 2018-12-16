/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef STATE_VECTOR_H_
#define STATE_VECTOR_H_

// TODO remove ...
//#pragma GCC diagnostic ignored "-Wterminate"

#include <include/spacetimevector.h>

namespace DOpE
{
  /**
   * This class represents the Statevector.
   *
   * @tparam <VECTOR>     Class in which we want to store the spatial vector
   *                      (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   */
  template<typename VECTOR>
    class StateVector : public SpaceTimeVector<VECTOR>
  {
  public:
    //FIXME this is not a real copyconstructor, it just
    //uses the information of ref about size and so on. Is this correct?
  StateVector(const StateVector<VECTOR> &ref) : SpaceTimeVector<VECTOR>(ref)
    {  
    }
    StateVector(const SpaceTimeHandlerBase<VECTOR> *STH,
                DOpEtypes::VectorStorageType behavior,
                ParameterReader &param_reader)
      : SpaceTimeVector<VECTOR>(STH,behavior, DOpEtypes::VectorType::state,param_reader)
    {
    }
    ~StateVector()
    {
    }

  };

}
#endif
