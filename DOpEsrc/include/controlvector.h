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

#ifndef CONTROL_VECTOR_H_
#define CONTROL_VECTOR_H_

// Quick fix for lots of warnings
// TODO remove ...
//#pragma GCC diagnostic ignored "-Wterminate"

#include <include/spacetimevector.h>


namespace DOpE
{

  /**
   * This class represents the controlvector.
   *
   * @tparam <VECTOR>     Class in which we want to store the spatial vector
   *                      (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   */
  template<typename VECTOR>
    class ControlVector : public SpaceTimeVector<VECTOR>
  {
  public:
    //TODO: Currently we only consider one fixed control
    //      for all timesteps, if more is desired one needs to augment the
    //      Spacetimehandler to have a time discretization for the control,
    //      Then one can update this vector similar to the statevector
    //      with different meshes for Vectors.
    //      Note that this requires to keep track of the interpolation
    //      between state and control time points...
    ControlVector(const ControlVector &ref)  : SpaceTimeVector<VECTOR>(ref)
    {  
    }
    ControlVector(const SpaceTimeHandlerBase<VECTOR> *STH,
		  DOpEtypes::VectorStorageType behavior,
		  ParameterReader &param_reader) : SpaceTimeVector<VECTOR>(STH,
									   behavior,
									   DOpEtypes::VectorType::control,
									   STH->GetControlActionType(),
									   param_reader)
    {
    }
    ~ControlVector()
    {
      
    }

  };

}

#endif
