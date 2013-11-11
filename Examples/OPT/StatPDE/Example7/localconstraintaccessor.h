/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _LOCAL_CONSTRAINT_ACCESSOR_H_
#define _LOCAL_CONSTRAINT_ACCESSOR_H_

namespace DOpE
{

  class LocalConstraintAccessor
  {
    public:
      LocalConstraintAccessor()
      {
        _rho_min = -500.;
        _rho_max = 500.;
      }

      /* Write Control Values into Constraint vector shifted by the lower or upper bounds respectively*/
      inline void
      ControlToLowerConstraint(const dealii::BlockVector<double>& control,
          dealii::BlockVector<double>& constraints)
      {
        assert(constraints.block(0).size() == 2 * control.block(0).size());
        for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          constraints.block(0)(i) = _rho_min - control.block(0)(i);
        }
      }

      inline void
      ControlToUpperConstraint(const dealii::BlockVector<double>& control,
          dealii::BlockVector<double>& constraints)
      {
        assert(constraints.block(0).size() == 2 * control.block(0).size());
        for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          constraints.block(0)(control.block(0).size() + i) = control.block(0)(
              i) - _rho_max;
        }
      }

      inline void
      GetControlBoxConstraints(dealii::BlockVector<double>& lb,
          dealii::BlockVector<double>& ub) const
      {
        lb = _rho_min;
        ub = _rho_max;
      }
      /************************************************************************************************/

    private:
      double _rho_min, _rho_max;
  };
}
#endif
