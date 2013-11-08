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

#ifndef _LOCAL_CONSTRAINT_H_
#define _LOCAL_CONSTRAINT_H_

#include "constraintinterface.h"
#include "localconstraintaccessor.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
    class LocalConstraint : public ConstraintInterface<CDC, FDC, DH, VECTOR,
        dopedim, dealdim>
    {
      public:
        LocalConstraint(LocalConstraintAccessor& CA) :
            LCA(CA)
        {
        }
        ~LocalConstraint()
        {
        }

        void
        EvaluateLocalControlConstraints(
            const dealii::BlockVector<double>& control,
            dealii::BlockVector<double>& constraints)
        {
          assert(constraints.block(0).size() == 2*control.block(0).size());

          for (unsigned int i = 0; i < control.block(0).size(); i++)
          {
            //Add Control Constraints, such that if control is feasible all  entries are not positive!
            // _rho_min <= control <= _rho_max
            LCA.ControlToLowerConstraint(control, constraints);
            LCA.ControlToUpperConstraint(control, constraints);
          }
        }
        void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
        {
          LCA.GetControlBoxConstraints(lb, ub);
        }

        std::string
        GetType() const
        {
          throw DOpEException("Unknown problem_type " + this->GetProblemType(),
              "LocalConstraints::GetType");
        }
        std::string
        GetName() const
        {
          throw DOpEException("Unknown problem_type " + this->GetProblemType(),
              "LocalConstraints::GetName");
        }

        dealii::UpdateFlags
        GetUpdateFlags() const
        {
          return update_values | update_quadrature_points;
        }

      private:
        LocalConstraintAccessor& LCA;

    };
}

#endif
