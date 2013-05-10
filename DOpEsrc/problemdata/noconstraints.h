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

#ifndef _NOCONSTRAINT_INTERFACE_H_
#define _NOCONSTRAINT_INTERFACE_H_

#include "constraintinterface.h"

namespace DOpE
{
  /**
   * This is a special case of additional (other then the PDE) constraint
   * when no such constraints are present.
   *
   * For details on the methods see ConstraintInterface
   */
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
    class NoConstraints : public ConstraintInterface<CDC, FDC, DH,
        VECTOR, dopedim, dealdim>
    {
      public:
        NoConstraints() :
            ConstraintInterface<CDC, FDC, DH, VECTOR, dopedim, dealdim>()
        {
        }
        ~NoConstraints()
        {
        }

        void
        EvaluateLocalControlConstraints(
	  const VECTOR& /*control*/,
	  VECTOR& /*constraints*/)
        {
          throw DOpEException("This should never be called!",
              "NoConstraints::EvaluateLocalControlConstraints");
          abort();
        }
        void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
        {
          lb = -1.e+20;
          ub = 1.e+20;
        }
        bool
        IsFeasible(
	  const ConstraintVector<VECTOR>& /*g*/) const
        {
          return true;
        }
        bool
	  IsLargerThan(const ConstraintVector<VECTOR>& /*g*/,
		       double p) const
        {
          if (p < 0)
            return true;
          return false;
        }
        bool
        IsEpsilonFeasible(const ConstraintVector<VECTOR>& /*g*/, double p) const
        {
          if (p >= 0)
            return true;
          return false;
        }
        void
        PostProcessConstraints(
	  ConstraintVector<VECTOR>& /*g*/) const
        {
        }
        double
        MaxViolation(
	  const ConstraintVector<VECTOR>& /*g*/) const
        {
          return 0.;
        }
        void
        FeasibilityShift(
	  const ControlVector<VECTOR>& /*g_hat*/,
	  ControlVector<VECTOR>& /*g*/,
	  double /*lambda*/) const
        {
        }
        double
        Complementarity(const ConstraintVector<VECTOR>& /*f*/,
            const ConstraintVector<VECTOR>& /*g*/) const
        {
          return 0.;
        }

      protected:
      private:

    };
}

#endif
