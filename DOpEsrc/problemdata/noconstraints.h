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

#ifndef NOCONSTRAINT_INTERFACE_H_
#define NOCONSTRAINT_INTERFACE_H_

#include <interfaces/constraintinterface.h>

namespace DOpE
{
  /**
   * This is a special case of additional (other then the PDE) constraint
   * when no such constraints are present.
   *
   * For details on the methods see ConstraintInterface
   */
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  class NoConstraints : public ConstraintInterface<EDC, FDC, DH,
    VECTOR, dopedim, dealdim>
  {
  public:
    NoConstraints() :
      ConstraintInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>()
    {
    }
    ~NoConstraints()
    {
    }

    void
    EvaluateLocalControlConstraints(
      const VECTOR & /*control*/,
      VECTOR & /*constraints*/)
    {
      throw DOpEException("This should never be called!",
                          "NoConstraints::EvaluateLocalControlConstraints");
      abort();
    }
    void
    GetControlBoxConstraints(VECTOR &lb, VECTOR &ub) const
    {
      lb = -1.e+20;
      ub = 1.e+20;
    }
    void
    PostProcessConstraints(
      ConstraintVector<VECTOR> & /*g*/) const
    {
    }

  protected:
  private:

  };
}

#endif
