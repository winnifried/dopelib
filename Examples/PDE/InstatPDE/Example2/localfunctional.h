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

#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
      dopedim, dealdim>
  {
    public:
      LocalFunctional() : _time(0)
      {
      }

      // include NeedTime
      void
      SetTime(double t) const
      {
        _time = t;
      }

      bool
      NeedTime() const
      {
        if (fabs(_time - 1.) < 1.e-13)
          return true;
        return false;
      }

      double
	ElementValue(const EDC<DH, VECTOR, dealdim>& /*edc*/)
      {
        return 0.0;
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_quadrature_points;
      }

      string
      GetType() const
      {
        return "domain time_local";
      }

      string
      GetName() const
      {
        return "dummy functional";
      }

    private:

      mutable double _time;

  };
#endif
