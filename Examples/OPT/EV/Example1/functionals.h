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

#ifndef LOCALFunctionalS_
#define LOCALFunctionalS_

#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalEigenvalueFunctional : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalEigenvalueFunctional : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
#endif
{
public:
  LocalEigenvalueFunctional()
  {
  }


  double
  AlgebraicValue(const std::map<std::string, const dealii::Vector<double>*> &param_values,
                 const std::map<std::string, const VECTOR *> &/*domain_values*/) override
  {
    typename std::map<std::string, const dealii::Vector<double>*>::const_iterator it = param_values.find("state_ev");
    assert(it!=param_values.end());
    double eigenvalue=(*(it->second))[0];
    return eigenvalue;
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_gradients | update_quadrature_points | update_JxW_values;
  }

  string
  GetType() const override
  {
    return "algebraic eigenvalue";
  }
  string
  GetName() const override
  {
    return "eigenvalue ";
  }

private:
  vector<double> uvalues_;
};

/****************************************************************************************/
#endif
