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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,dealdim>
#endif
{
public:
  LocalFunctional()
  {
  }

  double
  ElementValue(
    const EDC<DH, VECTOR, dealdim> &edc) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();

    {
      uvalues_.resize(n_q_points,Vector<double>(3));
      edc.GetValuesState("state", uvalues_);
    }
    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dealdim);

    double r = 0.;
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        const double alpha = 0.3;
        const double beta = 1;
        Tensor<1,dealdim> p = state_fe_values.quadrature_point(q_point);
        Vector<double> exact(3);
        exact(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
        exact(1) = alpha*p[0]*p[1];
        exact(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

        exact -= uvalues_[q_point];
        r += exact*exact*state_fe_values.JxW(q_point);
      }
    return r;
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const override
  {
    return "domain";
  }

  string
  GetName() const override
  {
    return "L^2-error";
  }

private:
  vector<Vector<double> > uvalues_;
};
#endif
