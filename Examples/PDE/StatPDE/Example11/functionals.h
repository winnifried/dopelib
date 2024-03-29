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

#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
class BoundaryFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class BoundaryFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dealdim>
#endif
{
public:
  BoundaryFunctional()
  {
  }

  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc) override
  {
    unsigned int n_q_points = fdc.GetNQPoints();

    double cw = 0;

    ufacevalues_.resize(n_q_points);

    fdc.GetFaceValuesState("state", ufacevalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        cw += 1. * fdc.GetFEFaceValuesState().JxW(q_point);
      }

    return cw / 2;
  }

  UpdateFlags
  GetFaceUpdateFlags() const override
  {
    return update_values | update_quadrature_points | update_normal_vectors;
  }

  string
  GetType() const override
  {
    return "boundary";
  }

  bool
  HasFaces() const override
  {
    return true;
  }

  string
  GetName() const override
  {
    return "BoundaryPI";
  }

private:
  vector<double> ufacevalues_;

};

#endif
