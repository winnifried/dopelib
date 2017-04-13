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


#ifndef FUNCTIONALS_H_
#define FUNCTIONALS_H_

#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalFunctional()
  {
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    unsigned int n_q_points = edc.GetNQPoints();

    double mean = 0;

    vector<Vector<double> > uvalues;
    uvalues.resize(n_q_points,Vector<double>(2));
    edc.GetValuesState("state", uvalues);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double v;

        v = uvalues[q_point][0];

        mean += v * edc.GetFEValuesState().JxW(q_point);
      }
    return mean;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    return "Mean-value";
  }

  bool
  NeedTime() const
  {
    if (fabs(this->GetTime() - 1.) < 1.e-13)
      return true;
    return false;
  }

};
#endif /* FUNCTIONALS_H_ */
