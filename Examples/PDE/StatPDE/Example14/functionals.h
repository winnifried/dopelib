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
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, 0, dealdim>
{
public:
  LocalFunctional(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("localpde parameters");
    w_in_   = param_reader.get_double("win");
    p_in_  = param_reader.get_double("pin");

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
        double w;

        w = uvalues[q_point][0];
        double w_ex = w_in_ + edc.GetFEValuesState().quadrature_point(q_point)[0];

        mean += fabs(w-w_ex) * edc.GetFEValuesState().JxW(q_point);
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
    return "W-Error";
  }

  bool
  NeedTime() const
  {
    return true;
  }

private:
  double w_in_, p_in_;
};

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalFunctional2 : public FunctionalInterface<EDC, FDC, DH, VECTOR, 0, dealdim>
{
public:
  LocalFunctional2(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("localpde parameters");
    w_in_   = param_reader.get_double("win");
    p_in_  = param_reader.get_double("pin");
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    unsigned int n_q_points = edc.GetNQPoints();

    double mean = 0;

    vector<Vector<double> > uvalues;
    uvalues.resize(n_q_points,Vector<double>(2));
    edc.GetValuesState("state", uvalues);
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double p = uvalues[q_point][1];
        double p_ex = p_in_ + 2.*(100.-edc.GetFEValuesState().quadrature_point(q_point)[0]);

        mean += fabs(p-p_ex) * state_fe_values.JxW(q_point);
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
    return "P-Error";
  }

  bool
  NeedTime() const
  {
    return true;
  }

private:
  double w_in_,p_in_;
};
#endif /* FUNCTIONALS_H_ */
