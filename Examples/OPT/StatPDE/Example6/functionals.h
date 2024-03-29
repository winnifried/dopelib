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
  bool DH, typename VECTOR, int dopedim, int dealdim = dopedim>
class LocalMeanValueFunctional : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim = dopedim>
class LocalMeanValueFunctional : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
#endif
{
public:
  LocalMeanValueFunctional()
  {
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      uvalues_.resize(n_q_points);
      edc.GetValuesState("state", uvalues_);
    }

    double r = 0.;
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        r += fabs(uvalues_[q_point]) * state_fe_values.JxW(q_point);
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
    return "L1-Norm";
  }

private:
  vector<double> uvalues_;
};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dopedim, int dealdim = dopedim>
class LocalPointFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim = dopedim>
class LocalPointFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#endif
{
public:

  double
  PointValue(
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dopedim> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
#else
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
#endif
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values) override
  {
    Point<2> p(0.125, 0.75);

    typename map<string, const BlockVector<double>*>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(1);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);

    return tmp_vector(0);
  }

  string
  GetType() const override
  {
    return "point";
  }
  string
  GetName() const override
  {
    return "PointValue";
  }

};

#endif
