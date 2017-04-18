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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <interfaces/functionalinterface.h>
#include "my_functions.h"

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
  LocalFunctional()
  {
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    unsigned int n_q_points = edc.GetNQPoints();
    double ret = 0.;

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);
    uvalues_.resize(n_q_points);
    qvalues_.reinit(1);
    edc.GetParamValues("control", qvalues_);

    edc.GetValuesState("state", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] =  my::ud(this->GetTime(),state_fe_values.quadrature_point(q_point));

        ret += 0.5 * (uvalues_[q_point] - fvalues_[q_point])
               * (uvalues_[q_point] - fvalues_[q_point])
               * state_fe_values.JxW(q_point);
        //In control correct for the volume integral.
        ret += 0.5/(M_PI*M_PI) * qvalues_(0) * qvalues_(0)
               * state_fe_values.JxW(q_point);
      }
    return ret;

  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    fvalues_.resize(n_q_points);
    uvalues_.resize(n_q_points);

    edc.GetValuesState("state", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] = my::ud(this->GetTime(),state_fe_values.quadrature_point(q_point));

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (uvalues_[q_point] - fvalues_[q_point])
                               * state_fe_values.shape_value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    unsigned int n_q_points = edc.GetNQPoints();
    assert(local_vector.size()==1);

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    qvalues_.reinit(1);
    edc.GetParamValues("control", qvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        local_vector(0) += scale * 1./(M_PI*M_PI) * qvalues_(0)
                           * state_fe_values.JxW(q_point);
      }
  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    duvalues_.resize(n_q_points);

    edc.GetValuesState("tangent", duvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * duvalues_[q_point]
                               * state_fe_values.shape_value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    unsigned int n_q_points = edc.GetNQPoints();
    assert(local_vector.size()==1);

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    dqvalues_.reinit(1);
    edc.GetParamValues("dq", dqvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        local_vector(0) += scale * 1./(M_PI*M_PI) * dqvalues_(0)
                           * state_fe_values.JxW(q_point);
      }

  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const
  {
    return "domain timedistributed";
  }

  std::string
  GetName() const
  {
    return "Cost-functional";
  }

private:
  Vector<double> qvalues_;
  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<double> duvalues_;
  Vector<double> dqvalues_;

};
#endif
