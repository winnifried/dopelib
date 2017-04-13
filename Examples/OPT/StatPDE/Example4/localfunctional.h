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

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional()
  {
    alpha_ = 10.;
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      qvalues_.reinit(5);
      fvalues_.resize(n_q_points, Vector<double>(2));
      uvalues_.resize(n_q_points, Vector<double>(2));

      edc.GetParamValues("control", qvalues_);
      edc.GetValuesState("state", uvalues_);
    }
    double r = 0.;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point](0) = (sin(
                                  M_PI * state_fe_values.quadrature_point(q_point)(0))
                                * sin(M_PI * state_fe_values.quadrature_point(q_point)(1)))
                               * (state_fe_values.quadrature_point(q_point)(0));
        fvalues_[q_point](1) = (state_fe_values.quadrature_point(q_point)(0));

        r += 0.5 * (uvalues_[q_point](0) - fvalues_[q_point](0))
             * (uvalues_[q_point](0) - fvalues_[q_point](0))
             * state_fe_values.JxW(q_point);
        r += 0.5 * (uvalues_[q_point](1) - fvalues_[q_point](1))
             * (uvalues_[q_point](1) - fvalues_[q_point](1))
             * state_fe_values.JxW(q_point);

        r += alpha_ * 0.5
             * (qvalues_(0) * qvalues_(0) + qvalues_(1) * qvalues_(1)
                + qvalues_(2) * qvalues_(2) + qvalues_(3) * qvalues_(3)
                + qvalues_(4) * qvalues_(4)) * state_fe_values.JxW(q_point);

      }
    return r;
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      fvalues_.resize(n_q_points, Vector<double>(2));
      uvalues_.resize(n_q_points, Vector<double>(2));

      edc.GetValuesState("state", uvalues_);
    }

    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point](0) = (sin(
                                  M_PI * state_fe_values.quadrature_point(q_point)(0))
                                * sin(M_PI * state_fe_values.quadrature_point(q_point)(1)))
                               * (state_fe_values.quadrature_point(q_point)(0));
        fvalues_[q_point](1) = (state_fe_values.quadrature_point(q_point)(0));

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (uvalues_[q_point](0) - fvalues_[q_point](0))
                               * state_fe_values[comp_0].value(i, q_point)
                               * state_fe_values.JxW(q_point);
            local_vector(i) += scale
                               * (uvalues_[q_point](1) - fvalues_[q_point](1))
                               * state_fe_values[comp_1].value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      qvalues_.reinit(local_vector.size());

      edc.GetParamValues("control", qvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * alpha_ * (qvalues_(i))
                               * state_fe_values.JxW(q_point);
          }
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
    {
      duvalues_.resize(n_q_points, Vector<double>(2));
      edc.GetValuesState("tangent", duvalues_);
    }

    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * duvalues_[q_point](0)
                               * state_fe_values[comp_0].value(i, q_point)
                               * state_fe_values.JxW(q_point);
            local_vector(i) += scale * duvalues_[q_point](1)
                               * state_fe_values[comp_1].value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      dqvalues_.reinit(local_vector.size());
      edc.GetParamValues("dq", dqvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * alpha_ * dqvalues_(i)
                               * state_fe_values.JxW(q_point);
          }
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
    return "domain";
  }

  string
  GetName() const
  {
    return "cost functional";
  }

private:
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  vector<Vector<double> > fvalues_;
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > duvalues_;
  double alpha_;
};
#endif
