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

//#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dopedim, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#endif
{
public:
  LocalFunctional()
  {
  }

  bool
  NeedTime() const override
  {
    if (fabs(this->GetTime() - 1.0) < 1.e-13)
      return true;
    if (fabs(this->GetTime()) < 1.e-13)
      return true;
    return false;
  }

  double
  ElementValue(
    const EDC<DH, VECTOR, dealdim> &edc) override
  {
    unsigned int n_q_points = edc.GetNQPoints();
    double ret = 0.;
    if (fabs(this->GetTime() - 1.0) < 1.e-13)
      {
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          edc.GetFEValuesState();
        //endtimevalue
        fvalues_.resize(n_q_points);
        uvalues_.resize(n_q_points);

        edc.GetValuesState("state", uvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = sin(
                                  state_fe_values.quadrature_point(q_point)(0))
                                * sin(state_fe_values.quadrature_point(q_point)(1));

            ret += 0.5 * (uvalues_[q_point] - fvalues_[q_point])
                   * (uvalues_[q_point] - fvalues_[q_point])
                   * state_fe_values.JxW(q_point);
          }
        return ret;
      }
    if (fabs(this->GetTime()) < 1.e-13)
      {
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          edc.GetFEValuesControl();
        //initialvalue
        fvalues_.resize(n_q_points);
        qvalues_.resize(n_q_points);
        edc.GetValuesControl("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = sin(
                                  state_fe_values.quadrature_point(q_point)(0))
                                * sin(state_fe_values.quadrature_point(q_point)(1));

            ret += 0.5 * (qvalues_[q_point] - fvalues_[q_point])
                   * (qvalues_[q_point] - fvalues_[q_point])
                   * state_fe_values.JxW(q_point);
          }
        return ret;
      }
    throw DOpEException("This should not be evaluated here!",
                        "LocalFunctional::Value");
  }

  void
  ElementValue_U(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    if (fabs(this->GetTime() - 1.0) < 1.e-13)
      {
        //endtimevalue
        fvalues_.resize(n_q_points);
        uvalues_.resize(n_q_points);

        edc.GetValuesState("state", uvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = sin(
                                  state_fe_values.quadrature_point(q_point)(0))
                                * sin(state_fe_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale
                                   * (uvalues_[q_point] - fvalues_[q_point])
                                   * state_fe_values.shape_value(i, q_point)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementValue_Q(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    if (fabs(this->GetTime()) < 1.e-13)
      {
        //endtimevalue
        fvalues_.resize(n_q_points);
        qvalues_.resize(n_q_points);

        edc.GetValuesControl("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = sin(
                                  state_fe_values.quadrature_point(q_point)(0))
                                * sin(state_fe_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale
                                   * (qvalues_[q_point] - fvalues_[q_point])
                                   * state_fe_values.shape_value(i, q_point)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementValue_UU(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    if (fabs(this->GetTime() - 1.0) < 1.e-13)
      {
        //endtimevalue
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
  }

  void
  ElementValue_QU(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double) override
  {
  }

  void
  ElementValue_UQ(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double) override
  {
  }

  void
  ElementValue_QQ(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    if (fabs(this->GetTime()) < 1.e-13)
      {
        //endtimevalue
        dqvalues_.resize(n_q_points);

        edc.GetValuesControl("dq", dqvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale * dqvalues_[q_point]
                                   * state_fe_values.shape_value(i, q_point)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const override
  {
    return "domain timelocal";
  }

  std::string
  GetName() const override
  {
    return "Cost-functional";
  }

private:
  vector<double> qvalues_;
  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<double> duvalues_;
  vector<double> dqvalues_;

};
#endif
