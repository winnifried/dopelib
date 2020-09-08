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

template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional(double alpha)
  {
    alpha_ = alpha;
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
      qvalues_.resize(n_q_points, Vector<double>(2));
      uvalues_.resize(n_q_points, Vector<double>(3));
      vector<Vector<double>> eigenvalue;
      eigenvalue.resize(n_q_points, Vector<double>(3));

      edc.GetValuesControl("control", qvalues_);
      edc.GetValuesState("state", uvalues_);
      edc.GetValuesState("eigenvalue", eigenvalue);

    double ev = eigenvalue[0][0];
//    std::cout << "eigenvalue in elementvalue = " << ev << std::endl;

    double r = 0.;
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
    	fvalues_= 1.;//* (state_fe_values.quadrature_point(q_point)(0)*state_fe_values.quadrature_point(q_point)(1));
        r += 0.5 * (ev - fvalues_) * (ev - fvalues_)
             * state_fe_values.JxW(q_point);
         r += 0.5 * alpha_ * (qvalues_[q_point][0] * qvalues_[q_point][0] + qvalues_[q_point][1] * qvalues_[q_point][1])
             * state_fe_values.JxW(q_point);
      }
    return r;
  }
//
//  void
//  ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
//                 dealii::Vector<double> &local_vector, double scale)
//  {
//    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
//      edc.GetFEValuesState();
//    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
//    unsigned int n_q_points = edc.GetNQPoints();
//    {
//      fvalues_.resize(n_q_points);
//      uvalues_.resize(n_q_points);
//
//      edc.GetValuesState("state", uvalues_);
//    }
//
//    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//      {
//        fvalues_[q_point] = 1.0;
//        for (unsigned int i = 0; i < n_dofs_per_element; i++)
//          {
//            local_vector(i) += scale
//                               * (uvalues_[q_point] - fvalues_[q_point])
//                               * state_fe_values.shape_value(i, q_point)
//                               * state_fe_values.JxW(q_point);
//          }
//      }
//  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
//      qvalues_.resize(n_q_points);
    	qvalues_.resize(n_q_points, Vector<double>(2));
      edc.GetValuesControl("control", qvalues_);
    }
	vector<Tensor<1, dealdim> > phi_q(n_dofs_per_element);

    const FEValuesExtractors::Vector dv(0);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
        	phi_q[i]=  control_fe_values[dv].value(i,q_point);

        	local_vector(i) += scale * alpha_
              * (qvalues_[q_point][0]*phi_q[i][0]+qvalues_[q_point][1]*phi_q[i][1])
              * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
//    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
//      edc.GetFEValuesState();
//    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
//    unsigned int n_q_points = edc.GetNQPoints();
//    {
//      duvalues_.resize(n_q_points);
//      edc.GetValuesState("tangent", duvalues_);
//    }
//
//    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//      {
//        for (unsigned int i = 0; i < n_dofs_per_element; i++)
//          {
//            local_vector(i) += scale * duvalues_[q_point]
//                               * state_fe_values.shape_value(i, q_point)
//                               * state_fe_values.JxW(q_point);
//          }
//      }
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
//    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
//      edc.GetFEValuesControl();
//    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
//    unsigned int n_q_points = edc.GetNQPoints();
//    {
//      dqvalues_.resize(n_q_points);
//      edc.GetValuesControl("dq", dqvalues_);
//    }
//
//    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//      {
//        for (unsigned int i = 0; i < n_dofs_per_element; i++)
//          {
//            local_vector(i) += scale * alpha_
//                               * (dqvalues_[q_point]
//                                  * control_fe_values.shape_value(i, q_point))
//                               * control_fe_values.JxW(q_point);
//          }
//      }
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
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > qvalues_;
  vector<Vector<double> > funcgradvalues_;
//  Vector<double>  fvalues_;
  double fvalues_;
//  vector<double> fvalues_;
//  vector<double> uvalues_;
//  vector<double> duvalues_;
//  vector<double> dqvalues_;
  double alpha_;
};
#endif
