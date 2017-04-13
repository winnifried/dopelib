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
#include "myfunctions.h"

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

  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    {
      //Reading data
      uvalues_.resize(n_q_points, Vector<double>(2));

      fdc.GetFaceValuesState("state", uvalues_);
      fvalues_.resize(2);
    }
    double ret = 0.;

    if (color == 3)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            MyFunctions::Forces(fvalues_,
                                state_fe_face_values.quadrature_point(q_point)(0),
                                state_fe_face_values.quadrature_point(q_point)(1));
            ret += (fvalues_[0] * uvalues_[q_point](0)
                    + fvalues_[1] * uvalues_[q_point](1))
                   * state_fe_face_values.JxW(q_point);
          }
      }
    return ret;
  }

  void
  BoundaryValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    {
      fvalues_.resize(2);
    }

    if (color == 3)
      {
        const FEValuesExtractors::Scalar comp_0(0);
        const FEValuesExtractors::Scalar comp_1(1);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            MyFunctions::Forces(fvalues_,
                                state_fe_face_values.quadrature_point(q_point)(0),
                                state_fe_face_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale
                                   * (fvalues_[0]
                                      * state_fe_face_values[comp_0].value(i, q_point)
                                      + fvalues_[1]
                                      * state_fe_face_values[comp_1].value(i, q_point))
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }

  }

  void
  BoundaryValue_Q(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryValue_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryValue_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryValue_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryValue_QQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }
  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  std::string
  GetType() const
  {
    return "boundary";
  }

  std::string
  GetName() const
  {
    return "cost functional";
  }

private:
  std::vector<double> fvalues_;
  std::vector<Vector<double> > uvalues_;
};
#endif
