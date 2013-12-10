/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "functionalinterface.h"

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
        _alpha = 10.;
      }

      double
      ElementValue(const EDC<DH, VECTOR, dealdim>& edc)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _qvalues.reinit(5);
          _fvalues.resize(n_q_points, Vector<double>(2));
          _uvalues.resize(n_q_points, Vector<double>(2));

          edc.GetParamValues("control", _qvalues);
          edc.GetValuesState("state", _uvalues);
        }
        double r = 0.;

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point](0) = (sin(
              M_PI * state_fe_values.quadrature_point(q_point)(0))
              * sin(M_PI * state_fe_values.quadrature_point(q_point)(1)))
              * (state_fe_values.quadrature_point(q_point)(0));
          _fvalues[q_point](1) = (state_fe_values.quadrature_point(q_point)(0));

          r += 0.5 * (_uvalues[q_point](0) - _fvalues[q_point](0))
              * (_uvalues[q_point](0) - _fvalues[q_point](0))
              * state_fe_values.JxW(q_point);
          r += 0.5 * (_uvalues[q_point](1) - _fvalues[q_point](1))
              * (_uvalues[q_point](1) - _fvalues[q_point](1))
              * state_fe_values.JxW(q_point);

          r += _alpha * 0.5
              * (_qvalues(0) * _qvalues(0) + _qvalues(1) * _qvalues(1)
                  + _qvalues(2) * _qvalues(2) + _qvalues(3) * _qvalues(3)
                  + _qvalues(4) * _qvalues(4)) * state_fe_values.JxW(q_point);

        }
        return r;
      }

      void
      ElementValue_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _fvalues.resize(n_q_points, Vector<double>(2));
          _uvalues.resize(n_q_points, Vector<double>(2));

          edc.GetValuesState("state", _uvalues);
        }

        const FEValuesExtractors::Scalar comp_0(0);
        const FEValuesExtractors::Scalar comp_1(1);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point](0) = (sin(
              M_PI * state_fe_values.quadrature_point(q_point)(0))
              * (sin(M_PI * state_fe_values.quadrature_point(q_point)(1))
                  + 0.5
                      * sin(
                          2 * M_PI
                              * state_fe_values.quadrature_point(q_point)(1))))
              * (state_fe_values.quadrature_point(q_point)(0));
          _fvalues[q_point](1) = (state_fe_values.quadrature_point(q_point)(0));

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_uvalues[q_point](0) - _fvalues[q_point](0))
                * state_fe_values[comp_0].value(i, q_point)
                * state_fe_values.JxW(q_point);
            local_vector(i) += scale
                * (_uvalues[q_point](1) - _fvalues[q_point](1))
                * state_fe_values[comp_1].value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementValue_Q(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _qvalues.reinit(local_vector.size());

          edc.GetParamValues("control", _qvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * _alpha * (_qvalues(i))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementValue_UU(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _duvalues.resize(n_q_points, Vector<double>(2));
          edc.GetValuesState("tangent", _duvalues);
        }

        const FEValuesExtractors::Scalar comp_0(0);
        const FEValuesExtractors::Scalar comp_1(1);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * _duvalues[q_point](0)
                * state_fe_values[comp_0].value(i, q_point)
                * state_fe_values.JxW(q_point);
            local_vector(i) += scale * _duvalues[q_point](1)
                * state_fe_values[comp_1].value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementValue_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
      }

      void
      ElementValue_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
      }

      void
      ElementValue_QQ(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _dqvalues.reinit(local_vector.size());
          edc.GetParamValues("dq", _dqvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * _alpha * _dqvalues(i)
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
      Vector<double> _qvalues;
      Vector<double> _dqvalues;
      vector<Vector<double> > _fvalues;
      vector<Vector<double> > _uvalues;
      vector<Vector<double> > _duvalues;
      double _alpha;
  };
#endif
