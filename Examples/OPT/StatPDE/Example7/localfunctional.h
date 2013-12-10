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
        _alpha = 1.e-3;
      }

      double
      ElementValue(const EDC<DH, VECTOR, dealdim>& edc)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();

        {
          _qvalues.resize(n_q_points);
          _fvalues.resize(n_q_points);
          _uvalues.resize(n_q_points);

          edc.GetValuesControl("control", _qvalues);
          edc.GetValuesState("state", _uvalues);
        }

        double r = 0.;

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point] = (1.
              * sin(4 * M_PI * state_fe_values.quadrature_point(q_point)(0))
              + 5. * M_PI * M_PI
                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
              * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1));

          r += 0.5 * (_uvalues[q_point] - _fvalues[q_point])
              * (_uvalues[q_point] - _fvalues[q_point])
              * state_fe_values.JxW(q_point);
          r += 0.5 * _alpha * (_qvalues[q_point] * _qvalues[q_point])
              * state_fe_values.JxW(q_point);
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
          _fvalues.resize(n_q_points);
          _uvalues.resize(n_q_points);

          edc.GetValuesState("state", _uvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point] = (1.
              * sin(4 * M_PI * state_fe_values.quadrature_point(q_point)(0))
              + 5. * M_PI * M_PI
                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
              * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1));
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_uvalues[q_point] - _fvalues[q_point])
                * state_fe_values.shape_value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementValue_Q(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _qvalues.resize(n_q_points);

          edc.GetValuesControl("control", _qvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) +=
                scale * _alpha
                    * (_qvalues[q_point]
                        * control_fe_values.shape_value(i, q_point))
                    * control_fe_values.JxW(q_point);
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
          _duvalues.resize(n_q_points);
          edc.GetValuesState("tangent", _duvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * _duvalues[q_point]
                * state_fe_values.shape_value(i, q_point)
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
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          _dqvalues.resize(n_q_points);
          edc.GetValuesControl("dq", _dqvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * _alpha
                * (_dqvalues[q_point]
                    * control_fe_values.shape_value(i, q_point))
                * control_fe_values.JxW(q_point);
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
      vector<double> _qvalues;
      vector<double> _fvalues;
      vector<double> _uvalues;
      vector<double> _duvalues;
      vector<double> _dqvalues;
      double _alpha;
  };
#endif
