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

//#include "pdeinterface.h"
#include "functionalinterface.h"

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
      LocalFunctional() :
          _time(0)
      {
      }

      // include NeedTime
      void
      SetTime(double t) const
      {
        _time = t;
      }

      bool
      NeedTime() const
      {
        if (fabs(_time - 1.0) < 1.e-13)
          return true;
        if (fabs(_time) < 1.e-13)
          return true;
        return false;
      }

      double
      ElementValue(const EDC<DH, VECTOR, dealdim>& edc)
      {
        unsigned int n_q_points = edc.GetNQPoints();
        double ret = 0.;
        if (fabs(_time - 1.0) < 1.e-13)
        {
          const DOpEWrapper::FEValues<dealdim> & state_fe_values =
              edc.GetFEValuesState();
          //endtimevalue
          _fvalues.resize(n_q_points);
          _uvalues.resize(n_q_points);

          edc.GetValuesState("state", _uvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            _fvalues[q_point] = sin(
                state_fe_values.quadrature_point(q_point)(0))
                * sin(state_fe_values.quadrature_point(q_point)(1));

            ret += 0.5 * (_uvalues[q_point] - _fvalues[q_point])
                * (_uvalues[q_point] - _fvalues[q_point])
                * state_fe_values.JxW(q_point);
          }
          return ret;
        }
        if (fabs(_time) < 1.e-13)
        {
          const DOpEWrapper::FEValues<dealdim> & state_fe_values =
              edc.GetFEValuesControl();
          //initialvalue
          _fvalues.resize(n_q_points);
          _qvalues.resize(n_q_points);
          edc.GetValuesControl("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            _fvalues[q_point] = sin(
                state_fe_values.quadrature_point(q_point)(0))
                * sin(state_fe_values.quadrature_point(q_point)(1));

            ret += 0.5 * (_qvalues[q_point] - _fvalues[q_point])
                * (_qvalues[q_point] - _fvalues[q_point])
                * state_fe_values.JxW(q_point);
          }
          return ret;
        }
        throw DOpEException("This should not be evaluated here!",
            "LocalFunctional::Value");
      }

      void
      ElementValue_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        if (fabs(_time - 1.0) < 1.e-13)
        {
          //endtimevalue
          _fvalues.resize(n_q_points);
          _uvalues.resize(n_q_points);

          edc.GetValuesState("state", _uvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            _fvalues[q_point] = sin(
                state_fe_values.quadrature_point(q_point)(0))
                * sin(state_fe_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) += scale
                  * (_uvalues[q_point] - _fvalues[q_point])
                  * state_fe_values.shape_value(i, q_point)
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ElementValue_Q(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        if (fabs(_time) < 1.e-13)
        {
          //endtimevalue
          _fvalues.resize(n_q_points);
          _qvalues.resize(n_q_points);

          edc.GetValuesControl("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            _fvalues[q_point] = sin(
                state_fe_values.quadrature_point(q_point)(0))
                * sin(state_fe_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) += scale
                  * (_qvalues[q_point] - _fvalues[q_point])
                  * state_fe_values.shape_value(i, q_point)
                  * state_fe_values.JxW(q_point);
            }
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

        if (fabs(_time - 1.0) < 1.e-13)
        {
          //endtimevalue
          _duvalues.resize(n_q_points);

          edc.GetValuesState("tangent", _duvalues);

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
      }

      void
      ElementValue_QU(const EDC<DH, VECTOR, dealdim>&,
          dealii::Vector<double> &, double)
      {
      }

      void
      ElementValue_UQ(const EDC<DH, VECTOR, dealdim>&,
          dealii::Vector<double> &, double)
      {
      }

      void
      ElementValue_QQ(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        if (fabs(_time) < 1.e-13)
        {
          //endtimevalue
          _dqvalues.resize(n_q_points);

          edc.GetValuesControl("dq", _dqvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) += scale * _dqvalues[q_point]
                  * state_fe_values.shape_value(i, q_point)
                  * state_fe_values.JxW(q_point);
            }
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
        return "domain timelocal";
      }

      std::string
      GetName() const
      {
        return "Cost-functional";
      }

    private:
      vector<double> _qvalues;
      vector<double> _fvalues;
      vector<double> _uvalues;
      vector<double> _duvalues;
      vector<double> _dqvalues;

      mutable double _time;

  };
#endif
