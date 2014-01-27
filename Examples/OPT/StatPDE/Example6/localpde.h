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

#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR,
      dealdim>
  {
    public:
      LocalPDE() :
          _block_components(1, 0)
      {
        _alpha = 1.e-3;
      }

      void
      ElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          //Reading data
          assert(this->_problem_type == "state");
          _qvalues.resize(n_q_points);
          _ugrads.resize(n_q_points);

          //Getting q
          edc.GetValuesControl("control", _qvalues);
          //Geting u
          edc.GetGradsState("last_newton_solution", _ugrads);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_ugrads[q_point] * state_fe_values.shape_grad(i, q_point)
                    - _qvalues[q_point]
                        * state_fe_values.shape_value(i, q_point))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "adjoint");
          _zgrads.resize(n_q_points);
          //We don't need u so we don't search for state
          edc.GetGradsState("last_newton_solution", _zgrads);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_zgrads[q_point] * state_fe_values.shape_grad(i, q_point))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_UT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "tangent");
          _dugrads.resize(n_q_points);
          edc.GetGradsState("last_newton_solution", _dugrads);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_dugrads[q_point] * state_fe_values.shape_grad(i, q_point))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_UTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "adjoint_hessian");
          _dzgrads.resize(n_q_points);
          edc.GetGradsState("last_newton_solution", _dzgrads);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_dzgrads[q_point] * state_fe_values.shape_grad(i, q_point))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_Q(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "gradient");
          _zvalues.resize(n_q_points);
          edc.GetValuesState("adjoint", _zvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (-_zvalues[q_point]
                    * control_fe_values.shape_value(i, q_point))
                * control_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_QT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "tangent");
          _dqvalues.resize(n_q_points);
          edc.GetValuesControl("dq", _dqvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) +=
                scale
                    * (-_dqvalues[q_point]
                        * state_fe_values.shape_value(i, q_point))
                    * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_QTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "hessian");
          _dzvalues.resize(n_q_points);
          edc.GetValuesState("adjoint_hessian", _dzvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (-_dzvalues[q_point]
                    * control_fe_values.shape_value(i, q_point))
                * control_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_UU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }
      void
      ElementEquation_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }
      void
      ElementEquation_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }
      void
      ElementEquation_QQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }

      void
      ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "state");
          _fvalues.resize(n_q_points);
        }
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          _fvalues[q_point] = ((20. * M_PI * M_PI
              * sin(4. * M_PI * state_fe_values.quadrature_point(q_point)(0))
              - 1. / _alpha
                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
              * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1)));

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_fvalues[q_point] * state_fe_values.shape_value(i, q_point))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(i, j) += scale
                  * state_fe_values.shape_grad(i, q_point)
                  * state_fe_values.shape_grad(j, q_point)
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ControlElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(
              (this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
          _funcgradvalues.resize(n_q_points);
          edc.GetValuesControl("last_newton_solution", _funcgradvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_funcgradvalues[q_point]
                    * control_fe_values.shape_value(i, q_point))
                * control_fe_values.JxW(q_point);
          }
        }
      }

      void
      ControlElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & control_fe_values =
            edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(i, j) += scale * control_fe_values.shape_value(i,
                  q_point) * control_fe_values.shape_value(j, q_point)
                  * control_fe_values.JxW(q_point);
            }
          }
        }
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        if ((this->_problem_type == "adjoint")
            || (this->_problem_type == "state")
            || (this->_problem_type == "tangent")
            || (this->_problem_type == "adjoint_hessian")
            || (this->_problem_type == "hessian"))
          return update_values | update_gradients | update_quadrature_points;
        else if ((this->_problem_type == "gradient"))
          return update_values | update_quadrature_points;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetUpdateFlags");
      }

      unsigned int
      GetControlNBlocks() const
      {
        return 1;
      }
      unsigned int
      GetStateNBlocks() const
      {
        return 1;
      }
      std::vector<unsigned int>&
      GetControlBlockComponent()
      {
        return _block_components;
      }
      const std::vector<unsigned int>&
      GetControlBlockComponent() const
      {
        return _block_components;
      }
      std::vector<unsigned int>&
      GetStateBlockComponent()
      {
        return _block_components;
      }
      const std::vector<unsigned int>&
      GetStateBlockComponent() const
      {
        return _block_components;
      }

    private:
      vector<double> _qvalues;
      vector<double> _dqvalues;
      vector<double> _funcgradvalues;
      vector<double> _fvalues;
      vector<double> _uvalues;
      vector<Tensor<1, dealdim> > _ugrads;
      vector<double> _zvalues;
      vector<Tensor<1, dealdim> > _zgrads;
      vector<double> _duvalues;
      vector<Tensor<1, dealdim> > _dugrads;
      vector<double> _dzvalues;
      vector<Tensor<1, dealdim> > _dzgrads;

      vector<unsigned int> _block_components;
      double _alpha;
  };
#endif
