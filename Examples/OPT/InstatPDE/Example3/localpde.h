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
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
  {
    public:

      LocalPDE() :
          _my_time(0), _state_block_components(1, 0), _block_components(1, 0)
      {

      }

      // Domain values for elements
      void
      ElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _ugrads.resize(n_q_points);
	_qvalues.reinit(1);
	_fvalues.resize(n_q_points);

	edc.GetParamValues("control", _qvalues);
	edc.GetGradsState("last_newton_solution", _ugrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
	  _fvalues[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                q_point);

            local_vector(i) += scale
                * ((_ugrads[q_point] * phi_i_grads)
		   - _qvalues(0) * _fvalues[q_point]* phi_i)
                * state_fe_values.JxW(q_point);
          }
        }
      }
      // Domain values for elements
      void
      ElementEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _zgrads.resize(n_q_points);

         edc.GetGradsState("last_newton_solution", _zgrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                q_point);

            local_vector(i) += scale
                * (_zgrads[q_point] * phi_i_grads)
                * state_fe_values.JxW(q_point);
          }
        }
      }
      // Domain values for elements
      void
      ElementEquation_UT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "tangent");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _dugrads.resize(n_q_points);

        edc.GetGradsState("last_newton_solution", _dugrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                q_point);

            local_vector(i) += scale
	      * (_dugrads[q_point] * phi_i_grads)
	      * state_fe_values.JxW(q_point);
          }
        }
      }
      // Domain values for elements
      void
      ElementEquation_UTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _dzgrads.resize(n_q_points);

        edc.GetGradsState("last_newton_solution", _dzgrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                q_point);

            local_vector(i) += scale
                * (_dzgrads[q_point] * phi_i_grads)
                * state_fe_values.JxW(q_point);
          }
        }
      }
      // Domain values for elements
      void
	ElementEquation_UU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
			   dealii::Vector<double> &/*local_vector*/, double /*scale*/,
			   double /*scale_ico*/)
      {
        
      }

      void
      ElementEquation_Q(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
	unsigned int n_q_points = edc.GetNQPoints();
	
	assert(this->_problem_type == "gradient");
	_fvalues.resize(n_q_points);
	_zvalues.resize(n_q_points);
	edc.GetValuesState("adjoint", _zvalues);


        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
	  _fvalues[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
	  local_vector(0) -= scale
	    * _zvalues[q_point]
	    * _fvalues[q_point]
	    * state_fe_values.JxW(q_point);
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
          _fvalues.resize(n_q_points);
	  _dqvalues.reinit(1);
          edc.GetParamValues("dq", _dqvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
	  _fvalues[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) -=
                scale
	      *_dqvalues(0)
	      * _fvalues[q_point]
	      * state_fe_values.shape_value(i, q_point)
	      * state_fe_values.JxW(q_point);
          }
        }
      }
      void
      ElementEquation_QTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          assert(this->_problem_type == "hessian");
          _fvalues.resize(n_q_points);
	  _dzvalues.resize(n_q_points);
          edc.GetValuesState("adjoint_hessian", _dzvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
	  _fvalues[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
	  local_vector(0) -= scale
	    *_dzvalues[q_point]
	    * _fvalues[q_point]
	    * state_fe_values.JxW(q_point);
        }
      }
      void
      ElementEquation_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
      }
      void
      ElementEquation_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
      }
      void
      ElementEquation_QQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double scale, double)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        std::vector<Tensor<1, dealdim> > phi_grads(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads[k] = state_fe_values.shape_grad(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(i, j) += scale
                  * (phi_grads[j] * phi_grads[i])
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
	ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& /*edc*/,
			     dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
    
      }

      void
      ElementTimeEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _uvalues.resize(n_q_points);

        edc.GetValuesState("last_newton_solution", _uvalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (_uvalues[q_point] * phi_i)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementTimeEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "adjoint");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _zvalues.resize(n_q_points);

        edc.GetValuesState("last_newton_solution", _zvalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (_zvalues[q_point] * phi_i)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementTimeEquation_UT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "tangent");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _duvalues.resize(n_q_points);

        edc.GetValuesState("last_newton_solution", _duvalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (_duvalues[q_point] * phi_i)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementTimeEquation_UTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "adjoint_hessian");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _dzvalues.resize(n_q_points);

        edc.GetValuesState("last_newton_solution", _dzvalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (_dzvalues[q_point] * phi_i)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementTimeMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        std::vector<double> phi(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi[k] = state_fe_values.shape_value(k, q_point);
          }
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(j, i) += (phi[i] * phi[j])
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &, double)
      {
      }
      void
      ElementTimeEquationExplicit_U(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &, double)
      {
      }
      void
      ElementTimeEquationExplicit_UT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &, double)
      {
      }
      void
      ElementTimeEquationExplicit_UTT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &, double)
      {
      }
      void
      ElementTimeEquationExplicit_UU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &, double)
      {
      }
      void
      ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          FullMatrix<double> &/*local_matrix*/)
      {
      }

      void
      ControlElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        {
          assert(
              (this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
          _funcgradvalues.reinit(1);
          edc.GetParamValues("last_newton_solution", _funcgradvalues);
        }

	local_vector(0) += scale * _funcgradvalues(0);
      }

      void
	ControlElementMatrix(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          FullMatrix<double> &local_matrix, double scale)
      {
	assert(local_matrix.m() == local_matrix.n());
	assert(local_matrix.m() == 1);
        for (unsigned int i = 0; i < local_matrix.m(); i++)
        {
          local_matrix(i, i) += scale * 1.;
        }
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        if (this->_problem_type == "state" || this->_problem_type == "adjoint"
            || this->_problem_type == "adjoint_hessian"
            || this->_problem_type == "tangent")
          return update_values | update_gradients | update_quadrature_points;
        else if (this->_problem_type == "gradient"
            || this->_problem_type == "hessian")
          return update_values | update_quadrature_points;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetUpdateFlags");
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        if (this->_problem_type == "state" || this->_problem_type == "adjoint"
            || this->_problem_type == "adjoint_hessian"
            || this->_problem_type == "tangent"
            || this->_problem_type == "gradient"
            || this->_problem_type == "hessian")
          return update_default;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetFaceUpdateFlags");
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
        return _state_block_components;
      }
      const std::vector<unsigned int>&
      GetStateBlockComponent() const
      {
        return _state_block_components;
      }

      void
      SetTime(double t) const
      {
        _my_time = t;
      }

    private:
      vector<double> _fvalues;
      vector<double> _uvalues;
      Vector<double> _qvalues;
      Vector<double> _dqvalues;
      vector<double> _zvalues;
      vector<double> _dzvalues;
      vector<double> _duvalues;
      Vector<double> _funcgradvalues;
      mutable double _my_time;

      vector<Tensor<1, dealdim> > _ugrads;
      vector<Tensor<1, dealdim> > _zgrads;
      vector<Tensor<1, dealdim> > _dugrads;
      vector<Tensor<1, dealdim> > _dzgrads;

      vector<unsigned int> _state_block_components;
      vector<unsigned int> _block_components;
  };
#endif
