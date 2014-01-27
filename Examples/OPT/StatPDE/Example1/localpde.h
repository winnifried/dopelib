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
  class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
  {
    public:
      LocalPDE(double alpha) :
          _block_components(1, 0)
      {
        _alpha = alpha;
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

      /******************************************************/
      void
      StrongElementResidual(const EDC<DH, VECTOR, dealdim>& edc,
          const EDC<DH, VECTOR, dealdim>& edc_w, double& sum, double scale)
      {
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        _qvalues.resize(n_q_points);
        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        _fvalues.resize(n_q_points);

        edc.GetLaplaciansState("state", _lap_u);
        edc.GetValuesControl("control", _qvalues);
        edc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        //make sure the binding of the function has worked
        assert(this->ResidualModifier);
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point] = ((20. * M_PI * M_PI
              * sin(4. * M_PI * state_fe_values.quadrature_point(q_point)(0))
              - 1. / _alpha
                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
              * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1)));

          double res;
          res = _qvalues[q_point] + _fvalues[q_point] + _lap_u[q_point];

          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }
      void
      StrongElementResidual_U(const EDC<DH, VECTOR, dealdim>& edc,
          const EDC<DH, VECTOR, dealdim>& edc_w, double& sum, double scale)
      {
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        _fvalues.resize(n_q_points);

        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        _uvalues.resize(n_q_points);

        edc.GetLaplaciansState("adjoint_for_ee", _lap_u);
        edc.GetValuesState("state", _uvalues);
        edc_w.GetValuesState("weight_for_dual_residual", _PI_h_z);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point] = (1.
              * sin(4 * M_PI * state_fe_values.quadrature_point(q_point)(0))
              + 5. * M_PI * M_PI
                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
              * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1));
          double res;
          res = _uvalues[q_point] - _fvalues[q_point] + _lap_u[q_point];
          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }
      void
      StrongElementResidual_Control(const EDC<DH, VECTOR, dealdim>& edc,
          const EDC<DH, VECTOR, dealdim>& edc_w, double& sum, double scale)
      {
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        _zvalues.resize(n_q_points);
        _qvalues.resize(n_q_points);

        edc.GetValuesControl("control", _qvalues);
        edc.GetLaplaciansState("adjoint_for_ee", _lap_u);
        edc.GetValuesState("adjoint_for_ee", _zvalues); //Same as z in this case!
        edc_w.GetValuesControl("weight_for_control_residual", _PI_h_z);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          double res;
          res = _alpha * _qvalues[q_point] + _zvalues[q_point];
          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }
      /******************************************************/

      void
      StrongFaceResidual(const FDC<DH, VECTOR, dealdim>& fdc,
          const FDC<DH, VECTOR, dealdim>& fdc_w, double& sum, double scale)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        _ugrads_nbr.resize(n_q_points, Tensor<1, dealdim>());
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetNbrFaceGradsState("state", _ugrads_nbr);
        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
        vector<double> jump(n_q_points);
        for (unsigned int q = 0; q < n_q_points; q++)
        {
          jump[q] = (_ugrads_nbr[q][0] - _ugrads[q][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + (_ugrads_nbr[q][1] - _ugrads[q][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }
        //make sure the binding of the function has worked
        assert(this->ResidualModifier);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          //Modify the residual as required by the error estimator
          double res;
          res = jump[q_point];
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }
      }

      void
      StrongFaceResidual_U(const FDC<DH, VECTOR, dealdim>& fdc,
          const FDC<DH, VECTOR, dealdim>& fdc_w, double& sum, double scale)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        _ugrads_nbr.resize(n_q_points, Tensor<1, dealdim>());
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("adjoint_for_ee", _ugrads);
        fdc.GetNbrFaceGradsState("adjoint_for_ee", _ugrads_nbr);
        fdc_w.GetFaceValuesState("weight_for_dual_residual", _PI_h_z);
        vector<double> jump(n_q_points);

        for (unsigned int q = 0; q < n_q_points; q++)
        {
          jump[q] = (_ugrads_nbr[q][0] - _ugrads[q][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + (_ugrads_nbr[q][1] - _ugrads[q][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          double res;
          res = jump[q_point];
          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }
      }

      void
      StrongFaceResidual_Control(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          const FDC<DH, VECTOR, dealdim>&/*fdc_w*/, double& sum, double)
      {
        sum = 0.;
      }
      /******************************************************/

      void
      StrongBoundaryResidual(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          const FDC<DH, VECTOR, dealdim>&/*fdc_w*/, double& sum, double)
      {
        sum = 0.;
      }

      void
      StrongBoundaryResidual_U(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          const FDC<DH, VECTOR, dealdim>&/*fdc_w*/, double& sum, double)
      {
        sum = 0.;
      }

      void
      StrongBoundaryResidual_Control(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          const FDC<DH, VECTOR, dealdim>&/*fdc_w*/, double& sum, double)
      {
        sum = 0.;
      }
      /******************************************************/

      UpdateFlags
      GetUpdateFlags() const
      {
        if ((this->_problem_type == "adjoint")
            || (this->_problem_type == "state")
            || (this->_problem_type == "tangent")
            || (this->_problem_type == "adjoint_hessian")
            || (this->_problem_type == "hessian")
            || (this->_problem_type == "adjoint_for_ee"))
          return update_values | update_gradients | update_quadrature_points;
        else if ((this->_problem_type == "error_evaluation"))
          return update_values | update_gradients | update_hessians
              | update_quadrature_points;
        else if ((this->_problem_type == "gradient"))
          return update_values | update_quadrature_points;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetUpdateFlags");
      }
      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_gradients | update_normal_vectors
            | update_quadrature_points;
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

    protected:

    private:
      vector<double> _qvalues;
      vector<double> _dqvalues;
      vector<double> _funcgradvalues;
      vector<double> _fvalues;
      vector<double> _uvalues;
      vector<double> _PI_h_z;
      vector<double> _lap_u;

      vector<Tensor<1, dealdim> > _ugrads;
      vector<double> _zvalues;
      vector<Tensor<1, dealdim> > _zgrads;
      vector<double> _duvalues;
      vector<Tensor<1, dealdim> > _dugrads;
      vector<double> _dzvalues;
      vector<Tensor<1, dealdim> > _dzgrads;
      vector<Tensor<1, dealdim> > _PI_h_z_grads;
      vector<Tensor<1, dealdim> > _ugrads_nbr;

      vector<unsigned int> _block_components;
      double _alpha;
  };
#endif
