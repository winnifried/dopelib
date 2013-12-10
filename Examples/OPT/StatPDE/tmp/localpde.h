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
#include "myfunctions.h"

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
          _control_block_components(1, 0), _state_block_components(2, 0)
      {

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
          _qvalues.resize(n_q_points, Vector<double>(1));
          _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(2));

          //Getting q
          edc.GetValuesControl("control", _qvalues);
          //Geting u
          edc.GetGradsState("last_newton_solution", _ugrads);
        }
        const FEValuesExtractors::Vector displacements(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> u_grad;
          u_grad.clear();
          u_grad[0][0] = _ugrads[q_point][0][0];
          u_grad[0][1] = _ugrads[q_point][0][1];
          u_grad[1][0] = _ugrads[q_point][1][0];
          u_grad[1][1] = _ugrads[q_point][1][1];

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_grads =
                state_fe_values[displacements].gradient(i, q_point);

            local_vector(i) += scale * _qvalues[q_point](0) * 0.25
                * scalar_product(u_grad + transpose(u_grad),
                    phi_grads + transpose(phi_grads))
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
          _qvalues.resize(n_q_points, Vector<double>(1));
          _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(2));

          //Getting q
          edc.GetValuesControl("control", _qvalues);
          //Geting u
          edc.GetGradsState("last_newton_solution", _zgrads);
        }
        const FEValuesExtractors::Vector displacements(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> z_grad;
          z_grad.clear();
          z_grad[0][0] = _zgrads[q_point][0][0];
          z_grad[0][1] = _zgrads[q_point][0][1];
          z_grad[1][0] = _zgrads[q_point][1][0];
          z_grad[1][1] = _zgrads[q_point][1][1];

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_grads =
                state_fe_values[displacements].gradient(i, q_point);

            local_vector(i) += scale * _qvalues[q_point](0) * 0.25
                * scalar_product(phi_grads + transpose(phi_grads),
                    z_grad + transpose(z_grad)) * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_UT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "tangent");
      }

      void
      ElementEquation_UTT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
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
          _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(2));
          _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(2));

          //Geting u
          edc.GetGradsState("adjoint", _zgrads);
          edc.GetGradsState("state", _ugrads);
        }
        const FEValuesExtractors::Vector displacements(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> z_grad;
          z_grad.clear();
          z_grad[0][0] = _zgrads[q_point][0][0];
          z_grad[0][1] = _zgrads[q_point][0][1];
          z_grad[1][0] = _zgrads[q_point][1][0];
          z_grad[1][1] = _zgrads[q_point][1][1];
          Tensor<2, 2> u_grad;
          u_grad.clear();
          u_grad[0][0] = _ugrads[q_point][0][0];
          u_grad[0][1] = _ugrads[q_point][0][1];
          u_grad[1][0] = _ugrads[q_point][1][0];
          u_grad[1][1] = _ugrads[q_point][1][1];

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * control_fe_values.shape_value(i, q_point) * 0.25
                * scalar_product(u_grad + transpose(u_grad),
                    z_grad + transpose(z_grad))
                * control_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementEquation_QT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "tangent");
      }

      void
      ElementEquation_QTT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }

      void
      ElementEquation_UU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }
      void
      ElementEquation_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }
      void
      ElementEquation_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }
      void
      ElementEquation_QQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }
      void
      ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/)
      {
        {
          assert(this->_problem_type == "state");
        }
      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double /*scale*/,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        {
          edc.GetValuesControl("control", _qvalues);
        }
        const FEValuesExtractors::Vector displacements(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_grads =
                state_fe_values[displacements].gradient(i, q_point);
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              const Tensor<2, 2> psi_grads =
                  state_fe_values[displacements].gradient(j, q_point);

              local_matrix(i, j) += _qvalues[q_point](0) * 0.25
                  * scalar_product(psi_grads + transpose(psi_grads),
                      phi_grads + transpose(phi_grads))
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
          _funcgradvalues.resize(n_q_points, Vector<double>(1));
          edc.GetValuesControl("last_newton_solution", _funcgradvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                * (_funcgradvalues[q_point](0)
                    * control_fe_values.shape_value(i, q_point))
                * control_fe_values.JxW(q_point);
          }
        }
      }

      void
      ControlElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix)
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
              local_matrix(i, j) += control_fe_values.shape_value(i,
                  q_point) * control_fe_values.shape_value(j, q_point)
                  * control_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryEquation(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();
        {
          _fvalues.resize(2);
        }

        if (color == 3)
        {
          const FEValuesExtractors::Scalar comp_0(0);
          const FEValuesExtractors::Scalar comp_1(1);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            MyFunctions::Forces(_fvalues,
                state_fe_face_values.quadrature_point(q_point)(0),
                state_fe_face_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) -= scale
                  * (_fvalues[0]
                      * state_fe_face_values[comp_0].value(i, q_point)
                      + _fvalues[1]
                          * state_fe_face_values[comp_1].value(i, q_point))
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryEquation_U(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_UT(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_UTT(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_Q(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_QT(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_QTT(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_UU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_QU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_UQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryEquation_QQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double/*scale*/)
      {
      }

      void
      BoundaryMatrix(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
		     dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/,
		     double /*scale_ico*/)
      {
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
          return update_values | update_gradients | update_quadrature_points;
        else if ((this->_problem_type == "hessian_inverse")
            || (this->_problem_type == "global_constraint_gradient")
            || (this->_problem_type == "global_constraint_hessian"))
          return update_values | update_quadrature_points;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetUpdateFlags");
      }
      UpdateFlags
      GetFaceUpdateFlags() const
      {
        if ((this->_problem_type == "adjoint")
            || (this->_problem_type == "state")
            || (this->_problem_type == "tangent")
            || (this->_problem_type == "adjoint_hessian")
            || (this->_problem_type == "hessian"))
          return update_values | update_quadrature_points;
        else if ((this->_problem_type == "hessian_inverse")
            || (this->_problem_type == "gradient")
            || (this->_problem_type == "global_constraint_gradient")
            || (this->_problem_type == "global_constraint_hessian"))
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
        return _control_block_components;
      }
      const std::vector<unsigned int>&
      GetControlBlockComponent() const
      {
        return _control_block_components;
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

    private:
      vector<double> _fvalues;

      vector<Vector<double> > _qvalues;
      vector<Vector<double> > _dqvalues;
      vector<Vector<double> > _funcgradvalues;

      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;
      vector<vector<Tensor<1, dealdim> > > _zgrads;

      vector<unsigned int> _control_block_components;
      vector<unsigned int> _state_block_components;
  };
#endif