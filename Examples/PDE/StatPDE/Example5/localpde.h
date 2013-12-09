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

#ifndef _LOCALPDE_H_
#define _LOCALPDE_H_

#include "pdeinterface.h"
#include "myfunctions.h"
#include <deal.II/base/numbers.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDELaplace : public PDEInterface<CDC, FDC, DH, VECTOR, dealdim>
  {
    public:
      LocalPDELaplace() :
          _state_block_components(1, 0)
      {
      }

      void
      ElementEquation(const CDC<DH, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_vector, double scale,
          double/*scale_ico*/)
      {
        unsigned int n_dofs_per_element = cdc.GetNDoFsPerElement();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        assert(this->_problem_type == "state");

        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        cdc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> vgrads;
          vgrads.clear();
          vgrads[0] = _ugrads[q_point][0];
          vgrads[1] = _ugrads[q_point][1];

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);

            local_vector(i) += scale * (vgrads * phi_i_grads_v)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      StrongElementResidual(const CDC<DH, VECTOR, dealdim>& cdc,
          const CDC<DH, VECTOR, dealdim>& cdc_w, double& sum, double scale)
      {
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        _fvalues.resize(n_q_points);

        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        cdc.GetLaplaciansState("state", _lap_u);
        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        const FEValuesExtractors::Scalar velocities(0);

        //make sure the binding of the function has worked
        assert(this->ResidualModifier);
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          _fvalues[q_point] = -_ex_sol.laplacian(
              state_fe_values.quadrature_point(q_point));
          double res;
          res = _fvalues[q_point] + _lap_u[q_point];

          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongElementResidual_U(const CDC<DH, VECTOR, dealdim>& cdc,
          const CDC<DH, VECTOR, dealdim>& cdc_w, double& sum, double scale)
      {
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        _fvalues.resize(n_q_points);

        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        cdc.GetLaplaciansState("adjoint_for_ee", _lap_u);
        cdc_w.GetValuesState("weight_for_dual_residual", _PI_h_z);

        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

          double res;
          res = _lap_u[q_point];
          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
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
      StrongFaceResidual_U(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        _ugrads_nbr.resize(n_q_points, Tensor<1, dealdim>());
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("adjoint_for_ee", _ugrads);
        fdc.GetNbrFaceGradsState("adjoint_for_ee", _ugrads_nbr);
        fdc_w.GetFaceValuesState("weight_for_dual_residual", _PI_h_z);
        vector<double> jump(n_q_points);
        double f = 0;

        unsigned int material_id = fdc.GetMaterialId();
        unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
        if ((material_id == 1 && material_id_neighbor == 2)
            || (material_id == 2 && material_id_neighbor == 1))
        {

          f = 1;

        }

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
          res = f + jump[q_point];
          //Modify the residual as required by the error estimator
          this->ResidualModifier(res);

          sum += scale * (res * _PI_h_z[q_point])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }
      }

      void
      StrongBoundaryResidual(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc*/,
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc_w*/,
          double& sum, double /*scale*/)
      {
        sum = 0;
      }

      void
      StrongBoundaryResidual_U(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc*/,
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc_w*/,
          double& sum, double /*scale*/)
      {
        sum = 0;
      }

      void
      FaceEquation_U(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double/*scale_ico*/)
      {
      }

      void
      FaceMatrix(
          const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&/*fdc*/,
          FullMatrix<double> & /*local_matrix*/, double /*scale*/,
          double/*scale_ico*/)
      {
      }

      void
      ElementEquation_U(const CDC<DH, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_vector, double scale,
          double/*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_element = cdc.GetNDoFsPerElement();
        unsigned int n_q_points = cdc.GetNQPoints();

        assert(this->_problem_type == "adjoint_for_ee");
        _zgrads.resize(n_q_points, Tensor<1, dealdim>());
        //We don't need u so we don't search for state
        cdc.GetGradsState("last_newton_solution", _zgrads);

        const FEValuesExtractors::Scalar velocities(0);
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> vgrads;
          vgrads.clear();
          vgrads[0] = _zgrads[q_point][0];
          vgrads[1] = _zgrads[q_point][1];
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            local_vector(i) += scale * vgrads * phi_i_grads_v
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementMatrix(const CDC<DH, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_matrix, double scale,
          double/*scale_ico*/)
      {
        unsigned int n_dofs_per_element = cdc.GetNDoFsPerElement();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        const FEValuesExtractors::Scalar velocities(0);

        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {

              local_matrix(i, j) += scale * phi_grads_v[j]
                  * phi_grads_v[i] * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ElementMatrix_T(const CDC<DH, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_matrix, double scale,
          double /*scale_ico*/)
      {
        unsigned int n_dofs_per_element = cdc.GetNDoFsPerElement();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        const FEValuesExtractors::Scalar velocities(0);

        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {

              local_matrix(i, j) += scale * phi_grads_v[j]
                  * phi_grads_v[i] * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ElementRightHandSide(const CDC<DH, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "state");
        unsigned int n_dofs_per_element = cdc.GetNDoFsPerElement();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        _fvalues.resize(n_q_points);
        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          _fvalues[q_point] = -_ex_sol.laplacian(
              state_fe_values.quadrature_point(q_point));

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * _fvalues[q_point]
                * state_fe_values[velocities].value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        } //endfor qpoint
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_gradients | update_hessians
            | update_quadrature_points;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_gradients | update_normal_vectors
            | update_quadrature_points;
      }

      unsigned int
      GetStateNBlocks() const
      {
        return 1;
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
      bool
      HasFaces() const
      {
        return false;
      }
      bool
      HasInterfaces() const
      {
        return false;
      }
    private:

      vector<double> _fvalues;
      vector<double> _PI_h_z;
      vector<double> _lap_u;

      vector<Tensor<1, dealdim> > _ugrads;
      vector<Tensor<1, dealdim> > _PI_h_z_grads;
      vector<Tensor<1, dealdim> > _ugrads_nbr;

      vector<Tensor<1, dealdim> > _zgrads;

      vector<unsigned int> _state_block_components;

      ExactSolution _ex_sol;
  }
  ;
//**********************************************************************************

#endif

