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

#ifndef LOCALPDE_H_
#define LOCALPDE_H_

#include <interfaces/pdeinterface.h>
#include "myfunctions.h"
#include <deal.II/base/numbers.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDELaplace : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalPDELaplace() :
    state_block_component_(1, 0)
  {
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    assert(this->problem_type_ == "state");

    ugrads_.resize(n_q_points, Tensor<1, dealdim>());
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Scalar velocities(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<1, 2> vgrads;
        vgrads.clear();
        vgrads[0] = ugrads_[q_point][0];
        vgrads[1] = ugrads_[q_point][1];

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
  StrongElementResidual(const EDC<DH, VECTOR, dealdim> &edc,
                        const EDC<DH, VECTOR, dealdim> &edc_w, double &sum, double scale)
  {
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);

    PI_h_z_.resize(n_q_points);
    lap_u_.resize(n_q_points);
    edc.GetLaplaciansState("state", lap_u_);
    edc_w.GetValuesState("weight_for_primal_residual", PI_h_z_);

    const FEValuesExtractors::Scalar velocities(0);

    //make sure the binding of the function has worked
    assert(this->ResidualModifier);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] = -ex_sol_.laplacian(
                              state_fe_values.quadrature_point(q_point));
        double res;
        res = fvalues_[q_point] + lap_u_[q_point];

        //Modify the residual as required by the error estimator
        this->ResidualModifier(res);

        sum += scale * (res * PI_h_z_[q_point])
               * state_fe_values.JxW(q_point);
      }
  }

  void
  StrongElementResidual_U(const EDC<DH, VECTOR, dealdim> &edc,
                          const EDC<DH, VECTOR, dealdim> &edc_w, double &sum, double scale)
  {
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);

    PI_h_z_.resize(n_q_points);
    lap_u_.resize(n_q_points);
    edc.GetLaplaciansState("adjoint_for_ee", lap_u_);
    edc_w.GetValuesState("weight_for_dual_residual", PI_h_z_);

    const FEValuesExtractors::Scalar velocities(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        double res;
        res = lap_u_[q_point];
        //Modify the residual as required by the error estimator
        this->ResidualModifier(res);

        sum += scale * (res * PI_h_z_[q_point])
               * state_fe_values.JxW(q_point);
      }
  }

  void
  StrongFaceResidual(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc,
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc_w,
    double &sum, double scale)
  {
    unsigned int n_q_points = fdc.GetNQPoints();
    ugrads_.resize(n_q_points, Tensor<1, dealdim>());
    ugrads_nbr_.resize(n_q_points, Tensor<1, dealdim>());
    PI_h_z_.resize(n_q_points);

    fdc.GetFaceGradsState("state", ugrads_);
    fdc.GetNbrFaceGradsState("state", ugrads_nbr_);
    fdc_w.GetFaceValuesState("weight_for_primal_residual", PI_h_z_);
    vector<double> jump(n_q_points);
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        jump[q] = (ugrads_nbr_[q][0] - ugrads_[q][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (ugrads_nbr_[q][1] - ugrads_[q][1])
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

        sum += scale * (res * PI_h_z_[q_point])
               * fdc.GetFEFaceValuesState().JxW(q_point);
      }
  }

  void
  StrongFaceResidual_U(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc,
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc_w,
    double &sum, double scale)
  {
    unsigned int n_q_points = fdc.GetNQPoints();
    ugrads_.resize(n_q_points, Tensor<1, dealdim>());
    ugrads_nbr_.resize(n_q_points, Tensor<1, dealdim>());
    PI_h_z_.resize(n_q_points);

    fdc.GetFaceGradsState("adjoint_for_ee", ugrads_);
    fdc.GetNbrFaceGradsState("adjoint_for_ee", ugrads_nbr_);
    fdc_w.GetFaceValuesState("weight_for_dual_residual", PI_h_z_);
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
        jump[q] = (ugrads_nbr_[q][0] - ugrads_[q][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (ugrads_nbr_[q][1] - ugrads_[q][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
      }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double res;
        res = f + jump[q_point];
        //Modify the residual as required by the error estimator
        this->ResidualModifier(res);

        sum += scale * (res * PI_h_z_[q_point])
               * fdc.GetFEFaceValuesState().JxW(q_point);
      }
  }

  void
  StrongBoundaryResidual(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc*/,
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc_w*/,
    double &sum, double /*scale*/)
  {
    sum = 0;
  }

  void
  StrongBoundaryResidual_U(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc*/,
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc_w*/,
    double &sum, double /*scale*/)
  {
    sum = 0;
  }

  void
  FaceEquation_U(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double/*scale_ico*/)
  {
  }

  void
  FaceMatrix(
    const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &/*fdc*/,
    FullMatrix<double> & /*local_matrix*/, double /*scale*/,
    double/*scale_ico*/)
  {
  }

  void
  ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double/*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    assert(this->problem_type_ == "adjoint_for_ee");
    zgrads_.resize(n_q_points, Tensor<1, dealdim>());
    //We don't need u so we don't search for state
    edc.GetGradsState("last_newton_solution", zgrads_);

    const FEValuesExtractors::Scalar velocities(0);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<1, 2> vgrads;
        vgrads.clear();
        vgrads[0] = zgrads_[q_point][0];
        vgrads[1] = zgrads_[q_point][1];
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
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    //unsigned int material_id = edc.GetMaterialId();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

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
  ElementMatrix_T(const EDC<DH, VECTOR, dealdim> &edc,
                  FullMatrix<double> &local_matrix, double scale,
                  double /*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    //unsigned int material_id = edc.GetMaterialId();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

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
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "state");
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);
    const FEValuesExtractors::Scalar velocities(0);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        fvalues_[q_point] = -ex_sol_.laplacian(
                              state_fe_values.quadrature_point(q_point));

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * fvalues_[q_point]
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
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_component_;
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

  vector<double> fvalues_;
  vector<double> PI_h_z_;
  vector<double> lap_u_;

  vector<Tensor<1, dealdim> > ugrads_;
  vector<Tensor<1, dealdim> > ugrads_nbr_;

  vector<Tensor<1, dealdim> > zgrads_;

  vector<unsigned int> state_block_component_;

  ExactSolution ex_sol_;
}
;
//**********************************************************************************

#endif

