/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
public:

  LocalPDE() :
    state_block_component_(1, 0)
  {

  }

  // Domain values for elements
  void
  ElementEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points);
    ugrads_.resize(n_q_points);

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale * (ugrads_[q_point] * phi_i_grads)
                               * state_fe_values.JxW(q_point);

          }
      }
  }

  void
  ElementMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix, double scale, double) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
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
                local_matrix(i, j) += scale * (phi_grads[j] * phi_grads[i])
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> & /*local_vector*/,
    double /*scale*/) override
  {
    assert(this->problem_type_ == "state");

  }

  void
  ElementTimeEquationExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> & /*local_vector*/,
    double /*scale*/) override
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector,
    double scale) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points);

    edc.GetValuesState("last_newton_solution", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (uvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeMatrixExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_matrix*/) override
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
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
  ///Error Estimation
  void
  StrongElementResidual(
    const EDC<DH, VECTOR, dealdim> &edc,
    const EDC<DH, VECTOR, dealdim> &edc_w,
    double &sum, double scale) override
  {
    if (this->GetTime() > 0.)
      {
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          edc.GetFEValuesState();

        fvalues_.resize(n_q_points);

        PI_h_z_.resize(n_q_points);
        lap_u_.resize(n_q_points);
        uvalues_.resize(n_q_points);
        uold_values_.resize(n_q_points);
        edc.GetLaplaciansState("state", lap_u_);
        edc.GetValuesState("state", uvalues_);
        edc.GetValuesState("last_time_state", uold_values_);
        edc_w.GetValuesState("weight_for_primal_residual", PI_h_z_);

        const FEValuesExtractors::Scalar velocities(0);

        //make sure the binding of the function has worked
        assert(this->ResidualModifier);
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = 0.;
            double res;
            res = fvalues_[q_point] + lap_u_[q_point] + (uvalues_[q_point]-uold_values_[q_point])/this->GetTimeStepSize();

            //Modify the residual as required by the error estimator
            this->ResidualModifier(res);

            sum += scale * (res * PI_h_z_[q_point])
                   * state_fe_values.JxW(q_point);
          }
      }
  }
  void
  StrongFaceResidual(
    const FDC<DH, VECTOR, dealdim> &fdc,
    const FDC<DH, VECTOR, dealdim> &fdc_w,
    double &sum, double scale) override
  {
    if (this->GetTime() > 0.)
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
  }

  void
  StrongBoundaryResidual(
    const FDC<DH, VECTOR, dealdim> &/*fdc*/,
    const FDC<DH, VECTOR, dealdim> &/*fdc_w*/,
    double &sum, double /*scale*/) override
  {
    sum = 0;
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    if (this->problem_type_ == "state" || this->problem_type_=="error_evaluation")
      return update_values | update_gradients | update_quadrature_points | update_hessians;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  UpdateFlags
  GetFaceUpdateFlags() const override
  {
    if (this->problem_type_ == "state" || this->problem_type_=="error_evaluation")
      return update_values | update_gradients | update_normal_vectors
             | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  unsigned int
  GetControlNBlocks() const override
  {
    return 1;
  }

  unsigned int
  GetStateNBlocks() const override
  {
    return 1;
  }

  std::vector<unsigned int> &
  GetControlBlockComponent() override
  {
    return control_block_component_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const override
  {
    return control_block_component_;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent() override
  {
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const override
  {
    return state_block_component_;
  }

private:
  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<double> uold_values_;
  vector<double> PI_h_z_;
  vector<double> lap_u_;

  vector<Tensor<1, dealdim> > ugrads_;
  vector<Tensor<1, dealdim> > ugrads_nbr_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_component_;

};
#endif
