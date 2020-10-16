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
#include "my_functions.h"

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
    state_block_components_(1, 0), control_block_components_(1, 0)
  {

  }

  // Domain values for elements
  void
  ElementEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    ugrads_.resize(n_q_points);
    qvalues_.reinit(1);
    fvalues_.resize(n_q_points);

    edc.GetParamValues("control", qvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale
                               * ((ugrads_[q_point] * phi_i_grads)
                                  - qvalues_(0) * fvalues_[q_point]* phi_i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  // Domain values for elements
  void
  ElementEquation_U(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    zgrads_.resize(n_q_points);

    edc.GetGradsState("last_newton_solution", zgrads_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale
                               * (zgrads_[q_point] * phi_i_grads)
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  // Domain values for elements
  void
  ElementEquation_UT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "tangent");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    dugrads_.resize(n_q_points);

    edc.GetGradsState("last_newton_solution", dugrads_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale
                               * (dugrads_[q_point] * phi_i_grads)
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  // Domain values for elements
  void
  ElementEquation_UTT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    dzgrads_.resize(n_q_points);

    edc.GetGradsState("last_newton_solution", dzgrads_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale
                               * (dzgrads_[q_point] * phi_i_grads)
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  // Domain values for elements
  void
  ElementEquation_UU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {

  }

  void
  ElementEquation_Q(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();

    assert(this->problem_type_ == "gradient");
    fvalues_.resize(n_q_points);
    zvalues_.resize(n_q_points);
    edc.GetValuesState("adjoint", zvalues_);


    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
        local_vector(0) -= scale
                           * zvalues_[q_point]
                           * fvalues_[q_point]
                           * state_fe_values.JxW(q_point);
      }
  }
  void
  ElementEquation_QT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "tangent");
      fvalues_.resize(n_q_points);
      dqvalues_.reinit(1);
      edc.GetParamValues("dq", dqvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) -=
              scale
              *dqvalues_(0)
              * fvalues_[q_point]
              * state_fe_values.shape_value(i, q_point)
              * state_fe_values.JxW(q_point);
          }
      }
  }
  void
  ElementEquation_QTT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "hessian");
      fvalues_.resize(n_q_points);
      dzvalues_.resize(n_q_points);
      edc.GetValuesState("adjoint_hessian", dzvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point] =  my::f(state_fe_values.quadrature_point(q_point));
        local_vector(0) -= scale
                           *dzvalues_[q_point]
                           * fvalues_[q_point]
                           * state_fe_values.JxW(q_point);
      }
  }
  void
  ElementEquation_QU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }
  void
  ElementEquation_UQ(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }
  void
  ElementEquation_QQ(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  ElementMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix, double scale, double)
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
                local_matrix(i, j) += scale
                                      * (phi_grads[j] * phi_grads[i])
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementTimeEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
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
  ElementTimeEquation_U(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "adjoint");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    zvalues_.resize(n_q_points);

    edc.GetValuesState("last_newton_solution", zvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (zvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeEquation_UT(
    const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "tangent");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    duvalues_.resize(n_q_points);

    edc.GetValuesState("last_newton_solution", duvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (duvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeEquation_UTT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "adjoint_hessian");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    dzvalues_.resize(n_q_points);

    edc.GetValuesState("last_newton_solution", dzvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const double phi_i = state_fe_values.shape_value(i, q_point);
            local_vector(i) += scale * (dzvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
                    FullMatrix<double> &local_matrix)
  {
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

  void
  ElementTimeEquationExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &, double)
  {
  }
  void
  ElementTimeEquationExplicit_U(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &, double)
  {
  }
  void
  ElementTimeEquationExplicit_UT(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &, double)
  {
  }
  void
  ElementTimeEquationExplicit_UTT(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &, double)
  {
  }
  void
  ElementTimeEquationExplicit_UU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &, double)
  {
  }
  void
  ElementTimeMatrixExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_matrix*/)
  {
  }

  void
  ControlElementEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
  {
    {
      assert(
        (this->problem_type_ == "gradient")||(this->problem_type_ == "hessian"));
      funcgradvalues_.reinit(1);
      edc.GetParamValues("last_newton_solution", funcgradvalues_);
    }

    local_vector(0) += scale * funcgradvalues_(0);
  }

  void
  ControlElementMatrix(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
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
    if (this->problem_type_ == "state" || this->problem_type_ == "adjoint"
        || this->problem_type_ == "adjoint_hessian"
        || this->problem_type_ == "tangent")
      return update_values | update_gradients | update_quadrature_points;
    else if (this->problem_type_ == "gradient"
             || this->problem_type_ == "hessian")
      return update_values | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    if (this->problem_type_ == "state" || this->problem_type_ == "adjoint"
        || this->problem_type_ == "adjoint_hessian"
        || this->problem_type_ == "tangent"
        || this->problem_type_ == "gradient"
        || this->problem_type_ == "hessian")
      return update_default;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
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

  std::vector<unsigned int> &
  GetControlBlockComponent()
  {
    return control_block_components_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return control_block_components_;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return state_block_components_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_components_;
  }

private:
  vector<double> fvalues_;
  vector<double> uvalues_;
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  vector<double> zvalues_;
  vector<double> dzvalues_;
  vector<double> duvalues_;
  Vector<double> funcgradvalues_;

  vector<Tensor<1, dealdim> > ugrads_;
  vector<Tensor<1, dealdim> > zgrads_;
  vector<Tensor<1, dealdim> > dugrads_;
  vector<Tensor<1, dealdim> > dzgrads_;

  vector<unsigned int> state_block_components_;
  vector<unsigned int> control_block_components_;
};
#endif
