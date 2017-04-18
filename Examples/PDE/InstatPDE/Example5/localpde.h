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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>

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
    state_block_component_(1, 0)
  {

  }

  // Domain values for elements
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
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

            const double phi_i = state_fe_values.shape_value(i, q_point);
            const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i,
                                                   q_point);

            local_vector(i) += scale
                               * ((ugrads_[q_point] * phi_i_grads)
                                  + uvalues_[q_point] * uvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);

          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale, double)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    edc.GetValuesState("last_newton_solution", uvalues_);

    std::vector<double> phi_values(n_dofs_per_element);
    std::vector<Tensor<1, dealdim> > phi_grads(n_dofs_per_element);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_values[k] = state_fe_values.shape_value(k, q_point);
            phi_grads[k] = state_fe_values.shape_grad(k, q_point);
          }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i, j) += scale
                                      * ((phi_grads[j] * phi_grads[i])
                                         + 2 * uvalues_[q_point] * phi_values[j] * phi_values[i])
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector,
                       double scale)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    RightHandSideFunction fvalues;
    fvalues.SetTime(this->GetTime());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Point<2> quadrature_point = fe_values.quadrature_point(q_point);
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {

            local_vector(i) += scale * fvalues.value(quadrature_point)
                               * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
          }
      }

  }

  void
  ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector,
                      double scale)
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
  ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                    FullMatrix<double> &local_matrix)
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

  // Values for boundary integrals
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/,
                   double /*scale*/,
                   double /*scale_ico*/)
  {

    assert(this->problem_type_ == "state");

  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    if (this->problem_type_ == "state")
      return update_values | update_gradients | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    if (this->problem_type_ == "state")
      return update_values | update_gradients | update_normal_vectors
             | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
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
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_component_;
  }

private:
  vector<double> fvalues_;
  vector<double> uvalues_;

  vector<Tensor<1, dealdim> > ugrads_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_components_;

};
#endif
