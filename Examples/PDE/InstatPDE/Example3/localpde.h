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
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>

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

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("interest rate", "0.", Patterns::Double(0));
    param_reader.declare_entry("volatility_1", "0.", Patterns::Double(0));
    param_reader.declare_entry("volatility_2", "0.", Patterns::Double(0));
    param_reader.declare_entry("rho", ".0", Patterns::Double(-1, 1));
    param_reader.declare_entry("strike price", "0.", Patterns::Double(0));
    param_reader.declare_entry("expiration date", "1.0",
                               Patterns::Double(0));
  }

  LocalPDE(ParameterReader &param_reader) :
    state_block_component_(1, 0)
  {
    param_reader.SetSubsection("Local PDE parameters");
    rate_ = param_reader.get_double("interest rate");
    volatility_(0) = param_reader.get_double("volatility_1");
    volatility_(1) = param_reader.get_double("volatility_2");
    rho_ = param_reader.get_double("rho");
    strike_ = param_reader.get_double("strike price");
  }

  void
  ElementEquation(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
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

    Tensor<2, 2> CoeffMatrix;
    Tensor<1, 2> CoeffVector;
    const double correlation = 2 * rho_ * 1. / (1 + rho_ * rho_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        const double x = state_fe_values.quadrature_point(q_point)[0];
        const double y = state_fe_values.quadrature_point(q_point)[1];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            CoeffMatrix.clear();
            CoeffVector.clear();

            CoeffMatrix[0][0] = volatility_(0) * volatility_(0) * x * x;
            CoeffMatrix[1][0] = volatility_(1) * volatility_(0) * x * y
                                * correlation;
            CoeffMatrix[0][1] = volatility_(0) * volatility_(1) * y * x
                                * correlation;
            CoeffMatrix[1][1] = volatility_(1) * volatility_(1) * y * y;
            CoeffMatrix.operator*=(0.5);

            CoeffVector[0] = x
                             * (volatility_(0) * volatility_(0)
                                + 0.5 * correlation * volatility_(0) * volatility_(1)
                                - rate_);
            CoeffVector[1] = y
                             * (volatility_(1) * volatility_(1)
                                + 0.5 * correlation * volatility_(0) * volatility_(1)
                                - rate_);

            const double phi_i = state_fe_values.shape_value(i, q_point);
            const Tensor<1, 2> phi_i_grads = state_fe_values.shape_grad(i,
                                                                        q_point);

            local_vector(i) += scale
                               * ((CoeffMatrix * ugrads_[q_point]) * phi_i_grads
                                  + (CoeffVector * ugrads_[q_point]) * phi_i
                                  + rate_ * uvalues_[q_point] * phi_i)
                               * state_fe_values.JxW(q_point);

          }
      }
  }

  void
  ElementMatrix(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale, double)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    std::vector<double> phi(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_grads(n_dofs_per_element);

    Tensor<2, 2> CoeffMatrix;
    Tensor<1, 2> CoeffVector;
    const double correlation = 2 * rho_ * 1. / (1 + rho_ * rho_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi[k] = state_fe_values.shape_value(k, q_point);
            phi_grads[k] = state_fe_values.shape_grad(k, q_point);
          }

        const double x = state_fe_values.quadrature_point(q_point)[0];
        const double y = state_fe_values.quadrature_point(q_point)[1];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            CoeffMatrix.clear();
            CoeffVector.clear();

            CoeffMatrix[0][0] = volatility_(0) * volatility_(0) * x * x;
            CoeffMatrix[1][0] = volatility_(1) * volatility_(0) * x * y
                                * correlation;
            CoeffMatrix[0][1] = volatility_(0) * volatility_(1) * y * x
                                * correlation;
            CoeffMatrix[1][1] = volatility_(1) * volatility_(1) * y * y;
            CoeffMatrix.operator *=(0.5);

            CoeffVector[0] = x
                             * (volatility_(0) * volatility_(0)
                                + 0.5 * correlation * volatility_(0) * volatility_(1)
                                - rate_);
            CoeffVector[1] = y
                             * (volatility_(1) * volatility_(1)
                                + 0.5 * correlation * volatility_(0) * volatility_(1)
                                - rate_);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i, j) += scale
                                      * ((phi_grads[j] * CoeffMatrix) * phi_grads[i]
                                         + (CoeffVector * phi_grads[j]) * phi[i]
                                         + rate_ * phi[j] * phi[i]) * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(const ElementDataContainer<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> & /*local_vector*/,
                       double /*scale*/)
  {
    assert(this->problem_type_ == "state");
    //i.e. f=0
  }

  void
  ElementTimeEquationExplicit(const ElementDataContainer<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
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
  ElementTimeMatrixExplicit(
    const ElementDataContainer<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
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
  BoundaryEquation(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                   double /*scale_ico*/)
  {

    assert(this->problem_type_ == "state");

  }

  void
  BoundaryRightHandSide(
    const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
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

  std::vector<unsigned int> &
  GetControlBlockComponent()
  {
    return control_block_component_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return control_block_component_;
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
  vector<double> uvalues_;

  vector<Tensor<1, dealdim> > ugrads_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_component_;

  double rate_, rho_, strike_;
  Point<dealdim> volatility_;

};
#endif
