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
#include "myfunctions.h"

using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool HP, template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<bool HP, template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  bool HP, template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<EDC, FDC, HP, DH, VECTOR, dealdim>
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
    control_block_components_(1, 0), state_block_components_(2, 0)
  {

  }

  void
  SetP(double p)
  {
    p_ = p;
  }

  void
  ElementEquation(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      //Reading data
      assert(this->problem_type_ == "state");
      qvalues_.resize(n_q_points, Vector<double>(1));
      ugrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(2));

      //Getting q
      edc.GetValuesControl("control", qvalues_);
      //Geting u
      edc.GetGradsState("last_newton_solution", ugrads_);
    }
    const FEValuesExtractors::Vector displacements(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> u_grad;
        u_grad.clear();
        u_grad[0][0] = ugrads_[q_point][0][0];
        u_grad[0][1] = ugrads_[q_point][0][1];
        u_grad[1][0] = ugrads_[q_point][1][0];
        u_grad[1][1] = ugrads_[q_point][1][1];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_grads =
              state_fe_values[displacements].gradient(i, q_point);

            local_vector(i) += scale * pow(qvalues_[q_point](0), p_) * 0.25
                               * scalar_product(u_grad + transpose(u_grad),
                                                phi_grads + transpose(phi_grads))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_U(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "adjoint");
      qvalues_.resize(n_q_points, Vector<double>(1));
      zgrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(2));

      //Getting q
      edc.GetValuesControl("control", qvalues_);
      //Geting u
      edc.GetGradsState("last_newton_solution", zgrads_);
    }
    const FEValuesExtractors::Vector displacements(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> z_grad;
        z_grad.clear();
        z_grad[0][0] = zgrads_[q_point][0][0];
        z_grad[0][1] = zgrads_[q_point][0][1];
        z_grad[1][0] = zgrads_[q_point][1][0];
        z_grad[1][1] = zgrads_[q_point][1][1];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_grads =
              state_fe_values[displacements].gradient(i, q_point);

            local_vector(i) += scale * pow(qvalues_[q_point](0), p_) * 0.25
                               * scalar_product(phi_grads + transpose(phi_grads),
                                                z_grad + transpose(z_grad)) * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "tangent");
  }

  void
  ElementEquation_UTT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  ElementEquation_Q(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "gradient");
      zgrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(2));
      ugrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(2));
      //Getting q
      edc.GetValuesControl("control", qvalues_);
      //Geting u
      edc.GetGradsState("adjoint", zgrads_);
      edc.GetGradsState("state", ugrads_);
    }
    const FEValuesExtractors::Vector displacements(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> z_grad;
        z_grad.clear();
        z_grad[0][0] = zgrads_[q_point][0][0];
        z_grad[0][1] = zgrads_[q_point][0][1];
        z_grad[1][0] = zgrads_[q_point][1][0];
        z_grad[1][1] = zgrads_[q_point][1][1];
        Tensor<2, 2> u_grad;
        u_grad.clear();
        u_grad[0][0] = ugrads_[q_point][0][0];
        u_grad[0][1] = ugrads_[q_point][0][1];
        u_grad[1][0] = ugrads_[q_point][1][0];
        u_grad[1][1] = ugrads_[q_point][1][1];

        if (p_ == 1.)
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) += scale
                                 * control_fe_values.shape_value(i, q_point) * 0.25
                                 * scalar_product(u_grad + transpose(u_grad),
                                                  z_grad + transpose(z_grad))
                                 * control_fe_values.JxW(q_point);
            }
        else
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              local_vector(i) += scale * p_
                                 * pow(qvalues_[q_point](0), p_ - 1.)
                                 * control_fe_values.shape_value(i, q_point) * 0.25
                                 * scalar_product(u_grad + transpose(u_grad),
                                                  z_grad + transpose(z_grad))
                                 * control_fe_values.JxW(q_point);
            }
      }
  }

  void
  ElementEquation_QT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "tangent");
  }

  void
  ElementEquation_QTT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }

  void
  ElementEquation_UU(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_QU(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_UQ(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }
  void
  ElementEquation_QQ(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }
  void
  ElementRightHandSide(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> & /*edc*/,
#else
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementMatrix(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    FullMatrix<double> &local_matrix, double /*scale*/,
    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      edc.GetValuesControl("control", qvalues_);
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

                local_matrix(i, j) += pow(qvalues_[q_point](0), p_) * 0.25
                                      * scalar_product(psi_grads + transpose(psi_grads),
                                                       phi_grads + transpose(phi_grads))
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ControlElementEquation(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(
        (this->problem_type_ == "gradient")||(this->problem_type_ == "hessian"));
      funcgradvalues_.resize(n_q_points, Vector<double>(1));
      edc.GetValuesControl("last_newton_solution", funcgradvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (funcgradvalues_[q_point](0)
                                  * control_fe_values.shape_value(i, q_point))
                               * control_fe_values.JxW(q_point);
          }
      }
  }

  void
  ControlElementMatrix(
#if DEAL_II_VERSION_GTE(9,3,0)
    const EDC<HP, DH, VECTOR, dealdim> &edc,
#else
    const EDC<DH, VECTOR, dealdim> &edc,
#endif
    FullMatrix<double> &local_matrix, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
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

  void
  BoundaryEquation(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> &fdc,
#else
    const FDC<DH, VECTOR, dealdim> &fdc,
#endif
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    {
      fvalues_.resize(2);
    }

    if (color == 3)
      {
        const FEValuesExtractors::Scalar comp_0(0);
        const FEValuesExtractors::Scalar comp_1(1);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            MyFunctions::Forces(fvalues_,
                                state_fe_face_values.quadrature_point(q_point)(0),
                                state_fe_face_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) -= scale
                                   * (fvalues_[0]
                                      * state_fe_face_values[comp_0].value(i, q_point)
                                      + fvalues_[1]
                                      * state_fe_face_values[comp_1].value(i, q_point))
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_U(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_UT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_UTT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_Q(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_QT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_QTT(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_UU(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_QU(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_UQ(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryEquation_QQ(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  void
  BoundaryRightHandSide(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryMatrix(
#if DEAL_II_VERSION_GTE(9,3,0)
    const FDC<HP, DH, VECTOR, dealdim> & /*fdc*/,
#else
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
#endif
    dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/,
    double /*scale_ico*/)
  {
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    if ((this->problem_type_ == "adjoint")
        || (this->problem_type_ == "state")
        || (this->problem_type_ == "tangent")
        || (this->problem_type_ == "adjoint_hessian")
        || (this->problem_type_ == "hessian"))
      return update_values | update_gradients | update_quadrature_points;
    else if ((this->problem_type_ == "gradient"))
      return update_values | update_gradients | update_quadrature_points;
    else if ((this->problem_type_ == "hessian_inverse")
             || (this->problem_type_ == "global_constraint_gradient")
             || (this->problem_type_ == "global_constraint_hessian"))
      return update_values | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }
  UpdateFlags
  GetFaceUpdateFlags() const
  {
    if ((this->problem_type_ == "adjoint")
        || (this->problem_type_ == "state")
        || (this->problem_type_ == "tangent")
        || (this->problem_type_ == "adjoint_hessian")
        || (this->problem_type_ == "hessian"))
      return update_values | update_quadrature_points;
    else if ((this->problem_type_ == "hessian_inverse")
             || (this->problem_type_ == "gradient")
             || (this->problem_type_ == "global_constraint_gradient")
             || (this->problem_type_ == "global_constraint_hessian"))
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
  std::vector<double> fvalues_;

  std::vector<Vector<double> > qvalues_;
  std::vector<Vector<double> > dqvalues_;
  std::vector<Vector<double> > funcgradvalues_;

  std::vector<Vector<double> > uvalues_;
  std::vector<std::vector<Tensor<1, dealdim> > > ugrads_;
  std::vector<std::vector<Tensor<1, dealdim> > > zgrads_;

  std::vector<unsigned int> control_block_components_;
  std::vector<unsigned int> state_block_components_;
  double p_;
};
#endif
