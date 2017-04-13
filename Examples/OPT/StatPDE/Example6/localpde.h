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

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR,
  dealdim>
{
public:
  LocalPDE() :
    block_component_(1, 0)
  {
    alpha_ = 1.e-3;
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
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
      qvalues_.resize(n_q_points);
      ugrads_.resize(n_q_points);

      //Getting q
      edc.GetValuesControl("control", qvalues_);
      //Geting u
      edc.GetGradsState("last_newton_solution", ugrads_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (ugrads_[q_point] * state_fe_values.shape_grad(i, q_point)
                                  - qvalues_[q_point]
                                  * state_fe_values.shape_value(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "adjoint");
      zgrads_.resize(n_q_points);
      //We don't need u so we don't search for state
      edc.GetGradsState("last_newton_solution", zgrads_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (zgrads_[q_point] * state_fe_values.shape_grad(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UT(const EDC<DH, VECTOR, dealdim> &edc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "tangent");
      dugrads_.resize(n_q_points);
      edc.GetGradsState("last_newton_solution", dugrads_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (dugrads_[q_point] * state_fe_values.shape_grad(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UTT(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "adjoint_hessian");
      dzgrads_.resize(n_q_points);
      edc.GetGradsState("last_newton_solution", dzgrads_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (dzgrads_[q_point] * state_fe_values.shape_grad(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_Q(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "gradient");
      zvalues_.resize(n_q_points);
      edc.GetValuesState("adjoint", zvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (-zvalues_[q_point]
                                  * control_fe_values.shape_value(i, q_point))
                               * control_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_QT(const EDC<DH, VECTOR, dealdim> &edc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "tangent");
      dqvalues_.resize(n_q_points);
      edc.GetValuesControl("dq", dqvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) +=
              scale
              * (-dqvalues_[q_point]
                 * state_fe_values.shape_value(i, q_point))
              * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_QTT(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "hessian");
      dzvalues_.resize(n_q_points);
      edc.GetValuesState("adjoint_hessian", dzvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (-dzvalues_[q_point]
                                  * control_fe_values.shape_value(i, q_point))
                               * control_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }
  void
  ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "state");
      fvalues_.resize(n_q_points);
    }
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        fvalues_[q_point] = ((20. * M_PI * M_PI
                              * sin(4. * M_PI * state_fe_values.quadrature_point(q_point)(0))
                              - 1. / alpha_
                              * sin(M_PI * state_fe_values.quadrature_point(q_point)(0)))
                             * sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1)));

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (fvalues_[q_point] * state_fe_values.shape_value(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
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
  ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(
        (this->problem_type_ == "gradient")||(this->problem_type_ == "hessian"));
      funcgradvalues_.resize(n_q_points);
      edc.GetValuesControl("last_newton_solution", funcgradvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (funcgradvalues_[q_point]
                                  * control_fe_values.shape_value(i, q_point))
                               * control_fe_values.JxW(q_point);
          }
      }
  }

  void
  ControlElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
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
      return update_values | update_quadrature_points;
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
    return block_component_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return block_component_;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return block_component_;
  }

private:
  vector<double> qvalues_;
  vector<double> dqvalues_;
  vector<double> funcgradvalues_;
  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<Tensor<1, dealdim> > ugrads_;
  vector<double> zvalues_;
  vector<Tensor<1, dealdim> > zgrads_;
  vector<double> duvalues_;
  vector<Tensor<1, dealdim> > dugrads_;
  vector<double> dzvalues_;
  vector<Tensor<1, dealdim> > dzgrads_;

  vector<unsigned int> block_component_;
  double alpha_;
};
#endif
