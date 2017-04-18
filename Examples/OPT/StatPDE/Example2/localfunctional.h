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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <interfaces/functionalinterface.h>
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional()
  {
    alpha_ = 0.;
    eval_points_.resize(3);
    //for q0
    eval_points_[0][0] = 0.5;
    eval_points_[0][1] = 0.5;
    //for q1
    eval_points_[1][0] = 0.5;
    eval_points_[1][1] = 0.25;
    //for q2
    eval_points_[2][0] = 0.25;
    eval_points_[2][1] = 0.25;
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      qvalues_.reinit(3);
      edc.GetParamValues("control", qvalues_);
    }
    double r = 0.;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        r += alpha_ * 0.5
             * (qvalues_(0) * qvalues_(0) + qvalues_(1) * qvalues_(1)
                + qvalues_(2) * qvalues_(2)) * state_fe_values.JxW(q_point);
      }

    return r;
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    double r = 0;
    //now we extract the solution u
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find state",
                            "localfunctional::PointValue");
      }
    VECTOR U(*(it->second));

    //J[i] = (u_h - \overline{u})(x_i)
    std::vector<Vector<double> > J(3, Vector<double>(2));

    for (unsigned int i = 0; i < eval_points_.size(); i++)
      {
        VectorTools::point_value(state_dof_handler, U, eval_points_[i], J[i]);
        Vector<double> u_ex(2);
        exact_u_.vector_value(eval_points_[i], u_ex);
        J[i].add(-1., u_ex);
        r += std::pow(J[i].l2_norm(), 2);
      }

    r *= 0.5;
    return r;
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  virtual void
  PointValue_U(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values,
    VECTOR &rhs, double scale)
  {
    VECTOR rhs_tmp_0, rhs_tmp_1;
    //lets extract the solution u
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find state",
                            "localfunctional::PointValue_U");
      }
    VECTOR U(*(it->second));

    //as obove, J[i] = (u_h - \overline{u})(x_i)
    std::vector<Vector<double> > J(3, Vector<double>(2));
    rhs.reinit(U);
    rhs_tmp_0.reinit(U);
    rhs_tmp_1.reinit(U);

    for (unsigned int i = 0; i < eval_points_.size(); i++)
      {
        VectorTools::point_value(state_dof_handler, U, eval_points_[i], J[i]);

        Vector<double> u_ex(2);
        exact_u_.vector_value(eval_points_[i], u_ex);
        J[i].add(-1., u_ex);

        create_point_source(state_dof_handler.GetDEALDoFHandler(),
                            eval_points_[i], 0, rhs_tmp_0);
        rhs_tmp_0 *= J[i][0];
        rhs += rhs_tmp_0;
        create_point_source(state_dof_handler.GetDEALDoFHandler(),
                            eval_points_[i], 1, rhs_tmp_1);
        rhs_tmp_1 *= J[i][1];
        rhs += rhs_tmp_1;
      }

    rhs *= scale;
  }

  virtual void
  PointValue_Q(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &,
    const std::map<std::string, const dealii::Vector<double>*> &,
    const std::map<std::string, const VECTOR *> &, VECTOR &, double)
  {
  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      qvalues_.reinit(3);

      edc.GetParamValues("control", qvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * alpha_ * (qvalues_(i))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> &, dealii::Vector<double> &,
                  double)
  {
  }

  virtual void
  PointValue_UU(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values,
    VECTOR &rhs, double scale)
  {
    VECTOR rhs_tmp_0, rhs_tmp_1;
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      domain_values.find("tangent");
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find tangent",
                            "localfunctional::PointValue_UU");
      }

    VECTOR DU(*(it->second));
    std::vector<Vector<double> > J(3, Vector<double>(2));
    rhs.reinit(DU);
    rhs_tmp_0.reinit(DU);
    rhs_tmp_1.reinit(DU);

    for (unsigned int i = 0; i < eval_points_.size(); i++)
      {
        VectorTools::point_value(state_dof_handler, DU, eval_points_[i],
                                 J[i]);

        create_point_source(state_dof_handler.GetDEALDoFHandler(),
                            eval_points_[i], 0, rhs_tmp_0);
        rhs_tmp_0 *= J[i][0];
        rhs += rhs_tmp_0;
        create_point_source(state_dof_handler.GetDEALDoFHandler(),
                            eval_points_[i], 1, rhs_tmp_1);
        rhs_tmp_1 *= J[i][1];
        rhs += rhs_tmp_1;
      }

    rhs *= scale;
  }

  virtual void
  PointValue_QU(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &/*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {

  }

  virtual void
  PointValue_UQ(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &/*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {

  }

  virtual void
  PointValue_QQ(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &/*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {

  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      dqvalues_.reinit(3);
      edc.GetParamValues("dq", dqvalues_);
    }

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < local_vector.size(); i++)
          {
            local_vector(i) += scale * alpha_ * dqvalues_(i)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const
  {
    return "point domain";
  }

  string
  GetName() const
  {
    return "cost functional";
  }

private:
  void
  create_point_source(const DH<dealdim, dealdim> &dof_handler,
                      const Point<dealdim> point, const unsigned int component,
                      VECTOR &rhs_vector)
  {
    Assert(rhs_vector.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));

    rhs_vector = 0;

    std::pair<typename DH<dealdim, dealdim>::active_cell_iterator,
        Point<dealdim> > element_point =
          GridTools::find_active_cell_around_point(
            StaticMappingQ1<dealdim>::mapping, dof_handler, point);

    Quadrature<dealdim> q(
      GeometryInfo<dealdim>::project_to_unit_cell(element_point.second));

    FEValues<dealdim> fe_values(dof_handler.get_fe(), q,
                                UpdateFlags(update_values));
    fe_values.reinit(element_point.first);

    const unsigned int dofs_per_element = dof_handler.get_fe().dofs_per_cell;

    std::vector<unsigned int> local_dof_indices(dofs_per_element);
    element_point.first->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_element; i++)
      rhs_vector(local_dof_indices[i]) = fe_values.shape_value_component(i,
                                         0, component);
  }
  vector<Point<2> > eval_points_;
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  double alpha_;
  ExactU exact_u_;
};
#endif
