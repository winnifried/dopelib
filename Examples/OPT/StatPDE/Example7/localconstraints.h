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

#ifndef LOCAL_CONSTRAINT_H_
#define LOCAL_CONSTRAINT_H_

#include <interfaces/constraintinterface.h>
#include "localconstraintaccessor.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
#if DEAL_II_VERSION_GTE(9,3,0)
  template<
    template<bool DH, typename VECTOR, int dealdim> class EDC,
    template<bool DH, typename VECTOR, int dealdim> class FDC,
    bool DH, typename VECTOR, int dopedim, int dealdim>
    class LocalConstraint : public ConstraintInterface<EDC, FDC, DH, VECTOR,
    dopedim, dealdim>
#else
    template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  class LocalConstraint : public ConstraintInterface<EDC, FDC, DH, VECTOR,
    dopedim, dealdim>
#endif
  {
  public:
    LocalConstraint(LocalConstraintAccessor &CA) :
      LCA_(CA)
    {
      vol_max_ = 0.5;
    }
    ~LocalConstraint()
    {
    }

    void
    EvaluateLocalControlConstraints(
      const dealii::BlockVector<double> &control,
      dealii::BlockVector<double> &constraints)
    {
      assert(constraints.block(0).size() == 2*control.block(0).size());

      for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          // rho_min_ <= control <= rho_max_
          LCA_.ControlToLowerConstraint(control, constraints);
          LCA_.ControlToUpperConstraint(control, constraints);
        }
    }
    void
    GetControlBoxConstraints(VECTOR &lb, VECTOR &ub) const
    {
      LCA_.GetControlBoxConstraints(lb, ub);
    }

    double
    ElementValue(
      const EDC<DH, VECTOR, dealdim> &edc)
    {
      if (this->GetProblemType() == "global_constraints"
          && this->GetProblemTypeNum() == 0)
        {
          const DOpEWrapper::FEValues<dealdim> &control_fe_values =
            edc.GetFEValuesControl();
          unsigned int n_q_points = edc.GetNQPoints();

          double ret = 0.;
          {
            qvalues_.resize(n_q_points, Vector<double>(1));
            edc.GetValuesControl("control", qvalues_);
          }
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              ret += (qvalues_[q_point](0) - vol_max_)
                     * control_fe_values.JxW(q_point);
            }
          return ret;
        }
      else
        {
          return 0;
        }
    }

    void
    ElementValue_U(
      const EDC<DH, VECTOR, dealdim> & /*edc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {
    }

    void
    ElementValue_Q(
      const EDC<DH, VECTOR, dealdim> &edc,
      dealii::Vector<double> &local_vector, double scale)
    {
      if (this->GetProblemType() == "global_constraint_gradient"
          && this->GetProblemTypeNum() == 0)
        {
          const DOpEWrapper::FEValues<dealdim> &control_fe_values =
            edc.GetFEValuesControl();
          unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
          unsigned int n_q_points = edc.GetNQPoints();

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int i = 0; i < n_dofs_per_element; i++)
                {
                  local_vector(i) += scale
                                     * control_fe_values.shape_value(i, q_point)
                                     * control_fe_values.JxW(q_point);
                }
            }
        }
      else
        {
          abort();
        }
    }

    void
    ElementValue_UU(
      const EDC<DH, VECTOR, dealdim> & /*edc*/,
      dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {
    }
    void
    ElementValue_QU(
      const EDC<DH, VECTOR, dealdim> & /*edc*/,
      dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {
    }
    void
    ElementValue_UQ(
      const EDC<DH, VECTOR, dealdim> & /*edc*/,
      dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {
    }
    void
    ElementValue_QQ(
      const EDC<DH, VECTOR, dealdim> & /*edc*/,
      dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {
    }

    std::string
    GetType() const
    {
      return "domain";
    }
    std::string
    GetName() const
    {
      return "volume_constraint";
    }

    dealii::UpdateFlags
    GetUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }

  private:
    double vol_max_;
    std::vector<dealii::Vector<double> > qvalues_;
    LocalConstraintAccessor &LCA_;

  };
}

#endif
