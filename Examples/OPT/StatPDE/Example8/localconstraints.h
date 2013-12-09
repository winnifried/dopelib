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

#ifndef _LOCAL_CONSTRAINT_H_
#define _LOCAL_CONSTRAINT_H_

#include "constraintinterface.h"
#include "localconstraintaccessor.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
    class LocalConstraint : public ConstraintInterface<CDC, FDC, DH, VECTOR,
        dopedim, dealdim>
    {
      public:
        LocalConstraint(LocalConstraintAccessor& CA) :
            LCA(CA)
        {
          _vol_max = 0.5;
        }
        ~LocalConstraint()
        {
        }

        void
        EvaluateLocalControlConstraints(
            const dealii::BlockVector<double>& control,
            dealii::BlockVector<double>& constraints)
        {
          assert(constraints.block(0).size() == 2*control.block(0).size());

          for (unsigned int i = 0; i < control.block(0).size(); i++)
          {
            //Add Control Constraints, such that if control is feasible all  entries are not positive!
            // _rho_min <= control <= _rho_max
            LCA.ControlToLowerConstraint(control, constraints);
            LCA.ControlToUpperConstraint(control, constraints);
          }
        }
        void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
        {
          LCA.GetControlBoxConstraints(lb, ub);
        }

        double
        ElementValue(const CDC<DH, VECTOR, dealdim>& cdc)
        {
          if (this->GetProblemType() == "global_constraints"
              && this->GetProblemTypeNum() == 0)
          {
            const DOpEWrapper::FEValues<dealdim> & control_fe_values =
                cdc.GetFEValuesControl();
            unsigned int n_q_points = cdc.GetNQPoints();

            double ret = 0.;
            {
              _qvalues.resize(n_q_points, Vector<double>(1));
              cdc.GetValuesControl("control", _qvalues);
            }
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              ret += (_qvalues[q_point](0) - _vol_max)
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
        ElementValue_U(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
        {
        }

        void
        ElementValue_Q(const CDC<DH, VECTOR, dealdim>& cdc,
            dealii::Vector<double> &local_cell_vector, double scale)
        {
          if (this->GetProblemType() == "global_constraint_gradient"
              && this->GetProblemTypeNum() == 0)
          {
            const DOpEWrapper::FEValues<dealdim> & control_fe_values =
                cdc.GetFEValuesControl();
            unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
            unsigned int n_q_points = cdc.GetNQPoints();

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; i++)
              {
                local_cell_vector(i) += scale
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
        ElementValue_UU(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_QU(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_UQ(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_QQ(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
        {
        }

        std::string
        GetType() const
        {
          if ((this->GetProblemType() == "global_constraints"
              || this->GetProblemType() == "global_constraint_gradient")
              && this->GetProblemTypeNum() == 0)
            return "domain";
          else
            throw DOpEException(
                "Unknown problem_type " + this->GetProblemType(),
                "LocalConstraints::GetType");
        }
        std::string
        GetName() const
        {
          if ((this->GetProblemType() == "global_constraints"
              || this->GetProblemType() == "global_constraint_gradient")
              && this->GetProblemTypeNum() == 0)
            return "volume_constraint";
          else
            throw DOpEException(
                "Unknown problem_type " + this->GetProblemType(),
                "LocalConstraints::GetName");
        }

        dealii::UpdateFlags
        GetUpdateFlags() const
        {
          return update_values | update_quadrature_points;
        }
	
      private:
        double _vol_max;
        std::vector<dealii::Vector<double> > _qvalues;
        LocalConstraintAccessor& LCA;

    };
}

#endif
