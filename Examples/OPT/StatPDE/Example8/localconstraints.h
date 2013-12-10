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

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
    class LocalConstraint : public ConstraintInterface<EDC, FDC, DH, VECTOR,
        dopedim, dealdim>
    {
      public:
        LocalConstraint() 
        {
          _vol_max = 0.5;
	  _rho_min = 1.e-4;
	  _rho_max = 1.;

        }
        ~LocalConstraint()
        {
        }
	
	void
	  SetRhoMin(double val)
	{
	  _rho_min = val;
	}

        void
        EvaluateLocalControlConstraints(
            const dealii::BlockVector<double>& control,
            dealii::BlockVector<double>& constraints)
        {
          assert(constraints.block(0).size() == 2*control.block(0).size());

	  //Add Control Constraints, such that if control is feasible all  entries are not positive!
	  // _rho_min <= control <= _rho_max
	  for (unsigned int i = 0; i < control.block(0).size(); i++)
	  {
	    constraints.block(0)(i) = _rho_min - control.block(0)(i);
	    constraints.block(0)(control.block(0).size() + i) = control.block(0)(i) - _rho_max;
	  }
        }
        void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
        {
	  lb = _rho_min;
	  ub = _rho_max;
        }

        double
        ElementValue(const EDC<DH, VECTOR, dealdim>& edc)
        {
          if (this->GetProblemType() == "global_constraints"
              && this->GetProblemTypeNum() == 0)
          {
            const DOpEWrapper::FEValues<dealdim> & control_fe_values =
                edc.GetFEValuesControl();
            unsigned int n_q_points = edc.GetNQPoints();

            double ret = 0.;
            {
              _qvalues.resize(n_q_points, Vector<double>(1));
              edc.GetValuesControl("control", _qvalues);
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
        ElementValue_U(const EDC<DH, VECTOR, dealdim>& /*edc*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/)
        {
        }

        void
        ElementValue_Q(const EDC<DH, VECTOR, dealdim>& edc,
            dealii::Vector<double> &local_vector, double scale)
        {
          if (this->GetProblemType() == "global_constraint_gradient"
              && this->GetProblemTypeNum() == 0)
          {
            const DOpEWrapper::FEValues<dealdim> & control_fe_values =
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
        ElementValue_UU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/)
        {
        }
        void
        ElementValue_QQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/)
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
        double _vol_max, _rho_min, _rho_max;
        std::vector<dealii::Vector<double> > _qvalues;

    };
}

#endif
