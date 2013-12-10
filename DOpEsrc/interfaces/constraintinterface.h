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

#ifndef _CONSTRAINT_INTERFACE_H_
#define _CONSTRAINT_INTERFACE_H_

#include <map>
#include <string>

#include <fe/fe_system.h>
#include <fe/fe_values.h>
#include <fe/mapping.h>

#include "fevalues_wrapper.h"
#include "dofhandler_wrapper.h"
#include "functionalinterface.h"

namespace DOpE
{
  /**
   * A template for an arbitrary control constraint.
   * GlobalConstraints are dealt with as a Functional, 
   * hence all functions from Functionals are inherited.
   *
   * @tparam <EDC>             The ElementDataContainer object
   *                           needed by the base class.
   * @tparam <FDC>             The FaceDataContainer object
   *                           needed by the base class.
   * @tparam <DH>              The DoFHandler object used by the 
   *                           FunctionalInterface.
   * @tparam <VECTOR>          The vector class on which 
   *                           ControlVector<> is based.
   * @tparam <dopedim>         The dimension of the domain for the control.
   * @tparam <dealdim>         The dimension of the domain for the state.
   *                           
   */
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
    class ConstraintInterface : public FunctionalInterface<EDC, FDC, DH,
        VECTOR, dopedim, dealdim>
    {
      public:
        ConstraintInterface()
        {
        }
        ~ConstraintInterface()
        {
        }

	/**
	 *  This function is used to evaluate all control constraints that 
	 *  are posed locally, i.e., those that can be evaluated by knowledge 
	 *  of the coefficient vector of the control.
	 *
	 *  @param control         The control in which the constraints should be 
	 *                         evaluated.
	 *  @param constraints     The vector in which the local constraints
	 *                         at the point control are stored. It is 
	 *                         assumed that a control is feasible, if all 
	 *                         entries of this vector are non positive.
	 *                         If any entry is positive, the control is 
	 *                         considered to be infeasible.
	 */
        virtual void
        EvaluateLocalControlConstraints(const VECTOR& control,
            VECTOR& constraints) = 0;

	/**
	 *  This function returns the lower and upper box constraints
	 *  on the control. This means for a problem with box constraints
	 *  The method EvaluateLocalControlConstraints computes
	 *  (lb - control, control - ub)
	 * 
	 *  @param lb    The vector where the lower bound is stored.
	 *               It is assumed that this vector is of the same 
	 *               size as the control.
	 *  @param ub    The vector where the upper bound is stored.
	 *               It is assumed that this vector is of the same 
	 *               size as the control.
	 */
        virtual void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const = 0;

        void
        SetProblemType(std::string type, unsigned int num)
        {
          _problem_type_num = num;
          _problem_type = type;
        }

	/**
	 * This function is called after the constraints are evaluated.
	 * The default is that nothing is done. However, certain
	 * Methods, e.g., penalty or barrier methods may need some 
	 * Postprocessing to convert the constraint vector into the 
	 * desired form.
	 *
	 * @param g   The vector to be transformed.
	 */
        virtual void
	  PostProcessConstraints(ConstraintVector<VECTOR>& /*g*/) const
	{
	}
	
      protected:
        std::string
        GetProblemType() const
        {
          return _problem_type;
        }
        unsigned int
        GetProblemTypeNum() const
        {
          return _problem_type_num;
        }
      private:
        std::string _problem_type;
        unsigned int _problem_type_num;
    };
}

#endif
