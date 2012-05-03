/*
 * constraintmaker.h
 *
 *  Created on: May 31, 2011
 *      Author: cgoll
 */

#ifndef _CONSTRAINTMAKER_H_
#define _CONSTRAINTMAKER_H_

#include <dofs/dof_tools.h>
#include <dofs/dof_handler.h>
#include <lac/constraint_matrix.h>

#include "dofhandler_wrapper.h"
#include "parameterreader.h"

namespace DOpE
{
  /**
   * This class is an interface which offers the user the possibility
   * to define some DoFConstraints for the state and/or control fe function.
   * To define non-standard constraints, one has to implement these in a
   * derived class of this one, and give then an instantiation
   * to the SpaceTimeHandler (via SetUserDefinedDoFConstraints).
   *
   * The constraints defined by MakeStateDoFConstrains and MakeControlDoFConstraints
   * are computed AFTER hanging_node_constraint is called, so if there are two
   * or more conflicting constraints on a DoF, the constraints coming from
   * hanging nodes win.
   *
   * FIXME: Just homogeneous dof constraints at the moment.
   * If we change distribution from global to local, this should
   * get changed.
   */
  template<typename DOFHANDLER, int dopedim, int dealdim = dopedim>
    class UserDefinedDoFConstraints
    {
      public:
        UserDefinedDoFConstraints()
        {
        }
        virtual
        ~UserDefinedDoFConstraints()
        {
        }
        virtual void
        MakeStateDoFConstraints(
            const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & dof_handler,
            dealii::ConstraintMatrix& dof_constraints) const;

#if dope_dimension > 0
        virtual void
        MakeControlDoFConstraints(
            const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & dof_handler,
            dealii::ConstraintMatrix& dof_constraints) const;
#endif

      private:
    };

  template<typename DOFHANDLER, int dopedim, int dealdim>
    void
    UserDefinedDoFConstraints<DOFHANDLER, dopedim, dealdim>::MakeStateDoFConstraints(
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & /*dof_handler*/,
        dealii::ConstraintMatrix& /*dof_constraints*/) const
    {
      throw DOpEException("Not Implemented.",
          "UserDefinedDoFConstraints::MakeStateDoFConstraints");
    }

#if dope_dimension > 0
template<typename DOFHANDLER, int dopedim, int dealdim>
void
UserDefinedDoFConstraints<DOFHANDLER, dopedim, dealdim>::MakeControlDoFConstraints(
    const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & /*dof_handler*/,
    dealii::ConstraintMatrix& /*dof_constraints*/) const
  {
    throw DOpEException("Not Implemented.",
        "UserDefinedDoFConstraints::MakeControlDoFConstraints");
  }

#endif
} //end of namespace
#endif /* CONSTRAINTMAKER_H_ */
