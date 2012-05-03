/*
 * constraintmaker.h
 *
 *  Created on: May 31, 2011
 *      Author: cgoll
 */

#ifndef CONSTRAINTMAKER_H_
#define CONSTRAINTMAKER_H_

#include <dofs/dof_tools.h>
#include <dofs/dof_handler.h>
#include <lac/constraint_matrix.h>

#include "dofhandler_wrapper.h"
#include "parameterreader.h"

namespace DOpE
{
  template<typename DOFHANDLER, int dim>
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
      MakeConstraints(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER> & dof_handler,
          dealii::ConstraintMatrix& dof_constraints) const;
      virtual void
      MakeHomogeneousConstraints(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER> & dof_handler,
          dealii::ConstraintMatrix& dof_constraints) const;


    private:
    };

  template<typename DOFHANDLER, int dim>
    void
    UserDefinedDoFConstraints<DOFHANDLER, dim>::MakeConstraints(
        const DOpEWrapper::DoFHandler<dim, DOFHANDLER> & dof_handler,
        dealii::ConstraintMatrix& dof_constraints) const
    {
      hanging_node_constraints.clear();
      DoFTools::make_hanging_node_constraints(
          static_cast<const DOFHANDLER&> (dof_handler),
          hanging_node_constraints);
      hanging_node_constraints.close();
    }
} //end of namespace
#endif /* CONSTRAINTMAKER_H_ */