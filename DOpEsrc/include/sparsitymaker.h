/*
 * sparsitymaker.h
 *
 *  Created on: May 31, 2011
 *      Author: cgoll
 */

#ifndef SPARSITYMAKER_H_
#define SPARSITYMAKER_H_

#include "dofhandler_wrapper.h"
#include <dofs/dof_tools.h>
#include <lac/constraint_matrix.h>

namespace DOpE
{
  /**
   * Constructs the sparsitypattern.
   */
  template<typename DOFHANDLER, int dim>
    class SparsityMaker
    {
    public:
      SparsityMaker()
      {
      }
      virtual
      ~SparsityMaker()
      {
      }
      virtual void
      ComputeSparsityPattern(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER>& dof_handler,
          dealii::BlockSparsityPattern & sparsity,
          const dealii::ConstraintMatrix& hanging_node_constraints,
          const std::vector<unsigned int>& blocks) const;

      virtual void
      ComputeSparsityPattern(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER>& dof_handler,
          dealii::SparsityPattern & sparsity,
          const dealii::ConstraintMatrix& hanging_node_constraints,
          const std::vector<unsigned int>& blocks) const;

      //TODO: If one wishes to change the sparsity-pattern of the control, one
      //has to implement this here.

    };

  template<typename DOFHANDLER, int dim>
    void
    SparsityMaker<DOFHANDLER, dim>::ComputeSparsityPattern(
        const DOpEWrapper::DoFHandler<dim, DOFHANDLER>& dof_handler,
        dealii::BlockSparsityPattern & sparsity,
        const dealii::ConstraintMatrix& hanging_node_constraints,
        const std::vector<unsigned int>& blocks) const
    {
      dealii::BlockCompressedSimpleSparsityPattern csp(blocks.size(),
          blocks.size());
      for (unsigned int i = 0; i < blocks.size(); i++)
        {
          for (unsigned int j = 0; j < blocks.size(); j++)
            {
              csp.block(i, j).reinit(blocks.at(i), blocks.at(j));
            }
        }
      csp.collect_sizes();
      dealii::DoFTools::make_sparsity_pattern(
          static_cast<const DOFHANDLER&> (dof_handler), csp);
      hanging_node_constraints.condense(csp);
      sparsity.copy_from(csp);
    }


  template<typename DOFHANDLER, int dim>
    void
    SparsityMaker<DOFHANDLER, dim>::ComputeSparsityPattern(
        const DOpEWrapper::DoFHandler<dim, DOFHANDLER>& dof_handler,
        dealii::SparsityPattern & sparsity,
        const dealii::ConstraintMatrix& hanging_node_constraints,
        const std::vector<unsigned int>& blocks) const
    {
      unsigned int total_dofs = 0;
      for (unsigned int j = 0; j < blocks.size(); j++)
        {
          total_dofs += blocks.at(j);
        }

      dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
      dealii::DoFTools::make_sparsity_pattern(
          static_cast<const DOFHANDLER&> (dof_handler), csp);
      hanging_node_constraints.condense(csp);
      sparsity.copy_from(csp);
    }


} //end of namespace

#endif /* SPARSITYMAKER_H_ */
