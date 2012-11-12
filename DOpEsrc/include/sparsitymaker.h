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
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
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
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
      sparsity.copy_from(csp);
    }


} //end of namespace

#endif /* SPARSITYMAKER_H_ */
