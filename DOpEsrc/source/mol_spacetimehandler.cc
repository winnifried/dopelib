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

#include "mol_spacetimehandler.h"

namespace DOpE
{
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
    void
    DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
        dealii::DoFHandler, dealii::BlockSparsityPattern,
        dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
        dealii::BlockSparsityPattern & sparsity) const
    {
      const std::vector<unsigned int>& blocks = this->GetControlDoFsPerBlock();
      dealii::BlockCompressedSimpleSparsityPattern csp(blocks.size(),
          blocks.size());
      for (unsigned int i = 0; i < blocks.size(); i++)
      {
        for (unsigned int j = 0; j < blocks.size(); j++)
        {
          csp.block(i, j).reinit(this->GetControlDoFsPerBlock(i),
              this->GetControlDoFsPerBlock(j));
        }
      }
      csp.collect_sizes();
#if dope_dimension > 0
      //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
      dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
      abort();
#endif
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

  /******************************************************/

  template<>
    void
    MethodOfLines_SpaceTimeHandler<dealii::FESystem,
        dealii::DoFHandler, dealii::SparsityPattern,
        dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
        dealii::SparsityPattern & sparsity) const
    {
      const unsigned int total_dofs = this->GetControlNDoFs();
      dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);

#if dope_dimension > 0
      dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
      abort();
#endif
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
    void
    DOpE::MethodOfLines_SpaceTimeHandler<
        dealii::hp::FECollection,
        dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
        dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
        dealii::BlockSparsityPattern & sparsity) const
    {
      const std::vector<unsigned int>& blocks = this->GetControlDoFsPerBlock();
      dealii::BlockCompressedSimpleSparsityPattern csp(blocks.size(),
          blocks.size());
      for (unsigned int i = 0; i < blocks.size(); i++)
      {
        for (unsigned int j = 0; j < blocks.size(); j++)
        {
          csp.block(i, j).reinit(this->GetControlDoFsPerBlock(i),
              this->GetControlDoFsPerBlock(j));
        }
      }
      csp.collect_sizes();
#if dope_dimension > 0
      //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
      dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
      abort();
#endif
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

  /******************************************************/

  template<>
    void
    MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
        dealii::hp::DoFHandler, dealii::SparsityPattern,
        dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
        dealii::SparsityPattern & sparsity) const
    {
      const unsigned int total_dofs = this->GetControlNDoFs();
      dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);

#if dope_dimension > 0
      dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
      abort();
#endif
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

}

template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
						    dealii::DoFHandler, 
						    dealii::BlockSparsityPattern,
						    dealii::BlockVector<double>, 
						    dope_dimension, 
						    deal_II_dimension>;
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
						    dealii::DoFHandler, 
						    dealii::SparsityPattern,
						    dealii::Vector<double>, 
						    dope_dimension, 
						    deal_II_dimension>;

template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
						    dealii::hp::DoFHandler, 
						    dealii::BlockSparsityPattern,
						    dealii::BlockVector<double>, 
						    dope_dimension, 
						    deal_II_dimension>;
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
						    dealii::hp::DoFHandler, 
						    dealii::SparsityPattern,
						    dealii::Vector<double>, 
						    dope_dimension, 
						    deal_II_dimension>;
