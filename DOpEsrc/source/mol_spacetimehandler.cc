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

#include <basic/mol_spacetimehandler.h>

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
         dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::BlockDynamicSparsityPattern csp(blocks.size(),
                                            blocks.size());
#else
    dealii::BlockCompressedSimpleSparsityPattern csp(blocks.size(),
                                                     blocks.size());
#endif

    for (unsigned int i = 0; i < blocks.size(); i++)
      {
        for (unsigned int j = 0; j < blocks.size(); j++)
          {
            csp.block(i, j).reinit(this->GetControlDoFsPerBlock()[i],
                                   this->GetControlDoFsPerBlock()[j]);
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
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::SparsityPattern,
       dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
         dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
#else
    dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
#endif

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
           dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::BlockDynamicSparsityPattern csp(blocks.size(),
                                            blocks.size());
#else
    dealii::BlockCompressedSimpleSparsityPattern csp(blocks.size(),
                                                     blocks.size());
#endif

    for (unsigned int i = 0; i < blocks.size(); i++)
      {
        for (unsigned int j = 0; j < blocks.size(); j++)
          {
            csp.block(i, j).reinit(this->GetControlDoFsPerBlock()[i],
                                   this->GetControlDoFsPerBlock()[j]);
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
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::SparsityPattern,
       dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
         dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
#else
    dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
#endif

#if dope_dimension > 0
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort();
#endif
    this->GetControlDoFConstraints().condense(csp);
    sparsity.copy_from(csp);
  }


///////////////////////////ResetTriangulation
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
         const dealii::Triangulation<deal_II_dimension> &tria)
  {
    state_dof_handler_.clear();
    triangulation_.clear();
    triangulation_.copy_triangulation(tria);
    state_dof_handler_.initialize(triangulation_, *state_fe_);
    this->IncrementControlTicket();
    this->IncrementStateTicket();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }

  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::SparsityPattern,
       dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
         const dealii::Triangulation<deal_II_dimension> &tria)
  {
    state_dof_handler_.clear();
    triangulation_.clear();
    triangulation_.copy_triangulation(tria);
    state_dof_handler_.initialize(triangulation_, *state_fe_);
    this->IncrementControlTicket();
    this->IncrementStateTicket();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }

  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<
  dealii::hp::FECollection,
         dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
           const dealii::Triangulation<deal_II_dimension> & /*tria*/)
  {
    abort();
  }

  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::SparsityPattern,
       dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
         const dealii::Triangulation<deal_II_dimension> & /*tria*/)
  {
    abort();
  }

}//End of namespace DOpE

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

