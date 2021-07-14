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

#include <basic/mol_spacetimehandler.h>

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
   /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, false,
       dealii::BlockSparsityPattern, dealii::BlockVector<double>,
       dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern (dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock ();
    dealii::BlockDynamicSparsityPattern csp (blocks.size (), blocks.size ());

    for (unsigned int i = 0; i < blocks.size (); i++)
      {
        for (unsigned int j = 0; j < blocks.size (); j++)
          {
            csp.block (i, j).reinit (this->GetControlDoFsPerBlock ()[i],
                                     this->GetControlDoFsPerBlock ()[j]);
          }
      }
    csp.collect_sizes ();
#if dope_dimension > 0
    //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }

  /******************************************************/

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, false,
       dealii::SparsityPattern, dealii::Vector<double>, dope_dimension,
       deal_II_dimension>::ComputeControlSparsityPattern (dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs ();
    dealii::DynamicSparsityPattern csp (total_dofs, total_dofs);

#if dope_dimension > 0
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }
#else
   /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, dealii::DoFHandler,
       dealii::BlockSparsityPattern, dealii::BlockVector<double>,
       dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern (dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock ();
    dealii::BlockDynamicSparsityPattern csp (blocks.size (), blocks.size ());

    for (unsigned int i = 0; i < blocks.size (); i++)
      {
        for (unsigned int j = 0; j < blocks.size (); j++)
          {
            csp.block (i, j).reinit (this->GetControlDoFsPerBlock ()[i],
                                     this->GetControlDoFsPerBlock ()[j]);
          }
      }
    csp.collect_sizes ();
#if dope_dimension > 0
    //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }

  /******************************************************/

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, dealii::DoFHandler,
       dealii::SparsityPattern, dealii::Vector<double>, dope_dimension,
       deal_II_dimension>::ComputeControlSparsityPattern (dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs ();
    dealii::DynamicSparsityPattern csp (total_dofs, total_dofs);

#if dope_dimension > 0
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }
#endif  

#if DEAL_II_VERSION_GTE(9,3,0)
   /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection, true,
       dealii::BlockSparsityPattern, dealii::BlockVector<double>,
       dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern (dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock ();
    dealii::BlockDynamicSparsityPattern csp (blocks.size (), blocks.size ());

    for (unsigned int i = 0; i < blocks.size (); i++)
      {
        for (unsigned int j = 0; j < blocks.size (); j++)
          {
            csp.block (i, j).reinit (this->GetControlDoFsPerBlock ()[i],
                                     this->GetControlDoFsPerBlock ()[j]);
          }
      }
    csp.collect_sizes ();
#if dope_dimension > 0
    //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }

  /******************************************************/

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection, true,
       dealii::SparsityPattern, dealii::Vector<double>, dope_dimension,
       deal_II_dimension>::ComputeControlSparsityPattern (dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs ();
    dealii::DynamicSparsityPattern csp (total_dofs, total_dofs);

#if dope_dimension > 0
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }
#else
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern (dealii::BlockSparsityPattern &sparsity) const
  {
    const std::vector<unsigned int> &blocks = this->GetControlDoFsPerBlock ();
    dealii::BlockDynamicSparsityPattern csp (blocks.size (), blocks.size ());

    for (unsigned int i = 0; i < blocks.size (); i++)
      {
        for (unsigned int j = 0; j < blocks.size (); j++)
          {
            csp.block (i, j).reinit (this->GetControlDoFsPerBlock ()[i],
                                     this->GetControlDoFsPerBlock ()[j]);
          }
      }
    csp.collect_sizes ();
#if dope_dimension > 0
    //We use here dealii::DoFHandler<dealdim>, because if dope_dim >0 then dopedim = dealdim.
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }

  /******************************************************/

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::SparsityPattern, dealii::Vector<double>,
       dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern (dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs ();
    dealii::DynamicSparsityPattern csp (total_dofs, total_dofs);

#if dope_dimension > 0
    dealii::DoFTools::make_sparsity_pattern (this->GetControlDoFHandler().GetDEALDoFHandler(),csp);
#else
    abort ();
#endif
    this->GetControlDoFConstraints ().condense (csp);
    sparsity.copy_from (csp);
  }
#endif//Endof dealii older than 9.3.0 

///////////////////////////ResetTriangulation
  template <>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, false,
#else
    DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, dealii::DoFHandler,
#endif
       dealii::BlockSparsityPattern, dealii::BlockVector<double>,
       dope_dimension, deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
           deal_II_dimension> &tria)
  {
    state_dof_handler_.clear ();
    triangulation_.clear ();
    triangulation_.copy_triangulation (tria);
#if DEAL_II_VERSION_GTE(9,3,0)
    state_dof_handler_.reinit (triangulation_);
    //FIXME: Only to assert that the hp_capabilities for the 'SetActiveIndes' methods are set
    // would be better to detect that from the fesystem than the DOFHandler.
    state_dof_handler_.distribute_dofs(*state_fe_);
#else
    state_dof_handler_.initialize (triangulation_, *state_fe_);
#endif
    this->IncrementControlTicket ();
    this->IncrementStateTicket ();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }

  template <>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, false,
#else
    DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem, dealii::DoFHandler,
#endif
       dealii::SparsityPattern, dealii::Vector<double>, dope_dimension,
       deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
                                               deal_II_dimension> &tria)
  {
    state_dof_handler_.clear ();
    triangulation_.clear ();
    triangulation_.copy_triangulation (tria);
#if DEAL_II_VERSION_GTE(9,3,0)
    state_dof_handler_.reinit (triangulation_);
    //FIXME: Only to assert that the hp_capabilities for the 'SetActiveIndes' methods are set
    // would be better to detect that from the fesystem than the DOFHandler.
    state_dof_handler_.distribute_dofs(*state_fe_);
#else
    state_dof_handler_.initialize (triangulation_, *state_fe_);
#endif
    this->IncrementControlTicket ();
    this->IncrementStateTicket ();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }
  
#if DEAL_II_VERSION_GTE(9,3,0)
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection, true,
       dealii::BlockSparsityPattern, dealii::BlockVector<double>,
       dope_dimension, deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
           deal_II_dimension> &tria)
  {
    state_dof_handler_.clear ();
    triangulation_.clear ();
    triangulation_.copy_triangulation (tria);
#if DEAL_II_VERSION_GTE(9,3,0)
    state_dof_handler_.reinit (triangulation_);
    //FIXME: Only to assert that the hp_capabilities for the 'SetActiveIndes' methods are set
    // would be better to detect that from the fesystem than the DOFHandler.
    state_dof_handler_.distribute_dofs(*state_fe_);
#else
    state_dof_handler_.initialize (triangulation_, *state_fe_);
#endif
    this->IncrementControlTicket ();
    this->IncrementStateTicket ();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection, true,
       dealii::SparsityPattern, dealii::Vector<double>, dope_dimension,
       deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
                                               deal_II_dimension> &tria)
  {
    state_dof_handler_.clear ();
    triangulation_.clear ();
    triangulation_.copy_triangulation (tria);
#if DEAL_II_VERSION_GTE(9,3,0)
    state_dof_handler_.reinit (triangulation_);
    //FIXME: Only to assert that the hp_capabilities for the 'SetActiveIndes' methods are set
    // would be better to detect that from the fesystem than the DOFHandler.
    state_dof_handler_.distribute_dofs(*state_fe_);
#else
    state_dof_handler_.initialize (triangulation_, *state_fe_);
#endif
    this->IncrementControlTicket ();
    this->IncrementStateTicket ();
    if (control_mesh_transfer_ != NULL)
      delete control_mesh_transfer_;
    control_mesh_transfer_ = NULL;
    if (state_mesh_transfer_ != NULL)
      delete state_mesh_transfer_;
    state_mesh_transfer_ = NULL;
  }
#else
  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
           deal_II_dimension> & /*tria*/)
  {
    abort ();
  }

  template <>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::SparsityPattern, dealii::Vector<double>,
       dope_dimension, deal_II_dimension>::ResetTriangulation (const dealii::Triangulation<
           deal_II_dimension> & /*tria*/)
  {
    abort ();
  }
#endif//Endof dealii older than 9.3.0 

} //End of namespace DOpE

#if DEAL_II_VERSION_GTE(9,3,0)
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
						    false,
						    dealii::BlockSparsityPattern,
                                                    dealii::BlockVector<double>,
                                                    dope_dimension,
                                                    deal_II_dimension>;
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
                                                    false,
						    dealii::SparsityPattern,
                                                    dealii::Vector<double>,
                                                    dope_dimension,
                                                    deal_II_dimension>;
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
						    true,
						    dealii::BlockSparsityPattern,
                                                    dealii::BlockVector<double>,
                                                    dope_dimension,
                                                    deal_II_dimension>;
template class DOpE::MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
                                                    true,
						    dealii::SparsityPattern,
                                                    dealii::Vector<double>,
                                                    dope_dimension,
                                                    deal_II_dimension>;
#else
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
#endif//Endof dealii older than 9.3.0 

