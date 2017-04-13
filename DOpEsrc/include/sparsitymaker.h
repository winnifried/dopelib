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

#ifndef SPARSITYMAKER_H_
#define SPARSITYMAKER_H_

#include <wrapper/dofhandler_wrapper.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>

// Multi-level routines (step-16 in deal.II)
//#include <deal.II/multigrid/mg_dof_handler.h>
//#include <deal.II/multigrid/mg_constrained_dofs.h>
//#include <deal.II/multigrid/multigrid.h>
//#include <deal.II/multigrid/mg_transfer.h>
//#include <deal.II/multigrid/mg_tools.h>
//#include <deal.II/multigrid/mg_coarse.h>
//#include <deal.II/multigrid/mg_smoother.h>
//#include <deal.II/multigrid/mg_matrix.h>



namespace DOpE
{
  /**
   * Standard implementation of the object responsible
   * to construct the sparsitypattern.
   */
  template<template<int, int> class DH, int dim>
  class SparsityMaker
  {
  public:
    SparsityMaker(bool flux_pattern = false)
    {
      flux_pattern_ = flux_pattern;
    }
    virtual
    ~SparsityMaker()
    {
    }
    virtual void
    ComputeSparsityPattern(
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
      dealii::BlockSparsityPattern &sparsity,
      const dealii::ConstraintMatrix &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;

    virtual void
    ComputeSparsityPattern(
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
      dealii::SparsityPattern &sparsity,
      const dealii::ConstraintMatrix &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;


//      /*
//       * Experimental status:
//       * Needed for MG prec.
//       */
//      virtual void
//      ComputeMGSparsityPattern(
//             const DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler>& dof_handler,
//             dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_pattern,
//             const dealii::ConstraintMatrix& hanging_node_constraints,
//             const std::vector<unsigned int>& blocks,
//             const unsigned int n_levels) const;
//
//      /*
//       * Experimental status:
//       * Needed for MG prec.
//       */
//      virtual void
//  ComputeMGSparsityPattern(
//          const DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler>& dof_handler,
//    dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_pattern,
//          const dealii::ConstraintMatrix& hanging_node_constraints,
//          const std::vector<unsigned int>& blocks,
//    const unsigned int n_levels) const;
//

    //TODO: If one wishes to change the sparsity-pattern of the control, one
    //has to implement this here.
  private:
    bool flux_pattern_;
  };


  /***********************************************************/


  template<template<int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern(
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::BlockSparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
  {
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
            csp.block(i, j).reinit(blocks.at(i), blocks.at(j));
          }
      }
    csp.collect_sizes();
    if ( flux_pattern_ )
      {
        dealii::DoFTools::make_flux_sparsity_pattern(
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
      }
    else
      {
        dealii::DoFTools::make_sparsity_pattern(
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
      }
    sparsity.copy_from(csp);
  }

  /***********************************************************/
  template<template<int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern(
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::SparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
  {
    unsigned int total_dofs = 0;
    for (unsigned int j = 0; j < blocks.size(); j++)
      {
        total_dofs += blocks.at(j);
      }

#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
#else
    dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
#endif
    if ( flux_pattern_ )
      {
        dealii::DoFTools::make_flux_sparsity_pattern(
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
      }
    else
      {
        dealii::DoFTools::make_sparsity_pattern(
          dof_handler.GetDEALDoFHandler(), csp, hanging_node_constraints);
      }
    sparsity.copy_from(csp);
  }

///***********************************************************/
//  template<template<int, int> class DH, int dim>
//    void
//    SparsityMaker<DH, dim>
//    ::ComputeMGSparsityPattern(
//             const DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler>& dof_handler,
//             dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
//             const dealii::ConstraintMatrix& /*hanging_node_constraints*/,
//             const std::vector<unsigned int>& blocks,
//             const unsigned int n_levels) const
//    {
//      // Hard coded for FE_System with 1 FE and 2 components:
//      // MUST be changed later!!!!!
//      std::vector<unsigned int> block_component (dim,0);
//
//
//     std::vector<std::vector<unsigned int> >   mg_dofs_per_block;
//      mg_dofs_per_block.resize (n_levels);
//      mg_sparsity_patterns.resize(0, n_levels-1);
//      mg_dofs_per_block.resize (n_levels);
//
//
//      for (unsigned int level=0; level<n_levels; ++level)
//  mg_dofs_per_block[level].resize (blocks.size());
//
//      dealii::MGTools::count_dofs_per_block (dof_handler.GetDEALDoFHandler(),
//               mg_dofs_per_block,
//               block_component);
//
//      for (unsigned int level=0; level<n_levels; ++level)
//  {
//    dealii::BlockCompressedSparsityPattern csp(mg_dofs_per_block[level],
//               mg_dofs_per_block[level]);
//
//    dealii::MGTools::make_sparsity_pattern(dof_handler.GetDEALDoFHandler(), csp, level);
//
//    mg_sparsity_patterns[level].copy_from (csp);
//
//  }
//    }
//
///***********************************************************/
//
//
// template<template<int, int> class DH, int dim>
//    void
//    SparsityMaker<DH, dim>
//    ::ComputeMGSparsityPattern(
//             const DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler>& dof_handler,
//             dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
//             const dealii::ConstraintMatrix& /*hanging_node_constraints*/,
//             const std::vector<unsigned int>& /*blocks*/,
//             const unsigned int n_levels) const
//    {
//
//      for (unsigned int level=0; level<n_levels; ++level)
//  {
//    dealii::CompressedSparsityPattern csp(dof_handler.GetDEALDoFHandler().n_dofs(level), dof_handler.GetDEALDoFHandler().n_dofs(level));
//
//    dealii::MGTools::make_sparsity_pattern(dof_handler.GetDEALDoFHandler(), csp, level);
//
//    mg_sparsity_patterns[level].copy_from (csp);
//
//  }
//    }
//



} //end of namespace

#endif /* SPARSITYMAKER_H_ */
