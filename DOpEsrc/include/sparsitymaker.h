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

#ifndef SPARSITYMAKER_H_
#define SPARSITYMAKER_H_


#include <deal.II/dofs/dof_tools.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#include <deal.II/lac/affine_constraints.h>
#else
#include <deal.II/lac/constraint_matrix.h>
#endif
#include <deal.II/lac/block_sparsity_pattern.h>
#include <wrapper/dofhandler_wrapper.h>
#include <include/helper.h>

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
#if DEAL_II_VERSION_GTE(9,3,0)
  template<int dim>
#else
  template<template<int, int> class DH, int dim>
#endif
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

#if DEAL_II_VERSION_GTE(9,1,1)
    virtual void
    ComputeSparsityPattern(
#if DEAL_II_VERSION_GTE(9,3,0)
      const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
      dealii::BlockSparsityPattern &sparsity,
      const dealii::AffineConstraints<double> &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;
#else
    virtual void
    ComputeSparsityPattern(
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
      dealii::BlockSparsityPattern &sparsity,
      const dealii::ConstraintMatrix &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;
#endif

#if DEAL_II_VERSION_GTE(9,1,1)
    virtual void
    ComputeSparsityPattern(
#if DEAL_II_VERSION_GTE(9,3,0)
      const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
      dealii::SparsityPattern &sparsity,
      const dealii::AffineConstraints<double> &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;
#else
    virtual void
    ComputeSparsityPattern(
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
      dealii::SparsityPattern &sparsity,
      const dealii::ConstraintMatrix &hanging_node_constraints,
      const std::vector<unsigned int> &blocks) const;
#endif

#ifdef DOPELIB_WITH_TRILINOS
#if DEAL_II_VERSION_GTE(9,1,1)
    virtual void
#if DEAL_II_VERSION_GTE(9,3,0)
      ComputeSparsityPattern (const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
      ComputeSparsityPattern (const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
                            dealii::TrilinosWrappers::BlockSparsityPattern &sparsity,
                            const dealii::AffineConstraints<double> &hanging_node_constraints,
                            const std::vector<unsigned int> &blocks,
                            const MPI_Comm mpi_comm = MPI_COMM_WORLD) const;

    virtual void
    ComputeSparsityPattern (			      
#if DEAL_II_VERSION_GTE(9,3,0)
      const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
      dealii::TrilinosWrappers::SparsityPattern &sparsity,
      const dealii::AffineConstraints<double> &hanging_node_constraints,
      const std::vector<unsigned int> &blocks,
      const MPI_Comm mpi_comm = MPI_COMM_WORLD) const;
#else
    virtual void
    ComputeSparsityPattern (const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
                            dealii::TrilinosWrappers::BlockSparsityPattern &sparsity,
                            const dealii::ConstraintMatrix &hanging_node_constraints,
                            const std::vector<unsigned int> &blocks,
                            const MPI_Comm mpi_comm = MPI_COMM_WORLD) const;

    virtual void
    ComputeSparsityPattern (const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
                            dealii::TrilinosWrappers::SparsityPattern &sparsity,
                            const dealii::ConstraintMatrix &hanging_node_constraints,
                            const std::vector<unsigned int> &blocks,
                            const MPI_Comm mpi_comm = MPI_COMM_WORLD) const;
#endif
#endif
    //TODO: If one wishes to change the sparsity-pattern of the control, one
    //has to implement this here.
  private:
    bool flux_pattern_;
  };


  /***********************************************************/

#if DEAL_II_VERSION_GTE(9,1,1)
#if DEAL_II_VERSION_GTE(9,3,0)
  template<int dim>
#else
  template<template<int, int> class DH, int dim>
#endif
  void
#if DEAL_II_VERSION_GTE(9,3,0)
    SparsityMaker<dim>::ComputeSparsityPattern(
#else
    SparsityMaker<DH, dim>::ComputeSparsityPattern(
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
    dealii::BlockSparsityPattern &sparsity,
    const dealii::AffineConstraints<double> &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
#else
  template<template<int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern(
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::BlockSparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
#endif
  {
    dealii::BlockDynamicSparsityPattern csp(blocks, blocks);

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
#if DEAL_II_VERSION_GTE(9,1,1)
#if DEAL_II_VERSION_GTE(9,3,0)
  template<int dim>
#else
  template<template<int, int> class DH, int dim>
#endif
  void
#if DEAL_II_VERSION_GTE(9,3,0)
    SparsityMaker<dim>::ComputeSparsityPattern(
#else
    SparsityMaker<DH, dim>::ComputeSparsityPattern(
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
    dealii::SparsityPattern &sparsity,
    const dealii::AffineConstraints<double> &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
#else
  template<template<int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern(
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::SparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<unsigned int> &blocks) const
#endif
  {
    unsigned int total_dofs = 0;
    for (unsigned int j = 0; j < blocks.size(); j++)
      {
        total_dofs += blocks.at(j);
      }

    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
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

#ifdef DOPELIB_WITH_TRILINOS
#if DEAL_II_VERSION_GTE(9,1,1)
#if DEAL_II_VERSION_GTE(9,3,0)
  template <int dim>
#else
  template <template <int, int> class DH, int dim>
#endif
  void
#if DEAL_II_VERSION_GTE(9,3,0)
    SparsityMaker<dim>::ComputeSparsityPattern (
#else
    SparsityMaker<DH, dim>::ComputeSparsityPattern (
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
    dealii::TrilinosWrappers::BlockSparsityPattern &sparsity,
    const dealii::AffineConstraints<double> &hanging_node_constraints,
    const std::vector<
    unsigned int> &blocks,
    const MPI_Comm mpi_comm) const
#else
  template <template <int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern (
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::TrilinosWrappers::BlockSparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<
    unsigned int> &blocks,
    const MPI_Comm mpi_comm) const
#endif
  {
    IndexSet locally_relevant;
    IndexSet locally_owned =
      dof_handler.GetDEALDoFHandler ().locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler.GetDEALDoFHandler (),
                                             locally_relevant);

    const auto block_owned = DOpEHelper::split_blockwise (locally_owned,
                                                          blocks);
    const auto block_relevant = DOpEHelper::split_blockwise (locally_relevant,
                                                             blocks);

    sparsity.reinit (block_owned, block_owned, block_relevant, mpi_comm);

    if (flux_pattern_)
      {
        dealii::DoFTools::make_flux_sparsity_pattern (
          dof_handler.GetDEALDoFHandler (), sparsity,
          hanging_node_constraints, false,
          dealii::Utilities::MPI::this_mpi_process (mpi_comm));
      }
    else
      {
        dealii::DoFTools::make_sparsity_pattern (
          dof_handler.GetDEALDoFHandler (), sparsity,
          hanging_node_constraints, false,
          dealii::Utilities::MPI::this_mpi_process (mpi_comm));
      }
    sparsity.compress ();
  }

  /***********************************************************/
#if DEAL_II_VERSION_GTE(9,1,1)
#if DEAL_II_VERSION_GTE(9,3,0)
  template <int dim>
#else
  template <template <int, int> class DH, int dim>
#endif
  void
#if DEAL_II_VERSION_GTE(9,3,0)
    SparsityMaker<dim>::ComputeSparsityPattern (
#else
    SparsityMaker<DH, dim>::ComputeSparsityPattern (
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dim> &dof_handler,
#else
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
    dealii::TrilinosWrappers::SparsityPattern &sparsity,
    const dealii::AffineConstraints<double> &hanging_node_constraints,
    const std::vector<
    unsigned int> &blocks,
    const MPI_Comm mpi_comm) const
#else
  template <template <int, int> class DH, int dim>
  void
  SparsityMaker<DH, dim>::ComputeSparsityPattern (
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::TrilinosWrappers::SparsityPattern &sparsity,
    const dealii::ConstraintMatrix &hanging_node_constraints,
    const std::vector<
    unsigned int> &blocks,
    const MPI_Comm mpi_comm) const
#endif
  {
    unsigned int total_dofs = 0;
    for (unsigned int j = 0; j < blocks.size (); j++)
      total_dofs += blocks.at (j);

    IndexSet locally_relevant;
    IndexSet locally_owned =
      dof_handler.GetDEALDoFHandler ().locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler.GetDEALDoFHandler (),
                                             locally_relevant);

    sparsity.reinit (locally_owned, locally_owned, locally_relevant,
                     mpi_comm);

    if (flux_pattern_)
      {
        dealii::DoFTools::make_flux_sparsity_pattern (
          dof_handler.GetDEALDoFHandler (), sparsity,
          hanging_node_constraints, false,
          dealii::Utilities::MPI::this_mpi_process (mpi_comm));
      }
    else
      {
        dealii::DoFTools::make_sparsity_pattern (
          dof_handler.GetDEALDoFHandler (), sparsity,
          hanging_node_constraints, false,
          dealii::Utilities::MPI::this_mpi_process (mpi_comm));
      }
    sparsity.compress ();
  }
#endif //Endof Dopelib_with_Trilinos

} //end of namespace

#endif /* SPARSITYMAKER_H_ */
