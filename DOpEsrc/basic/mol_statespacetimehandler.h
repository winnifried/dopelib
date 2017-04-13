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

#ifndef MOL_STATESPACE_TIME_HANDLER_H_
#define MOL_STATESPACE_TIME_HANDLER_H_

#include <basic/statespacetimehandler.h>
#include <basic/constraints.h>
#include <include/sparsitymaker.h>
#include <include/userdefineddofconstraints.h>
#include <basic/sth_internals.h>
#include <container/refinementcontainer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/grid/grid_refinement.h>

namespace DOpE
{
  /**
   * Implements a Space Time Handler with a Method of Lines discretization.
   * This means there is only one fixed mesh for the spatial domain.
   * This Space Time Handler has knowlege of only one variable, namely the
   * solution to a PDE.
   *
   * For the detailed comments, please see the documentation of MethodOfLines_SpaceTimeHandler
   */
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN, typename VECTOR, int dealdim>
  class MethodOfLines_StateSpaceTimeHandler : public StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>
  {
  public:
    MethodOfLines_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation, const FE<dealdim, dealdim> &state_fe,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
        dealdim>(index_setter), sparse_mkr_dynamic_(true), triangulation_(
          triangulation), state_dof_handler_(triangulation_), state_fe_(
          &state_fe), mapping_(
          &DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), state_mesh_transfer_(
          NULL)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }
    MethodOfLines_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation, const FE<dealdim, dealdim> &state_fe,
      dealii::Triangulation<1> &times,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
        dealdim>(times, index_setter), sparse_mkr_dynamic_(true), triangulation_(
          triangulation), state_dof_handler_(triangulation_), state_fe_(
          &state_fe), mapping_(
          &DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), state_mesh_transfer_(
          NULL)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    MethodOfLines_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation,
      const DOpEWrapper::Mapping<dealdim, DH> &mapping,
      const FE<dealdim, dealdim> &state_fe,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
        dealdim>(index_setter), sparse_mkr_dynamic_(true), triangulation_(
          triangulation), state_dof_handler_(triangulation_), state_fe_(
          &state_fe), mapping_(&mapping), state_mesh_transfer_(NULL)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }
    MethodOfLines_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation,
      const DOpEWrapper::Mapping<dealdim, DH> &mapping,
      const FE<dealdim, dealdim> &state_fe,
      dealii::Triangulation<1> &times,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
        dealdim>(times, index_setter), sparse_mkr_dynamic_(true), triangulation_(
          triangulation), state_dof_handler_(triangulation_), state_fe_(
          &state_fe), mapping_(&mapping), state_mesh_transfer_(NULL)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    virtual
    ~MethodOfLines_StateSpaceTimeHandler()
    {
      state_dof_handler_.clear();

      if (state_mesh_transfer_ != NULL)
        {
          delete state_mesh_transfer_;
        }
      if (sparsitymaker_ != NULL && sparse_mkr_dynamic_ == true)
        {
          delete sparsitymaker_;
        }
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    void
    ReInit(unsigned int state_n_blocks,
           const std::vector<unsigned int> &state_block_component,
           const DirichletDescriptor &DD  )
    {

      StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>::SetActiveFEIndicesState(
        state_dof_handler_);
      state_dof_handler_.distribute_dofs(GetFESystem("state"));
//          DoFRenumbering::Cuthill_McKee(
//              static_cast<DH<dealdim, dealdim>&>(state_dof_handler_));
      DoFRenumbering::component_wise(
        static_cast<DH<dealdim, dealdim>&>(state_dof_handler_));

      state_dof_constraints_.clear();
      DoFTools::make_hanging_node_constraints(static_cast<DH<dealdim, dealdim>&>(state_dof_handler_), state_dof_constraints_);
      //TODO Dirichlet ueber Constraints
      if (GetUserDefinedDoFConstraints() != NULL) GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(state_dof_handler_, state_dof_constraints_);


      std::vector<unsigned int> dirichlet_colors = DD.GetDirichletColors();
      for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = DD.GetDirichletCompMask(color);

          //TODO: mapping[0] is a workaround, as deal does not support interpolate
          // boundary_values with a mapping collection at this point.
          VectorTools::interpolate_boundary_values(GetMapping()[0], state_dof_handler_.GetDEALDoFHandler(), color, dealii::ZeroFunction<dealdim>(comp_mask.size()),
                                                   state_dof_constraints_, comp_mask);
        }

      state_dof_constraints_.close();
      state_dofs_per_block_.resize(state_n_blocks);

      DoFTools::count_dofs_per_block(static_cast<DH<dealdim, dealdim>&>(state_dof_handler_), state_dofs_per_block_, state_block_component);

      support_points_.clear();

      //Initialize also the timediscretization.
      this->ReInitTime();

      //There where changes invalidate tickets
      this->IncrementStateTicket();
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const DOpEWrapper::DoFHandler<dealdim, DH> &
    GetStateDoFHandler() const
    {
      //There is only one mesh, hence always return this
      return state_dof_handler_;
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const DOpEWrapper::Mapping<dealdim, DH> &
    GetMapping() const
    {
      return *mapping_;
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandlerBase
     */
    const std::vector<unsigned int> &
    GetStateDoFsPerBlock(int /*time_point*/= -1) const
    {
      return state_dofs_per_block_;
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const dealii::ConstraintMatrix &
    GetStateDoFConstraints() const
    {
      return state_dof_constraints_;
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandlerBase
     */
    virtual void InterpolateState(VECTOR &result, const std::vector<VECTOR *> &local_vectors, double t, const TimeIterator &it) const
    {
      assert(it.get_left() <= t);
      assert(it.get_right() >= t);
      if (local_vectors.size() != 2) throw DOpEException("This function is currently not implemented for anything other than"
                                                           " linear interpolation of 2 DoFs.",
                                                           "MethodOfLine_SpaceTimeHandler::InterpolateState");

      double lambda_l = (it.get_right() - t) / it.get_k();
      double lambda_r = (t - it.get_left()) / it.get_k();

      //Here we assume that the numbering of dofs goes from left to right!
      result = *local_vectors[0];

      result.sadd(lambda_l, lambda_r, *local_vectors[1]);
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandlerBase
     */
    unsigned int GetStateNDoFs(int /*time_point*/= -1) const
    {
      return GetStateDoFHandler().n_dofs();
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const std::vector<Point<dealdim> > &
    GetMapDoFToSupportPoints()
    {
      support_points_.resize(GetStateNDoFs());
      DOpE::STHInternals::MapDoFsToSupportPoints<std::vector<Point<dealdim> >, dealdim>(this->GetMapping(), GetStateDoFHandler(), support_points_);
      return support_points_;
    }

    /******************************************************/
    void ComputeStateSparsityPattern(SPARSITYPATTERN &sparsity) const
    {
      this->GetSparsityMaker()->ComputeSparsityPattern(this->GetStateDoFHandler(), sparsity, this->GetStateDoFConstraints(),
                                                       this->GetStateDoFsPerBlock());
    }

    /******************************************************/

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const FE<dealdim, dealdim> &
    GetFESystem(std::string name) const
    {
      if (name == "state")
        {
          return *state_fe_;
        }
      else
        {
          abort();
          throw DOpEException("Not implemented for name =" + name, "MethodOfLines_StateSpaceTimeHandler::GetFESystem");
        }

    }
    /******************************************************/
    /**
     * This Function is used to refine the spatial and temporal mesh globally.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method, else one has to specify
     *                       additionally a RefinementContainer
     */
    void
    RefineSpaceTime(DOpEtypes::RefinementType ref_type =
                      DOpEtypes::RefinementType::global)
    {
      assert(ref_type == DOpEtypes::RefinementType::global);
      RefineSpace(ref_type);
      SpaceTimeHandlerBase<VECTOR>::RefineTime(ref_type);
    }

    /******************************************************/
    /**
     * This Function is used to refine the spatial mesh globally.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method, else one has to specify
     *                       additionally a RefinementContainer, see the alternative
     *                       RefineSpace method.
     */
    void RefineSpace(DOpEtypes::RefinementType /*ref_type*/= DOpEtypes::RefinementType::global)
    {
      //assert(ref_type == DOpEtypes::RefinementType::global);
      RefinementContainer ref_con_dummy;
      RefineSpace(ref_con_dummy);
    }

    /******************************************************/
    /**
     * This Function is used to refine the spatial mesh.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_container   Steers the local mesh refinement. Currently availabe are
     *                        RefinementContainer (for global refinement), RefineFixedFraction,
     *                        RefineFixedNumber and RefineOptimized.
     */

    void RefineSpace(const RefinementContainer &ref_container)
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

      //make sure that we do not use any coarsening
      assert( !ref_container.UsesCoarsening());

      if (state_mesh_transfer_ != NULL)
        {
          delete state_mesh_transfer_;
          state_mesh_transfer_ = NULL;
        }
      state_mesh_transfer_ = new dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> >(state_dof_handler_);

      if (DOpEtypes::RefinementType::global == ref_type)
        {
          triangulation_.set_all_refine_flags();
        }
      else if (DOpEtypes::RefinementType::fixed_number == ref_type)
        {
          GridRefinement::refine_and_coarsen_fixed_number(triangulation_, ref_container.GetLocalErrorIndicators(), ref_container.GetTopFraction(),
                                                          ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::fixed_fraction == ref_type)
        {

          GridRefinement::refine_and_coarsen_fixed_fraction(triangulation_, ref_container.GetLocalErrorIndicators(), ref_container.GetTopFraction(),
                                                            ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::optimized == ref_type)
        {

          GridRefinement::refine_and_coarsen_optimize(triangulation_, ref_container.GetLocalErrorIndicators(),
                                                      ref_container.GetConvergenceOrder());
        }
      else
        {
          throw DOpEException("Not implemented for name =" + DOpEtypesToString(ref_type),
                              "MethodOfLines_StateSpaceTimeHandler::RefineStateSpace");
        }
      triangulation_.prepare_coarsening_and_refinement();
      if (state_mesh_transfer_ != NULL) state_mesh_transfer_->prepare_for_pure_refinement();
      triangulation_.execute_coarsening_and_refinement();
    }
    /******************************************************/

    /**
     * Implementation of virtual function in StateSpaceTimeHandlerBase
     */
    unsigned int NewTimePointToOldTimePoint(unsigned int t) const
    {
      //TODO this has to be implemented when temporal refinement is possible!
      //At present the temporal grid can't be refined
      return t;
    }

    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    void SpatialMeshTransferState(const VECTOR &old_values, VECTOR &new_values) const
    {
      if (state_mesh_transfer_ != NULL) state_mesh_transfer_->refine_interpolate(old_values, new_values);
    }

    /******************************************************/
    /**
     * Through this function one commits a UserDefinedDoFConstraints
     * object to the class. With the help of the user_defined_dof_constr
     * one has the capability to  impose additional constraints
     * on the state-dofs (for example a pressure filter for the
     * stokes problem). This function must be called prior to
     * ReInit.
     */
    void SetUserDefinedDoFConstraints(UserDefinedDoFConstraints<DH, dealdim> &user_defined_dof_constr)
    {
      user_defined_dof_constr_ = &user_defined_dof_constr;
      user_defined_dof_constr_->RegisterMapping(this->GetMapping());
    }
    /******************************************************/
    /**
     * Through this function one commits a sparsity_maker
     * to the class. With the help of the sparsity_maker
     * one has the capability to create non-standard sparsity
     * patterns. This function must be called prior to
     * ReInit.
     */
    void SetSparsityMaker(SparsityMaker<DH, dealdim> &sparsity_maker)
    {
      assert(sparse_mkr_dynamic_ == true);  //If not true, we already set the sparsity maker
      if (sparsitymaker_ != NULL && sparse_mkr_dynamic_) delete sparsitymaker_;
      sparsitymaker_ = &sparsity_maker;
      sparse_mkr_dynamic_ = false;
    }

  private:
    const SparsityMaker<DH, dealdim> *
    GetSparsityMaker() const
    {
      return sparsitymaker_;
    }
    const UserDefinedDoFConstraints<DH, dealdim> *
    GetUserDefinedDoFConstraints() const
    {
      return user_defined_dof_constr_;
    }
    SparsityMaker<DH, dealdim> *sparsitymaker_;
    UserDefinedDoFConstraints<DH, dealdim> *user_defined_dof_constr_;
    bool sparse_mkr_dynamic_;

    dealii::Triangulation<dealdim> &triangulation_;
    DOpEWrapper::DoFHandler<dealdim, DH> state_dof_handler_;

    std::vector<unsigned int> state_dofs_per_block_;

    dealii::ConstraintMatrix state_dof_constraints_;

    const dealii::SmartPointer<const FE<dealdim, dealdim> > state_fe_; //TODO is there a reason that this is not a reference?
    const dealii::SmartPointer<
    const DOpEWrapper::Mapping<dealdim, DH> > mapping_;

    std::vector<Point<dealdim> > support_points_;
    dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> > *state_mesh_transfer_;

  };

}
#endif
