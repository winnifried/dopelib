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

#ifndef MOL_MULTIMESH_SPACE_TIME_HANDLER_H_
#define MOL_MULTIMESH_SPACE_TIME_HANDLER_H_

#include <basic/mol_spacetimehandler.h>
#include <basic/constraints.h>
#include <include/sparsitymaker.h>
#include <include/userdefineddofconstraints.h>
#include <basic/sth_internals.h>
#include <container/refinementcontainer.h>
#include <wrapper/solutiontransfer_wrapper.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>

namespace DOpE
{
  /**
   * Implements a Space Time Handler with a Method of Lines discretization.
   * This means there is only one fixed mesh for the spatial domain.
   * However, this space time handler allows to have different meshes for
   * control and state variable based upon the same coarse triangulation.
   *
   * Note that this makes sense only if dealdim=dopedim. This is why this
   * class has only one dim template argument
   *
   * For the detailed documentation, see MethodOfLines_SpaceTimeHandler
   */
  template<template<int, int> class FE, template<int, int> class DH,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  class MethodOfLines_MultiMesh_SpaceTimeHandler : public SpaceTimeHandler<FE,
    DH, SPARSITYPATTERN, VECTOR, dim, dim>
  {
  public:
    MethodOfLines_MultiMesh_SpaceTimeHandler(
      dealii::Triangulation<dim> &triangulation,
      const FE<dim, dim> &control_fe, const FE<dim, dim> &state_fe,
      DOpEtypes::ControlType type,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dim, dim> &index_setter =
        ActiveFEIndexSetterInterface<dim, dim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>(type,
                                                                  index_setter), state_triangulation_(triangulation), control_dof_handler_(
                                                                      control_triangulation_), state_dof_handler_(
                                                                          state_triangulation_), control_fe_(&control_fe), state_fe_(
                                                                              &state_fe), mapping_(
                                                                                  DOpEWrapper::StaticMappingQ1<dim, DH>::mapping_q1), constraints_(), control_mesh_transfer_(
                                                                                      NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(true)
    {
      control_triangulation_.copy_triangulation(state_triangulation_);
      sparsitymaker_ = new SparsityMaker<DH, dim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    MethodOfLines_MultiMesh_SpaceTimeHandler(
      dealii::Triangulation<dim> &triangulation,
      const FE<dim, dim> &control_fe, const FE<dim, dim> &state_fe,
      dealii::Triangulation<1> &times, DOpEtypes::ControlType type,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dim, dim> &index_setter =
        ActiveFEIndexSetterInterface<dim, dim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>(times,
                                                                  type, index_setter), state_triangulation_(triangulation), control_dof_handler_(
                                                                      control_triangulation_), state_dof_handler_(
                                                                          state_triangulation_), control_fe_(&control_fe), state_fe_(
                                                                              &state_fe), mapping_(
                                                                                  DOpEWrapper::StaticMappingQ1<dim, DH>::mapping_q1), constraints_(), control_mesh_transfer_(
                                                                                      NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(true)
    {
      control_triangulation_.copy_triangulation(state_triangulation_);
      sparsitymaker_ = new SparsityMaker<DH, dim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    MethodOfLines_MultiMesh_SpaceTimeHandler(
      dealii::Triangulation<dim> &triangulation,
      const FE<dim, dim> &control_fe, const FE<dim, dim> &state_fe,
      const Constraints &c, DOpEtypes::ControlType type,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dim, dim> &index_setter =
        ActiveFEIndexSetterInterface<dim, dim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>(type,
                                                                  index_setter), state_triangulation_(triangulation), control_dof_handler_(
                                                                      control_triangulation_), state_dof_handler_(
                                                                          state_triangulation_), control_fe_(&control_fe), state_fe_(
                                                                              &state_fe), mapping_(
                                                                                  DOpEWrapper::StaticMappingQ1<dim, DH>::mapping_q1), constraints_(
                                                                                      c), control_mesh_transfer_(NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(
                                                                                          true)
    {
      control_triangulation_.copy_triangulation(state_triangulation_);
      sparsitymaker_ = new SparsityMaker<DH, dim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    MethodOfLines_MultiMesh_SpaceTimeHandler(
      dealii::Triangulation<dim> &triangulation,
      const FE<dim, dim> &control_fe, const FE<dim, dim> &state_fe,
      dealii::Triangulation<1> &times, const Constraints &c,
      DOpEtypes::ControlType type,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dim, dim> &index_setter =
        ActiveFEIndexSetterInterface<dim, dim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>(times,
                                                                  type, index_setter), state_triangulation_(triangulation), control_dof_handler_(
                                                                      control_triangulation_), state_dof_handler_(
                                                                          state_triangulation_), control_fe_(&control_fe), state_fe_(
                                                                              &state_fe), mapping_(
                                                                                  DOpEWrapper::StaticMappingQ1<dim, DH>::mapping_q1), constraints_(
                                                                                      c), control_mesh_transfer_(NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(
                                                                                          true)
    {
      control_triangulation_.copy_triangulation(state_triangulation_);
      sparsitymaker_ = new SparsityMaker<DH, dim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
    }

    ~MethodOfLines_MultiMesh_SpaceTimeHandler()
    {
      control_dof_handler_.clear();

      state_dof_handler_.clear();

      if (control_mesh_transfer_ != NULL)
        {
          delete control_mesh_transfer_;
        }
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
     * Implementation of virtual function in SpaceTimeHandler
     */
    void
    ReInit(unsigned int control_n_blocks,
           const std::vector<unsigned int> &control_block_component,
           const DirichletDescriptor &DD_control,
           unsigned int state_n_blocks,
           const std::vector<unsigned int> &state_block_component,
           const DirichletDescriptor &DD_state)
    {
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>::SetActiveFEIndicesControl(
        control_dof_handler_);
      control_dof_handler_.distribute_dofs(*control_fe_);
      DoFRenumbering::component_wise(static_cast<DH<dim, dim>&>(control_dof_handler_));

      control_dof_constraints_.clear();
      DoFTools::make_hanging_node_constraints(
        static_cast<DH<dim, dim>&>(control_dof_handler_), control_dof_constraints_);
      if (GetUserDefinedDoFConstraints() != NULL)
        GetUserDefinedDoFConstraints()->MakeControlDoFConstraints(
          control_dof_handler_, control_dof_constraints_);

      {
        std::vector<unsigned int> dirichlet_colors = DD_control.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
          {
            unsigned int color = dirichlet_colors[i];
            std::vector<bool> comp_mask = DD_control.GetDirichletCompMask(color);

            //TODO: mapping[0] is a workaround, as deal does not support interpolate
            // boundary_values with a mapping collection at this point.
            dealii::VectorTools::interpolate_boundary_values(GetMapping()[0], control_dof_handler_.GetDEALDoFHandler(), color, dealii::ZeroFunction<dim>(comp_mask.size()),
                                                             control_dof_constraints_, comp_mask);
          }
      }

      control_dof_constraints_.close();

      control_dofs_per_block_.resize(control_n_blocks);
      {
        DoFTools::count_dofs_per_block(
          static_cast<DH<dim, dim>&>(control_dof_handler_), control_dofs_per_block_,
          control_block_component);
      }

      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dim, dim>::SetActiveFEIndicesState(
        state_dof_handler_);
      state_dof_handler_.distribute_dofs(GetFESystem("state"));
      DoFRenumbering::component_wise(static_cast<DH<dim, dim>&>(state_dof_handler_));

      state_dof_constraints_.clear();
      DoFTools::make_hanging_node_constraints(
        static_cast<DH<dim, dim>&>(state_dof_handler_), state_dof_constraints_);
      //TODO Dirichlet Daten hierueber.
      if (GetUserDefinedDoFConstraints() != NULL)
        GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(
          state_dof_handler_, state_dof_constraints_);
      {
        std::vector<unsigned int> dirichlet_colors = DD_state.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
          {
            unsigned int color = dirichlet_colors[i];
            std::vector<bool> comp_mask = DD_state.GetDirichletCompMask(color);

            //TODO: mapping[0] is a workaround, as deal does not support interpolate
            // boundary_values with a mapping collection at this point.
            VectorTools::interpolate_boundary_values(GetMapping()[0], state_dof_handler_.GetDEALDoFHandler(), color, dealii::ZeroFunction<dim>(comp_mask.size()),
                                                     state_dof_constraints_, comp_mask);
          }
      }

      state_dof_constraints_.close();

      state_dofs_per_block_.resize(state_n_blocks);
      DoFTools::count_dofs_per_block(static_cast<DH<dim, dim>&>(state_dof_handler_),
                                     state_dofs_per_block_, state_block_component);

      support_points_.clear();

      constraints_.ReInit(control_dofs_per_block_);
      //constraints_.ReInit(control_dofs_per_block_, state_dofs_per_block_);

      //Initialize also the timediscretization.
      this->ReInitTime();

      //There where changes invalidate tickets
      this->IncrementControlTicket();
      this->IncrementStateTicket();
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const DOpEWrapper::DoFHandler<dim, DH> &
    GetControlDoFHandler() const
    {
      //There is only one mesh, hence always return this
      return control_dof_handler_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const DOpEWrapper::DoFHandler<dim, DH> &
    GetStateDoFHandler() const
    {
      //There is only one mesh, hence always return this
      return state_dof_handler_;
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const DOpEWrapper::Mapping<dim, DH> &
    GetMapping() const
    {
      return mapping_;
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    unsigned int
    GetConstraintDoFsPerBlock(std::string name, unsigned int b) const
    {
      return (constraints_.GetDoFsPerBlock(name))[b];
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    const std::vector<unsigned int> &
    GetControlDoFsPerBlock(int /*time_point*/= -1) const
    {
      return control_dofs_per_block_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    const std::vector<unsigned int> &
    GetStateDoFsPerBlock(int /*time_point*/= -1) const
    {
      return state_dofs_per_block_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    const std::vector<unsigned int> &
    GetConstraintDoFsPerBlock(std::string name) const
    {
      return constraints_.GetDoFsPerBlock(name);
    }
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const dealii::ConstraintMatrix &
    GetControlDoFConstraints() const
    {
      return control_dof_constraints_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const dealii::ConstraintMatrix &
    GetStateDoFConstraints() const
    {
      return state_dof_constraints_;
    }

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */

    virtual void
    InterpolateControl(VECTOR &result,
                       const std::vector<VECTOR *> &local_vectors, double t,
                       const TimeIterator &it) const
    {
      assert(it.get_left() <= t);
      assert(it.get_right() >= t);
      if (local_vectors.size() != 2)
        throw DOpEException(
          "This function is currently not implemented for anything other than"
          " linear interpolation of 2 DoFs.",
          "MethodOfLine_SpaceTimeHandler::InterpolateControl");

      double lambda_l = (it.get_right() - t) / it.get_k();
      double lambda_r = (t - it.get_left()) / it.get_k();

      //Here we assume that the numbering of dofs goes from left to right!
      result = *local_vectors[0];

      result.sadd(lambda_l, lambda_r, *local_vectors[1]);
    }

    virtual void
    InterpolateState(VECTOR &result,
                     const std::vector<VECTOR *> &local_vectors, double t,
                     const TimeIterator &it) const
    {
      assert(it.get_left() <= t);
      assert(it.get_right() >= t);
      if (local_vectors.size() != 2)
        throw DOpEException(
          "This function is currently not implemented for anything other than"
          " linear interpolation of 2 DoFs.",
          "MethodOfLine_SpaceTimeHandler::InterpolateState");

      double lambda_l = (it.get_right() - t) / it.get_k();
      double lambda_r = (t - it.get_left()) / it.get_k();

      //Here we assume that the numbering of dofs goes from left to right!
      result = *local_vectors[0];

      result.sadd(lambda_l, lambda_r, *local_vectors[1]);
    }

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetControlNDoFs(int /*time_point*/= -1) const
    {
      return GetControlDoFHandler().n_dofs();
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetStateNDoFs(int /*time_point*/= -1) const
    {
      return GetStateDoFHandler().n_dofs();
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetConstraintNDoFs(std::string name) const
    {
      return constraints_.n_dofs(name);
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetNGlobalConstraints() const
    {
      return constraints_.n_dofs("global");
      //return constraints_.global_n_dofs();
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetNLocalConstraints() const
    {
      return constraints_.n_dofs("local");
      //return constraints_.local_n_dofs();
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const std::vector<Point<dim> > &
    GetMapDoFToSupportPoints()
    {
      support_points_.resize(GetStateNDoFs());
      DOpE::STHInternals::MapDoFsToSupportPoints(this->GetMapping(),
                                                 GetStateDoFHandler(), support_points_);
      return support_points_;
    }

    /******************************************************/
    void
    ComputeControlSparsityPattern(SPARSITYPATTERN &sparsity) const;

    /******************************************************/
    void
    ComputeStateSparsityPattern(SPARSITYPATTERN &sparsity) const
    {
      this->GetSparsityMaker()->ComputeSparsityPattern(
        this->GetStateDoFHandler(), sparsity,
        this->GetStateDoFConstraints(), this->GetStateDoFsPerBlock());
    }

    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const FE<dim, dim> &
    GetFESystem(std::string name) const
    {
      if (name == "state")
        {
          return *state_fe_;
        }
      else if (name == "control")
        {
          return *control_fe_;
        }
      else
        {
          throw DOpEException("Not implemented for name =" + name,
                              "MethodOfLines_MultiMesh_SpaceTimeHandler::GetFESystem");
        }

    }

    /**
     * This Function is used to refine the spatial mesh for both the state and the control.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method, else one has to specify
     *                       additionally a RefinementContainer, see the alternative
     *                       RefineSpace method.
     */
    void
    RefineSpace(DOpEtypes::RefinementType ref_type =
                  DOpEtypes::RefinementType::global)
    {
      assert(ref_type == DOpEtypes::RefinementType::global);
      RefinementContainer ref_con_dummy;
      RefineStateSpace(ref_con_dummy);
      RefineControlSpace(ref_con_dummy);
    }

    /**
     * This Function is used to refine the spatial mesh for both the state.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method, else one has to specify
     *                       additionally a RefinementContainer, see the alternative
     *                       RefineSpace method.
     */
    void
    RefineStateSpace(DOpEtypes::RefinementType /*ref_type*/ =
                       DOpEtypes::RefinementType::global)
    {
      //assert(ref_type == DOpEtypes::RefinementType::global);
      RefinementContainer ref_con_dummy;
      RefineStateSpace(ref_con_dummy);
    }

    /**
     * This Function is used to refine the spatial mesh for  the control.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method, else one has to specify
     *                       additionally a RefinementContainer, see the alternative
     *                       RefineSpace method.
     */
    void
    RefineControlSpace(DOpEtypes::RefinementType /*ref_type*/ =
                         DOpEtypes::RefinementType::global)
    {
      //assert(ref_type == DOpEtypes::RefinementType::global);
      RefinementContainer ref_con_dummy;
      RefineControlSpace(ref_con_dummy);
    }

    /**
     * This Function is used to refine the spatial mesh for both the state and the control.
     * After calling a refinement function a reinitialization is required!
     *
     *
     * @param ref_container   Steers the local mesh refinement. Currently availabe are
     *                        RefinementContainer (for global refinement), RefineFixedFraction,
     *                        RefineFixedNumber and RefineOptimized.
     */
    template<typename NUMBER>
    void
    RefineSpace(const RefinementContainer &ref_container)
    {
      RefineStateSpace(ref_container);
      RefineControlSpace(ref_container);
    }

    /**
     * This Function is used to refine the spatial mesh for the state.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_container   Steers the local mesh refinement. Currently availabe are
     *                        RefinementContainer (for global refinement), RefineFixedFraction,
     *                        RefineFixedNumber and RefineOptimized.
     */
    void
    RefineStateSpace(const RefinementContainer &ref_container)
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

      //make sure that we do not use any coarsening
      assert(!ref_container.UsesCoarsening());

      if (state_mesh_transfer_ != NULL)
        {
          delete state_mesh_transfer_;
          state_mesh_transfer_ = NULL;
        }
      state_mesh_transfer_ = new DOpEWrapper::SolutionTransfer<dim, VECTOR,
      DH>(state_dof_handler_);

      if (DOpEtypes::RefinementType::global == ref_type)
        {
          state_triangulation_.set_all_refine_flags();
        }
      else if (DOpEtypes::RefinementType::fixed_number == ref_type)
        {

          GridRefinement::refine_and_coarsen_fixed_number(
            state_triangulation_, ref_container.GetLocalErrorIndicators(),
            ref_container.GetTopFraction(),
            ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::fixed_fraction == ref_type)
        {

          GridRefinement::refine_and_coarsen_fixed_fraction(
            state_triangulation_, ref_container.GetLocalErrorIndicators(),
            ref_container.GetTopFraction(),
            ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::optimized == ref_type)
        {

          GridRefinement::refine_and_coarsen_optimize(state_triangulation_,
                                                      ref_container.GetLocalErrorIndicators(),
                                                      ref_container.GetConvergenceOrder());
        }
      else if (DOpEtypes::RefinementType::finest_of_both == ref_type)
        {
          this->FlagIfLeftIsNotFinest(state_triangulation_,
                                      control_triangulation_);
        }
      else
        {
          throw DOpEException("Not implemented for name =" + DOpEtypesToString(ref_type),
                              "MethodOfLines_MultiMesh_SpaceTimeHandler::RefineSpace");
        }
      state_triangulation_.prepare_coarsening_and_refinement();
      if (state_mesh_transfer_ != NULL)
        state_mesh_transfer_->prepare_for_pure_refinement();

      state_triangulation_.execute_coarsening_and_refinement();
    }

    /**
     * This Function is used to refine the spatial mesh for control.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_container   Steers the local mesh refinement. Currently availabe are
     *                        RefinementContainer (for global refinement), RefineFixedFraction,
     *                        RefineFixedNumber and RefineOptimized.
     */
    void
    RefineControlSpace(const RefinementContainer &ref_container)
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

      //make sure that we do not use any coarsening
      assert(!ref_container.UsesCoarsening());

      if (control_mesh_transfer_ != NULL)
        {
          delete control_mesh_transfer_;
          control_mesh_transfer_ = NULL;
        }
#if dope_dimension == deal_II_dimension
      control_mesh_transfer_ = new DOpEWrapper::SolutionTransfer<dim, VECTOR,
      DH>(control_dof_handler_);
#endif
      if (DOpEtypes::RefinementType::global == ref_type)
        {
          control_triangulation_.set_all_refine_flags();
        }

      else if (DOpEtypes::RefinementType::fixed_number == ref_type)
        {

          GridRefinement::refine_and_coarsen_fixed_number(
            control_triangulation_, ref_container.GetLocalErrorIndicators(),
            ref_container.GetTopFraction(),
            ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::fixed_fraction == ref_type)
        {

          GridRefinement::refine_and_coarsen_fixed_fraction(
            control_triangulation_, ref_container.GetLocalErrorIndicators(),
            ref_container.GetTopFraction(),
            ref_container.GetBottomFraction());
        }
      else if (DOpEtypes::RefinementType::optimized == ref_type)
        {

          GridRefinement::refine_and_coarsen_optimize(control_triangulation_,
                                                      ref_container.GetLocalErrorIndicators(),
                                                      ref_container.GetConvergenceOrder());
        }
      else if ("finest-of-both")
        {
          this->FlagIfLeftIsNotFinest(control_triangulation_,
                                      state_triangulation_);
        }
      else
        {
          throw DOpEException("Not implemented for name =" + DOpEtypesToString(ref_type),
                              "MethodOfLines_MultiMesh_SpaceTimeHandler::RefineSpace");
        }
      control_triangulation_.prepare_coarsening_and_refinement();

      //FIXME: works only if no coarsening happens, because we do not have the vectors to be interpolated availiable...
      if (control_mesh_transfer_ != NULL)
        control_mesh_transfer_->prepare_for_pure_refinement();

      control_triangulation_.execute_coarsening_and_refinement();
    }
    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    NewTimePointToOldTimePoint(unsigned int t) const
    {
      //TODO this has to be implemented when temporal refinement is possible!
      //At present the temporal grid can't be refined
      return t;
    }

    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    void
    SpatialMeshTransferControl(const VECTOR &old_values,
                               VECTOR &new_values) const
    {
      if (control_mesh_transfer_ != NULL)
        control_mesh_transfer_->refine_interpolate(old_values, new_values);
    }
    /******************************************************/

    /**
     * Transfer of the State Vectors
     */
    void
    SpatialMeshTransferState(const VECTOR &old_values,
                             VECTOR &new_values) const
    {
      if (state_mesh_transfer_ != NULL)
        state_mesh_transfer_->refine_interpolate(old_values, new_values);
    }
    /******************************************************/
    /**
     * Through this function one commits a constraints_maker
     * to the class. With the help of the constraints_maker
     * one has the capability to  impose additional constraints
     * on the state-dofs (for example a pressure filter for the
     * stokes problem). This function must be called prior to
     * ReInit.
     */
    void
    SetUserDefinedDoFConstraints(
      UserDefinedDoFConstraints<DH, dim> &constraints_maker)
    {
      user_defined_dof_constr_ = &constraints_maker;
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
    void
    SetSparsityMaker(SparsityMaker<DH, dim> &sparsity_maker)
    {
      assert(sparse_mkr_dynamic_==true); //If not true, we already set the sparsity maker
      if (sparsitymaker_ != NULL && sparse_mkr_dynamic_)
        delete sparsitymaker_;
      sparsitymaker_ = &sparsity_maker;
      sparse_mkr_dynamic_ = false;
    }
    /******************************************************/
    /**
     * Through this function one can reinitialize the
     * triangulation for the state variable to be a copy of the
     * given argument.
     */

    void
    ResetStateTriangulation(const dealii::Triangulation<dim> &tria)
    {
      state_dof_handler_.clear();
      state_triangulation_.clear();
      state_triangulation_.copy_triangulation(tria);
      state_dof_handler_.initialize(state_triangulation_, *state_fe_);
      this->IncrementStateTicket();
      if (state_mesh_transfer_ != NULL)
        delete state_mesh_transfer_;
      state_mesh_transfer_ = NULL;
    }

    /******************************************************/
    /**
     * Through this function one can reinitialize the
     * triangulation for the state variable to be a copy of the
     * given argument.
     */

    void
    ResetControlTriangulation(const dealii::Triangulation<dim> &tria)
    {
      control_dof_handler_.clear();
      control_triangulation_.clear();
      control_triangulation_.copy_triangulation(tria);
      control_dof_handler_.initialize(control_triangulation_, *control_fe_);
      this->IncrementControlTicket();
      if (control_mesh_transfer_ != NULL)
        delete control_mesh_transfer_;
      control_mesh_transfer_ = NULL;
    }

  private:
    const SparsityMaker<DH, dim> *
    GetSparsityMaker() const
    {
      return sparsitymaker_;
    }
    const UserDefinedDoFConstraints<DH, dim> *
    GetUserDefinedDoFConstraints() const
    {
      return user_defined_dof_constr_;
    }

    void
    FlagIfLeftIsNotFinest(dealii::Triangulation<dim> &left,
                          const dealii::Triangulation<dim> &right);

    SparsityMaker<DH, dim> *sparsitymaker_;
    UserDefinedDoFConstraints<DH, dim> *user_defined_dof_constr_;

    dealii::Triangulation<dim> control_triangulation_;
    dealii::Triangulation<dim> &state_triangulation_;
    DOpEWrapper::DoFHandler<dim, DH> control_dof_handler_;
    DOpEWrapper::DoFHandler<dim, DH> state_dof_handler_;

    std::vector<unsigned int> control_dofs_per_block_;
    std::vector<unsigned int> state_dofs_per_block_;

    dealii::ConstraintMatrix control_dof_constraints_;
    dealii::ConstraintMatrix state_dof_constraints_;

    const dealii::SmartPointer<const FE<dim, dim>> control_fe_;
    const dealii::SmartPointer<const FE<dim, dim>> state_fe_;

    DOpEWrapper::Mapping<dim, DH> mapping_;

    std::vector<Point<dim> > support_points_;

    Constraints constraints_;
    DOpEWrapper::SolutionTransfer<dim, VECTOR,DH> *control_mesh_transfer_;
    DOpEWrapper::SolutionTransfer<dim, VECTOR,DH> *state_mesh_transfer_;
    bool sparse_mkr_dynamic_;
  };

  /**************************explicit instantiation*************/
#if dope_dimension == deal_II_dimension
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
  void
  DOpE::MethodOfLines_MultiMesh_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension>::ComputeControlSparsityPattern(
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

    dealii::DoFTools::make_sparsity_pattern(
      static_cast<const dealii::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
      csp);
    this->GetControlDoFConstraints().condense(csp);
    sparsity.copy_from(csp);
  }

  /******************************************************/

  template<>
  void
  MethodOfLines_MultiMesh_SpaceTimeHandler<dealii::FESystem,
                                           dealii::DoFHandler, dealii::SparsityPattern, dealii::Vector<double>,
                                           dope_dimension>::ComputeControlSparsityPattern(
                                             dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
#else
    dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
#endif
    dealii::DoFTools::make_sparsity_pattern(
      static_cast<const dealii::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
      csp);
    this->GetControlDoFConstraints().condense(csp);
    sparsity.copy_from(csp);
  }

  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
  void
  DOpE::MethodOfLines_MultiMesh_SpaceTimeHandler<dealii::hp::FECollection,
       dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension>::ComputeControlSparsityPattern(
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

    dealii::DoFTools::make_sparsity_pattern(
      static_cast<const dealii::hp::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
      csp);
    this->GetControlDoFConstraints().condense(csp);
    sparsity.copy_from(csp);
  }

  /******************************************************/

  template<>
  void
  MethodOfLines_MultiMesh_SpaceTimeHandler<dealii::hp::FECollection,
                                           dealii::hp::DoFHandler, dealii::SparsityPattern, dealii::Vector<double>,
                                           dope_dimension>::ComputeControlSparsityPattern(
                                             dealii::SparsityPattern &sparsity) const
  {
    const unsigned int total_dofs = this->GetControlNDoFs();
#if DEAL_II_VERSION_GTE(8,3,0)
    dealii::DynamicSparsityPattern csp(total_dofs, total_dofs);
#else
    dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);
#endif
    dealii::DoFTools::make_sparsity_pattern(
      static_cast<const dealii::hp::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
      csp);
    this->GetControlDoFConstraints().condense(csp);
    sparsity.copy_from(csp);
  }
#endif //Endof explicit instanciation
  /*******************************************************/
  template<template<int, int> class FE, template<int, int> class DH,
  typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  MethodOfLines_MultiMesh_SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
                                           dim>::FlagIfLeftIsNotFinest(dealii::Triangulation<dim> &left,
                                               const dealii::Triangulation<dim> &right)
  {
    auto element_list = GridTools::get_finest_common_cells(left, right);
    auto element_iter = element_list.begin();
    for (; element_iter != element_list.end(); element_iter++)
      {
        if (element_iter->second->has_children())
          {
            //left is not finest, so we should flag the left element to be refined
            element_iter->first->set_refine_flag();
          }
      }
  }
}

#endif
