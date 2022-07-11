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

#ifndef MOL_SPACE_TIME_HANDLER_H_
#define MOL_SPACE_TIME_HANDLER_H_

#include <basic/spacetimehandler.h>
#include <basic/constraints.h>
#include <include/sparsitymaker.h>
#include <include/userdefineddofconstraints.h>
#include <basic/sth_internals.h>
#include <wrapper/mapping_wrapper.h>
#include <container/refinementcontainer.h>
#include <wrapper/solutiontransfer_wrapper.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/hp/mapping_collection.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#include <deal.II/lac/affine_constraints.h>
#else
#include <deal.II/lac/constraint_matrix.h>
#endif
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

namespace DOpE
{
  /**
   * Implements a Space Time Handler with a Method of Lines discretization.
   * This means there is only one fixed mesh for the spatial domain.
   */
#if DEAL_II_VERSION_GTE(9,3,0)
  template<template <int, int> class FE, bool DH, typename SPARSITYPATTERN,
#else
  template<template <int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
#endif
  typename VECTOR, int dopedim, int dealdim>
    class MethodOfLines_SpaceTimeHandler : public SpaceTimeHandler<FE, DH,
    SPARSITYPATTERN, VECTOR, dopedim, dealdim>
  {
  public:
    /**
    * Constructor used for stationary PDEs and stationary optimization problems without any
    * further constraints beyond the PDE.
     *
     * @param triangulation     The coars triangulation to be used.
     * @param control_fe        The finite elements used for the discretization of the control variable.
     * @param state_fe          The finite elements used for the discretization of the state variable.
     * @param type              The type of the control, see dopetypes.h for more information.
    * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
     * @param index_setter      The index setter object (only needed in case of hp elements).
     */
    MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim> &triangulation,
                                   const FE<dealdim, dealdim> &control_fe,
                                   const FE<dealdim, dealdim> &state_fe,
                                   DOpEtypes::VectorAction type,
                                   bool flux_pattern = false,
                                   const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter =
                                     ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
    SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(type, index_setter),
      triangulation_(triangulation),
      control_dof_handler_(triangulation_),
      state_dof_handler_(triangulation_),
      control_fe_(&control_fe),
      state_fe_(&state_fe),
      mapping_(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1),
      constraints_(),
      control_mesh_transfer_(NULL),
      state_mesh_transfer_(NULL),
      sparse_mkr_dynamic_(true)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      sparsitymaker_ = new SparsityMaker<dealdim>(flux_pattern);
#else
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
#endif
      user_defined_dof_constr_ = NULL;
    }

    /**
    * Constructor used for nonstationary PDEs and nonstationary optimization problems without any
    * further constraints beyond the PDE.
     *
     * @param triangulation     The coars triangulation to be used.
     * @param control_fe        The finite elements used for the discretization of the control variable.
     * @param state_fe          The finite elements used for the discretization of the state variable.
     * @param times             The timegrid for instationary problems.
     * @param type              The type of the control, see dopetypes.h for more information.
     * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
     * @param index_setter      The index setter object (only needed in case of hp elements).
     */
    MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim> &triangulation,
                                   const FE<dealdim, dealdim> &control_fe,
                                   const FE<dealdim, dealdim> &state_fe,
                                   dealii::Triangulation<1> &times,
                                   DOpEtypes::VectorAction type,
                                   bool flux_pattern = false,
                                   const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter =
                                     ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(times, type, index_setter),
      triangulation_(triangulation),
      control_dof_handler_(triangulation_),
      state_dof_handler_(triangulation_),
      control_fe_(&control_fe),
      state_fe_(&state_fe),
      mapping_(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1),
      constraints_(),
      control_mesh_transfer_(NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(true)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      sparsitymaker_ = new SparsityMaker<dealdim>(flux_pattern);
#else
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
#endif
      user_defined_dof_constr_ = NULL;
    }

    /**
     * Constructor used for stationary optimization problems with additional constraints.
           *
           * @param triangulation     The coars triangulation to be used.
           * @param control_fe        The finite elements used for the discretization of the control variable.
           * @param state_fe          The finite elements used for the discretization of the state variable.
           * @param constraints       An object describing the constraints imposed on an optimization
     *                          problem in term of the number of unknowns in the control and state.
           * @param type              The type of the control, see dopetypes.h for more information.
           * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
           * @param index_setter      The index setter object (only needed in case of hp elements).
           */
    MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim> &triangulation,
                                   const FE<dealdim, dealdim> &control_fe,
                                   const FE<dealdim, dealdim> &state_fe,
                                   const Constraints &c,
                                   DOpEtypes::VectorAction type,
                                   bool flux_pattern = false,
                                   const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter =
                                     ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(type, index_setter),
      triangulation_(triangulation),
      control_dof_handler_(triangulation_),
      state_dof_handler_(triangulation_),
      control_fe_(&control_fe),
      state_fe_(&state_fe),
      mapping_(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1),
      constraints_(c),
      control_mesh_transfer_(NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(true)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      sparsitymaker_ = new SparsityMaker<dealdim>(flux_pattern);
#else      
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
#endif
      user_defined_dof_constr_ = NULL;
    }

    /**
    * Constructor used for nonstationary optimization problems with additional constraints.
     *
     * @param triangulation     The coars triangulation to be used.
     * @param control_fe        The finite elements used for the discretization of the control variable.
     * @param state_fe          The finite elements used for the discretization of the state variable.
     * @param times             The timegrid for instationary problems.
     * @param constraints       An object describing the constraints imposed on an optimization
    *                          problem in term of the number of unknowns in the control and state.
     * @param type              The type of the control, see dopetypes.h for more information.
     * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
     * @param index_setter      The index setter object (only needed in case of hp elements).
     */
    MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim> &triangulation,
                                   const FE<dealdim, dealdim> &control_fe,
                                   const FE<dealdim, dealdim> &state_fe,
                                   dealii::Triangulation<1> &times,
                                   const Constraints &c,
                                   DOpEtypes::VectorAction type,
                                   bool flux_pattern = false,
                                   const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter =
                                     ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(times, type, index_setter),
      triangulation_(triangulation),
      control_dof_handler_(triangulation_),
      state_dof_handler_(triangulation_),
      control_fe_(&control_fe),
      state_fe_(&state_fe),
      mapping_(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1),
      constraints_(c),
      control_mesh_transfer_(NULL), state_mesh_transfer_(NULL), sparse_mkr_dynamic_(true)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      sparsitymaker_ = new SparsityMaker<dealdim>(flux_pattern);
#else
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
#endif
      user_defined_dof_constr_ = NULL;
    }

        virtual
    ~MethodOfLines_SpaceTimeHandler()
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
#if dope_dimension > 0
           const DirichletDescriptor &DD_control,
#else
           const DirichletDescriptor &,
#endif
           unsigned int state_n_blocks,
           const std::vector<unsigned int> &state_block_component,
           const DirichletDescriptor &DD_state)
    {

#if dope_dimension > 0
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::SetActiveFEIndicesControl(control_dof_handler_);
#endif
      control_dof_handler_.distribute_dofs(GetFESystem("control"));

#if dope_dimension > 0
#if DEAL_II_VERSION_GTE(9,3,0)
      DoFRenumbering::component_wise (static_cast<dealii::DoFHandler<dopedim, dopedim>&>(control_dof_handler_),control_block_component);
#else
      DoFRenumbering::component_wise (static_cast<DH<dopedim, dopedim>&>(control_dof_handler_),control_block_component);
#endif
      if (dopedim==dealdim)
        {
	  control_hn_constraints_.clear ();
	  control_hn_constraints_.reinit(this->GetLocallyRelevantDoFs(DOpEtypes::VectorType::control));
          DoFTools::make_hanging_node_constraints (
#if DEAL_II_VERSION_GTE(9,3,0)
	    static_cast<dealii::DoFHandler<dopedim, dopedim>&>(control_dof_handler_),
#else
	    static_cast<DH<dopedim, dopedim>&>(control_dof_handler_),
#endif
	    control_hn_constraints_);
	  
          control_dof_constraints_.clear ();
	  control_dof_constraints_.reinit(this->GetLocallyRelevantDoFs(DOpEtypes::VectorType::control));
          DoFTools::make_hanging_node_constraints (
#if DEAL_II_VERSION_GTE(9,3,0)
	    static_cast<dealii::DoFHandler<dopedim, dopedim>&>(control_dof_handler_),
#else
	    static_cast<DH<dopedim, dopedim>&>(control_dof_handler_),
#endif
	    control_dof_constraints_);
          if (GetUserDefinedDoFConstraints() != NULL)
            GetUserDefinedDoFConstraints()->MakeControlDoFConstraints(control_dof_handler_,
                                                                      control_dof_constraints_);

          std::vector<unsigned int> dirichlet_colors = DD_control.GetDirichletColors();
          for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
            {
              unsigned int color = dirichlet_colors[i];
              std::vector<bool> comp_mask = DD_control.GetDirichletCompMask(color);

              //TODO: mapping[0] is a workaround, as deal does not support interpolate
              // boundary_values with a mapping collection at this point.
#if DEAL_II_VERSION_GTE(9,0,0)
              dealii::VectorTools::interpolate_boundary_values(GetMapping()[0], control_dof_handler_.GetDEALDoFHandler(), color, dealii::Functions::ZeroFunction<dopedim>(comp_mask.size()),
                                                               control_dof_constraints_, comp_mask);
#else
              dealii::VectorTools::interpolate_boundary_values(GetMapping()[0], control_dof_handler_.GetDEALDoFHandler(), color, dealii::ZeroFunction<dopedim>(comp_mask.size()),
                                                               control_dof_constraints_, comp_mask);
#endif	      
            }
          control_hn_constraints_.close ();
          control_dof_constraints_.close ();
        }
      else
        {
          throw DOpEException("Not implemented for dopedim != dealdim","MethodOfLines_SpaceTimeHandler::ReInit");
        }
#endif
      control_dofs_per_block_.resize(control_n_blocks);
#if dope_dimension > 0
      {
#if DEAL_II_VERSION_GTE(9,2,0)
	control_dofs_per_block_ = DoFTools::count_dofs_per_fe_block (
#if DEAL_II_VERSION_GTE(9,3,0)
	  static_cast<dealii::DoFHandler<dopedim, dopedim>&>(control_dof_handler_),
#else
	  static_cast<DH<dopedim, dopedim>&>(control_dof_handler_),
#endif
	  control_block_component);
#else
	DoFTools::count_dofs_per_block (
	  static_cast<DH<dopedim, dopedim>&>(control_dof_handler_),
	  control_dofs_per_block_,control_block_component);
#endif //dealii older than 9.2.0
        
      }
#else
      {
        for (unsigned int i = 0; i < control_dofs_per_block_.size(); i++)
          {
            control_dofs_per_block_[i] = 0;
          }
        for (unsigned int i = 0; i < control_block_component.size(); i++)
          {
            control_dofs_per_block_[control_block_component[i]]++;
          }
      }
#endif
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::SetActiveFEIndicesState(
        state_dof_handler_);
      state_dof_handler_.distribute_dofs(GetFESystem("state"));
#if DEAL_II_VERSION_GTE(9,3,0)
      DoFRenumbering::component_wise(static_cast<dealii::DoFHandler<dealdim, dealdim>&>(state_dof_handler_),state_block_component);
#else
      DoFRenumbering::component_wise(static_cast<DH<dealdim, dealdim>&>(state_dof_handler_),state_block_component);
#endif
      
      state_hn_constraints_.clear();
      state_hn_constraints_.reinit (
        this->GetLocallyRelevantDoFs (DOpEtypes::VectorType::state));
      DoFTools::make_hanging_node_constraints(
#if DEAL_II_VERSION_GTE(9,3,0)
	static_cast<dealii::DoFHandler<dealdim, dealdim>&>(state_dof_handler_),
#else
	static_cast<DH<dealdim, dealdim>&>(state_dof_handler_),
#endif
        state_hn_constraints_);

      state_dof_constraints_.clear();
      state_dof_constraints_.reinit (
        this->GetLocallyRelevantDoFs (DOpEtypes::VectorType::state));
      DoFTools::make_hanging_node_constraints(
#if DEAL_II_VERSION_GTE(9,3,0)
	static_cast<dealii::DoFHandler<dealdim, dealdim>&>(state_dof_handler_),
#else
	static_cast<DH<dealdim, dealdim>&>(state_dof_handler_),
#endif
        state_dof_constraints_);
      //TODO Dirichlet ueber Constraints
      if (GetUserDefinedDoFConstraints() != NULL)
        GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(
          state_dof_handler_, state_dof_constraints_);


      std::vector<unsigned int> dirichlet_colors = DD_state.GetDirichletColors();
      for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = DD_state.GetDirichletCompMask(color);

          //TODO: mapping[0] is a workaround, as deal does not support interpolate
          // boundary_values with a mapping collection at this point.
#if DEAL_II_VERSION_GTE(9,0,0)
          VectorTools::interpolate_boundary_values(GetMapping()[0], state_dof_handler_.GetDEALDoFHandler(), color, dealii::Functions::ZeroFunction<dealdim>(comp_mask.size()),
                                                   state_dof_constraints_, comp_mask);
#else
          VectorTools::interpolate_boundary_values(GetMapping()[0], state_dof_handler_.GetDEALDoFHandler(), color, dealii::ZeroFunction<dealdim>(comp_mask.size()),
                                                   state_dof_constraints_, comp_mask);
#endif	  
        }
      state_hn_constraints_.close();
      state_dof_constraints_.close();

      state_dofs_per_block_.resize(state_n_blocks);
#if DEAL_II_VERSION_GTE(9,2,0)
      state_dofs_per_block_ = DoFTools::count_dofs_per_fe_block(
#if DEAL_II_VERSION_GTE(9,3,0)
	static_cast<dealii::DoFHandler<dealdim, dealdim>&>(state_dof_handler_),
#else
	static_cast<DH<dealdim, dealdim>&>(state_dof_handler_),
#endif
        state_block_component);
#else
      DoFTools::count_dofs_per_block(
	static_cast<DH<dealdim, dealdim>&>(state_dof_handler_),
        state_dofs_per_block_, state_block_component);
#endif //dealii older than 9.2.0
     
      support_points_.clear();
      n_neighbour_to_vertex_.clear();

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
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dopedim> &
#else
    const DOpEWrapper::DoFHandler<dopedim, DH> &
#endif
      GetControlDoFHandler(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      //There is only one mesh, hence always return this
      return control_dof_handler_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
#if DEAL_II_VERSION_GTE(9,3,0)
    const DOpEWrapper::DoFHandler<dealdim> &
#else
    const DOpEWrapper::DoFHandler<dealdim, DH> &
#endif
      GetStateDoFHandler(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
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
    GetControlDoFsPerBlock(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return control_dofs_per_block_;
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    const std::vector<unsigned int> &
    GetStateDoFsPerBlock(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
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
#if DEAL_II_VERSION_GTE(9,1,1)
    const dealii::AffineConstraints<double> &
    GetControlDoFConstraints() const
    {
      return control_dof_constraints_;
    }
#else
    const dealii::ConstraintMatrix &
    GetControlDoFConstraints() const
    {
      return control_dof_constraints_;
    }
#endif
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
#if DEAL_II_VERSION_GTE(9,1,1)
    const dealii::AffineConstraints<double> &
    GetStateDoFConstraints(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return state_dof_constraints_;
    }
#else
    const dealii::ConstraintMatrix &
    GetStateDoFConstraints(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return state_dof_constraints_;
    }
#endif
    
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
#if DEAL_II_VERSION_GTE(9,1,1)
    const dealii::AffineConstraints<double> &
    GetControlHNConstraints() const
    {
      return control_hn_constraints_;
    }
#else
    const dealii::ConstraintMatrix &
    GetControlHNConstraints() const
    {
      return control_hn_constraints_;
    }
#endif
    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
#if DEAL_II_VERSION_GTE(9,1,1)
    const dealii::AffineConstraints<double> &
    GetStateHNConstraints(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return state_hn_constraints_;
    }
#else
    const dealii::ConstraintMatrix &
    GetStateHNConstraints(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return state_hn_constraints_;
    }
#endif
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
        {
          throw DOpEException(
            "This function is currently not implemented for anything other than"
            " linear interpolation of 2 DoFs.",
            "MethodOfLine_SpaceTimeHandler::InterpolateControl");
        }
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
        {
          throw DOpEException(
            "This function is currently not implemented for anything other than"
            " linear interpolation of 2 DoFs.",
            "MethodOfLine_SpaceTimeHandler::InterpolateState");
        }
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
    GetControlNDoFs(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      return GetControlDoFHandler().n_dofs();
    }
    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    unsigned int
    GetStateNDoFs(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
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
      //return constraints_.local_n_dofs();
      return constraints_.n_dofs("local");
    }

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const std::vector<Point<dealdim> > &
    GetMapDoFToSupportPoints(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max())
    {
      support_points_.resize(GetStateNDoFs());
      DOpE::STHInternals::MapDoFsToSupportPoints(this->GetMapping(),
                                                 GetStateDoFHandler(), support_points_);
      return support_points_;
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const std::vector<unsigned int>* GetNNeighbourElements(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max())
    {
      if(n_neighbour_to_vertex_.size()!=triangulation_.n_vertices())
      {
	DOpE::STHInternals::CalculateNeigbourElementsToVertices(triangulation_,n_neighbour_to_vertex_);
      }
      return &n_neighbour_to_vertex_;
    }

    /******************************************************/
    /**
     * Computes the SparsityPattern for the stiffness matrix
     * of the scalar product in the control space.
     */
    void
    ComputeControlSparsityPattern(SPARSITYPATTERN &sparsity) const;

    /******************************************************/
    /**
     * Computes the SparsityPattern for the stiffness matrix
     * of the PDE.
     */
    void
      ComputeStateSparsityPattern(SPARSITYPATTERN &sparsity, unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      this->GetSparsityMaker()->ComputeSparsityPattern(
        this->GetStateDoFHandler(), sparsity,
        this->GetStateDoFConstraints(), this->GetStateDoFsPerBlock());
    }

    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandler
     */
    const FE<dealdim, dealdim> &
    GetFESystem (std::string name) const // TODO enum
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
                              "MethodOfLines_SpaceTimeHandler::GetFESystem");
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
    void
    RefineSpace(DOpEtypes::RefinementType /*ref_type*/ =
                  DOpEtypes::RefinementType::global)
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

    void
    RefineSpace(const RefinementContainer &ref_container)
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

      //make sure that we do not use any coarsening
      assert(!ref_container.UsesCoarsening());

      if (control_mesh_transfer_ != NULL)
        {
          delete control_mesh_transfer_;
          control_mesh_transfer_ = NULL;
        }
      if (state_mesh_transfer_ != NULL)
        {
          delete state_mesh_transfer_;
          state_mesh_transfer_ = NULL;
        }
#if dope_dimension == deal_II_dimension
          control_mesh_transfer_ =
#if DEAL_II_VERSION_GTE(9,3,0)
	    new DOpEWrapper::SolutionTransfer<dopedim, VECTOR> (control_dof_handler_);
#else
	    new DOpEWrapper::SolutionTransfer<dopedim, VECTOR, DH> (control_dof_handler_);
#endif
#endif
          state_mesh_transfer_ =
#if DEAL_II_VERSION_GTE(9,3,0)
	    new DOpEWrapper::SolutionTransfer<dealdim, VECTOR> (state_dof_handler_);
#else
	    new DOpEWrapper::SolutionTransfer<dealdim, VECTOR, DH> (state_dof_handler_);
#endif
	  
          switch (ref_type)
        {
          case DOpEtypes::RefinementType::global:
          triangulation_.set_all_refine_flags();
            break;

          case DOpEtypes::RefinementType::fixed_number:
          GridRefinement::refine_and_coarsen_fixed_number(triangulation_,
                                                          ref_container.GetLocalErrorIndicators(),
                                                          ref_container.GetTopFraction(),
                                                          ref_container.GetBottomFraction());
            break;

          case DOpEtypes::RefinementType::fixed_fraction:
          GridRefinement::refine_and_coarsen_fixed_fraction(triangulation_,
                                                            ref_container.GetLocalErrorIndicators(),
                                                            ref_container.GetTopFraction(),
                                                            ref_container.GetBottomFraction());
            break;

          case DOpEtypes::RefinementType::optimized:
          //FIXME: refine_and_coarse_optimize takes an unsigned int argument
          // for the convergence order. We thus have to convert the double
          // stored in ref_container to an unsigned int keeping the "floor
          // rounding" in mind that is performed by type casting:
          GridRefinement::refine_and_coarsen_optimize(
            triangulation_,
            ref_container.GetLocalErrorIndicators(),
            static_cast<unsigned int>(ref_container.GetConvergenceOrder() + 0.5));
            break;

	  case DOpEtypes::RefinementType::geometry:
	    dynamic_cast<const RefineByGeometry<dealdim>&>(ref_container).MarkElements(triangulation_);
	    break;

	  default:
            throw DOpEException (
                "Not implemented for name =" + DOpEtypesToString (ref_type),
                              "MethodOfLines_SpaceTimeHandler::RefineSpace");
        }

      triangulation_.prepare_coarsening_and_refinement();

      //FIXME: works only if no coarsening happens, because we do not have the vectors to be interpolated availiable...
      if (control_mesh_transfer_ != NULL)
        control_mesh_transfer_->prepare_for_pure_refinement();
      if (state_mesh_transfer_ != NULL)
        state_mesh_transfer_->prepare_for_pure_refinement();

      triangulation_.execute_coarsening_and_refinement();
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
    void
    SpatialMeshTransferState(const VECTOR &old_values,
                             VECTOR &new_values,unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      if (state_mesh_transfer_ != NULL)
        state_mesh_transfer_->refine_interpolate(old_values, new_values);
    }

     /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */

    virtual bool TemporalMeshTransferControl( VECTOR & /*new_values*/, unsigned int /*from_time_dof*/, unsigned int /*to_time_dof*/) const
    {
      return false;
    }

     /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */

    virtual bool TemporalMeshTransferState(VECTOR & /*new_values*/ , unsigned int /*from_time_dof*/, unsigned int /*to_time_dof*/) const
    {
      return false;
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
      UserDefinedDoFConstraints<DH, dopedim, dealdim> &constraints_maker)
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
#if DEAL_II_VERSION_GTE(9,3,0)
      SetSparsityMaker(SparsityMaker<dealdim> &sparsity_maker)
#else
      SetSparsityMaker(SparsityMaker<DH, dealdim> &sparsity_maker)
#endif
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
    ResetTriangulation(const dealii::Triangulation<dealdim> &tria);

  private:
#if DEAL_II_VERSION_GTE(9,3,0)
    const SparsityMaker<dealdim> *
#else
    const SparsityMaker<DH, dealdim> *
#endif
    GetSparsityMaker() const
    {
      return sparsitymaker_;
    }
    const UserDefinedDoFConstraints<DH, dopedim, dealdim> *
    GetUserDefinedDoFConstraints() const
    {
      return user_defined_dof_constr_;
    }
#if DEAL_II_VERSION_GTE(9,3,0)
    SparsityMaker<dealdim> *sparsitymaker_;
#else
    SparsityMaker<DH, dealdim> *sparsitymaker_;
#endif
    UserDefinedDoFConstraints<DH, dopedim, dealdim> *user_defined_dof_constr_;

    dealii::Triangulation<dealdim> &triangulation_;
#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::DoFHandler<dopedim> control_dof_handler_;
#else
    DOpEWrapper::DoFHandler<dopedim, DH> control_dof_handler_;
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::DoFHandler<dealdim> state_dof_handler_;
#else
    DOpEWrapper::DoFHandler<dealdim, DH> state_dof_handler_;
#endif

    std::vector<unsigned int> control_dofs_per_block_;
    std::vector<unsigned int> state_dofs_per_block_;

#if DEAL_II_VERSION_GTE(9,1,1)
    dealii::AffineConstraints<double> control_hn_constraints_;
    dealii::AffineConstraints<double> state_hn_constraints_;
    dealii::AffineConstraints<double> control_dof_constraints_;
    dealii::AffineConstraints<double> state_dof_constraints_;
#else
    dealii::ConstraintMatrix control_hn_constraints_;
    dealii::ConstraintMatrix state_hn_constraints_;
    dealii::ConstraintMatrix control_dof_constraints_;
    dealii::ConstraintMatrix state_dof_constraints_;
#endif

    const dealii::SmartPointer<const FE<dealdim, dealdim> > control_fe_;
    const dealii::SmartPointer<const FE<dealdim, dealdim> > state_fe_;

    const dealii::SmartPointer<const DOpEWrapper::Mapping<dealdim, DH> > mapping_;
    std::vector<Point<dealdim> > support_points_;

    Constraints constraints_;
#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::SolutionTransfer<dealdim, VECTOR> *control_mesh_transfer_;
#else
    DOpEWrapper::SolutionTransfer<dealdim, VECTOR,DH> *control_mesh_transfer_;
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::SolutionTransfer<dealdim, VECTOR> *state_mesh_transfer_;
#else
    DOpEWrapper::SolutionTransfer<dealdim, VECTOR,DH> *state_mesh_transfer_;
#endif
    bool sparse_mkr_dynamic_;

    std::vector<unsigned int> n_neighbour_to_vertex_;

  };

  /**************************explicit instantiation*************/

#if DEAL_II_VERSION_GTE(9,3,0)
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
    false, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
         dealii::BlockSparsityPattern &sparsity) const;
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
    false, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
         const dealii::Triangulation<deal_II_dimension> &tria);

  /******************************************************/

  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::FESystem,
    false, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
                                   dealii::SparsityPattern &sparsity) const;
  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::FESystem,
    false, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
                                   const dealii::Triangulation<deal_II_dimension> &tria);
#else
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
         dealii::BlockSparsityPattern &sparsity) const;
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<dealii::FESystem,
       dealii::DoFHandler, dealii::BlockSparsityPattern,
       dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
         const dealii::Triangulation<deal_II_dimension> &tria);

  /******************************************************/

  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::FESystem,
                                 dealii::DoFHandler, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
                                   dealii::SparsityPattern &sparsity) const;
  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::FESystem,
                                 dealii::DoFHandler, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
                                   const dealii::Triangulation<deal_II_dimension> &tria);
#endif

  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
#if DEAL_II_VERSION_GTE(9,3,0)
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<
  dealii::hp::FECollection,
    true, dealii::BlockSparsityPattern,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
           dealii::BlockSparsityPattern &sparsity) const;
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<
  dealii::hp::FECollection,
    true, dealii::BlockSparsityPattern,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
           const dealii::Triangulation<deal_II_dimension> &tria);

  /******************************************************/

  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
    true, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
                                   dealii::SparsityPattern &sparsity) const;
  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
    true, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
                                   const dealii::Triangulation<deal_II_dimension> &tria);
#else
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<
  dealii::hp::FECollection,
         dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
           dealii::BlockSparsityPattern &sparsity) const;
  template<>
  void
  DOpE::MethodOfLines_SpaceTimeHandler<
  dealii::hp::FECollection,
         dealii::hp::DoFHandler, dealii::BlockSparsityPattern,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
           const dealii::Triangulation<deal_II_dimension> &tria);

  /******************************************************/

  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
                                 dealii::hp::DoFHandler, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
                                   dealii::SparsityPattern &sparsity) const;
  template<>
  void
  MethodOfLines_SpaceTimeHandler<dealii::hp::FECollection,
                                 dealii::hp::DoFHandler, dealii::SparsityPattern,
                                 dealii::Vector<double>, dope_dimension, deal_II_dimension>::ResetTriangulation(
                                   const dealii::Triangulation<deal_II_dimension> &tria);

#endif
}

#endif
