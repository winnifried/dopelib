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

#ifndef _MOL_SPACE_TIME_HANDLER_H_
#define _MOL_SPACE_TIME_HANDLER_H_

#include "spacetimehandler.h"
#include "constraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "sth_internals.h"
#include "mapping_wrapper.h"
#include "refinementcontainer.h"
#include "solutiontransfer_wrapper.h"

#include <dofs/dof_handler.h>
#include <dofs/dof_renumbering.h>
#include <dofs/dof_tools.h>
#include <hp/mapping_collection.h>
#include <lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>

namespace DOpE
{
  /**
   * Implements a Space Time Handler with a Method of Lines discretization.
   * This means there is only one fixed mesh for the spatial domain.
   */
  template<template <int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
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
         * @param index_setter      The index setter object (only needed in case of hp elements).
         */
        MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim>& triangulation, 
				       const FE<dealdim, dealdim>& control_fe,
				       const FE<dealdim, dealdim>& state_fe, 
				       DOpEtypes::ControlType type,
				       const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter =
				       ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(type, index_setter), 
	_triangulation(triangulation), 
	_control_dof_handler(_triangulation), 
	_state_dof_handler(_triangulation), 
	_control_fe(&control_fe), 
	_state_fe(&state_fe), 
	_mapping(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), 
	_constraints(), 
	_control_mesh_transfer(NULL), 
	_state_mesh_transfer(NULL), 
	_sparse_mkr_dynamic(true)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
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
         * @param index_setter      The index setter object (only needed in case of hp elements).
         */      
        MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim>& triangulation, 
				       const FE<dealdim, dealdim>& control_fe,
				       const FE<dealdim, dealdim>& state_fe, 
				       dealii::Triangulation<1> & times,
				       DOpEtypes::ControlType type,
				       const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter =
				       ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(times, type, index_setter), 
	_triangulation(triangulation), 
	_control_dof_handler(_triangulation), 
	_state_dof_handler(_triangulation), 
	_control_fe(&control_fe), 
	_state_fe(&state_fe), 
	_mapping(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), 
	_constraints(), 
	_control_mesh_transfer(NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(true)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
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
         * @param index_setter      The index setter object (only needed in case of hp elements).
         */
         MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim>& triangulation, 
					const FE<dealdim, dealdim>& control_fe,
					const FE<dealdim, dealdim>& state_fe, 
					const Constraints& c,
					DOpEtypes::ControlType type,
					const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter =
					ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(type, index_setter), 
	_triangulation(triangulation), 
	_control_dof_handler(_triangulation), 
	_state_dof_handler(_triangulation), 
	_control_fe(&control_fe), 
	_state_fe(&state_fe), 
	_mapping(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), 
	_constraints(c), 
	_control_mesh_transfer(NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(true)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
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
         * @param index_setter      The index setter object (only needed in case of hp elements).
         */
        MethodOfLines_SpaceTimeHandler(dealii::Triangulation<dealdim>& triangulation, 
				       const FE<dealdim, dealdim>& control_fe,
				       const FE<dealdim, dealdim>& state_fe, 
				       dealii::Triangulation<1> & times,
				       const Constraints& c, 
				       DOpEtypes::ControlType type,
				       const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter =
				       ActiveFEIndexSetterInterface<dopedim, dealdim>()) :
      SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(times, type, index_setter), 
	_triangulation(triangulation), 
	_control_dof_handler(_triangulation), 
	_state_dof_handler(_triangulation), 
	_control_fe(&control_fe), 
	_state_fe(&state_fe), 
	_mapping(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), 
	_constraints(c), 
	_control_mesh_transfer(NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(true)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
        }

        ~MethodOfLines_SpaceTimeHandler()
        {
          _control_dof_handler.clear();

          _state_dof_handler.clear();

          if (_control_mesh_transfer != NULL)
          {
            delete _control_mesh_transfer;
          }
          if (_state_mesh_transfer != NULL)
          {
            delete _state_mesh_transfer;
          }
          if (_sparsitymaker != NULL && _sparse_mkr_dynamic == true)
          {
            delete _sparsitymaker;
          }
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        void
        ReInit(unsigned int control_n_blocks,
            const std::vector<unsigned int>& control_block_component,
            unsigned int state_n_blocks,
            const std::vector<unsigned int>& state_block_component)
        {
#if dope_dimension > 0
          SpaceTimeHandler<FE, DH, SPARSITYPATTERN,
          VECTOR, dopedim, dealdim>::SetActiveFEIndicesControl(_control_dof_handler);
#endif
          _control_dof_handler.distribute_dofs(*_control_fe);

#if dope_dimension > 0
          DoFRenumbering::component_wise (static_cast<DH<dopedim, dopedim>&>(_control_dof_handler));
          if(dopedim==dealdim)
          {
            _control_dof_constraints.clear ();
            DoFTools::make_hanging_node_constraints (static_cast<DH<dopedim, dopedim>&>(_control_dof_handler),
                _control_dof_constraints);
            if (GetUserDefinedDoFConstraints() != NULL)
            GetUserDefinedDoFConstraints()->MakeControlDoFConstraints(_control_dof_handler,
                _control_dof_constraints);
            _control_dof_constraints.close ();
          }
          else
          {
            throw DOpEException("Not implemented for dopedim != dealdim","MethodOfLines_SpaceTimeHandler::ReInit");
          }
#endif
          _control_dofs_per_block.resize(control_n_blocks);
#if dope_dimension > 0
          {
            DoFTools::count_dofs_per_block (static_cast<DH<dopedim, dopedim>&>(_control_dof_handler),
                _control_dofs_per_block,control_block_component);
          }
#else
          {
            for (unsigned int i = 0; i < _control_dofs_per_block.size(); i++)
            {
              _control_dofs_per_block[i] = 0;
            }
            for (unsigned int i = 0; i < control_block_component.size(); i++)
            {
              _control_dofs_per_block[control_block_component[i]]++;
            }
          }
#endif
          SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::SetActiveFEIndicesState(
              _state_dof_handler);
          _state_dof_handler.distribute_dofs(GetFESystem("state"));
          DoFRenumbering::component_wise(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler));

          _state_dof_constraints.clear();
          DoFTools::make_hanging_node_constraints(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler),
              _state_dof_constraints);
          //TODO Dirichlet ueber Constraints
          if (GetUserDefinedDoFConstraints() != NULL)
            GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(
                _state_dof_handler, _state_dof_constraints);
          _state_dof_constraints.close();

          _state_dofs_per_block.resize(state_n_blocks);
          DoFTools::count_dofs_per_block(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler),
              _state_dofs_per_block, state_block_component);

          _support_points.clear();

          _constraints.ReInit(_control_dofs_per_block);
          //_constraints.ReInit(_control_dofs_per_block, _state_dofs_per_block);

          //Initialize also the timediscretization.
          this->ReInitTime();

          //There where changes invalidate tickets
          this->IncrementControlTicket();
          this->IncrementStateTicket();
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const DOpEWrapper::DoFHandler<dopedim, DH>&
        GetControlDoFHandler() const
        {
          //There is only one mesh, hence always return this
          return _control_dof_handler;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const DOpEWrapper::DoFHandler<dealdim, DH>&
        GetStateDoFHandler() const
        {
          //There is only one mesh, hence always return this
          return _state_dof_handler;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const DOpEWrapper::Mapping<dealdim, DH>&
        GetMapping() const
        {
          return *_mapping;
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        unsigned int
        GetControlDoFsPerBlock(unsigned int b, int /*time_point*/= -1) const
        {
          return _control_dofs_per_block[b];
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        unsigned int
        GetStateDoFsPerBlock(unsigned int b, int /*time_point*/= -1) const
        {
          return _state_dofs_per_block[b];
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        unsigned int
        GetConstraintDoFsPerBlock(std::string name, unsigned int b) const
        {
          return (_constraints.GetDoFsPerBlock(name))[b];
        }
        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        const std::vector<unsigned int>&
        GetControlDoFsPerBlock(int /*time_point*/= -1) const
        {
          return _control_dofs_per_block;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        const std::vector<unsigned int>&
        GetStateDoFsPerBlock(int /*time_point*/= -1) const
        {
          return _state_dofs_per_block;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        const std::vector<unsigned int>&
        GetConstraintDoFsPerBlock(std::string name) const
        {
          return _constraints.GetDoFsPerBlock(name);
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const dealii::ConstraintMatrix&
        GetControlDoFConstraints() const
        {
          return _control_dof_constraints;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const dealii::ConstraintMatrix&
        GetStateDoFConstraints() const
        {
          return _state_dof_constraints;
        }

        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        virtual void
        InterpolateControl(VECTOR& result,
            const std::vector<VECTOR*> & local_vectors, double t,
            const TimeIterator& it) const
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
        InterpolateState(VECTOR& result,
            const std::vector<VECTOR*> & local_vectors, double t,
            const TimeIterator& it) const
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
          return _constraints.n_dofs(name);
        }
        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        unsigned int
        GetNGlobalConstraints() const
        {
          return _constraints.n_dofs("global");
          //return _constraints.global_n_dofs();
        }
        /**
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        unsigned int
        GetNLocalConstraints() const
        {
          //return _constraints.local_n_dofs();
          return _constraints.n_dofs("local");
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const std::vector<Point<dealdim> >&
        GetMapDoFToSupportPoints()
        {
          _support_points.resize(GetStateNDoFs());
          DOpE::STHInternals::MapDoFsToSupportPoints(this->GetMapping(),
              GetStateDoFHandler(), _support_points);
          return _support_points;
        }

        /******************************************************/
	/**
	 * Computes the SparsityPattern for the stiffness matrix
	 * of the scalar product in the control space. 
	 */
        void
        ComputeControlSparsityPattern(SPARSITYPATTERN & sparsity) const;

        /******************************************************/
	/**
	 * Computes the SparsityPattern for the stiffness matrix
	 * of the PDE. 
	 */
        void
        ComputeStateSparsityPattern(SPARSITYPATTERN & sparsity) const
        {
          this->GetSparsityMaker()->ComputeSparsityPattern(
              this->GetStateDoFHandler(), sparsity,
              this->GetStateDoFConstraints(), this->GetStateDoFsPerBlock());
        }

        /******************************************************/

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const FE<dealdim, dealdim>&
        GetFESystem(std::string name) const
        {
          if (name == "state")
          {
            return *_state_fe;
          }
          else if (name == "control")
          {
            return *_control_fe;
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
        RefineSpace(DOpEtypes::RefinementType ref_type =
            DOpEtypes::RefinementType::global)
        {
          assert(ref_type == DOpEtypes::RefinementType::global);
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
        RefineSpace(const RefinementContainer& ref_container)
        {
          DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

          //make sure that we do not use any coarsening
          assert(!ref_container.UsesCoarsening());

          if (_control_mesh_transfer != NULL)
          {
            delete _control_mesh_transfer;
            _control_mesh_transfer = NULL;
          }
          if (_state_mesh_transfer != NULL)
          {
            delete _state_mesh_transfer;
            _state_mesh_transfer = NULL;
          }
#if dope_dimension == deal_II_dimension
          _control_mesh_transfer = new DOpEWrapper::SolutionTransfer<dopedim, VECTOR,
	    DH>(_control_dof_handler);
#endif
          _state_mesh_transfer = new DOpEWrapper::SolutionTransfer<dealdim, VECTOR,
	    DH>(_state_dof_handler);
          if (DOpEtypes::RefinementType::global == ref_type)
          {
            _triangulation.set_all_refine_flags();
          }
          else if (DOpEtypes::RefinementType::fixed_number == ref_type)
          {
            GridRefinement::refine_and_coarsen_fixed_number(_triangulation,
                ref_container.GetLocalErrorIndicators(),
                ref_container.GetTopFraction(),
                ref_container.GetBottomFraction());
          }
          else if (DOpEtypes::RefinementType::fixed_fraction == ref_type)
          {
            GridRefinement::refine_and_coarsen_fixed_fraction(_triangulation,
                ref_container.GetLocalErrorIndicators(),
                ref_container.GetTopFraction(),
                ref_container.GetBottomFraction());
          }
          else if (DOpEtypes::RefinementType::optimized == ref_type)
          {
            GridRefinement::refine_and_coarsen_optimize(_triangulation,
                ref_container.GetLocalErrorIndicators(),
                ref_container.GetConvergenceOrder());
          }
          else
          {
            throw DOpEException("Not implemented for name =" + ref_type,
                "MethodOfLines_SpaceTimeHandler::RefineSpace");
          }
          _triangulation.prepare_coarsening_and_refinement();

          //FIXME: works only if no coarsening happens, because we do not have the vectors to be interpolated availiable...
          if (_control_mesh_transfer != NULL)
            _control_mesh_transfer->prepare_for_pure_refinement();
          if (_state_mesh_transfer != NULL)
            _state_mesh_transfer->prepare_for_pure_refinement();

          _triangulation.execute_coarsening_and_refinement();
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
        SpatialMeshTransferControl(const VECTOR& old_values,
            VECTOR& new_values) const
        {
          if (_control_mesh_transfer != NULL)
            _control_mesh_transfer->refine_interpolate(old_values, new_values);
        }
        void
        SpatialMeshTransferState(const VECTOR& old_values,
            VECTOR& new_values) const
        {
          if (_state_mesh_transfer != NULL)
            _state_mesh_transfer->refine_interpolate(old_values, new_values);
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
            UserDefinedDoFConstraints<DH, dopedim, dealdim>& constraints_maker)
        {
          _user_defined_dof_constr = &constraints_maker;
          _user_defined_dof_constr->RegisterMapping(this->GetMapping());
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
        SetSparsityMaker(SparsityMaker<DH, dealdim>& sparsity_maker)
        {
          if (_sparsitymaker != NULL && _sparse_mkr_dynamic)
            delete _sparsitymaker;
          _sparsitymaker = &sparsity_maker;
          _sparse_mkr_dynamic = false;
        }

        /******************************************************/
        /**
         * Through this function one can reinitialize the 
         * triangulation for the state variable to be a copy of the
         * given argument.
         */

        void
        ResetTriangulation(const dealii::Triangulation<dealdim>& tria)
        {
          _state_dof_handler.clear();
          _triangulation.clear();
          _triangulation.copy_triangulation(tria);
          _state_dof_handler.initialize(_triangulation, *_state_fe);
          this->IncrementControlTicket();
          this->IncrementStateTicket();
          if (_control_mesh_transfer != NULL)
            delete _control_mesh_transfer;
          _control_mesh_transfer = NULL;
          if (_state_mesh_transfer != NULL)
            delete _state_mesh_transfer;
          _state_mesh_transfer = NULL;

        }

      private:
        const SparsityMaker<DH, dealdim>*
        GetSparsityMaker() const
        {
          return _sparsitymaker;
        }
        const UserDefinedDoFConstraints<DH, dopedim, dealdim>*
        GetUserDefinedDoFConstraints() const
        {
          return _user_defined_dof_constr;
        }
        SparsityMaker<DH, dealdim>* _sparsitymaker;
        UserDefinedDoFConstraints<DH, dopedim, dealdim>* _user_defined_dof_constr;

        dealii::Triangulation<dealdim>& _triangulation;
        DOpEWrapper::DoFHandler<dopedim, DH> _control_dof_handler;
        DOpEWrapper::DoFHandler<dealdim, DH> _state_dof_handler;

        std::vector<unsigned int> _control_dofs_per_block;
        std::vector<unsigned int> _state_dofs_per_block;

        dealii::ConstraintMatrix _control_dof_constraints;
        dealii::ConstraintMatrix _state_dof_constraints;

        const dealii::SmartPointer<const FE<dealdim, dealdim> > _control_fe;
        const dealii::SmartPointer<const FE<dealdim, dealdim> > _state_fe;

        const dealii::SmartPointer<const DOpEWrapper::Mapping<dealdim, DH> > _mapping;

        std::vector<Point<dealdim> > _support_points;

        Constraints _constraints;
        DOpEWrapper::SolutionTransfer<dealdim, VECTOR,DH>* _control_mesh_transfer;
        DOpEWrapper::SolutionTransfer<dealdim, VECTOR,DH>* _state_mesh_transfer;
        bool _sparse_mkr_dynamic;
    };

  /**************************explicit instantiation*************/

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

#endif
