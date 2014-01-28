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

#ifndef _MOL_STATESPACE_TIME_HANDLER_H_
#define _MOL_STATESPACE_TIME_HANDLER_H_

#include "statespacetimehandler.h"
#include "constraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "sth_internals.h"
#include "refinementcontainer.h"

#include <dofs/dof_handler.h>
#include <dofs/dof_renumbering.h>
#include <dofs/dof_tools.h>
#include <lac/constraint_matrix.h>
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
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
      typename VECTOR, int dealdim>
    class MethodOfLines_StateSpaceTimeHandler : public StateSpaceTimeHandler<FE,
        DH, SPARSITYPATTERN, VECTOR, dealdim>
    {
      public:
        MethodOfLines_StateSpaceTimeHandler(
            dealii::Triangulation<dealdim>& triangulation, const FE<dealdim, dealdim>& state_fe,
            const ActiveFEIndexSetterInterface<dealdim>& index_setter =
                ActiveFEIndexSetterInterface<dealdim>())
            : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
                dealdim>(index_setter), _sparse_mkr_dynamic(true), _triangulation(
                triangulation), _state_dof_handler(_triangulation), _state_fe(
                &state_fe), _mapping(
                &DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), _state_mesh_transfer(
                NULL)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
        }
        MethodOfLines_StateSpaceTimeHandler(
            dealii::Triangulation<dealdim>& triangulation, const FE<dealdim, dealdim>& state_fe,
            dealii::Triangulation<1> & times,
            const ActiveFEIndexSetterInterface<dealdim>& index_setter =
                ActiveFEIndexSetterInterface<dealdim>())
            : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
                dealdim>(times, index_setter), _sparse_mkr_dynamic(true), _triangulation(
                triangulation), _state_dof_handler(_triangulation), _state_fe(
                &state_fe), _mapping(
                &DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1), _state_mesh_transfer(
                NULL)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
        }

        MethodOfLines_StateSpaceTimeHandler(
            dealii::Triangulation<dealdim>& triangulation,
            const DOpEWrapper::Mapping<dealdim, DH>& mapping,
            const FE<dealdim, dealdim>& state_fe,
            const ActiveFEIndexSetterInterface<dealdim>& index_setter =
                ActiveFEIndexSetterInterface<dealdim>())
            : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
                dealdim>(index_setter), _sparse_mkr_dynamic(true), _triangulation(
                triangulation), _state_dof_handler(_triangulation), _state_fe(
                &state_fe), _mapping(&mapping), _state_mesh_transfer(NULL)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
        }
        MethodOfLines_StateSpaceTimeHandler(
            dealii::Triangulation<dealdim>& triangulation,
            const DOpEWrapper::Mapping<dealdim, DH>& mapping,
            const FE<dealdim, dealdim>& state_fe, 
	    dealii::Triangulation<1> & times,
            const ActiveFEIndexSetterInterface<dealdim>& index_setter =
                ActiveFEIndexSetterInterface<dealdim>())
            : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
                dealdim>(times, index_setter), _sparse_mkr_dynamic(true), _triangulation(
                triangulation), _state_dof_handler(_triangulation), _state_fe(
                &state_fe), _mapping(&mapping), _state_mesh_transfer(NULL)
        {
          _sparsitymaker = new SparsityMaker<DH, dealdim>;
          _user_defined_dof_constr = NULL;
        }

        virtual
        ~MethodOfLines_StateSpaceTimeHandler()
        {
          _state_dof_handler.clear();

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
         * Implementation of virtual function in StateSpaceTimeHandler
         */
        void
        ReInit(unsigned int state_n_blocks,
            const std::vector<unsigned int>& state_block_component)
        {

          StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>::SetActiveFEIndicesState(
              _state_dof_handler);
          _state_dof_handler.distribute_dofs(GetFESystem("state"));
          DoFRenumbering::Cuthill_McKee(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler));
          DoFRenumbering::component_wise(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler));

          _state_dof_constraints.clear();
          DoFTools::make_hanging_node_constraints(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler),
              _state_dof_constraints);
          //TODO Dirichlet ueber Constraints
          if (GetUserDefinedDoFConstraints() != NULL
          )
            GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(
                _state_dof_handler, _state_dof_constraints);
          _state_dof_constraints.close();
          _state_dofs_per_block.resize(state_n_blocks);

          DoFTools::count_dofs_per_block(
              static_cast<DH<dealdim, dealdim>&>(_state_dof_handler),
              _state_dofs_per_block, state_block_component);

          _support_points.clear();

          //Initialize also the timediscretization.
          this->ReInitTime();

          //There where changes invalidate tickets
          this->IncrementStateTicket();
        }

        /**
         * Implementation of virtual function in StateSpaceTimeHandler
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
         * Implementation of virtual function in StateSpaceTimeHandler
         */
        unsigned int
        GetStateDoFsPerBlock(unsigned int b, int /*time_point*/= -1) const
        {
          return _state_dofs_per_block[b];
        }

        /**
         * Implementation of virtual function in StateSpaceTimeHandlerBase
         */
        const std::vector<unsigned int>&
        GetStateDoFsPerBlock(int /*time_point*/= -1) const
        {
          return _state_dofs_per_block;
        }

        /**
         * Implementation of virtual function in StateSpaceTimeHandler
         */
        const dealii::ConstraintMatrix&
        GetStateDoFConstraints() const
        {
          return _state_dof_constraints;
        }

        /**
         * Implementation of virtual function in StateSpaceTimeHandlerBase
         */
        virtual void
        InterpolateState(VECTOR& result,
            const std::vector<VECTOR*> & local_vectors, double t,
            const TimeIterator& it) const
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
         * Implementation of virtual function in StateSpaceTimeHandlerBase
         */
        unsigned int
        GetStateNDoFs(int /*time_point*/= -1) const
        {
          return GetStateDoFHandler().n_dofs();
        }

        /**
         * Implementation of virtual function in StateSpaceTimeHandler
         */
        const std::vector<Point<dealdim> >&
        GetMapDoFToSupportPoints()
        {
          _support_points.resize(GetStateNDoFs());
          DOpE::STHInternals::MapDoFsToSupportPoints
              < std::vector<Point<dealdim> >, dealdim
              > (this->GetMapping(), GetStateDoFHandler(), _support_points);
          return _support_points;
        }

        /******************************************************/
        void
        ComputeStateSparsityPattern(SPARSITYPATTERN & sparsity) const
        {
          this->GetSparsityMaker()->ComputeSparsityPattern(
              this->GetStateDoFHandler(), sparsity,
              this->GetStateDoFConstraints(), this->GetStateDoFsPerBlock());
        }

        /******************************************************/

        /**
         * Implementation of virtual function in StateSpaceTimeHandler
         */
        const FE<dealdim, dealdim>&
        GetFESystem(std::string name) const
        {
          if (name == "state")
          {
            return *_state_fe;
          }
          else
          {
            abort();
            throw DOpEException("Not implemented for name =" + name,
                "MethodOfLines_StateSpaceTimeHandler::GetFESystem");
          }

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

          if (_state_mesh_transfer != NULL)
          {
            delete _state_mesh_transfer;
            _state_mesh_transfer = NULL;
          }
          _state_mesh_transfer = new dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> >(
              _state_dof_handler);

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
                "MethodOfLines_StateSpaceTimeHandler::RefineStateSpace");
          }
          _triangulation.prepare_coarsening_and_refinement();
          if (_state_mesh_transfer != NULL
            )
            _state_mesh_transfer->prepare_for_pure_refinement();
          _triangulation.execute_coarsening_and_refinement();
        }
        /******************************************************/

        /**
         * Implementation of virtual function in StateSpaceTimeHandlerBase
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
        SpatialMeshTransferState(const VECTOR& old_values,
            VECTOR& new_values) const
        {
          if (_state_mesh_transfer != NULL
          )
            _state_mesh_transfer->refine_interpolate(old_values, new_values);
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
        void
        SetUserDefinedDoFConstraints(
            UserDefinedDoFConstraints<DH, dealdim>& user_defined_dof_constr)
        {
          _user_defined_dof_constr = &user_defined_dof_constr;
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

      private:
        const SparsityMaker<DH, dealdim>*
        GetSparsityMaker() const
        {
          return _sparsitymaker;
        }
        const UserDefinedDoFConstraints<DH, dealdim>*
        GetUserDefinedDoFConstraints() const
        {
          return _user_defined_dof_constr;
        }
        SparsityMaker<DH, dealdim>* _sparsitymaker;
        UserDefinedDoFConstraints<DH, dealdim>* _user_defined_dof_constr;
        bool _sparse_mkr_dynamic;

        dealii::Triangulation<dealdim>& _triangulation;
        DOpEWrapper::DoFHandler<dealdim, DH> _state_dof_handler;

        std::vector<unsigned int> _state_dofs_per_block;

        dealii::ConstraintMatrix _state_dof_constraints;

        const dealii::SmartPointer<const FE<dealdim, dealdim> > _state_fe; //TODO is there a reason that this is not a reference?
        const dealii::SmartPointer<
            const DOpEWrapper::Mapping<dealdim, DH> > _mapping;

        std::vector<Point<dealdim> > _support_points;
        dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> >* _state_mesh_transfer;

    };

}
#endif
