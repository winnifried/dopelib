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

#ifndef ROTHE_STATESPACE_TIME_HANDLER_H_
#define ROTHE_STATESPACE_TIME_HANDLER_H_

#include <basic/statespacetimehandler.h>
#include <basic/constraints.h>
#include <include/sparsitymaker.h>
#include <include/userdefineddofconstraints.h>
#include <basic/sth_internals.h>
#include <container/refinementcontainer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

namespace DOpE
{
  /**
   * Implements a Space Time Handler with a Rothe discretization.
   * This means there is only one fixed mesh for the spatial domain.
   * This Space Time Handler has knowlege of only one variable, namely the
   * solution to a PDE.
   *
   * For the detailed comments, please see the documentation of Rothe_SpaceTimeHandler
   */
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN, typename VECTOR, int dealdim>
  class Rothe_StateSpaceTimeHandler : public StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>
  {
  public:
    /**
     * Only Constructors with time dependence, as there is no sense in a 
     * Rothe discretization for non-timedependent problems
     */
    Rothe_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation, const FE<dealdim, dealdim> &state_fe,
      dealii::Triangulation<1> &times,
      std::vector<unsigned int> &time_to_dofhandler,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
        dealdim>(times, index_setter), sparse_mkr_dynamic_(true), state_fe_(
          &state_fe), mapping_(&DOpEWrapper::StaticMappingQ1<dealdim, DH>::mapping_q1)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;

      InitSpaceTime(triangulation,time_to_dofhandler);
    }

    Rothe_StateSpaceTimeHandler(
      dealii::Triangulation<dealdim> &triangulation,
      const DOpEWrapper::Mapping<dealdim, DH> &mapping,
      const FE<dealdim, dealdim> &state_fe,
      dealii::Triangulation<1> &times,
      std::vector<unsigned int> &time_to_dofhandler,
      bool flux_pattern = false,
      const ActiveFEIndexSetterInterface<dealdim> &index_setter =
        ActiveFEIndexSetterInterface<dealdim>())
      : StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,
      dealdim>(times, index_setter), sparse_mkr_dynamic_(true), state_fe_(
	&state_fe), mapping_(&mapping)
    {
      sparsitymaker_ = new SparsityMaker<DH, dealdim>(flux_pattern);
      user_defined_dof_constr_ = NULL;
      InitSpaceTime(triangulation,time_to_dofhandler);
    }

    virtual
    ~Rothe_StateSpaceTimeHandler()
    {
      assert(triangulations_.size()==n_dof_handlers_);
      assert(state_dof_handlers_.size()==n_dof_handlers_);
      assert(state_mesh_transfers_.size()==n_dof_handlers_);
      assert(state_dof_constraints_.size()==n_dof_handlers_);
      for(unsigned int i = 0; i < n_dof_handlers_; i++)
      {
	assert(state_dof_handlers_[i]!= NULL);
	state_dof_handlers_[i]->clear();
	delete state_dof_handlers_[i];
	
	if (state_mesh_transfers_[i] != NULL)
        {
          delete state_mesh_transfers_[i];
        }
	if (state_dof_constraints_[i] != NULL)
	{
	  delete state_dof_constraints_[i];
	}
	if ( i != 0 )
	{
	  assert(triangulations_[i]!=NULL);
	  delete triangulations_[i];
	}
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
	assert(state_dof_constraints_.size()==n_dof_handlers_);
	
     state_dofs_per_block_.resize(n_dof_handlers_,std::vector<unsigned int>(state_n_blocks));
     for(unsigned int j = 0; j < n_dof_handlers_; j++)
      {
	this->SetInterval(this->GetTimeDoFHandler().first_interval(),dofhandler_to_time_[j]);
      	StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>::SetActiveFEIndicesState(
        *state_dof_handlers_[j]);
      	state_dof_handlers_[j]->distribute_dofs(GetFESystem("state"));
      	DoFRenumbering::component_wise(
        static_cast<DH<dealdim, dealdim>&>(*state_dof_handlers_[j]));

      	state_dof_constraints_[j]->clear();
      	state_dof_constraints_[j]->reinit (
	  this->GetLocallyRelevantDoFs (DOpEtypes::VectorType::state,dofhandler_to_time_[j]));
      	DoFTools::make_hanging_node_constraints (
        	static_cast<DH<dealdim, dealdim>&> (*state_dof_handlers_[j]),
        *state_dof_constraints_[j]);
      	//TODO Dirichlet ueber Constraints
      	if (GetUserDefinedDoFConstraints() != NULL)
	  GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(*state_dof_handlers_[j], *state_dof_constraints_[j]);
	
	std::vector<unsigned int> dirichlet_colors = DD.GetDirichletColors();
	for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = DD.GetDirichletCompMask(color);
	  
          //TODO: mapping[0] is a workaround, as deal does not support interpolate
          // boundary_values with a mapping collection at this point.
#if DEAL_II_VERSION_GTE(9,0,0)
          VectorTools::interpolate_boundary_values(GetMapping()[0], state_dof_handlers_[j]->GetDEALDoFHandler(),color, dealii::Functions::ZeroFunction<dealdim>(comp_mask.size()), *state_dof_constraints_[j], comp_mask);
#else
	  VectorTools::interpolate_boundary_values(GetMapping()[0], *state_dof_handlers_[j].GetDEALDoFHandler(),  
						   state_dof_constraints_[j], comp_mask);
#endif
        }
      
      	state_dof_constraints_[j]->close();
      	state_dofs_per_block_[j].resize(state_n_blocks);

      	DoFTools::count_dofs_per_block(static_cast<DH<dealdim, dealdim>&>(*state_dof_handlers_[j]),
      	state_dofs_per_block_[j], state_block_component);
	if(support_points_.size() > j)
	{
	  support_points_[j].clear();
	}
	if(n_neighbour_to_vertex_.size() > j)
	{
	  n_neighbour_to_vertex_[j].clear();
	}  
      }
     //Initialize also the timediscretization.
     this->ReInitTime();
     
     //There where changes invalidate tickets
     this->IncrementStateTicket();
     this->SetInterval(this->GetTimeDoFHandler().first_interval(),0);
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const DOpEWrapper::DoFHandler<dealdim, DH> &
    GetStateDoFHandler(unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      //Logic is as follows: if time_point is given, we take this value
      if(time_point != std::numeric_limits<unsigned int>::max())
      {
	if(time_point > time_to_dofhandler_.size())
	{
	  throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetStateDoFHandler");
	}
	return *state_dof_handlers_[time_to_dofhandler_[time_point]];
      }
      //otherwise, take the internaly stored time (asserting that it has been set)
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetStateDoFHandler");
      }
      
      return *state_dof_handlers_[time_to_dofhandler_[this->GetTimeDoFNumber()]];    
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
    GetStateDoFsPerBlock(unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      //Logic is as follows: if time_point is given, we take this value
      if(time_point != std::numeric_limits<unsigned int>::max())
      {
	if(time_point > time_to_dofhandler_.size())
	{
	  throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetStateDoFHandler");
	}
	assert(time_to_dofhandler_[time_point]<state_dofs_per_block_.size());
	return state_dofs_per_block_[time_to_dofhandler_[time_point]];
      }
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetStateDoFsPerBlock");
      }
      assert(time_to_dofhandler_[this->GetTimeDoFNumber()]<state_dofs_per_block_.size());
      return state_dofs_per_block_[time_to_dofhandler_[this->GetTimeDoFNumber()]];
      
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const dealii::ConstraintMatrix &
    GetStateDoFConstraints(unsigned int /*time_point = std::numeric_limits<unsigned int>::max()*/) const
    {
      assert(time_point == std::numeric_limits<unsigned int>::max() || time_point == this->GetTimeDoFNumber());
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetStateDoFConstraints");
      }
      return *state_dof_constraints_[time_to_dofhandler_[this->GetTimeDoFNumber()]];
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
                                                           "Rothe_SpaceTimeHandler::InterpolateState");

      double lambda_l = (it.get_right() - t) / it.get_k();
      double lambda_r = (t - it.get_left()) / it.get_k();

      //Here we assume that the numbering of dofs goes from left to right!
      result = *local_vectors[0];

      result.sadd(lambda_l, lambda_r, *local_vectors[1]);
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandlerBase
     */
    unsigned int GetStateNDoFs(unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      return GetStateDoFHandler(time_point).n_dofs();
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const std::vector<Point<dealdim> > &
    GetMapDoFToSupportPoints(unsigned int time_point = std::numeric_limits<unsigned int>::max())
    {
      assert(time_point == std::numeric_limits<unsigned int>::max() || time_point == this->GetTimeDoFNumber());
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetMapDoFToSupportPoints");
      }
      assert(time_to_dofhandler_[this->GetTimeDoFNumber()]<support_points_.size());
      support_points_[time_to_dofhandler_[this->GetTimeDoFNumber()]].resize(GetStateNDoFs(time_point));
      DOpE::STHInternals::MapDoFsToSupportPoints<std::vector<Point<dealdim> >, dealdim>(this->GetMapping(), GetStateDoFHandler(time_point), support_points_[time_to_dofhandler_[this->GetTimeDoFNumber()]]);
      return support_points_[time_to_dofhandler_[this->GetTimeDoFNumber()]];
    }

    /**
     * Implementation of virtual function in StateSpaceTimeHandler
     */
    const std::vector<unsigned int>* GetNNeighbourElements(unsigned int /*time_point = std::numeric_limits<unsigned int>::max()*/)
    {
      assert(time_point == std::numeric_limits<unsigned int>::max() || time_point == this->GetTimeDoFNumber());
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::GetNNeighbourElements");
      }
      if(n_neighbour_to_vertex_.size() <= time_to_dofhandler_[this->GetTimeDoFNumber()])
      {
	n_neighbour_to_vertex_.resize(n_dof_handlers_);
	assert(time_to_dofhandler_[this->GetTimeDoFNumber()] < n_dof_handlers_);
      }
      if(n_neighbour_to_vertex_[time_to_dofhandler_[this->GetTimeDoFNumber()]].size()!=triangulations_[time_to_dofhandler_[this->GetTimeDoFNumber()]]->n_vertices())
      {
	DOpE::STHInternals::CalculateNeigbourElementsToVertices(*triangulations_[time_to_dofhandler_[this->GetTimeDoFNumber()]],n_neighbour_to_vertex_[time_to_dofhandler_[this->GetTimeDoFNumber()]]);
      }
      return &n_neighbour_to_vertex_[time_to_dofhandler_[this->GetTimeDoFNumber()]];
    }

    /******************************************************/
    void ComputeStateSparsityPattern(SPARSITYPATTERN &sparsity,unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      this->GetSparsityMaker()->ComputeSparsityPattern(this->GetStateDoFHandler(time_point), sparsity, this->GetStateDoFConstraints(time_point),
                                                       this->GetStateDoFsPerBlock(time_point));
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
          throw DOpEException("Not implemented for name =" + name, "Rothe_StateSpaceTimeHandler::GetFESystem");
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
      //std::cout<<"Not implemented yet: RefineSpace"<<std::endl;
      //abort();
      ///Time refinement needs potentially a change in the dofhandler number?
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

    void RefineSpace(const RefinementContainer &ref_container) //Refine function for cell refinement depending on the error on the cell of the current grid
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();
      
      //make sure that we do not use any coarsening
      assert( !ref_container.UsesCoarsening());
	// i loop through all n dof handlers
      for(unsigned int i = 0; i < n_dof_handlers_; i++) // n dof handlers is the number of dof handlers necessary to create different spatial meshes in different timesteps
      {
      if (state_mesh_transfers_[i] != NULL)
        {
          delete state_mesh_transfers_[i];
          state_mesh_transfers_[i] = NULL;
        }
      state_mesh_transfers_[i] = new dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> >(*state_dof_handlers_[i]);
	dealii::Vector<float> Indicators;
	// j goes through every timestep
	  for(unsigned int j = 0; j < time_to_dofhandler_.size(); j++) // time to dofhandler is a vector which has the size of number of timesteps and contains 0 where no dofhandler needed and 1 where dofhandler needed, in main called Rothe_time_to_dof
          {
	  if(j == i) // eigentlich will ich abfragen, wenn aktueller zeipunkt mit index i uebereinstimmt
	   {
	   if(Indicators.size()==0)
	     {
		Indicators=ref_container.GetLocalErrorIndicators(j); //then j would be the current timepoint!
	     }
	   else
	     {
	   assert(Indicators.size()==ref_container.GetLocalErrorIndicators(j).size());
		Indicators+=ref_container.GetLocalErrorIndicators(j);
		std::cout << Indicators.size() << std::endl;
	     }
	   }
	  }
	  switch (ref_type)
            {
            case DOpEtypes::RefinementType::global:
             triangulations_[i]->set_all_refine_flags();
            break;

            case DOpEtypes::RefinementType::fixed_number:
             GridRefinement::refine_and_coarsen_fixed_number (*triangulations_[i],
                                                           Indicators,
                                                           ref_container.GetTopFraction (),
                                                          ref_container.GetBottomFraction());
            break;

            case DOpEtypes::RefinementType::fixed_fraction:
             GridRefinement::refine_and_coarsen_fixed_fraction (*triangulations_[i],
                                                             Indicators,
                                                             ref_container.GetTopFraction (),
                                                            ref_container.GetBottomFraction());
            break;

            case DOpEtypes::RefinementType::optimized: // we use optimized refinement
             GridRefinement::refine_and_coarsen_optimize (*triangulations_[i],
                                                       Indicators,
                                                      ref_container.GetConvergenceOrder());
            break;

            default:
             throw DOpEException (
              "Not implemented for name =" + DOpEtypesToString (ref_type),
                              "Rothe_StateSpaceTimeHandler::RefineStateSpace");
	    }
	triangulations_[i]->prepare_coarsening_and_refinement();
      	if (state_mesh_transfers_[i] != NULL) state_mesh_transfers_[i]->prepare_for_pure_refinement();
      	triangulations_[i]->execute_coarsening_and_refinement();
        }
    }
    /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    void SpatialMeshTransferState(const VECTOR &old_values, VECTOR &new_values, unsigned int /*time_point = std::numeric_limits<unsigned int>::max()*/) const
    {
      assert(time_point == std::numeric_limits<unsigned int>::max() || time_point == this->GetTimeDoFNumber());
      if(this->GetTimeDoFNumber() > time_to_dofhandler_.size() || this->GetTimeDoFNumber() == std::numeric_limits<unsigned int>::max())
      {
	throw DOpEException("Invalid Timepoint", "Rothe_SpaceTimeHandler::SpatialMeshTransferState");
      }
      if (state_mesh_transfers_[time_to_dofhandler_[this->GetTimeDoFNumber()]] != NULL) state_mesh_transfers_[time_to_dofhandler_[this->GetTimeDoFNumber()]]->refine_interpolate(old_values, new_values);
     }

      /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */

    virtual void TemporalMeshTransferControl( VECTOR & /*new_values*/, unsigned int /*from_time_dof*/, unsigned int /*to_time_dof*/) const
    {
	abort();
    }

     /******************************************************/

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */

    virtual void TemporalMeshTransferState(VECTOR & new_values, unsigned int from_time_dof, unsigned int to_time_dof) const
    {
      assert(time_to_dofhandler_.size() > std::max(from_time_dof,to_time_dof));
      assert(state_dof_handlers_.size() > std::max(time_to_dofhandler_[from_time_dof],time_to_dofhandler_[to_time_dof]));
      if (time_to_dofhandler_[from_time_dof] == time_to_dofhandler_[to_time_dof])
      {
	return;
      }
      VECTOR temp = new_values;
      this->ReinitVector(new_values, DOpEtypes::VectorType::state, to_time_dof);
      VectorTools::interpolate_to_different_mesh(state_dof_handlers_[time_to_dofhandler_[from_time_dof]]->GetDEALDoFHandler(),
						 temp,
						 state_dof_handlers_[time_to_dofhandler_[to_time_dof]]->GetDEALDoFHandler(),
						 *state_dof_constraints_[time_to_dofhandler_[to_time_dof]],
						 new_values);	
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
    /**
     * Initialize the map of time to dof_handler
     * and the corresponding triangulations ...
     */
    void InitSpaceTime(dealii::Triangulation<dealdim> &triangulation,
		       std::vector<unsigned int> &time_to_dofhandler)
    {
      //Create Map DoF To 
      if(time_to_dofhandler.size()==0)
      {
	time_to_dofhandler_.resize(this->GetMaxTimePoint()+1,0);
	n_dof_handlers_ = 1;
      }
      else
      {
	if(time_to_dofhandler.size() != this->GetMaxTimePoint()+1)
	{
	  throw DOpEException("Invalid given time_to_dofhandler map! Needs to have the same length as number of time-points.", "Rothe_SpaceTimeHandler::InitSpaceTime");
	}
	time_to_dofhandler_.resize(this->GetMaxTimePoint()+1);
	//Input needs to be ordered without missing numbers!
	for(unsigned int i = 0; i <= this->GetMaxTimePoint(); i++)
	{
 	  time_to_dofhandler_[i] = time_to_dofhandler[i];
	  if(i==0)
	  {
	    if(time_to_dofhandler_[i] != 0)
	    {
	      throw DOpEException("Invalid given time_to_dofhandler map! Must start with number zero!", "Rothe_SpaceTimeHandler::InitSpaceTime");
	    }
	    n_dof_handlers_=1;
	  }
	  else
	  {
	    if(time_to_dofhandler_[i] >= n_dof_handlers_)
	    {
	      n_dof_handlers_++;
	      if(time_to_dofhandler_[i] >= n_dof_handlers_)
	      {
		throw DOpEException("Invalid given time_to_dofhandler map! DoF-Handler numbers must be given sequentially!", "Rothe_SpaceTimeHandler::InitSpaceTime");
	      }
	    }
	  }
	}
      }
      dofhandler_to_time_.resize(n_dof_handlers_);
      for(unsigned int i = 0; i < n_dof_handlers_; i++)
      {
	dofhandler_to_time_[i]=*(find(time_to_dofhandler_.begin(),time_to_dofhandler_.end(),i));
      }
      //Initialize triangulations, ...
      triangulations_.resize(n_dof_handlers_,NULL);
      state_dof_handlers_.resize(n_dof_handlers_,NULL);
      state_dof_constraints_.resize(n_dof_handlers_,NULL);
      state_mesh_transfers_.resize(n_dof_handlers_,NULL);

      for(unsigned int i = 0; i < n_dof_handlers_; i++)
      {
	if(i == 0)
	{
	  triangulations_[0]= &triangulation;
	}
	else
	{
	  //TODO: Not shure if this works if the original
	  //Triangulation is a derived class such as
	  //parallel::distributed::triangulation...
	  triangulations_[i]=new dealii::Triangulation<dealdim>;
	  assert(triangulations_[i] != NULL);
	  assert(triangulations_[i-1] != NULL);
	  triangulations_[i]->copy_triangulation(*(triangulations_[i-1]));
	}
	assert(triangulations_[i] != NULL);	  
	state_dof_handlers_[i] = new DOpEWrapper::DoFHandler<dealdim, DH>(*(triangulations_[i]));
	state_dof_constraints_[i]= new dealii::ConstraintMatrix;
      }
    }
    
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

    std::vector<dealii::Triangulation<dealdim> *> triangulations_; 
    std::vector<DOpEWrapper::DoFHandler<dealdim, DH> *> state_dof_handlers_;

    std::vector<std::vector<unsigned int> > state_dofs_per_block_;

    std::vector<dealii::ConstraintMatrix*> state_dof_constraints_;

    const dealii::SmartPointer<const FE<dealdim, dealdim> > state_fe_;
    const dealii::SmartPointer<const DOpEWrapper::Mapping<dealdim, DH> > mapping_;

    std::vector<std::vector<Point<dealdim> > > support_points_;
    std::vector<dealii::SolutionTransfer<dealdim, VECTOR, DH<dealdim, dealdim> > *> state_mesh_transfers_;

    std::vector<std::vector<unsigned int> > n_neighbour_to_vertex_;

    std::vector<unsigned int> time_to_dofhandler_;
    std::vector<unsigned int> dofhandler_to_time_;
    unsigned int n_dof_handlers_;
  };

}
#endif
