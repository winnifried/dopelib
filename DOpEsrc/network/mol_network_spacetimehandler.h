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

#ifndef Network_MOL_SPACE_TIME_HANDLER_H_
#define Network_MOL_SPACE_TIME_HANDLER_H_

#include <basic/spacetimehandler.h>
#include <basic/mol_statespacetimehandler.h>
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
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <network/networkinterface.h>

namespace DOpE
{
  namespace Networks
  {
    /**
     * Implements a Space Time Handler with a Method of Lines discretization.
     * This means there is only one fixed mesh for the spatial domain.
     *
     * Networks only work with dealdim = 1
     */
    template<template <int, int> class FE, template<int, int> class DH,
             typename VECTOR, int dopedim, int dealdim>
    class MethodOfLines_Network_SpaceTimeHandler : public SpaceTimeHandler<FE, DH, dealii::BlockSparsityPattern, VECTOR, dopedim, dealdim>
    {
    public:
      /**
      * Constructor used for stationary PDEs and stationary optimization problems without any
      * further constraints beyond the PDE.
       *
       * @param triangulation     The coarse triangulations to be used.
       * @param control_fe        The finite elements used for the discretization of the control variable.
       * @param state_fe          The finite elements used for the discretization of the state variable.
       * @param type              The type of the control, see dopetypes.h for more information.
       * @param network           An object describing the considered network
       * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
       */
      MethodOfLines_Network_SpaceTimeHandler(std::vector<dealii::Triangulation<dealdim> *> &triangulation,
                                             const FE<dealdim, dealdim> &control_fe,
                                             const FE<dealdim, dealdim> &state_fe,
                                             DOpEtypes::ControlType type,
                                             NetworkInterface &network,
                                             bool flux_pattern = false) :
        SpaceTimeHandler<FE, DH, dealii::BlockSparsityPattern,  VECTOR, dopedim, dealdim>(type),
        control_dof_handler_(*triangulation[0]),
        control_fe_(&control_fe),
        network_(network)
      {
        if (triangulation.size() != GetNPipes())
          {
            throw DOpEException("Mismatch between given triangulations and number of pipes.","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }

        if (dealdim != 1)
          {
            throw DOpEException("Not implemented for dealdim != 1","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }
        if (dopedim != 0)
          {
            throw DOpEException("Not implemented for dopedim != 0","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }
        sth_s_.resize(GetNPipes(),NULL);
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i] = new MethodOfLines_StateSpaceTimeHandler<FE,DH,dealii::SparsityPattern,dealii::Vector<double>,dealdim>
            (*triangulation[i],state_fe, flux_pattern);
          }
        selected_pipe_=sth_s_.size();

        state_dofs_per_block_.resize(sth_s_.size()+1,0);
      }

      /**
      * Constructor used for nonstationary PDEs and nonstationary optimization problems without any
      * further constraints beyond the PDE.
       *
       * @param triangulation     The coarse triangulations to be used.
       * @param control_fe        The finite elements used for the discretization of the control variable.
       * @param state_fe          The finite elements used for the discretization of the state variable.
       * @param times             The timegrid for instationary problems.
       * @param type              The type of the control, see dopetypes.h for more information.
       * @param network           An object describing the considered network
       * @param flux_pattern      True if a flux sparsity pattern is needed (for DG discretizations)
       */
      MethodOfLines_Network_SpaceTimeHandler(std::vector<dealii::Triangulation<dealdim> *> &triangulation,
                                             const FE<dealdim, dealdim> &control_fe,
                                             const FE<dealdim, dealdim> &state_fe,
                                             dealii::Triangulation<1> &times,
                                             DOpEtypes::ControlType type,
                                             NetworkInterface &network,
                                             bool flux_pattern = false) :
        SpaceTimeHandler<FE, DH, dealii::BlockSparsityPattern,  VECTOR, dopedim, dealdim>(times, type),
        control_dof_handler_(*triangulation[0]),
        control_fe_(&control_fe),
        network_(network)
      {

        if (triangulation.size() != GetNPipes())
          {
            throw DOpEException("Mismatch between given triangulations and number of pipes.","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }
        if (dealdim != 1)
          {
            throw DOpEException("Not implemented for dealdim != 1","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }
        if (dopedim != 0)
          {
            throw DOpEException("Not implemented for dopedim != 0","MethodOfLines_Network_SpaceTimeHandler::MethodOfLines_Network_SpaceTimeHandler");
          }
        sth_s_.resize(GetNPipes(),NULL);
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i] = new MethodOfLines_StateSpaceTimeHandler<FE,DH,dealii::SparsityPattern,dealii::Vector<double>,dealdim>
            (*triangulation[i],state_fe,flux_pattern);
          }
        selected_pipe_=sth_s_.size();

        state_dofs_per_block_.resize(sth_s_.size()+1,0);
      }

      ~MethodOfLines_Network_SpaceTimeHandler()
      {
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            if (sth_s_[i] != NULL)
              {
                delete sth_s_[i];
                sth_s_[i] = NULL;
              }
          }
      }

      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      void
      ReInit(unsigned int control_n_blocks,
             const std::vector<unsigned int> &control_block_component,
             const DirichletDescriptor &,
             unsigned int state_n_blocks,
             const std::vector<unsigned int> &state_block_component,
             const DirichletDescriptor &DD_state)
      {
        control_dof_handler_.distribute_dofs(*control_fe_);
        control_dofs_per_block_.resize(control_n_blocks);
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

        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i]->ReInit(state_n_blocks,state_block_component,DD_state);
          }
        //Initialize also the timediscretization.
        this->ReInitTime();

        //There where changes invalidate tickets
        this->IncrementControlTicket();
        this->IncrementStateTicket();
        total_state_dofs_ = CountStateDoFs();
        InitializePipeBoundaryMapping();
      }

      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      const DOpEWrapper::DoFHandler<dopedim, DH> &
      GetControlDoFHandler() const
      {
        //There is only one control mesh, hence always return this
        return control_dof_handler_;
      }
      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      const DOpEWrapper::DoFHandler<dealdim, DH> &
      GetStateDoFHandler() const
      {
        if (selected_pipe_ < sth_s_.size())
          return sth_s_[selected_pipe_]->GetStateDoFHandler();

        abort();
        //No pipe selected
        throw DOpEException("No pipe selected","MethodOfLines_Network_SpaceTimeHandler::GetStateDoFHandler");
      }
      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      const DOpEWrapper::Mapping<dealdim, DH> &
      GetMapping() const
      {
        if (selected_pipe_ < sth_s_.size())
          return sth_s_[selected_pipe_]->GetMapping();

        //No pipe selected
        abort();
        throw DOpEException("No pipe selected","MethodOfLines_Network_SpaceTimeHandler::GetMapping");
      }

      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      unsigned int
      GetConstraintDoFsPerBlock(std::string /*name*/, unsigned int /*b*/) const
      {
        //Not implemented
        abort();
        return 0.;
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
      GetConstraintDoFsPerBlock(std::string /*name*/) const
      {
        //not implemented
        abort();
        return control_dofs_per_block_ ;//Wrong value just to avoid warnings!
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
        if (selected_pipe_ < sth_s_.size())
          return sth_s_[selected_pipe_]->GetStateDoFConstraints();

        //No pipe selected
        abort();
        throw DOpEException("No pipe selected","MethodOfLines_Network_SpaceTimeHandler::GetStateDoFConstraints");

//      return state_dof_constraints_;
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
      InterpolateState( VECTOR &/*result*/,
                        const std::vector<VECTOR *> &/*local_vectors*/, double /*t*/,
                        const TimeIterator &/*it*/) const
      {
        abort();
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
        return total_state_dofs_;
      }
      /**
       * Implementation of virtual function in SpaceTimeHandlerBase
       */
      unsigned int
      GetConstraintNDoFs(std::string /*name*/) const
      {
        return 0;
      }
      /**
       * Implementation of virtual function in SpaceTimeHandlerBase
       */
      unsigned int
      GetNGlobalConstraints() const
      {
        return 0;
        //return constraints_.global_n_dofs();
      }
      /**
       * Implementation of virtual function in SpaceTimeHandlerBase
       */
      unsigned int
      GetNLocalConstraints() const
      {
        //return constraints_.local_n_dofs();
        return 0;
      }

      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      const std::vector<Point<dealdim> > &
      GetMapDoFToSupportPoints()
      {
        if (selected_pipe_ < sth_s_.size())
          return sth_s_[selected_pipe_]->GetMapDoFToSupportPoints();

        //No pipe selected
        abort();
        throw DOpEException("No pipe selected","MethodOfLines_Network_SpaceTimeHandler::GetMapDoFToSupportPoints");
      }

      /******************************************************/
      /**
       * Computes the SparsityPattern for the stiffness matrix
       * of the scalar product in the control space.
       */
      void
      ComputeControlSparsityPattern(dealii::BlockSparsityPattern &sparsity) const;

      /******************************************************/
      /**
       * Computes the SparsityPattern for the stiffness matrix
       * of the PDE.
       */
      void
      ComputeStateSparsityPattern(dealii::BlockSparsityPattern &sparsity) const
      {
        unsigned int n_pipes = GetNPipes();
        unsigned int n_comp = GetFESystem("state").n_components();
        sparsity.reinit(n_pipes+1,n_pipes+1);
        //PDE-Blocks
        for ( unsigned int i = 0; i < n_pipes ; i++)
          {
            for ( unsigned int j = 0; j < n_pipes; j++)
              {
                if ( j == i )
                  {
                    //Diagonal
                    dealii::SparsityPattern tmp;
                    sth_s_[i]->ComputeStateSparsityPattern(tmp);
                    sparsity.block(i,i).copy_from(tmp);
                  }
                else
                  {
                    //Off-diagonal zeros
                    sparsity.block(i,j).reinit(sth_s_[i]->GetStateNDoFs(),sth_s_[j]->GetStateNDoFs(),0);
                  }
              }
          }
        //final column; i.e., PDE to influx coupling
        const std::vector<std::vector<unsigned int> > &left_vals = GetPipeToLeftDoF();
        const std::vector<std::vector<unsigned int> > &right_vals = GetPipeToRightDoF();
        for ( unsigned int i = 0; i < n_pipes ; i++ )
          {
            //Anzahl der Zeilen
            std::vector<unsigned int> row_lenght(sth_s_[i]->GetStateNDoFs(),0);
            //Find-coupling lines
            assert(left_vals[i].size() == right_vals[i].size());
            assert(n_comp == right_vals[i].size());
            for ( unsigned int n =0; n < left_vals[i].size(); n++)
              {
                row_lenght[left_vals[i][n]]+=n_comp;//Can couple to all n_comp fluxes
                row_lenght[right_vals[i][n]]+=n_comp;
                row_lenght[right_vals[i][n]]+=n_comp;//Add one more to have symmetrie with the final row
              }
            //Reinit-size
            sparsity.block(i,n_pipes).reinit(sth_s_[i]->GetStateNDoFs(),2*n_pipes*n_comp,row_lenght);
            //Add couplings (Left-Pipes are stored in [0,n_pipes*n_comp) and right in [n_pipes*n_comp, 2*n_pipes*n_comp])
            //For the i-th pipe the offset for the first component-flux is i*n_comp
            for ( unsigned int n =0; n < left_vals[i].size(); n++)
              {
                for ( unsigned int c=0; c < n_comp; c++)
                  {
                    //In each line coupling to all corresponding fluxes at the boundary are possible
                    //Left Values
                    sparsity.block(i,n_pipes).add(left_vals[i][n],i*n_comp+c);
                    //Right Values
                    sparsity.block(i,n_pipes).add(right_vals[i][n],n_pipes*n_comp+i*n_comp+c);
                    //For symmetry
                    sparsity.block(i,n_pipes).add(right_vals[i][n],i*n_comp+c);
                  }
              }
            sparsity.block(i,n_pipes).compress();
          }
        //Final row, outflow coupling:
        for ( unsigned int i = 0; i < n_pipes ; i++ )
          {
            //Anzahl der Zeilen
            std::vector<unsigned int> row_lenght(2*n_pipes*n_comp,0);
            //Howmany-coupling lines
            for (unsigned int n=0; n < n_comp; n++)
              {
                row_lenght[i*n_comp+n] = 2*n_comp;//one left and right value + symmetrie to final column!
                row_lenght[n_pipes*n_comp+i*n_comp+n] = n_comp;//For symmetrie with the final column
              }
            //Reinit-size
            sparsity.block(n_pipes,i).reinit(2*n_pipes*n_comp,sth_s_[i]->GetStateNDoFs(),row_lenght);
            //Add couplings (Left-Pipes are stored in [0,n_pipes*n_comp) and right in [n_pipes*n_comp, 2*n_pipes*n_comp])
            //For the i-th pipe the offset for the first component-flux is i*n_comp
            for ( unsigned int n =0; n < left_vals[i].size(); n++)
              {
                for (unsigned int c=0; c < n_comp; c++)
                  {
                    //Left Values
                    sparsity.block(n_pipes,i).add(i*n_comp+n,left_vals[i][c]);
                    //Right Values
                    sparsity.block(n_pipes,i).add(i*n_comp+n,right_vals[i][c]);
                    //Symmetrie with final collumn
                    sparsity.block(n_pipes,i).add(n_pipes*n_comp+i*n_comp+n,right_vals[i][c]);
                  }
              }
            sparsity.block(n_pipes,i).compress();
          }
        //Final block Flux Couplings
        GetNetwork().GetFluxSparsityPattern(sparsity.block(n_pipes,n_pipes));
        sparsity.block(n_pipes,n_pipes).symmetrize();
        sparsity.collect_sizes();
        sparsity.compress();
      }

      /******************************************************/

      /**
       * Implementation of virtual function in SpaceTimeHandler
       */
      const FE<dealdim, dealdim> &
      GetFESystem(std::string name) const
      {
        if (name == "state")
          {
            //all FESystems for state are the same
            return sth_s_[0]->GetFESystem(name);
          }
        else if (name == "control")
          {
            return *control_fe_;
          }
        else
          {
            throw DOpEException("Not implemented for name =" + name,
                                "MethodOfLines_Network_SpaceTimeHandler::GetFESystem");
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
        SpaceTimeHandlerBase<VECTOR >::RefineTime(ref_type);
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
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i]->RefineSpace(ref_container);
          }
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
      SpatialMeshTransferControl(const VECTOR &/*old_values*/,
                                 VECTOR &/*new_values*/) const
      {
        //should never be called
        abort();
      }
      void
      SpatialMeshTransferState(const  VECTOR &old_values,
                               VECTOR &new_values) const
      {
        assert(old_values.n_blocks() == new_values.n_blocks());
        assert(old_values.n_blocks() == sth_s_.size()+1);

        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i]->SpatialMeshTransferState(old_values.block(i),new_values.block(i));
          }
        if (new_values.block(sth_s_.size()).size() == old_values.block(sth_s_.size()).size())
          {
            new_values.block(sth_s_.size()) = old_values.block(sth_s_.size());
          }
        else
          {
            //N-Pipes should never increase, so the only case when this happens is at initialization!
            new_values.block(sth_s_.size()) = 0.;
          }
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
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i]->SetUserDefinedDoFConstraints(constraints_maker);
          }
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
      SetSparsityMaker(SparsityMaker<DH, dealdim> &sparsity_maker)
      {
        for (unsigned int i = 0; i < sth_s_.size(); i++)
          {
            sth_s_[i]->SetSparsityMaker(sparsity_maker);
          }
      }

      /******************************************************/
      /**
       * Through this function one can reinitialize the
       * triangulation for the state variable to be a copy of the
       * given argument.
       */
      void
      ResetTriangulation(const dealii::Triangulation<dealdim> &/*tria*/)
      {
        //not yet working for networks!
        abort();
      }

      /*************************************************************/
      /*Special functions for the use on networks*/
      unsigned int GetNPipes() const
      {
        return network_.GetNPipes();
      }
      void SelectPipe(unsigned int i)
      {
        if (i >= GetNPipes())
          selected_pipe_ = GetNPipes();
        else
          selected_pipe_ = i;
      }

      MethodOfLines_StateSpaceTimeHandler<FE,DH,dealii::SparsityPattern,dealii::Vector<double>,dealdim>      *GetPipeSTH()
      {
        if (selected_pipe_ < sth_s_.size())
          return sth_s_[selected_pipe_];

        abort();
        //No pipe selected
        throw DOpEException("No pipe selected","MethodOfLines_Network_SpaceTimeHandler::GetPipeSTH");
      }

      const std::vector<std::vector<unsigned int> > &GetPipeToLeftDoF() const
      {
        return pipe_to_leftdofs_;
      }
      const std::vector<std::vector<unsigned int> > &GetPipeToRightDoF() const
      {
        return pipe_to_rightdofs_;
      }

      const NetworkInterface &GetNetwork() const
      {
        return network_;
      }

    private:
      unsigned int CountStateDoFs()
      {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < sth_s_.size(); i++ )
          {
            assert(sth_s_[i]->GetStateDoFsPerBlock().size() == 1);
            assert(sth_s_[i]->GetStateDoFsPerBlock()[0] == sth_s_[i]->GetStateNDoFs());
            state_dofs_per_block_[i] = sth_s_[i]->GetStateNDoFs();
            sum += sth_s_[i]->GetStateNDoFs();
          }
        state_dofs_per_block_[sth_s_.size()] = 2*GetNPipes()*GetFESystem("state").n_components();
        sum += 2*GetNPipes()*GetFESystem("state").n_components(); //n-flow variables
        return sum;
      }

      //Initializes the mapping from the dofs on the pipe to the
      //coupling variables
      void InitializePipeBoundaryMapping()
      {
        pipe_to_leftdofs_.clear();
        pipe_to_rightdofs_.clear();
        pipe_to_leftdofs_.resize(GetNPipes(),std::vector<unsigned int>(GetFESystem("state").n_components(),0));
        pipe_to_rightdofs_.resize(GetNPipes(),std::vector<unsigned int>(GetFESystem("state").n_components(),0));


        assert(sth_s_.size()==GetNPipes());
        std::set<dealii::types::boundary_id> left;
        std::set<dealii::types::boundary_id> right;
        left.insert(0);
        right.insert(1);
        dealii::ComponentMask comp_mask(GetFESystem("state").n_components(),true);
        for (unsigned int i = 0; i < GetNPipes(); i++ )
          {
            std::vector<bool> selected_left;
            DoFTools::extract_dofs_with_support_on_boundary(sth_s_[i]->GetStateDoFHandler().GetDEALDoFHandler(),
                                                            comp_mask, selected_left, left);
            std::vector<bool> selected_right;
            DoFTools::extract_dofs_with_support_on_boundary(sth_s_[i]->GetStateDoFHandler().GetDEALDoFHandler(),
                                                            comp_mask, selected_right, right);
            //Sorting into Vector
            unsigned int l = 0;
            unsigned int r = 0;
            assert(selected_right.size()==selected_left.size());
            for ( unsigned int d = 0; d < selected_right.size(); d ++)
              {
                if (selected_left[d])
                  {
                    assert(l < GetFESystem("state").n_components());
                    pipe_to_leftdofs_[i][l] = d;
                    l++;
                  }
                if (selected_right[d])
                  {
                    assert(r < GetFESystem("state").n_components());
                    pipe_to_rightdofs_[i][r] = d;
                    r++;
                  }
              }
            assert(l == r);
            assert(l == GetFESystem("state").n_components());
          }
      }

      DOpEWrapper::DoFHandler<dopedim, DH> control_dof_handler_;
      const dealii::SmartPointer<const FE<dealdim, dealdim> > control_fe_;

      std::vector<unsigned int> control_dofs_per_block_;
      dealii::ConstraintMatrix control_dof_constraints_;
      std::vector<unsigned int> state_dofs_per_block_;

      std::vector<MethodOfLines_StateSpaceTimeHandler<FE,DH,dealii::SparsityPattern,dealii::Vector<double>,dealdim>*> sth_s_;
      NetworkInterface &network_;
      unsigned int selected_pipe_;
      unsigned int total_state_dofs_;
      std::vector<std::vector<unsigned int> > pipe_to_leftdofs_;
      std::vector<std::vector<unsigned int> > pipe_to_rightdofs_;
    };

    /**************************explicit instantiation*************/

    /**
     * The default case - is not implemented
     */
    template<template <int, int> class FE, template<int, int> class DH,
             typename VECTOR, int dopedim, int dealdim>
    void
    MethodOfLines_Network_SpaceTimeHandler<FE,DH,VECTOR,dopedim,dealdim>::ComputeControlSparsityPattern(
      dealii::BlockSparsityPattern &/*sparsity*/) const
    {
      abort();
    }

    /******************************************************/
    /**
     * The only reasonable case for networks - BlockVectors with 'normal' Sparsity pattern
     */
    template<>
    void
    MethodOfLines_Network_SpaceTimeHandler<dealii::FESystem,
                                           dealii::DoFHandler,
                                           dealii::BlockVector<double>, dope_dimension, deal_II_dimension>::ComputeControlSparsityPattern(
                                             dealii::BlockSparsityPattern &/*sparsity*/) const
    {
      abort();
    }


  }
}

#endif
