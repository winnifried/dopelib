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

#ifndef _STATE_SPACE_TIME_HANDLER_H_
#define _STATE_SPACE_TIME_HANDLER_H_

#include "spacetimehandler_base.h"
#include "active_fe_index_setter_interface.h"

#include <numerics/data_out.h>
#include <lac/vector.h>
#include <lac/block_vector_base.h>
#include <lac/block_vector.h>
#include <lac/constraint_matrix.h>
#include <dofs/dof_handler.h>

#include <vector>
#include <iostream>
#include <sstream>

namespace DOpE
{
  /**
   * Interface to the dimension depended functionality of a
   * StateSpaceTimeDoFHandler.
   *
   * @tparam <FE>               The finite element type we use (i.e. 'normal' finite elements vs. hp::FECollections)
   * @tparam <DH>       The dofhandler type we use (i.e. 'normal' dofhandler vs. hp::dofhandler)
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state. This is needed as a class template, because
   *                            member function templates are not allowed for virtual member functions.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam<dealdim>           The dimension for the state variable. This is the dimension the
   *                            mesh is in.
   */
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
      typename VECTOR, int dealdim>
    class StateSpaceTimeHandler : public SpaceTimeHandlerBase<VECTOR>
    {
      public:
        StateSpaceTimeHandler() :
            SpaceTimeHandlerBase<VECTOR>()
        {
          _domain_dofhandler_vector.resize(1);
        }
        StateSpaceTimeHandler(const dealii::Triangulation<1> & times) :
            SpaceTimeHandlerBase<VECTOR>(times)
        {
          _domain_dofhandler_vector.resize(1);
        }
        StateSpaceTimeHandler(
            const ActiveFEIndexSetterInterface<dealdim>& index_setter) :
            SpaceTimeHandlerBase<VECTOR>(), _fe_index_setter(&index_setter)
        {
          _domain_dofhandler_vector.resize(1);
        }
        StateSpaceTimeHandler(const dealii::Triangulation<1> & times,
            const ActiveFEIndexSetterInterface<dealdim>& index_setter) :
            SpaceTimeHandlerBase<VECTOR>(times), _fe_index_setter(&index_setter)
        {
          _domain_dofhandler_vector.resize(1);
        }
        virtual
        ~StateSpaceTimeHandler()
        {
        }

        /**
         * Initializes the dof handlers corresponding to the finite elements.
         *
         * @param control_n_blocks          Number of Blocks for the control variable
         * @param control_block_components  Component to Block mapping for the control
         * @param state_n_blocks            Number of Blocks for the state variable
         * @param state_block_components    Component to Block mapping for the state
         */
        virtual void
        ReInit(unsigned int state_n_blocks,
            const std::vector<unsigned int>& state_block_component) =0;

        /******************************************************/

        /**
         * Returns a reference to the DoF Handler for the State at the current time point.
         */
        virtual const DOpEWrapper::DoFHandler<dealdim, DH>&
        GetStateDoFHandler() const =0;

        /******************************************************/

        /**
         * Returns a reference to the Mapping in use.
         */
        virtual const DOpEWrapper::Mapping<dealdim, DH>&
        GetMapping() const = 0;

        /******************************************************/

        /**
         * Returns a reference to a vector of DoFHandlers, the order of the DoFHandlers must
         * be set prior by SetDoFHandlerOrdering
         */
        const std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*>&
        GetDoFHandler() const
        {
          _domain_dofhandler_vector[0] = &GetStateDoFHandler();
          return _domain_dofhandler_vector;
        }

        /******************************************************/

        /**
         * Returns a vector of the begin_active-celliterators of the
         * DoFHandlers in use.
         */
        std::vector<
            typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator>
        GetDoFHandlerBeginActive() const
        {
          std::vector<
              typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator> ret(
              this->GetDoFHandler().size());
          for (unsigned int dh = 0; dh < this->GetDoFHandler().size(); dh++)
          {
            ret[dh] = this->GetDoFHandler()[dh]->begin_active();
          }
          return ret;
        }

        /******************************************************/

        /**
         * Returns a vector of the end-celliterators of the
         * DoFHandlers in use.
         */

        std::vector<
            typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator>
        GetDoFHandlerEnd() const
        {
          std::vector<
              typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator> ret(
              this->GetDoFHandler().size());
          for (unsigned int dh = 0; dh < this->GetDoFHandler().size(); dh++)
          {
            ret[dh] = this->GetDoFHandler()[dh]->end();
          }
          return ret;
        }

        /******************************************************/

        /**
         * Returns the order of the StateDofHandler set by SetDoFHandlerOrdering.
         *
         */
        unsigned int
        GetStateIndex()
        {
          return 0;
        }

        /******************************************************/
        /**
         * Returns a const reference to the ActiveFEIndexSetterInterface object stored in
         * this class. This function is only useful in the hp case.
         */
        const ActiveFEIndexSetterInterface<dealdim>&
        GetFEIndexSetter() const
        {
          //makes only sense in the hp case.
          return *_fe_index_setter;
        }

        /******************************************************/
        /*
         * This function sets for every cell the right fe index for the state variable.
         * This is only useful in the hp case!
         *
         * @param dof_handler   The dof_handler for which the fe indices have to be set.
         */
        void
        SetActiveFEIndicesState(
            DOpEWrapper::DoFHandler<dealdim, DH>& dof_handler)
        {
          if (dof_handler.NeedIndexSetter())
          {
            for (typename DH<dealdim, dealdim>::active_cell_iterator cell =
                dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
            {
              this->GetFEIndexSetter().SetActiveFEIndexState(cell);
            }
          }
        }

        /******************************************************/

        /**
         * Returns the state dofs in Block b at at the time given through time_point. Default value for the time is now.
         */
        virtual unsigned int
        GetStateDoFsPerBlock(unsigned int b, int time_point = -1) const =0;

        /******************************************************/

        /**
         * Returns the state dofs per Block at the  time indicated by time_point.
         *
         * \\TODO
         */
        virtual const std::vector<unsigned int>&
        GetStateDoFsPerBlock(int time_point = -1) const =0;

        /******************************************************/

        /**
         * Returns the state HN-Constraints at the current time
         */
        virtual const dealii::ConstraintMatrix &
        GetStateDoFConstraints() const=0;

        /*******************************************************/

        /**
         * Returns a Reference to a vector of points where the FEs have their support points.
         * on the current spatial mesh (if they do have that compare dealii::DoFTools>>map_dofs_to_support_points!).
         */
        virtual const std::vector<dealii::Point<dealdim> > &
        GetMapDoFToSupportPoints()=0;

        /******************************************************/

        /**
         * Computes the current sparsity pattern for the state variable
         */
        virtual void
        ComputeStateSparsityPattern(SPARSITYPATTERN & sparsity) const=0;

        /******************************************************/
        /**
         * Returns a const Reference to the FESystem indicated by the string 'name', i.e. state oder control.
         */

        virtual const FE<dealdim, dealdim>&
        GetFESystem(std::string name) const=0;

        /******************************************************/

        dealii::DataOut<dealdim, DH<dealdim, dealdim> >&
        GetDataOut()
        {
          _data_out.clear();
          return _data_out;
        }

      protected:
        //we need this here, because we know the type of the DoFHandler in use.
        //This saves us a template argument for statpdeproblem etc.
        dealii::DataOut<dealdim, DH<dealdim, dealdim> > _data_out;
        const ActiveFEIndexSetterInterface<dealdim>* _fe_index_setter;
        mutable std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*> _domain_dofhandler_vector;

    };
}

#endif
