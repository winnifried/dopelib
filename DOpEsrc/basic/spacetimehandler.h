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

#ifndef SPACE_TIME_HANDLER_H_
#define SPACE_TIME_HANDLER_H_

#include <basic/spacetimehandler_base.h>
#include <interfaces/active_fe_index_setter_interface.h>
#include <wrapper/mapping_wrapper.h>
#include <wrapper/dataout_wrapper.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/dofs/dof_handler.h>


// Multi-level routines
//#include <deal.II/multigrid/mg_constrained_dofs.h>
//#include <deal.II/multigrid/multigrid.h>
//#include <deal.II/multigrid/mg_transfer.h>
//#include <deal.II/multigrid/mg_tools.h>
//#include <deal.II/multigrid/mg_coarse.h>
//#include <deal.II/multigrid/mg_smoother.h>
//#include <deal.II/multigrid/mg_matrix.h>



#include <vector>
#include <iostream>
#include <sstream>

namespace DOpE
{
  /**
   * Interface to the dimension depended functionality of a
   * SpaceTimeDoFHandler.
   *
   * @tparam <FE>               The finite element type we use (i.e. 'normal' finite elements vs. hp::FECollections)
   * @tparam <DH>       The dofhandler type we use (i.e. 'normal' dofhandler vs. hp::dofhandler)
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state. This is needed as a class template, because
   *                            member function templates are not allowed for virtual member functions.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam<dopedim>           The dimension for the control variable.
   * @tparam<dealdim>           The dimension for the state variable. This is the dimension the
   *                            mesh is in.
   */
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
           typename VECTOR, int dopedim, int dealdim>
  class SpaceTimeHandler : public SpaceTimeHandlerBase<VECTOR>
  {
  public:
    SpaceTimeHandler(DOpEtypes::ControlType type) :
      SpaceTimeHandlerBase<VECTOR>(type), control_index_(
        dealii::numbers::invalid_unsigned_int), state_index_(
          dealii::numbers::invalid_unsigned_int)
    {
    }
    SpaceTimeHandler(dealii::Triangulation<1> &times,
                     DOpEtypes::ControlType type) :
      SpaceTimeHandlerBase<VECTOR>(times, type), control_index_(
        dealii::numbers::invalid_unsigned_int), state_index_(
          dealii::numbers::invalid_unsigned_int)
    {
    }
    SpaceTimeHandler(DOpEtypes::ControlType type,
                     const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter) :
      SpaceTimeHandlerBase<VECTOR>(type), control_index_(
        dealii::numbers::invalid_unsigned_int), state_index_(
          dealii::numbers::invalid_unsigned_int), fe_index_setter_(
            &index_setter)
    {
    }
    SpaceTimeHandler(dealii::Triangulation<1> &times,
                     DOpEtypes::ControlType type,
                     const ActiveFEIndexSetterInterface<dopedim, dealdim> &index_setter) :
      SpaceTimeHandlerBase<VECTOR>(times, type), control_index_(
        dealii::numbers::invalid_unsigned_int), state_index_(
          dealii::numbers::invalid_unsigned_int), fe_index_setter_(
            &index_setter)
    {
    }
    virtual
    ~SpaceTimeHandler()
    {

    }

    /**
     * Initializes the dof handlers corresponding to the finite elements.
     *
     * @param control_n_blocks          Number of Blocks for the control variable
     * @param control_block_components  Component to Block mapping for the control
     * @param DD_control                Description of the Dirichlet Boundaries
    *                                  for the control
    * @param state_n_blocks            Number of Blocks for the state variable
     * @param state_block_components    Component to Block mapping for the state
     * @param DD_state                  Description of the Dirichlet Boundaries
    *                                  for the state
     */
    virtual void
    ReInit(unsigned int control_n_blocks,
           const std::vector<unsigned int> &control_block_component,
           const DirichletDescriptor &DD_control,
           unsigned int state_n_blocks,
           const std::vector<unsigned int> &state_block_component,
           const DirichletDescriptor &DD_state) =0;

    /******************************************************/

    /**
     * Returns a reference to the DoF Handler for the Control at the current time point.
     */
    virtual const DOpEWrapper::DoFHandler<dopedim, DH> &
    GetControlDoFHandler() const =0;

    /******************************************************/

    /**
     * Returns a reference to the DoF Handler for the State at the current time point.
     */
    virtual const DOpEWrapper::DoFHandler<dealdim, DH> &
    GetStateDoFHandler() const = 0;

    /******************************************************/

    /**
     * Returns a reference to the Mapping in use.
     */
    virtual const DOpEWrapper::Mapping<dealdim, DH> &
    GetMapping() const = 0;

    /******************************************************/

    /**
     * Returns a reference to a vector of DoFHandlers, the order of the DoFHandlers must
     * be set prior by SetDoFHandlerOrdering
     */
    const std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*> &
    GetDoFHandler() const
    {
      assert(state_index_ != dealii::numbers::invalid_unsigned_int);
#if dope_dimension > 0
      assert(control_index_ != dealii::numbers::invalid_unsigned_int);
      domain_dofhandler_vector_[control_index_] = &GetControlDoFHandler();
      domain_dofhandler_vector_[state_index_] = &GetStateDoFHandler();
#else
      domain_dofhandler_vector_[state_index_] = &GetStateDoFHandler();
#endif
      return domain_dofhandler_vector_;
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
    * Experimental status:
     * Returns a vector of the begin_celliterators of the
     * DoFHandlers in use.
    * Iterator for multigrid's matrix assembling running
    * over all elements on all levels.
     */
    std::vector<
    typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator>
    GetDoFHandlerBeginActiveAllLevels() const
    {
      std::vector<
      typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator> ret(
        this->GetDoFHandler().size());
      for (unsigned int dh = 0; dh < this->GetDoFHandler().size(); dh++)
        {
          ret[dh] = this->GetDoFHandler()[dh]->begin_active();
        }
      return ret;
    }

    /******************************************************/

    /**
    * Experimental status:
     * Returns a vector of the end-celliterators of the
     * DoFHandlers in use.
    * Iterator for multigrid's matrix assembling running
    * over all elements on all levels.
     */

    std::vector<
    typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator>
    GetDoFHandlerEndAllLevels() const
    {
      std::vector<
      typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator> ret(
        this->GetDoFHandler().size());
      for (unsigned int dh = 0; dh < this->GetDoFHandler().size(); dh++)
        {
          ret[dh] = this->GetDoFHandler()[dh]->end();
        }
      return ret;
    }



    /******************************************************/

    /**
     * Sets the ordering of the DofHandlers in GetDoFHandler.
     * Indices must start at zero and be consecutive numbers. If dealdim != dopedim the
     * control_index doesn't matter.
     *
     * @param control_index      Index for the control
     * @param state_index        Index for the state
     */
    void
    SetDoFHandlerOrdering(unsigned int control_index,
                          unsigned int state_index)
    {
      control_index_ = control_index;
      state_index_ = state_index;
#if dope_dimension > 0
      {
        assert(( control_index_ ==0 && state_index_ ==1 )||( control_index_ ==1 && state_index_ ==0 ));
        domain_dofhandler_vector_.clear();
        if (domain_dofhandler_vector_.size() != 2)
          {
            domain_dofhandler_vector_.resize(2,NULL);
          }
      }
#else
      {
        assert(state_index_ == 0);
        domain_dofhandler_vector_.clear();
        if (domain_dofhandler_vector_.size() != 1)
          {
            domain_dofhandler_vector_.resize(1, NULL);
          }
      }
#endif
    }
    /******************************************************/

    /**
     * Returns the order of the StateDofHandler set by SetDoFHandlerOrdering.
     *
     */
    unsigned int
    GetStateIndex()
    {
      return state_index_;
    }

    /******************************************************/
    /**
     * Returns a const reference to the ActiveFEIndexSetterInterface object stored in
     * this class. This function is only useful in the hp case.
     */
    const ActiveFEIndexSetterInterface<dopedim, dealdim> &
    GetFEIndexSetter() const
    {
      //makes only sense in the hp case.
      return *fe_index_setter_;
    }

    /******************************************************/
    /*
     * This function sets for every element the right fe index for the state variable.
     * This is only useful in the hp case!
     *
     * @param dof_handler   The dof_handler for which the fe indices have to be set.
     */
    void
    SetActiveFEIndicesState(
      DOpEWrapper::DoFHandler<dealdim, DH> &dof_handler)
    {
      if (dof_handler.NeedIndexSetter()) //with this we distinguish between hp and classic
        {
          for (typename DH<dealdim, dealdim>::active_cell_iterator element =
                 dof_handler.begin_active(); element != dof_handler.end(); ++element)
            {
              this->GetFEIndexSetter().SetActiveFEIndexState(element);
            }
        }
    }

    /******************************************************/
    /*
     * This function sets for every element the right fe index for the state variable.
     * This is only useful in the hp case!
     *
     * @param dof_handler   The dof_handler for which the fe indices have to be set.
     */
    void
    SetActiveFEIndicesControl(
      DOpEWrapper::DoFHandler<dopedim, DH> &dof_handler)
    {
      if (dof_handler.NeedIndexSetter())
        {
          for (typename DH<dopedim, dopedim>::active_cell_iterator element =
                 dof_handler.begin_active(); element != dof_handler.end(); ++element)
            {
              this->GetFEIndexSetter().SetActiveFEIndexState(element);
            }
        }
    }

    /******************************************************/

    /**
     * Returns the constraint dofs in Block b at the current time
     */
    virtual unsigned int
    GetConstraintDoFsPerBlock(std::string name, unsigned int b) const=0;

    /******************************************************/

    /**
     * Returns the control dofs per Block at the current time
     */
    virtual const std::vector<unsigned int> &
    GetControlDoFsPerBlock(int time_point = -1) const =0;

    /******************************************************/

    /**
     * Returns the state dofs per Block at the  time indicated by time_point.
     *
     * \\TODO
     */
    virtual const std::vector<unsigned int> &
    GetStateDoFsPerBlock(int time_point = -1) const =0;

    /******************************************************/
    /**
     * Returns the DoFs per  block for the constraint vector at the current
     * time which has to be set prior to calling this function using SetTime.
     */
    virtual const std::vector<unsigned int> &
    GetConstraintDoFsPerBlock(std::string name) const = 0;

    /******************************************************/

    /**
     * Returns the control HN-Constraints  at the current time
     */
    virtual const dealii::ConstraintMatrix
    &
    GetControlDoFConstraints() const=0;

    /******************************************************/

    /**
     * Returns the state HN-Constraints at the current time
     */
    virtual const dealii::ConstraintMatrix
    &
    GetStateDoFConstraints() const=0;

    /*******************************************************/

    /**
     * Returns a Reference to a vector of points where the FEs have their support points.
     * on the current spatial mesh (if they do have that compare dealii::DoFTools>>map_dofs_to_support_points!).
     */
    virtual const std::vector<dealii::Point<dealdim> >
    &
    GetMapDoFToSupportPoints()=0;

    /******************************************************/

    /**
     * Computes the current sparsity pattern for the control variable
     */
    virtual void
    ComputeControlSparsityPattern(SPARSITYPATTERN &sparsity) const=0;

    /******************************************************/

    /**
     * Computes the current sparsity pattern for the state variable
     */
    virtual void
    ComputeStateSparsityPattern(SPARSITYPATTERN &sparsity) const=0;

    /******************************************************/

//        /**
//   * Experimental status:
//         * Needed for MG prec.
//         */
//        virtual void
//    ComputeMGStateSparsityPattern(dealii::MGLevelObject<dealii::BlockSparsityPattern> & /*mg_sparsity_patterns*/,
//          unsigned int /*n_levels*/) const
//  {
//     throw DOpEException(
//                "Not used for normal DofHandler",
//                "StateSpaceTimeHandler.h");
//  }

    /******************************************************/

//      /**
//       * Experimental status:
//       * Needed for MG prec.
//       */
//      virtual void
//    ComputeMGStateSparsityPattern(dealii::MGLevelObject<dealii::SparsityPattern> & /*mg_sparsity_patterns*/,
//          unsigned int /*n_levels*/) const
//  {
//     throw DOpEException(
//              "Not used for normal DofHandler",
//              "StateSpaceTimeHandler.h");
//  }
//
    /******************************************************/
    /**
     * Returns a const Smartpointer to the FESystem indicated by the string 'name', i.e. state oder control.
     */

    virtual const FE<dealdim, dealdim> &
    GetFESystem(std::string name) const=0;

    /******************************************************/

    DOpEWrapper::DataOut<dealdim, DH> &
    GetDataOut()
    {
      data_out_.clear();
      return data_out_;
    }

    /******************************************************/

  protected:
    //we need this here, because we know the type of the DoFHandler in use.
    //This saves us a template argument for statpdeproblem etc.
    DOpEWrapper::DataOut<dealdim, DH> data_out_;
    unsigned int control_index_, state_index_;
    const ActiveFEIndexSetterInterface<dopedim, dealdim> *fe_index_setter_;
    mutable std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*> domain_dofhandler_vector_;
    //TODO What if control and state have different dofhandlertypes??

  };
}

#endif
