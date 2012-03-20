#ifndef _SPACE_TIME_HANDLER_H_
#define _SPACE_TIME_HANDLER_H_

#include "spacetimehandler_base.h"
#include "active_fe_index_setter_interface.h"

#include <numerics/data_out.h>
#include <lac/vector.h>
#include <lac/block_vector_base.h>
#include <lac/block_vector.h>
#include <dofs/dof_constraints.h>
#include <dofs/dof_handler.h>

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
   * @tparam <DOFHANDLER>       The dofhandler type we use (i.e. 'normal' dofhandler vs. hp::dofhandler)
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state. This is needed as a class template, because
   *                            member function templates are not allowed for virtual member functions.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam<dopedim>           The dimension for the control variable.
   * @tparam<dealdim>           The dimension for the state variable. This is the dimension the
   *                            mesh is in.
   */
  template<typename FE, typename DOFHANDLER, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim>
    class SpaceTimeHandler : public SpaceTimeHandlerBase<VECTOR>
    {
    public:
      SpaceTimeHandler() :
        SpaceTimeHandlerBase<VECTOR> ()
      {
      }
      SpaceTimeHandler(const dealii::Triangulation<1> & times) :
        SpaceTimeHandlerBase<VECTOR> (times)
      {
      }
      SpaceTimeHandler(
          const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter) :
        SpaceTimeHandlerBase<VECTOR> (), _fe_index_setter(&index_setter)
      {
      }
      SpaceTimeHandler(const dealii::Triangulation<1> & times,
          const ActiveFEIndexSetterInterface<dopedim, dealdim>& index_setter) :
        SpaceTimeHandlerBase<VECTOR> (times), _fe_index_setter(&index_setter)
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
       * @param state_n_blocks            Number of Blocks for the state variable
       * @param state_block_components    Component to Block mapping for the state
       */
      virtual void
      ReInit(unsigned int control_n_blocks,
          const std::vector<unsigned int>& control_block_component,
          unsigned int state_n_blocks,
          const std::vector<unsigned int>& state_block_component) =0;

      /******************************************************/

      /**
       * Returns a reference to the DoF Handler for the Control at the current time point.
       */
      virtual const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER>&
      GetControlDoFHandler() const =0;

      /******************************************************/

      /**
       * Returns a reference to the DoF Handler for the State at the current time point.
       */
      virtual const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>&
      GetStateDoFHandler() const =0;


      /******************************************************/

      /**
       * Returns a reference to a vector of DoFHandlers, the order of the DoFHandlers must
       * be set prior by SetDoFHandlerOrdering
       */
      const std::vector<const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>*>&
      GetDoFHandler() const
      {
#if dope_dimension > 0
        _domain_dofhandler_vector[_control_index] = &GetControlDoFHandler();
        _domain_dofhandler_vector[_state_index] = &GetStateDoFHandler();
#else
        _domain_dofhandler_vector[_state_index] = &GetStateDoFHandler();
#endif
        return _domain_dofhandler_vector;
      }

      /******************************************************/

      /**
       * Returns a vector of the begin_active-celliterators of the
       * DoFHandlers in use.
       */
      std::vector<
            typename DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>::active_cell_iterator>
        GetDoFHandlerBeginActive() const
        {
          std::vector<
              typename DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>::active_cell_iterator>
              ret(this->GetDoFHandler().size());
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
            typename DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>::active_cell_iterator>
        GetDoFHandlerEnd() const
        {
          std::vector<
              typename DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>::active_cell_iterator>
              ret(this->GetDoFHandler().size());
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
        _control_index = control_index;
        _state_index = state_index;
#if dope_dimension > 0
          {
            assert(( _control_index ==0 && _state_index ==1 )||( _control_index ==1 && _state_index ==0 ));
            _domain_dofhandler_vector.clear();
            if(_domain_dofhandler_vector.size() != 2)
              {
                _domain_dofhandler_vector.resize(2,NULL);
              }
          }
#else
          {
            assert(_state_index==0);
            _domain_dofhandler_vector.clear();
            if (_domain_dofhandler_vector.size() != 1)
              {
                _domain_dofhandler_vector.resize(1, NULL);
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
        return _state_index;
      }

      /******************************************************/
      /**
       * Returns a const reference to the ActiveFEIndexSetterInterface object stored in
       * this class. This function is only useful in the hp case.
       */
      const ActiveFEIndexSetterInterface<dopedim, dealdim>&
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
          DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>& dof_handler)
      {
        if (dof_handler.NeedIndexSetter())//with this we distinguish between hp and classic
          {
            for (typename DOFHANDLER::active_cell_iterator cell =
                dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
              {
                this->GetFEIndexSetter().SetActiveFEIndexState(cell);
              }
          }
      }

      /******************************************************/
      /*
       * This function sets for every cell the right fe index for the state variable.
       * This is only useful in the hp case!
       *
       * @param dof_handler   The dof_handler for which the fe indices have to be set.
       */
      void
      SetActiveFEIndicesControl(
          DOpEWrapper::DoFHandler<dopedim, DOFHANDLER>& dof_handler)
      {
        if (dof_handler.NeedIndexSetter())
          {
            for (typename DOFHANDLER::active_cell_iterator cell =
                dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
              {
                this->GetFEIndexSetter().SetActiveFEIndexState(cell);
              }
          }
      }

      /******************************************************/

      /**
       * Returns the control dofs in Block b at the time given through time_point. Default value for the time is now.
       */
      virtual unsigned int
      GetControlDoFsPerBlock(unsigned int b, int time_point = -1) const=0;

      /******************************************************/

      /**
       * Returns the state dofs in Block b at at the time given through time_point. Default value for the time is now.
       */
      virtual unsigned int
      GetStateDoFsPerBlock(unsigned int b, int time_point = -1) const =0;

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
      virtual const std::vector<unsigned int>&
      GetControlDoFsPerBlock() const =0;

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
       * Returns the DoFs per  block for the constraint vector at the current
       * time which has to be set prior to calling this function using SetTime.
       */
      virtual const std::vector<unsigned int>&
      GetConstraintDoFsPerBlock(std::string name) const = 0;

      /******************************************************/

      /**
       * Returns the control HN-Constraints  at the current time
       */
      virtual const dealii::ConstraintMatrix
      &
      GetControlHangingNodeConstraints() const=0;

      /******************************************************/

      /**
       * Returns the state HN-Constraints at the current time
       */
      virtual const dealii::ConstraintMatrix
      &
      GetStateHangingNodeConstraints() const=0;

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
      ComputeControlSparsityPattern(SPARSITYPATTERN & sparsity) const=0;

      /******************************************************/

      /**
       * Computes the current sparsity pattern for the state variable
       */
      virtual void
      ComputeStateSparsityPattern(SPARSITYPATTERN & sparsity) const=0;

      /******************************************************/
      /**
       * Returns a const Smartpointer to the FESystem indicated by the string 'name', i.e. state oder control.
       */

      virtual const FE&
      GetFESystem(std::string name) const=0;


      /******************************************************/

      dealii::DataOut<dealdim, DOFHANDLER>&
        GetDataOut()
        {
        _data_out.clear();
          return _data_out;
        }

      /******************************************************/

    protected:
      //we need this here, because we know the type of the DoFHandler in use.
      //This saves us a template argument for statpdeproblem etc.
      dealii::DataOut<dealdim, DOFHANDLER> _data_out;
      unsigned int _control_index, _state_index;
      const ActiveFEIndexSetterInterface<dopedim, dealdim>* _fe_index_setter;
      mutable std::vector<const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER>*>
          _domain_dofhandler_vector;
      //TODO What if control and state have different dofhandlertypes??

    };
}

#endif
