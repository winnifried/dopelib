#ifndef _MOL_MULTIMESH_SPACE_TIME_HANDLER_H_
#define _MOL_MULTIMESH_SPACE_TIME_HANDLER_H_

#include "mol_spacetimehandler.h"
#include "constraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "sth_internals.h"

#include <dofs/dof_handler.h>
#include <dofs/dof_renumbering.h>
#include <dofs/dof_tools.h>
#include <hp/mapping_collection.h>
#include <lac/constraint_matrix.h>
#include <deal.II/numerics/solution_transfer.h>
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
   */
  template<typename FE, typename DOFHANDLER, typename SPARSITYPATTERN,
      typename VECTOR, int dim>
    class MethodOfLines_MultiMesh_SpaceTimeHandler : public SpaceTimeHandler<FE,
        DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>
    {
      public:
        /**
         * Constructors.
         *
         * @param triangulation     The triangulation in use.
         * @param control_fe        The finite elements used for the discretization of the control variable.
         * @param state_fe          The finite elements used for the discretization of the state variable.
         * @param type              The type of the control, see dopetypes.h for more information.
         * @param times             The timegrid for instationary problems.
         * @param constraints       ?
         * @param index_setter      The index setter object (only needed in case of hp elements).
         */

        MethodOfLines_MultiMesh_SpaceTimeHandler(
            dealii::Triangulation<dim>& triangulation, const FE& control_fe,
            const FE& state_fe, DOpEtypes::ControlType type,
            const ActiveFEIndexSetterInterface<dim, dim>& index_setter =
                ActiveFEIndexSetterInterface<dim, dim>())
            : SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>(
                type, index_setter), _state_triangulation(triangulation), _control_dof_handler(
                _control_triangulation), _state_dof_handler(
                _state_triangulation), _control_fe(&control_fe), _state_fe(
                &state_fe), _mapping(
                DOpEWrapper::StaticMappingQ1<dim, DOFHANDLER>::mapping_q1), _constraints(), _control_mesh_transfer(
                NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(true)
        {
          _control_triangulation.copy_triangulation(_state_triangulation);
          _sparsitymaker = new SparsityMaker<DOFHANDLER, dim>;
          _user_defined_dof_constr = NULL;
        }
        MethodOfLines_MultiMesh_SpaceTimeHandler(
            dealii::Triangulation<dim>& triangulation, const FE& control_fe,
            const FE& state_fe, const dealii::Triangulation<1> & times,
            DOpEtypes::ControlType type,
            const ActiveFEIndexSetterInterface<dim, dim>& index_setter =
                ActiveFEIndexSetterInterface<dim, dim>())
            : SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>(
                times, type, index_setter), _state_triangulation(triangulation), _control_dof_handler(
                _control_triangulation), _state_dof_handler(
                _state_triangulation), _control_fe(&control_fe), _state_fe(
                &state_fe), _mapping(
                DOpEWrapper::StaticMappingQ1<dim, DOFHANDLER>::mapping_q1), _constraints(), _control_mesh_transfer(
                NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(true)
        {
          _control_triangulation.copy_triangulation(_state_triangulation);
          _sparsitymaker = new SparsityMaker<DOFHANDLER, dim>;
          _user_defined_dof_constr = NULL;
        }

        MethodOfLines_MultiMesh_SpaceTimeHandler(
            dealii::Triangulation<dim>& triangulation, const FE& control_fe,
            const FE& state_fe, const Constraints& c,
            DOpEtypes::ControlType type,
            const ActiveFEIndexSetterInterface<dim, dim>& index_setter =
                ActiveFEIndexSetterInterface<dim, dim>())
            : SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>(
                type, index_setter), _state_triangulation(triangulation), _control_dof_handler(
                _control_triangulation), _state_dof_handler(
                _state_triangulation), _control_fe(&control_fe), _state_fe(
                &state_fe), _mapping(
                DOpEWrapper::StaticMappingQ1<dim, DOFHANDLER>::mapping_q1), _constraints(
                c), _control_mesh_transfer(NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(
                true)
        {
          _control_triangulation.copy_triangulation(_state_triangulation);
          _sparsitymaker = new SparsityMaker<DOFHANDLER, dim>;
          _user_defined_dof_constr = NULL;
        }

        MethodOfLines_MultiMesh_SpaceTimeHandler(
            dealii::Triangulation<dim>& triangulation, const FE& control_fe,
            const FE& state_fe, const dealii::Triangulation<1> & times,
            const Constraints& c, DOpEtypes::ControlType type,
            const ActiveFEIndexSetterInterface<dim, dim>& index_setter =
                ActiveFEIndexSetterInterface<dim, dim>())
            : SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>(
                times, type, index_setter), _state_triangulation(triangulation), _control_dof_handler(
                _control_triangulation), _state_dof_handler(
                _state_triangulation), _control_fe(&control_fe), _state_fe(
                &state_fe), _mapping(
                DOpEWrapper::StaticMappingQ1<dim, DOFHANDLER>::mapping_q1), _constraints(
                c), _control_mesh_transfer(NULL), _state_mesh_transfer(NULL), _sparse_mkr_dynamic(
                true)
        {
          _control_triangulation.copy_triangulation(_state_triangulation);
          _sparsitymaker = new SparsityMaker<DOFHANDLER, dim>;
          _user_defined_dof_constr = NULL;
        }

        ~MethodOfLines_MultiMesh_SpaceTimeHandler()
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
          SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>::SetActiveFEIndicesControl(
              _control_dof_handler);
          _control_dof_handler.distribute_dofs(*_control_fe);
          DoFRenumbering::component_wise(
              static_cast<DOFHANDLER&>(_control_dof_handler),
              control_block_component);

          _control_dof_constraints.clear();
          DoFTools::make_hanging_node_constraints(
              static_cast<DOFHANDLER&>(_control_dof_handler),
              _control_dof_constraints);
          if (GetUserDefinedDoFConstraints() != NULL)
            GetUserDefinedDoFConstraints()->MakeControlDoFConstraints(
                _control_dof_handler, _control_dof_constraints);
          _control_dof_constraints.close();

          _control_dofs_per_block.resize(control_n_blocks);
          {
            DoFTools::count_dofs_per_block(
                static_cast<DOFHANDLER&>(_control_dof_handler),
                _control_dofs_per_block, control_block_component);
          }

          SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dim, dim>::SetActiveFEIndicesState(
              _state_dof_handler);
          _state_dof_handler.distribute_dofs(GetFESystem("state"));
          DoFRenumbering::component_wise(
              static_cast<DOFHANDLER&>(_state_dof_handler),
              state_block_component);

          _state_dof_constraints.clear();
          DoFTools::make_hanging_node_constraints(
              static_cast<DOFHANDLER&>(_state_dof_handler),
              _state_dof_constraints);
          //TODO Dirichlet Daten hierueber.
          if (GetUserDefinedDoFConstraints() != NULL)
            GetUserDefinedDoFConstraints()->MakeStateDoFConstraints(
                _state_dof_handler, _state_dof_constraints);
          _state_dof_constraints.close();

          _state_dofs_per_block.resize(state_n_blocks);
          DoFTools::count_dofs_per_block(
              static_cast<DOFHANDLER&>(_state_dof_handler),
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
        const DOpEWrapper::DoFHandler<dim, DOFHANDLER>&
        GetControlDoFHandler() const
        {
          //There is only one mesh, hence always return this
          return _control_dof_handler;
        }
        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const DOpEWrapper::DoFHandler<dim, DOFHANDLER>&
        GetStateDoFHandler() const
        {
          //There is only one mesh, hence always return this
          return _state_dof_handler;
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const DOpEWrapper::Mapping<dim, DOFHANDLER>&
        GetMapping() const
        {
          return _mapping;
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
        GetControlDoFsPerBlock() const
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
         * Implementation of virtual function in SpaceTimeHandlerBase
         */
        unsigned int
        GetControlNDoFs() const
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
          return _constraints.n_dofs("local");
          //return _constraints.local_n_dofs();
        }

        /**
         * Implementation of virtual function in SpaceTimeHandler
         */
        const std::vector<Point<dim> >&
        GetMapDoFToSupportPoints()
        {
          _support_points.resize(GetStateNDoFs());
          DOpE::STHInternals::MapDoFsToSupportPoints(this->GetMapping(),
              GetStateDoFHandler(), _support_points);
          return _support_points;
        }

        /******************************************************/
        void
        ComputeControlSparsityPattern(SPARSITYPATTERN & sparsity) const;

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
         * Implementation of virtual function in SpaceTimeHandler
         */
        const FE&
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
                "MethodOfLines_MultiMesh_SpaceTimeHandler::GetFESystem");
          }

        }

        /**
         * This Function is used to refine the spatial mesh for both the state and the control.
         * After calling a refinement function a reinitialization is required!
         *
         * @param ref_type          A string telling how to refine, feasible values are at present
         *                          'global', 'fixedfraction', 'fixednumber', 'optimized'
         * @param indicators        A set of positive values, used to guide refinement.
         * @param topfraction       In a fixed fraction strategy, wich part should be refined
         * @param bottomfraction    In a fixed fraction strategy, wich part should be coarsened
         */
        void
        RefineSpace(std::string ref_type,
            const Vector<float>* indicators = NULL, double topfraction = 0.1,
            double bottomfraction = 0.0)
        {
          RefineStateSpace(ref_type, indicators, topfraction, bottomfraction);
          RefineControlSpace(ref_type, indicators, topfraction, bottomfraction);
        }

        /**
         * This Function is used to refine the spatial mesh for the state.
         * After calling a refinement function a reinitialization is required!
         *
         * @param ref_type          A string telling how to refine, feasible values are at present
         *                          'global', 'fixedfraction', 'fixednumber', 'optimized'
         * @param indicators        A set of positive values, used to guide refinement.
         * @param topfraction       In a fixed fraction strategy, wich part should be refined
         * @param bottomfraction    In a fixed fraction strategy, wich part should be coarsened
         */
        void
        RefineStateSpace(std::string ref_type, const Vector<float>* indicators =
            NULL, double topfraction = 0.1, double bottomfraction = 0.0)
        {
          assert(bottomfraction == 0.0);

          if (_state_mesh_transfer != NULL)
          {
            delete _state_mesh_transfer;
            _state_mesh_transfer = NULL;
          }
          _state_mesh_transfer = new dealii::SolutionTransfer<dim, VECTOR>(
              _state_dof_handler);

          if ("global" == ref_type)
          {
            _state_triangulation.set_all_refine_flags();
          }
          else if ("fixednumber" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_fixed_number(
                _state_triangulation, *indicators, topfraction, bottomfraction);
          }
          else if ("fixedfraction" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_fixed_fraction(
                _state_triangulation, *indicators, topfraction, bottomfraction);
          }
          else if ("optimized" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_optimize(_state_triangulation,
                *indicators);
            //TODO: how can we prevent coarsening here ?
          }
          else if ("finest-of-both")
          {
            this->FlagIfLeftIsNotFinest(_state_triangulation,
                _control_triangulation);
          }
          else
          {
            throw DOpEException("Not implemented for name =" + ref_type,
                "MethodOfLines_MultiMesh_SpaceTimeHandler::RefineSpace");
          }
          _state_triangulation.prepare_coarsening_and_refinement();
          if (_state_mesh_transfer != NULL)
            _state_mesh_transfer->prepare_for_pure_refinement();

          _state_triangulation.execute_coarsening_and_refinement();
        }

        /**
         * This Function is used to refine the spatial mesh for control.
         * After calling a refinement function a reinitialization is required!
         *
         * @param ref_type          A string telling how to refine, feasible values are at present
         *                          'global', 'fixedfraction', 'fixednumber', 'optimized'
         * @param indicators        A set of positive values, used to guide refinement.
         * @param topfraction       In a fixed fraction strategy, wich part should be refined
         * @param bottomfraction    In a fixed fraction strategy, wich part should be coarsened
         */
        void
        RefineControlSpace(std::string ref_type,
            const Vector<float>* indicators = NULL, double topfraction = 0.1,
            double bottomfraction = 0.0)
        {
          assert(bottomfraction == 0.0);

          if (_control_mesh_transfer != NULL)
          {
            delete _control_mesh_transfer;
            _control_mesh_transfer = NULL;
          }
#if dope_dimension == deal_II_dimension
          _control_mesh_transfer = new dealii::SolutionTransfer<dim, VECTOR>(
              _control_dof_handler);
#endif
          if ("global" == ref_type)
          {
            _control_triangulation.set_all_refine_flags();
          }
          else if ("fixednumber" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_fixed_number(
                _control_triangulation, *indicators, topfraction,
                bottomfraction);
          }
          else if ("fixedfraction" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_fixed_fraction(
                _control_triangulation, *indicators, topfraction,
                bottomfraction);
          }
          else if ("optimized" == ref_type)
          {
            assert(indicators != NULL);
            GridRefinement::refine_and_coarsen_optimize(_control_triangulation,
                *indicators);
            //TODO: how can we prevent coarsening here ?
          }
          else if ("finest-of-both")
          {
            this->FlagIfLeftIsNotFinest(_control_triangulation,
                _state_triangulation);
          }
          else
          {
            throw DOpEException("Not implemented for name =" + ref_type,
                "MethodOfLines_MultiMesh_SpaceTimeHandler::RefineSpace");
          }
          _control_triangulation.prepare_coarsening_and_refinement();

          //FIXME: works only if no coarsening happens, because we do not have the vectors to be interpolated availiable...
          if (_control_mesh_transfer != NULL)
            _control_mesh_transfer->prepare_for_pure_refinement();

          _control_triangulation.execute_coarsening_and_refinement();
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
        /******************************************************/

        /**
         * Transfer of the State Vectors
         */
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
            UserDefinedDoFConstraints<DOFHANDLER, dim>& constraints_maker)
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
        SetSparsityMaker(SparsityMaker<DOFHANDLER, dim>& sparsity_maker)
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
        ResetStateTriangulation(const dealii::Triangulation<dim>& tria)
        {
          _state_dof_handler.clear();
          _state_triangulation.clear();
          _state_triangulation.copy_triangulation(tria);
          _state_dof_handler.initialize(_state_triangulation, *_state_fe);
          this->IncrementStateTicket();
          if (_state_mesh_transfer != NULL)
            delete _state_mesh_transfer;
          _state_mesh_transfer = NULL;
        }

        /******************************************************/
        /**
         * Through this function one can reinitialize the 
         * triangulation for the state variable to be a copy of the
         * given argument.
         */

        void
        ResetControlTriangulation(const dealii::Triangulation<dim>& tria)
        {
          _control_dof_handler.clear();
          _control_triangulation.clear();
          _control_triangulation.copy_triangulation(tria);
          _control_dof_handler.initialize(_control_triangulation, *_control_fe);
          this->IncrementControlTicket();
          if (_control_mesh_transfer != NULL)
            delete _control_mesh_transfer;
          _control_mesh_transfer = NULL;
        }

      private:
        const SparsityMaker<DOFHANDLER, dim>*
        GetSparsityMaker() const
        {
          return _sparsitymaker;
        }
        const UserDefinedDoFConstraints<DOFHANDLER, dim>*
        GetUserDefinedDoFConstraints() const
        {
          return _user_defined_dof_constr;
        }

        void
        FlagIfLeftIsNotFinest(dealii::Triangulation<dim>& left,
            const dealii::Triangulation<dim>& right);

        SparsityMaker<DOFHANDLER, dim>* _sparsitymaker;
        UserDefinedDoFConstraints<DOFHANDLER, dim>* _user_defined_dof_constr;

        dealii::Triangulation<dim> _control_triangulation;
        dealii::Triangulation<dim>& _state_triangulation;
        DOpEWrapper::DoFHandler<dim, DOFHANDLER> _control_dof_handler;
        DOpEWrapper::DoFHandler<dim, DOFHANDLER> _state_dof_handler;

        std::vector<unsigned int> _control_dofs_per_block;
        std::vector<unsigned int> _state_dofs_per_block;

        dealii::ConstraintMatrix _control_dof_constraints;
        dealii::ConstraintMatrix _state_dof_constraints;

        const dealii::SmartPointer<const FE> _control_fe;
        const dealii::SmartPointer<const FE> _state_fe;

        DOpEWrapper::Mapping<dim, DOFHANDLER> _mapping;

        std::vector<Point<dim> > _support_points;

        Constraints _constraints;
        dealii::SolutionTransfer<dim, VECTOR>* _control_mesh_transfer;
        dealii::SolutionTransfer<dim, VECTOR>* _state_mesh_transfer;
        bool _sparse_mkr_dynamic;
    };

  /**************************explicit instantiation*************/
#if dope_dimension == deal_II_dimension
  /**
   * Implementation of virtual function in SpaceTimeHandler
   */
  template<>
    void
    DOpE::MethodOfLines_MultiMesh_SpaceTimeHandler<
        dealii::FESystem<deal_II_dimension>,
        dealii::DoFHandler<deal_II_dimension>, dealii::BlockSparsityPattern,
        dealii::BlockVector<double>, dope_dimension>::ComputeControlSparsityPattern(
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

      dealii::DoFTools::make_sparsity_pattern(
          static_cast<const dealii::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
          csp);
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

  /******************************************************/

  template<>
    void
    MethodOfLines_MultiMesh_SpaceTimeHandler<
        dealii::FESystem<deal_II_dimension>,
        dealii::DoFHandler<deal_II_dimension>, dealii::SparsityPattern,
        dealii::Vector<double>, dope_dimension>::ComputeControlSparsityPattern(
        dealii::SparsityPattern & sparsity) const
    {
      const unsigned int total_dofs = this->GetControlNDoFs();
      dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);

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
    DOpE::MethodOfLines_MultiMesh_SpaceTimeHandler<
        dealii::hp::FECollection<deal_II_dimension>,
        dealii::hp::DoFHandler<deal_II_dimension>, dealii::BlockSparsityPattern,
        dealii::BlockVector<double>, dope_dimension>::ComputeControlSparsityPattern(
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

      dealii::DoFTools::make_sparsity_pattern(
          static_cast<const dealii::hp::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
          csp);
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }

  /******************************************************/

  template<>
    void
    MethodOfLines_MultiMesh_SpaceTimeHandler<
        dealii::hp::FECollection<deal_II_dimension>,
        dealii::hp::DoFHandler<deal_II_dimension>, dealii::SparsityPattern,
        dealii::Vector<double>, dope_dimension>::ComputeControlSparsityPattern(
        dealii::SparsityPattern & sparsity) const
    {
      const unsigned int total_dofs = this->GetControlNDoFs();
      dealii::CompressedSimpleSparsityPattern csp(total_dofs, total_dofs);

      dealii::DoFTools::make_sparsity_pattern(
          static_cast<const dealii::hp::DoFHandler<deal_II_dimension>&>(this->GetControlDoFHandler()),
          csp);
      this->GetControlDoFConstraints().condense(csp);
      sparsity.copy_from(csp);
    }
#endif //Endof explicit instanciation
  /*******************************************************/
  template<typename FE, typename DOFHANDLER, typename SPARSITYPATTERN,
      typename VECTOR, int dim>
    void
    MethodOfLines_MultiMesh_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, dim>::FlagIfLeftIsNotFinest(dealii::Triangulation<dim>& left,
        const dealii::Triangulation<dim>& right)
    {
      auto cell_list = GridTools::get_finest_common_cells(left, right);
      auto cell_iter = cell_list.begin();
      for (; cell_iter != cell_list.end(); cell_iter++)
      {
        if (cell_iter->second->has_children())
        {
          //left is not finest, so we should flag the left cell to be refined
          cell_iter->first->set_refine_flag();
        }
      }
    }
}

#endif
