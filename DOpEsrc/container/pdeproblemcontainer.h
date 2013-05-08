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

#ifndef _PDEProblemContainer_H_
#define _PDEProblemContainer_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "functionalinterface.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "function_wrapper.h"
#include "statespacetimehandler.h"
#include "primaldirichletdata.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "stateproblem.h"
#include "problemcontainer_internal.h"

#include "dopetypes.h"
#include "dwrdatacontainer.h"

#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <grid/tria_iterator.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <fe/fe_system.h>
#include <fe/fe_values.h>
#include <fe/mapping.h>
#include <base/quadrature_lib.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/compressed_simple_sparsity_pattern.h>

#include <assert.h>
#include <string>
#include <vector>

namespace DOpE
{
  //Predeclaration necessary
  template<typename VECTOR>
    class DOpEOutputHandler;
  template<typename VECTOR>
    class DOpEExceptionHandler;
  /////////////////////////////

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE = dealii::FESystem,
      template<int, int> class DH = dealii::DoFHandler>
    class PDEProblemContainer : public ProblemContainerInternal<PDE>
    {
      public:
        PDEProblemContainer(PDE& pde,
            StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>& STH);

        /******************************************************/

        virtual
        ~PDEProblemContainer();

        /******************************************************/

        virtual std::string
        GetName() const
        {
          return "PDEProblemContainer";
        }

        /******************************************************/
        StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
        GetStateProblem()
        {
          if (_state_problem == NULL)
          {
            _state_problem = new StateProblem<
                PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim,
                    FE, DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(*this,
                this->GetPDE());
          }
          return *_state_problem;
        }

        //TODO This is Pfush needed to split into different subproblems and allow optproblem to
        //be substituted as any of these problems. Can be removed once the splitting is complete.
        PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>&
        GetBaseProblem()
        {
          return *this;
        }
        /******************************************************/

        /**
         * This function calls the ReInit function of the StateSpaceTimeHandler
         *
         * @param algo_type      Specifies the type of the algorithm
         *                       Actually, only the `reduced' algorithm is
         *                       implemented.
         */
        void
        ReInit(std::string algo_type);

        /******************************************************/

        void
        RegisterOutputHandler(DOpEOutputHandler<VECTOR>* OH)
        {
          _OutputHandler = OH;
        }

        /******************************************************/

        void
        RegisterExceptionHandler(DOpEExceptionHandler<VECTOR>* OH)
        {
          _ExceptionHandler = OH;
        }

        /******************************************************/

        /**
         * Sets the type of the problem.
         *
         * @param type      Specifies the type of the problem, like
         *                  'state', 'adjoint' etc.
         */

        void
        SetType(std::string type, unsigned int num = 0);

        /******************************************************/

        /**
         * This function returns a functional value on a cell.
         * Different types of functionals
         * have been implemented so far: `cost_functional', `aux_functional'
         * and 'functional_for_ee'.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined on cells, e.g., drag and lift computation. Or
         * computations of deflections and deformations. The last
         * one is needed if one wants to use the goal oriented adaptive
         * grid refinement based on the DWR method.
         *
         *
         * @template DATACONTAINER    Class of the datacontainer, distinguishes
         *                            between hp- and classical case.
         *
         * @param fdc                 A DataContainer holding all the needed information
         *                            of the face.
         */
        template<typename DATACONTAINER>
          double
          CellFunctional(const DATACONTAINER& cdc);

        /******************************************************/

        /**
         * This function returns a functional value of a point.
         * Different types of functionals
         * have been implemented so far:  `cost_functional', `aux_functional'
         * and 'functional_for_ee.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined in points. For example, deflection or pressure values in
         * points.  The last
         * one is needed if one watns to use the goal oriented adaptive
         * grid refinement based on the DWR method.
         */
        double
        PointFunctional(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values);

        /******************************************************/

        /**
         * This function returns a functional value of a part of the outer boundary
         * or the whole boundary.
         * Different types of functionals
         * have been implemented so far:  `cost_functional', `aux_functional'
         * and 'functional_for_ee.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined on the boundary, e.g., stresses.  The last
         * one is needed if one watns to use the goal oriented adaptive
         * grid refinement based on the DWR method.
         *
         * @template FACEDATACONTAINER    Class of the datacontainer in use, distinguishes
         *                                between hp- and classical case.
         *
         * @param fdc                     A DataContainer holding all the needed information
         *                                of the face.
         *
         */
        template<typename FACEDATACONTAINER>
          double
          BoundaryFunctional(const FACEDATACONTAINER& fdc);

        /******************************************************/

        /**
         * This function returns a functional value of quantities that
         * are defined on faces. This function is very similar to the
         * BoundaryFunctional and has the same functionality.
         */

        template<typename FACEDATACONTAINER>
          double
          FaceFunctional(const FACEDATACONTAINER& fdc);

        /******************************************************/

        double
        AlgebraicFunctional(
            const std::map<std::string, const dealii::Vector<double>*> &values,
            const std::map<std::string, const VECTOR*> &block_values);

        /******************************************************/

        /**
         * Computes the value of the cell equation which corresponds
         * to the residuum in nonlinear cases. This function is the basis
         * for all stationary examples and unsteady configurations as well.
         * However, in unsteady computations one has to differentiate
         * between explicit (diffusion, convection) and implicit terms
         * (pressure, incompressibility). For that reason a second
         * function CellEquationImplicit also exists.
         *
         * If no differentiation between explicit and implicit terms is needed
         * this function should be used.
         *
         * @template DATACONTAINER        Class of the datacontainer in use, distinguishes
         *                                between hp- and classical case.
         *
         * @param cdc                     A DataContainer holding all the needed information
         *                                of the cell
         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<typename DATACONTAINER>
          void
          CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        /******************************************************/

        /******************************************************/

        /**
         * This function has the same functionality as the CellEquation function.
         * It is needed for time derivatives when working with
         * time dependent problems.
         */
        template<typename DATACONTAINER>
          void
          CellTimeEquation(const DATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * This function has the same functionality as the CellTimeEquation function.
         * It is only needed for fluid-structure interaction problems and has
         * special structure. Please talk to Thomas Wick when you would like to
         * use this function.
         */
        template<typename DATACONTAINER>
          void
          CellTimeEquationExplicit(const DATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the right-hand side of the problem at hand.
         *
         * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
         *                                 between hp- and classical case.
         *
         * @param cdc                      A DataContainer holding all the needed information
         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        template<typename DATACONTAINER>
          void
          CellRhs(const DATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the right-hand side of the problem at hand, if it
         * contains pointevaluations.
         *
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param rhs_vector               This vector contains the complete point-rhs.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the cell matrix which is derived
         * by computing the directional derivatives of the residuum equation of the PDE
         * problem under consideration.
         *
         * The differentiation between explicit and implicit terms is
         * equivalent to the CellEquation. We refer to its documentation.
         *
         * Moreover, you find an extensive explication in the
         * time step algorithms, e.g., backward_euler_problem.h.
         *
         * @template DATACONTAINER      Class of the datacontainer in use, distinguishes
         *                              between hp- and classical case.
         *
         * @param cdc                   A DataContainer holding all the needed information
         *

         * @param local_entry_matrix    The local matrix is quadratic and has size local DoFs times local DoFs and is
         *                              filled by the locally computed values. For more information of its functionality, please
         *                              search for the keyword `FullMatrix' in the deal.ii manual.
         * @param scale                 A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<typename DATACONTAINER>
          void
          CellMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
              double scale_ico = 1.);

        /******************************************************/

        /**
         * Computes the value of the cell matrix which is derived
         * by computing the directional derivatives of the time residuum of the PDE
         * problem under consideration.
         *
         * The differentiation between explicit and implicit terms is
         * equivalent to the CellTimeEquation. We refer to its documentation.
         *
         * Moreover, you find an extensive explication in the
         * time step algorithms, e.g., backward_euler_problem.h.
         */
        template<typename DATACONTAINER>
          void
          CellTimeMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        /******************************************************/

        /**
         * Computes the value of the cell matrix which is derived
         * by computing the directional derivatives of the time residuum of the PDE
         * problem under consideration.
         *
         * This function is only needed for fluid-structure interaction problems.
         * Please ask Thomas Wick WHY and HOW to use this function.
         *
         */
        template<typename DATACONTAINER>
          void
          CellTimeMatrixExplicit(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        /******************************************************/

        /**
         * Computes the value of face on a cell.
         * It has the same functionality as CellEquation. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          FaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the product of two different finite elements
         * on a interior face. It has the same functionality as CellEquation.
         * We refer to its documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          InterfaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the value of face on a cell.
         * It has the same functionality as CellRhs. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          FaceRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of face on a cell.
         * It has the same functionality as CellMatrix. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          FaceMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
              double scale_ico = 1.);

        /******************************************************/
        /**
         * Computes the product of two different finite elements
         * on an interior face. It has the same functionality as
         * CellMatrix. We refer to its documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          InterfaceMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
              double scale_ico = 1.);

        /******************************************************/

        /**
         * Computes the value of face on a boundary.
         * It has the same functionality as CellEquation. We refer to its
         * documentation.
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the value of the boundary on a cell.
         * It has the same functionality as CellRhs. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the boundary on a cell.
         * It has the same functionality as CellMatrix. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_cell_matrix, double scale = 1.,
              double scale_ico = 1.);

        /******************************************************/

        const FE<dealdim, dealdim>&
        GetFESystem() const;

        /******************************************************/
        /**
         * This function determines whether a loop over all faces is required or not.
         *
         * @return Returns whether or not this functional has components on faces between elements.
         *         The default value is false.
         */
        bool
        HasFaces() const;
        /******************************************************/
        /**
         * Do we need the evaluation of PointRhs?
         */
        bool
        HasPoints() const;

        /******************************************************/
        /**
         * This function determines whether a loop over all faces is required or not.
         *
         * @return Returns whether or not this functional needs to compute the product
         * of two different finite element functions across an internal face.
         */
        bool
        HasInterfaces() const;

        /******************************************************/

        dealii::UpdateFlags
        GetUpdateFlags() const;

        /******************************************************/

        dealii::UpdateFlags
        GetFaceUpdateFlags() const;

        /******************************************************/

        void
        SetDirichletBoundaryColors(unsigned int color,
            const std::vector<bool>& comp_mask, const DD* values);

        /******************************************************/

        const std::vector<unsigned int>&
        GetDirichletColors() const;
        const std::vector<bool>&
        GetDirichletCompMask(unsigned int color) const;

        /******************************************************/

        const dealii::Function<dealdim> &
        GetDirichletValues(unsigned int color,
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values) const;

        /******************************************************/

        void
        SetInitialValues(const dealii::Function<dealdim>* values)
        {
          _initial_values = values;
        }
        const dealii::Function<dealdim>&
        GetInitialValues() const
        {
          return *_initial_values;
        }

        /******************************************************/

        void
        SetBoundaryEquationColors(unsigned int color);
        const std::vector<unsigned int>&
        GetBoundaryEquationColors() const;
        void
        SetBoundaryFunctionalColors(unsigned int color);
        const std::vector<unsigned int>&
        GetBoundaryFunctionalColors() const;

        /******************************************************/

        /**
         * Adds a functional which will be evaluated after/during (in
         * time dependent equations) the computation of the state solution.
         *
         * Note that you have to specify a the GetName() method for
         * your functional, and that you can not add two functionals
         * with the same name!
         */
        void
        AddFunctional(
            FunctionalInterface<CellDataContainer, FaceDataContainer, DH,
                VECTOR, dealdim>* F)
        {
          _aux_functionals.push_back(F);
          if (_functional_position.find(F->GetName())
              != _functional_position.end())
          {
            throw DOpEException(
                "You cant add two functionals with the same name.",
                "PDEProblemContainer::AddFunctional");
          }
          _functional_position[F->GetName()] = _aux_functionals.size() - 1;
          //-1 because we are in the pde case and have therefore no cost functional
        }

        /******************************************************/

        /**
         * Through this function one sets the functional for the
         * error estimation. The name given by functional_name is
         * looked up in _aux_functionals, so the function assumes
         * that the functional intended for the error estimation
         * is set prior by AddFunctional!
         *
         * @param functional_name     The name of the functional
         *                            for the error estimation.
         */
        void
        SetFunctionalForErrorEstimation(std::string functional_name)
        {
          bool found = false;
          //we go through all aux functionals.
          for (unsigned int i = 0; i < this->GetNFunctionals(); i++)
          {
            if (_aux_functionals[i]->GetName() == functional_name)
            {
              //if the names match, we have found our functional.
              found = true;
              _functional_for_ee_num = i;
              break;
            }
          }
          //If we have not found a functional with the given name,
          //we throw an error.
          if (!found)
          {
            throw DOpEException(
                "Can't find functional " + functional_name
                    + " in _aux_functionals",
                "PDEProblemContainer::SetFunctionalForErrorEstimation");
          }
        }

        /******************************************************/

        unsigned int
        GetNFunctionals() const
        {
          return _aux_functionals.size();
        }

        /******************************************************/

        unsigned int
        GetStateNBlocks() const;

        /******************************************************/

        unsigned int
        GetNBlocks() const;

        /******************************************************/

        unsigned int
        GetDoFsPerBlock(unsigned int b) const;

        /******************************************************/

        const std::vector<unsigned int>&
        GetDoFsPerBlock() const;

        /******************************************************/

        const dealii::ConstraintMatrix&
        GetDoFConstraints() const;

        /******************************************************/

        std::string
        GetDoFType() const;
        std::string
        GetFunctionalType() const;
        std::string
        GetFunctionalName() const;

        /******************************************************/

        bool
        NeedTimeFunctional() const;

        /******************************************************/

        DOpEExceptionHandler<VECTOR>*
        GetExceptionHandler()
        {
          assert(_ExceptionHandler);
          return _ExceptionHandler;
        }

        /******************************************************/

        DOpEOutputHandler<VECTOR>*
        GetOutputHandler()
        {
          assert(_OutputHandler);
          return _OutputHandler;
        }

        /******************************************************/

        void
        SetTime(double time, const TimeIterator& interval);

        /******************************************************/

        const StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>*
        GetSpaceTimeHandler() const
        {
          return _STH;
        }

        /******************************************************/

        StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>*
        GetSpaceTimeHandler()
        {
          return _STH;
        }

        /******************************************************/

        void
        ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

        /******************************************************/

        template<typename INTEGRATOR>
          void
          AddAuxiliaryToIntegrator(INTEGRATOR& /*integrator*/)
          {

          }

        /******************************************************/

        template<typename INTEGRATOR>
          void
          DeleteAuxiliaryFromIntegrator(INTEGRATOR& /*integrator*/)
          {

          }

        /******************************************************/

        unsigned int
        GetStateNBlocks()
        {
          return this->GetPDE().GetStateNBlocks();
        }

        /******************************************************/

        std::vector<unsigned int>&
        GetStateBlockComponent()
        {
          return this->GetPDE().GetStateBlockComponent();
        }

        /******************************************************/

        /**
         * FunctionalPosition maps the name of the cost/auxiliary functionals
         * to their position in ReducedProblemInterface_Base::_functional_values.
         *
         * TODO This should not be public!
         */
        const std::map<std::string, unsigned int>&
        GetFunctionalPosition() const
        {
          return _functional_position;
        }

        /******************************************************/
      private:
        DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
        DOpEOutputHandler<VECTOR>* _OutputHandler;

        std::string _algo_type;

        std::vector<
            FunctionalInterface<CellDataContainer, FaceDataContainer, DH,
                VECTOR, dealdim>*> _aux_functionals;
        std::map<std::string, unsigned int> _functional_position;

        unsigned int _functional_for_ee_num;
        StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim> * _STH;

        std::vector<unsigned int> _dirichlet_colors;
        std::vector<std::vector<bool> > _dirichlet_comps;
        std::vector<PrimalDirichletData<DD, VECTOR, dealdim>*> _primal_dirichlet_values;
        const dealii::Function<dealdim>* _zero_dirichlet_values;

        const dealii::Function<dealdim>* _initial_values;

        std::vector<unsigned int> _state_boundary_equation_colors;
        std::vector<unsigned int> _adjoint_boundary_equation_colors;

        std::vector<unsigned int> _boundary_functional_colors;

        StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>* _state_problem;

        friend class StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    };
  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::PDEProblemContainer(
        PDE& pde,
        StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>& STH) :
        ProblemContainerInternal<PDE>(pde), _STH(&STH), _state_problem(NULL)
    {
      _ExceptionHandler = NULL;
      _OutputHandler = NULL;
      _zero_dirichlet_values = new ZeroFunction<dealdim>(
          this->GetPDE().GetStateNComponents());
      _algo_type = "";
      _functional_for_ee_num = dealii::numbers::invalid_unsigned_int;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::~PDEProblemContainer()
    {
      if (_zero_dirichlet_values != NULL)
      {
        delete _zero_dirichlet_values;
      }
      for (unsigned int i = 0; i < _primal_dirichlet_values.size(); i++)
      {
        if (_primal_dirichlet_values[i] != NULL)
          delete _primal_dirichlet_values[i];
      }
      if (_state_problem != NULL)
      {
        delete _state_problem;
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ReInit(
        std::string algo_type)
    {
      if (_state_problem != NULL)
      {
        delete _state_problem;
        _state_problem = NULL;
      }

      if (_algo_type != algo_type && _algo_type != "")
      {
        throw DOpEException("Conflicting Algorithms!",
            "PDEProblemContainer::ReInit");
      }
      else
      {
        _algo_type = algo_type;
        this->SetTypeInternal("");

        if (_algo_type == "reduced")
        {
          GetSpaceTimeHandler()->ReInit(this->GetPDE().GetStateNBlocks(),
              this->GetPDE().GetStateBlockComponent());
        }
        else
        {
          throw DOpEException("Unknown Algorithm " + _algo_type,
              "PDEProblemContainer::ReInit");
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetType(
        std::string type, unsigned int num)
    {
      if (this->GetType() != type || this->GetTypeNum() != num)
      {
        this->SetTypeInternal(type);
        this->SetTypeNumInternal(num);
        this->GetPDE().SetProblemType(type);
        if (_functional_for_ee_num != dealii::numbers::invalid_unsigned_int)
          _aux_functionals[_functional_for_ee_num]->SetProblemType(type);
      }
      //Nothing to do.
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellFunctional(
          const DATACONTAINER& cdc)
      {

        if (this->GetType() == "cost_functional")
        {
          return 0;
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->Value(cdc);
        }
        else if (this->GetType() == "error_evaluation")
        {
          return _aux_functionals[_functional_for_ee_num]->Value(cdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellFunctional");
        }
      }

  /******************************************************/
  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    double
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::PointFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (this->GetType() == "cost_functional")
      {
        return 0.;
      } //endif cost_functional
      else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[this->GetTypeNum()]->PointValue(
            this->GetSpaceTimeHandler()->GetStateDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif aux_functional
      else if (this->GetType() == "error_evaluation")
      {
        return _aux_functionals[_functional_for_ee_num]->PointValue(
            this->GetSpaceTimeHandler()->GetStateDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "PDEProblemContainer::PointFunctional");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::BoundaryFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (this->GetType() == "cost_functional")
        {
          // state values in quadrature points
          return 0.;
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->BoundaryValue(fdc);
        }
        else if (this->GetType() == "error_evaluation")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->BoundaryValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::BoundaryFunctional");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (this->GetType() == "cost_functional")
        {
          // state values in quadrature points
          return 0.;
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->FaceValue(fdc);
        }
        else if (this->GetType() == "error_evaluation")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->FaceValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::FaceFunctional");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    double
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::AlgebraicFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (this->GetType() == "cost_functional")
      {
        // state values in quadrature points
        return 0.;
      }
      else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[this->GetTypeNum()]->AlgebraicValue(
            param_values, domain_values);
      }
      else if (this->GetType() == "error_evaluation")
      //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
      {
        return _aux_functionals[_functional_for_ee_num]->AlgebraicValue(
            param_values, domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "PDEProblemContainer::AlgebraicFunctional");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellEquation(cdc, local_cell_vector, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().CellEquation_U(cdc, local_cell_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellTimeEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellTimeEquation(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented",
              "OptProblem::CellTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellTimeEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellTimeEquationExplicit(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellTimeEquationExplicit(cdc, local_cell_vector,
              scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented",
              "OptProblem::CellTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellTimeEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation(fdc, local_cell_vector, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::FaceEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::InterfaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().InterfaceEquation(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().InterfaceEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::InterfaceEquation");
        }
      }
  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::BoundaryEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellBoundaryEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellRhs(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellRightHandSide(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //TODO currently, pointvalue is not working for dual error estimation!
          //values of the derivative of the functional for error estimation.
          //Check, if we have to evaluate an integral over a domain.
          if (_aux_functionals[_functional_for_ee_num]->GetType().find("domain")
              != std::string::npos)
            _aux_functionals[_functional_for_ee_num]->Value_U(cdc,
                local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellRhs");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::PointRhs(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values,
        VECTOR& rhs_vector, double scale)
    {
      if (this->GetType() == "adjoint_for_ee")
      {
        //values of the derivative of the functional for error estimation
        if (_aux_functionals[_functional_for_ee_num]->GetType().find("point")
            != std::string::npos)
          _aux_functionals[_functional_for_ee_num]->PointValue_U(
              this->GetSpaceTimeHandler()->GetStateDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
      }
      else
      {
        throw DOpEException("Not implemented", "OptProblem::CellRhs");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().FaceRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (_aux_functionals[_functional_for_ee_num]->GetType().find("face")
              != std::string::npos)
            _aux_functionals[_functional_for_ee_num]->FaceValue_U(fdc,
                local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellFaceRhs");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::BoundaryRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (_aux_functionals[_functional_for_ee_num]->GetType().find(
              "boundary") != std::string::npos)
            _aux_functionals[_functional_for_ee_num]->BoundaryValue_U(fdc,
                local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellBoundaryRhs");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellMatrix(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().CellMatrix_T(cdc, local_entry_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellTimeMatrix(
          const DATACONTAINER& cdc, FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellTimeMatrix(cdc, local_entry_matrix);
        }
        else if (this->GetType() == "dual_for_ee")
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellTimeMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::CellTimeMatrixExplicit(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellTimeMatrixExplicit(cdc, local_entry_matrix);
        }
        else if (this->GetType() == "dual_for_ee")
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellTimeMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().FaceMatrix(fdc, local_entry_matrix, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          this->GetPDE().FaceMatrix_T(fdc, local_entry_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonFaceMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::InterfaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().InterfaceMatrix(fdc, local_entry_matrix, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          this->GetPDE().InterfaceMatrix_T(fdc, local_entry_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonInterfaceMatrix");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::BoundaryMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_cell_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryMatrix(fdc, local_cell_matrix, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryMatrix_T(fdc, local_cell_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellBoundaryMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDoFType() const
    {
      if (this->GetType() == "state" || this->GetType() == "adjoint_for_ee"
          || this->GetType() == "error_evaluation")
      {
        return "state";
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDoFType");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const FE<dealdim, dealdim>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetFESystem() const
    {
      if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
      {
        return this->GetSpaceTimeHandler()->GetFESystem("state");
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetFESystem");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    UpdateFlags
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetUpdateFlags() const
    {

      UpdateFlags r;
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[this->GetTypeNum()]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetUpdateFlags();
        if (this->GetType() == "adjoint_for_ee"
            || this->GetType() == "error_evaluation")
        {
          if (_functional_for_ee_num != dealii::numbers::invalid_unsigned_int)
            r = r | _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    UpdateFlags
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[this->GetTypeNum()]->GetFaceUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetFaceUpdateFlags();
        if (this->GetType() == "adjoint_for_ee"
            || this->GetType() == "error_evaluation")
        {
          if (_functional_for_ee_num != dealii::numbers::invalid_unsigned_int)
            r =
                r
                    | _aux_functionals[_functional_for_ee_num]->GetFaceUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetFunctionalType() const
    {
      if (this->GetType() == "aux_functional")
      {
        return _aux_functionals[this->GetTypeNum()]->GetType();
      }
      else if (this->GetType() == "error_evaluation")
      {
        return _aux_functionals[_functional_for_ee_num]->GetType();
      }
      return "none";
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetFunctionalName() const
    {
      if (this->GetType() == "aux_functional")
      {
        return _aux_functionals[this->GetTypeNum()]->GetName();
      }
      else if (this->GetType() == "error_evaluation")
      {
        return _aux_functionals[_functional_for_ee_num]->GetName();
      }
      return "";
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetTime(
        double time, const TimeIterator& interval)
    {
      GetSpaceTimeHandler()->SetInterval(interval);

      { //Zeit an Dirichlet Werte uebermitteln
        for (unsigned int i = 0; i < _primal_dirichlet_values.size(); i++)
          _primal_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i < _aux_functionals.size(); i++)
          _aux_functionals[i]->SetTime(time);
        //PDE
        this->GetPDE().SetTime(time);
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      if (this->GetType() == "state" || this->GetType() == "adjoint_for_ee")
      {
        this->GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
      }
      else
      {
        throw DOpEException("Unknown type " + this->GetType(),
            "PDEProblemContainer::ComputeSparsityPattern");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::HasFaces() const
    {
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        return _aux_functionals[this->GetTypeNum()]->HasFaces();
      }
      else
      {
        if ((this->GetType() == "state"))
        {
          return this->GetPDE().HasFaces();
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          return this->GetPDE().HasFaces()
              || _aux_functionals[_functional_for_ee_num]->HasFaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
              "PDEProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::HasPoints() const
    {
      if ((this->GetType() == "state") || this->GetType() == "aux_functional")
      {
        //We dont need PointRhs in these cases
        return false;
      }
      else if (this->GetType() == "adjoint_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->HasPoints();
      }
      else
      {
        throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
            "PDEProblemContainer::HasFaces");
      }

    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::HasInterfaces() const
    {
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        return false;
      }
      else
      {
        if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
        {
          return this->GetPDE().HasInterfaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
              "PDEProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetDirichletBoundaryColors(
        unsigned int color, const std::vector<bool>& comp_mask,
        const DD* values)
    {
      assert(values);
      assert(values->n_components() == comp_mask.size());

      unsigned int comp = _dirichlet_colors.size();
      for (unsigned int i = 0; i < _dirichlet_colors.size(); ++i)
      {
        if (_dirichlet_colors[i] == color)
        {
          comp = i;
          break;
        }
      }
      if (comp != _dirichlet_colors.size())
      {
        std::stringstream s;
        s << "DirichletColor" << color << " has multiple occurrences !";
        throw DOpEException(s.str(),
            "PDEProblemContainer::SetDirichletBoundary");
      }
      _dirichlet_colors.push_back(color);
      _dirichlet_comps.push_back(comp_mask);
      PrimalDirichletData<DD, VECTOR, dealdim> *data = new PrimalDirichletData<
          DD, VECTOR, dealdim>(*values);
      _primal_dirichlet_values.push_back(data);
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDirichletColors() const
    {
      if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
      {
        return _dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDirichletColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<bool>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDirichletCompMask(
        unsigned int color) const
    {
      if ((this->GetType() == "state" || this->GetType() == "adjoint_for_ee"))
      {
        unsigned int comp = _dirichlet_colors.size();
        for (unsigned int i = 0; i < _dirichlet_colors.size(); ++i)
        {
          if (_dirichlet_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp == _dirichlet_colors.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::GetDirichletCompMask");
        }
        return _dirichlet_comps[comp];
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDirichletCompMask");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const Function<dealdim>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDirichletValues(
        unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {

      unsigned int col = _dirichlet_colors.size();
      if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
      {
        for (unsigned int i = 0; i < _dirichlet_colors.size(); ++i)
        {
          if (_dirichlet_colors[i] == color)
          {
            col = i;
            break;
          }
        }
        if (col == _dirichlet_colors.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::GetDirichletValues");
        }
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDirichletValues");
      }

      if (this->GetType() == "state")
      {
        _primal_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_primal_dirichlet_values[col]);
      }
      else if (this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "adjoint_hessian"))
      {
        return *(_zero_dirichlet_values);
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDirichletValues");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetBoundaryEquationColors() const
    {
      if (this->GetType() == "state")
      {
        return _state_boundary_equation_colors;
      }
      else if (this->GetType() == "adjoint_for_ee")
      {
        return _adjoint_boundary_equation_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetBoundaryEquationColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetBoundaryEquationColors(
        unsigned int color)
    {
      { //State Boundary Equation colors are simply inserted
        unsigned int comp = _state_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _state_boundary_equation_colors.size();
            ++i)
        {
          if (_state_boundary_equation_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != _state_boundary_equation_colors.size())
        {
          std::stringstream s;
          s << "Boundary Equation Color" << color
              << " has multiple occurences !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::SetBoundaryEquationColors");
        }
        _state_boundary_equation_colors.push_back(color);
      }
      { //For the  adjoint they are added with the boundary functional colors
        unsigned int comp = _adjoint_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _adjoint_boundary_equation_colors.size();
            ++i)
        {
          if (_adjoint_boundary_equation_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != _adjoint_boundary_equation_colors.size())
        {
          //Seems this color is already added, however it might have been a functional color
          //so we don't  do anything.
        }
        else
        {
          _adjoint_boundary_equation_colors.push_back(color);
        }

      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetBoundaryFunctionalColors() const
    {
      //FIXME cost_functional?? This is pdeproblemcontainer, we should not have a cost functional! ~cg
      if (this->GetType() == "cost_functional"
          || this->GetType() == "aux_functional"
          || this->GetType() == "error_evaluation")
      {
        return _boundary_functional_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetBoundaryFunctionalColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetBoundaryFunctionalColors(
        unsigned int color)
    {
      { //Boundary Functional colors are simply inserted
        unsigned int comp = _boundary_functional_colors.size();
        for (unsigned int i = 0; i < _boundary_functional_colors.size(); ++i)
        {
          if (_boundary_functional_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != _boundary_functional_colors.size())
        {
          std::stringstream s;
          s << "Boundary Functional Color" << color
              << " has multiple occurences !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::SetBoundaryFunctionalColors");
        }
        _boundary_functional_colors.push_back(color);
      }
      { //For the  adjoint they are added  to the boundary equation colors
        unsigned int comp = _adjoint_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _adjoint_boundary_equation_colors.size();
            ++i)
        {
          if (_adjoint_boundary_equation_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != _adjoint_boundary_equation_colors.size())
        {
          //Seems this color is already added, however it might have been a equation color
          //so we don't  do anything.
        }
        else
        {
          _adjoint_boundary_equation_colors.push_back(color);
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetStateNBlocks() const
    {
      return this->GetPDE().GetStateNBlocks();
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetNBlocks() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint_for_ee"))
      {
        return this->GetStateNBlocks();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetNBlocks");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDoFsPerBlock(
        unsigned int b) const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock(b);
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDoFsPerBlock() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const dealii::ConstraintMatrix&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDoFConstraints() const
    {
//      std::cout << "Constraints:"
//          << GetSpaceTimeHandler()->GetStateDoFConstraints().n_constraints()
//          << " and type is " << this->GetType() << std::endl;
      if ((this->GetType() == "state") || (this->GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateDoFConstraints();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "PDEProblemContainer::GetDoFConstraints");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::NeedTimeFunctional() const
    {
      if (this->GetType() == "cost_functional")
        return false;
      else if (this->GetType() == "aux_functional")
        return _aux_functionals[this->GetTypeNum()]->NeedTime();
      else if (this->GetType() == "error_evaluation")
        return _aux_functionals[_functional_for_ee_num]->NeedTime();
      else
        throw DOpEException("Not implemented",
            "PDEProblemContainer::NeedTimeFunctional");
    }

}
#endif
