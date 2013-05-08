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

#ifndef _OptProblemContainer_H_
#define _OptProblemContainer_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "functionalinterface.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "function_wrapper.h"
#include "spacetimehandler.h"
#include "primaldirichletdata.h"
#include "tangentdirichletdata.h"
#include "transposeddirichletdatainterface.h"
#include "transposedgradientdirichletdata.h"
#include "transposedhessiandirichletdata.h"
#include "constraintvector.h"
#include "controlvector.h"
#include "statevector.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "stateproblem.h"
#include "dopetypes.h"
#include "dwrdatacontainer.h"
#include "problemcontainer_internal.h"

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

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim,
      template<int, int> class FE = dealii::FESystem,
      template<int, int> class DH = dealii::DoFHandler>
    class OptProblemContainer : public ProblemContainerInternal<PDE>
    {
      public:
        OptProblemContainer(FUNCTIONAL& functional, PDE& pde,
            CONSTRAINTS& constraints,
            SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>& STH);

        /******************************************************/

        ~OptProblemContainer();

        /******************************************************/

        virtual std::string
        GetName() const
        {
          return "OptProblem";
        }

        /******************************************************/
        StateProblem<
            OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
            PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>&
        GetStateProblem()
        {
          if (_state_problem == NULL)
          {
            _state_problem = new StateProblem<
                OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                    CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
                    DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>(
                *this, this->GetPDE());
          }
          return *_state_problem;
        }

        //TODO This is Pfush needed to split into different subproblems and allow optproblem to
        //be substituted as any of these problems. Can be removed once the splitting is complete.
        OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
            CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>&
        GetBaseProblem()
        {
          return *this;
        }
        /******************************************************/

        /**
         * This function calls the ReInit function of the SpaceTimeHandler
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
         * have been implemented so far: `cost_functional' and `aux_functional'.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined on cells, e.g., drag and lift computation.
         * Or computations of deflections and deformations.
         *
         * @template FACEDATACONTAINER    Class of the datacontainer, distinguishes
         *                                between hp- and classical case.
         *
         * @param fdc                     A DataContainer holding all the needed information
         *                                of the face.
         */
        template<typename DATACONTAINER>
          double
          CellFunctional(const DATACONTAINER& cdc);

        /******************************************************/

        /**
         * This function returns a functional value of a point.
         * Different types of functionals
         * have been implemented so far: `cost_functional' and `aux_functional'.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined in points. For example, deflection or pressure values in
         * points.
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
         * have been implemented so far: `cost_functional' and `aux_functional'.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined on the boundary, e.g., stresses.
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

        void
        AlgebraicResidual(VECTOR& residual,
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
         *                                of the cell.
         * @param local_cell_vector       This vector contains the locally computed values of the cell equation. For more information
         *                                on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                   A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<typename DATACONTAINER>
          void
          CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

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
         * It is mainly needed for fluid-structure interaction problems and should
         * be used when the term of the time derivative contains
         * nonlinear terms, i.e. $\partial_t u v + ...$
         * in which u and v denote solution variables.
         * Secondly, this function should be used when the densities
         * are not constant: $\partial_t \rho v + ...$
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
         * @param cdc                      A DataContainer holding all the needed information of the cell.
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
         * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
         *                                 between hp- and classical case.
         *
         * @param cdc                      A DataContainer holding all the needed information
         *
         * @param scale                 A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
         *                                 filled by the locally computed values. For more information of its functionality, please
         *                                 search for the keyword `FullMatrix' in the deal.ii manual.
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
        //FIXME maybe InterfaceEquation and InterfaceMatrix could get
        //integrated into FaceEquation and FaceMatrix?
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
         * Computes the value of the boundary on a cell.
         * It has the same functionality as CellEquation. We refer to its
         * documentation.
         *
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
        void
        ComputeLocalControlConstraints(VECTOR& constraints,
            const std::map<std::string, const dealii::Vector<double>*> &values,
            const std::map<std::string, const VECTOR*> &block_values);
        /******************************************************/

        /**
         * This is to get the lower and upper control box constraints
         *

         */
        void
        GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
        {
          this->GetConstraints()->GetControlBoxConstraints(lb, ub);
        }

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
         * This function determines whether point evaluations are required or not.
         *
         * @return Returns whether or not this functional needs evaluations of
         *         point values.
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
        SetControlDirichletBoundaryColors(unsigned int color,
            const std::vector<bool>& comp_mask,
            const DOpEWrapper::Function<dealdim>* values);

        /******************************************************/

        void
        SetDirichletBoundaryColors(unsigned int color,
            const std::vector<bool>& comp_mask, const DD* values);

        /******************************************************/

        const std::vector<unsigned int>&
        GetDirichletColors() const;
        const std::vector<unsigned int>&
        GetTransposedDirichletColors() const;
        const std::vector<bool>&
        GetDirichletCompMask(unsigned int color) const;
        const std::vector<bool>&
        GetTransposedDirichletCompMask(unsigned int color) const;

        /******************************************************/

        const dealii::Function<dealdim> &
        GetDirichletValues(unsigned int color,
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values) const;

        /******************************************************/

        const TransposedDirichletDataInterface<dopedim, dealdim> &
        GetTransposedDirichletValues(unsigned int color,
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
        SetControlBoundaryEquationColors(unsigned int color);
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
        AddFunctional(FUNCTIONAL_INTERFACE* F)
        {
          _aux_functionals.push_back(F);
          if (_functional_position.find(F->GetName())
              != _functional_position.end())
          {
            throw DOpEException(
                "You cant add two functionals with the same name.",
                "OPTProblemContainer::AddFunctional");
          }
          _functional_position[F->GetName()] = _aux_functionals.size();
          //remember! At _functional_values[0] we store always the cost functional!
        }

        /******************************************************/

        /**
         * Through this function one sets the functional for the
         * error estimation. The name given by functional_name is
         * looked up in _aux_functionals, so the function assumes
         * that the functional intended for the error estimation
         * is set prior by AddFunctional.
         *
         * @param functional_name     The name of the functional for the error estimation.
         */
        void
        SetFunctionalForErrorEstimation(std::string functional_name)
        {
          if (GetFunctional()->GetName() == functional_name)
          {
            _functional_for_ee_is_cost = true;
            _functional_for_ee_num = dealii::numbers::invalid_unsigned_int;
          }
          else
          {
            _functional_for_ee_is_cost = false;
            bool found = false;
            //we go through all aux functionals.
            for (unsigned int i = 0; i < this->GetNFunctionals(); i++)
            {
              if (_aux_functionals[i]->GetName() == functional_name)
              {
                //if the names match, we have found our functional.
                found = true;
                _functional_for_ee_num = i;
              }
            }
            //If we have not found a functional with the given name,
            //we throw an error.
            if (!found)
            {
              throw DOpEException(
                  "Can't find functional " + functional_name
                      + " in _aux_functionals",
                  "Optproblem::SetFunctionalForErrorEstimation");
            }
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
        GetControlNBlocks() const;

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
        std::string
        GetConstraintType() const;

        /******************************************************/

        bool
        NeedTimeFunctional() const;

        /******************************************************/

        bool
        HasControlInDirichletData() const;

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

        /**
         * Sets the actual time.
         *
         * @param time      The actual time.
         * @param interval  The actual interval. Make sure that time
         *                  lies in interval!
         */
        void
        SetTime(double time, const TimeIterator& interval);

        /******************************************************/

        const SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>*
        GetSpaceTimeHandler() const
        {
          return _STH;
        }

        /******************************************************/

        SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>*
        GetSpaceTimeHandler()
        {
          return _STH;
        }

        /******************************************************/

        void
        ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

        /******************************************************/

        bool
        IsFeasible(const ConstraintVector<VECTOR>& g) const;
        bool
        IsLargerThan(const ConstraintVector<VECTOR>& g, double p) const
        {
          return this->GetConstraints()->IsLargerThan(g, p);
        }
        bool
        IsEpsilonFeasible(const ConstraintVector<VECTOR>& g, double p) const
        {
          return this->GetConstraints()->IsEpsilonFeasible(g, p);
        }
        double
        MaxViolation(const ConstraintVector<VECTOR>& g) const
        {
          return this->GetConstraints()->MaxViolation(g);
        }
        void
        FeasibilityShift(const ControlVector<VECTOR>& g_hat,
            ControlVector<VECTOR>& g, double lambda) const
        {
          this->GetConstraints()->FeasibilityShift(g_hat, g, lambda);
        }
        double
        Complementarity(const ConstraintVector<VECTOR>& f,
            const ConstraintVector<VECTOR>& g) const
        {
          return this->GetConstraints()->Complementarity(f, g);
        }
        /******************************************************/

        //      void
        //      PostProcessConstraints(ConstraintVector<VECTOR>& g,
        //          bool process_global_in_time_constraints) const;
        void
        PostProcessConstraints(ConstraintVector<VECTOR>& g) const;

        /******************************************************/

        void
        AddAuxiliaryControl(const ControlVector<VECTOR>* c, std::string name);

        /******************************************************/

        void
        AddAuxiliaryState(const StateVector<VECTOR>* c, std::string name);

        /******************************************************/

        void
        AddAuxiliaryConstraint(const ConstraintVector<VECTOR>* c,
            std::string name);

        /******************************************************/

        const ControlVector<VECTOR>*
        GetAuxiliaryControl(std::string name) const;

        /******************************************************/

        const StateVector<VECTOR>*
        GetAuxiliaryState(std::string name) const;

        /******************************************************/

        void
        DeleteAuxiliaryControl(std::string name);

        /******************************************************/

        void
        DeleteAuxiliaryState(std::string name);

        /******************************************************/

        void
        DeleteAuxiliaryConstraint(std::string name);

        /*****************************************************************/

        const ConstraintVector<VECTOR>*
        GetAuxiliaryConstraint(std::string name)
        {
          typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
              _auxiliary_constraints.find(name);
          if (it == _auxiliary_constraints.end())
          {
            throw DOpEException("Could not find data" + name,
                "OptProblemContainer::GetAuxiliaryConstraint");
          }
          return it->second;
        }
        /*****************************************************************/

        template<typename INTEGRATOR>
          void
          AddAuxiliaryToIntegrator(INTEGRATOR& integrator)
          {
            {
              typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
                  _auxiliary_controls.begin();
              for (; it != _auxiliary_controls.end(); it++)
              {
                if (dopedim == dealdim)
                {
                  integrator.AddDomainData(it->first,
                      &(it->second->GetSpacialVector()));
                }
                else if (dopedim == 0)
                {
                  integrator.AddParamData(it->first,
                      &(it->second->GetSpacialVectorCopy()));
                }
                else
                {
                  throw DOpEException("dopedim not implemented",
                      "OptProblemContainer::AddAuxiliaryToIntegrator");
                }
              }
            }
            {
              typename std::map<std::string, const StateVector<VECTOR> *>::iterator it =
                  _auxiliary_state.begin();
              for (; it != _auxiliary_state.end(); it++)
              {
                integrator.AddDomainData(it->first,
                    &(it->second->GetSpacialVector()));
              }
            }
            {
              typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
                  _auxiliary_constraints.begin();
              for (; it != _auxiliary_constraints.end(); it++)
              {
                integrator.AddDomainData(it->first + "_local",
                    &(it->second->GetSpacialVector("local")));
                integrator.AddParamData(it->first + "_global",
                    &(it->second->GetGlobalConstraints()));
              }
            }
          }

        /******************************************************/

        template<typename INTEGRATOR>
          void
          DeleteAuxiliaryFromIntegrator(INTEGRATOR& integrator)
          {
            {
              typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
                  _auxiliary_controls.begin();
              for (; it != _auxiliary_controls.end(); it++)
              {
                if (dopedim == dealdim)
                {
                  integrator.DeleteDomainData(it->first);
                }
                else if (dopedim == 0)
                {
                  integrator.DeleteParamData(it->first);
                  it->second->UnLockCopy();
                }
                else
                {
                  throw DOpEException("dopedim not implemented",
                      "OptProblemContainer::AddAuxiliaryToIntegrator");
                }
              }
            }
            {
              typename std::map<std::string, const StateVector<VECTOR> *>::iterator it =
                  _auxiliary_state.begin();
              for (; it != _auxiliary_state.end(); it++)
              {
                integrator.DeleteDomainData(it->first);
              }
            }
            {
              typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
                  _auxiliary_constraints.begin();
              for (; it != _auxiliary_constraints.end(); it++)
              {
                integrator.DeleteDomainData(it->first + "_local");
                integrator.DeleteParamData(it->first + "_global");
              }
            }
          }

        /******************************************************/
        const std::map<std::string, unsigned int>&
        GetFunctionalPosition() const
        {
          return _functional_position;
        }

        unsigned int
        GetStateNBlocks()
        {
          return this->GetPDE().GetStateNBlocks();
        }

        /******************************************************/

        std::vector<unsigned int>&
        GetControlBlockComponent()
        {
          return this->GetPDE().GetControlBlockComponent();
        }
        /******************************************************/

        std::vector<unsigned int>&
        GetStateBlockComponent()
        {
          return this->GetPDE().GetStateBlockComponent();
        }

        /******************************************************/
        /******************************************************/
        /****For the initial values ***************/
        template<typename DATACONTAINER>
          void
          Init_CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico)
          {
            if (this->GetType() == "adjoint" || this->GetType() == "tangent"
                || this->GetType() == "adjoint_hessian")
            {
              this->GetPDE().Init_CellEquation(cdc, local_cell_vector, scale,
                  scale_ico);
            }
            else
            {
              throw DOpEException("Not implemented",
                  "OptProblemContainer::Init_CellEquation");
            }
          }

        template<typename DATACONTAINER>
          void
          Init_CellRhs(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            //FIXME: We should take care of the cases of boundary and face functionals...
            if (this->GetType() == "adjoint")
            {
              if (GetFunctional()->NeedTime())
              {
                if (GetFunctional()->GetType().find("timelocal")
                    != std::string::npos)
                {
                  if (GetFunctional()->GetType().find("domain")
                      != std::string::npos)
                  {
                    GetFunctional()->Value_U(cdc, local_cell_vector, scale);
                  }
                }
              }
            }
            else if (this->GetType() == "tangent")
            {
              this->GetPDE().Init_CellRhs_QT(cdc, local_cell_vector, scale);
            }
            else if (this->GetType() == "adjoint_hessian")
            {
              if (GetFunctional()->NeedTime())
              {
                if (GetFunctional()->GetType().find("timelocal")
                    != std::string::npos)
                {
                  if (GetFunctional()->GetType().find("domain")
                      != std::string::npos)
                  {
                    GetFunctional()->Value_UU(cdc, local_cell_vector, scale);
                  }
                }
              }
            }
            else
            {
              throw DOpEException("Not implemented",
                  "OptProblemContainer::Init_CellRhs");
            }
          }

        void
        Init_PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1.)
        {
          if (this->GetType() == "adjoint")
          {
            if (GetFunctional()->NeedTime())
            {
              if (GetFunctional()->GetType().find("timelocal")
                  != std::string::npos)
              {
                if (GetFunctional()->GetType().find("point")
                    != std::string::npos)
                {
                  GetFunctional()->PointValue_U(param_values, domain_values,
                      rhs_vector, scale);
                }
              }
            }
          }
          else if (this->GetType() == "tangent")
          {
            //Nothing to do for tangent, since no point sources are allowed in the initial condition.
          }
          else if (this->GetType() == "adjoint_hessian")
          {
            if (GetFunctional()->NeedTime())
            {
              if (GetFunctional()->GetType().find("timelocal")
                  != std::string::npos)
              {
                if (GetFunctional()->GetType().find("point")
                    != std::string::npos)
                {
                  GetFunctional()->PointValue_UU(param_values, domain_values,
                      rhs_vector, scale);
                }
              }
            }
          }
          else
          {
            throw DOpEException("Not implemented",
                "OptProblemContainer::Init_PointRhs");
          }
        }

        template<typename DATACONTAINER>
          void
          Init_CellMatrix(const DATACONTAINER& cdc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale,
              double scale_ico)
          {
            if (this->GetType() == "adjoint" || this->GetType() == "tangent"
                || this->GetType() == "adjoint_hessian")
            {
              this->GetPDE().Init_CellMatrix(cdc, local_entry_matrix, scale,
                  scale_ico);
            }
            else
            {
              throw DOpEException("Not implemented",
                  "OptProblemContainer::Init_CellMatrix");
            }
          }

        /******************************************************/

        /**
         * Returns whether the functional for error estimation
         * is the costfunctional
         */
        bool
        EEFunctionalIsCost() const
        {
          return _functional_for_ee_is_cost;
        }

      protected:
        FUNCTIONAL*
        GetFunctional();
        const FUNCTIONAL*
        GetFunctional() const;
        CONSTRAINTS *
        GetConstraints()
        {
          return _constraints;
        }
        const CONSTRAINTS *
        GetConstraints() const
        {
          return _constraints;
        }

        const VECTOR*
        GetBlockVector(const std::map<std::string, const VECTOR*>& values,
            std::string name)
        {
          typename std::map<std::string, const VECTOR*>::const_iterator it =
              values.find(name);
          if (it == values.end())
          {
            throw DOpEException("Did not find " + name,
                "OptProblemContainer::GetBlockVector");
          }
          return it->second;
        }
        const dealii::Vector<double>*
        GetVector(const std::map<std::string, const Vector<double>*>& values,
            std::string name)
        {
          typename std::map<std::string, const Vector<double>*>::const_iterator it =
              values.find(name);
          if (it == values.end())
          {
            throw DOpEException("Did not find " + name,
                "OptProblemContainer::GetVector");
          }
          return it->second;
        }
        /******************************************************/
      private:
        DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
        DOpEOutputHandler<VECTOR>* _OutputHandler;
        std::string _algo_type;

        bool _functional_for_ee_is_cost;
        unsigned int _functional_for_ee_num;
        std::vector<FUNCTIONAL_INTERFACE*> _aux_functionals;
        std::map<std::string, unsigned int> _functional_position;
        FUNCTIONAL* _functional;
        CONSTRAINTS* _constraints;
        SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>* _STH;

        std::vector<unsigned int> _control_dirichlet_colors;
        std::vector<unsigned int> _control_transposed_dirichlet_colors;
        std::vector<std::vector<bool> > _control_dirichlet_comps;
        std::vector<const DOpEWrapper::Function<dealdim>*> _control_dirichlet_values;
        std::vector<
            TransposedGradientDirichletData<DD, VECTOR, dopedim, dealdim>*> _transposed_control_gradient_dirichlet_values;
        std::vector<
            TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim>*> _transposed_control_hessian_dirichlet_values;

        std::vector<unsigned int> _dirichlet_colors;
        std::vector<std::vector<bool> > _dirichlet_comps;
        std::vector<PrimalDirichletData<DD, VECTOR, dopedim, dealdim>*> _primal_dirichlet_values;
        std::vector<TangentDirichletData<DD, VECTOR, dopedim, dealdim>*> _tangent_dirichlet_values;
        const dealii::Function<dealdim>* _zero_dirichlet_values;

        const dealii::Function<dealdim>* _initial_values;

        std::vector<unsigned int> _control_boundary_equation_colors;
        std::vector<unsigned int> _state_boundary_equation_colors;
        std::vector<unsigned int> _adjoint_boundary_equation_colors;

        std::vector<unsigned int> _boundary_functional_colors;

        std::map<std::string, const ControlVector<VECTOR>*> _auxiliary_controls;
        std::map<std::string, const StateVector<VECTOR>*> _auxiliary_state;
        std::map<std::string, const ConstraintVector<VECTOR>*> _auxiliary_constraints;

        StateProblem<
            OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
            PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim> * _state_problem;

        friend class StateProblem<
            OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
            PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim> ;
    };
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::OptProblemContainer(
        FUNCTIONAL& functional, PDE& pde, CONSTRAINTS& constraints,
        SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim>& STH) :
        ProblemContainerInternal<PDE>(pde), _functional(&functional), _constraints(
            &constraints), _STH(&STH), _state_problem(NULL)
    {
      _ExceptionHandler = NULL;
      _OutputHandler = NULL;
      _zero_dirichlet_values = new ZeroFunction<dealdim>(
          this->GetPDE().GetStateNComponents());
      _algo_type = "";
      _functional_position[_functional->GetName()] = 0;
      //remember! At _functional_values[0] we store always the cost functional!
      _functional_for_ee_num = dealii::numbers::invalid_unsigned_int;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::~OptProblemContainer()
    {
      if (_zero_dirichlet_values != NULL)
      {
        delete _zero_dirichlet_values;
      }

      for (unsigned int i = 0;
          i < _transposed_control_gradient_dirichlet_values.size(); i++)
      {
        if (_transposed_control_gradient_dirichlet_values[i] != NULL)
          delete _transposed_control_gradient_dirichlet_values[i];
      }
      for (unsigned int i = 0;
          i < _transposed_control_hessian_dirichlet_values.size(); i++)
      {
        if (_transposed_control_hessian_dirichlet_values[i] != NULL)
          delete _transposed_control_hessian_dirichlet_values[i];
      }
      for (unsigned int i = 0; i < _primal_dirichlet_values.size(); i++)
      {
        if (_primal_dirichlet_values[i] != NULL)
          delete _primal_dirichlet_values[i];
      }
      for (unsigned int i = 0; i < _tangent_dirichlet_values.size(); i++)
      {
        if (_tangent_dirichlet_values[i] != NULL)
          delete _tangent_dirichlet_values[i];
      }
      if (_state_problem != NULL)
      {
        delete _state_problem;
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ReInit(
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
            "OptProblemContainer::ReInit");
      }
      else
      {
        _algo_type = algo_type;
        this->SetTypeInternal("");

        if (_algo_type == "reduced")
        {
          GetSpaceTimeHandler()->ReInit(this->GetPDE().GetControlNBlocks(),
              this->GetPDE().GetControlBlockComponent(),
              this->GetPDE().GetStateNBlocks(),
              this->GetPDE().GetStateBlockComponent());
        }
        else
        {
          throw DOpEException("Unknown Algorithm " + _algo_type,
              "OptProblemContainer::ReInit");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetType(
        std::string type, unsigned int num)
    {
      if (this->GetType() != type || this->GetTypeNum() != num)
      {
        this->SetTypeNumInternal(num);
        this->SetTypeInternal(type);
        this->GetPDE().SetProblemType(type);
        if (_functional_for_ee_num != dealii::numbers::invalid_unsigned_int)
          _aux_functionals[_functional_for_ee_num]->SetProblemType(type);
        this->GetConstraints()->SetProblemType(type, num);

#if dope_dimension > 0
        if(dealdim == dopedim)
        {
          //Prepare DoFHandlerPointer

          {
            if(this->GetType() == "state" ||this->GetType() == "adjoint"
                || this->GetType() == "adjoint_for_ee" || this->GetType() == "cost_functional"
                || this->GetType() == "aux_functional" || this->GetType() == "functional_for_ee"
                || this->GetType() == "tangent" || this->GetType() == "adjoint_hessian"
                || this->GetType() == "error_evaluation"
                || this->GetType().find("constraints") != std::string::npos)
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(1,0);
            }
            else if (this->GetType() == "gradient"||this->GetType() == "hessian"||this->GetType() == "hessian_inverse" || this->GetType() == "global_constraint_gradient"|| this->GetType() == "global_constraint_hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0,1);
            }
            else
            {
              throw DOpEException("_problem_type : "+this->GetType()+" not implemented!", "OptProblemContainer::SetType");
            }
          }
        }
        else
        {
          throw DOpEException("dopedim not implemented", "OptProblemContainer::SetType");
        }
#else
        //dopedim ==0
        {
          //Prepare DoFHandlerPointer
          {

            if (this->GetType() == "state" || this->GetType() == "adjoint"
                || this->GetType() == "adjoint_for_ee"
                || this->GetType() == "functional_for_ee"
                || this->GetType() == "cost_functional"
                || this->GetType() == "aux_functional"
                || this->GetType() == "tangent"
                || this->GetType() == "error_evaluation"
                || this->GetType() == "adjoint_hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0, 0);
            }
            else if (this->GetType() == "gradient"
                || this->GetType() == "hessian_inverse"
                || this->GetType() == "hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0, 0);
            }
            else
            {
              throw DOpEException(
                  "_problem_type : " + this->GetType() + " not implemented!",
                  "OptProblemContainer::SetType");
            }
          }
        }
#endif
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      double
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellFunctional(
          const DATACONTAINER& cdc)
      {

        if (this->GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->Value(cdc);
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->Value(cdc);
        }
        else if (this->GetType() == "functional_for_ee")
        { //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
          return _aux_functionals[_functional_for_ee_num]->Value(cdc);
        }
        else if (this->GetType().find("constraints") != std::string::npos)
        {
          return GetConstraints()->Value(cdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellFunctional");
        }
      }

  /******************************************************/
  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    double
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::PointFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (this->GetType() == "cost_functional")
      {
        // state values in quadrature points
        return GetFunctional()->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif cost_functional
      else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[this->GetTypeNum()]->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif aux_functional
      else if (this->GetType() == "functional_for_ee")
      {
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        return _aux_functionals[_functional_for_ee_num]->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);
      } //endif functional_for_ee
      else if (this->GetType().find("constraints") != std::string::npos)
      {
        return GetConstraints()->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif constraints
      else
      {
        throw DOpEException("Not implemented",
            "OptProblemContainer::PointFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      double
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::BoundaryFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (this->GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->BoundaryValue(fdc);
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->BoundaryValue(fdc);
        }
        else if (this->GetType() == "functional_for_ee")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->BoundaryValue(fdc);
        }
        else if (this->GetType().find("constraints") != std::string::npos)
        {
          return GetConstraints()->BoundaryValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::BoundaryFunctional");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      double
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (this->GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->FaceValue(fdc);
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[this->GetTypeNum()]->FaceValue(fdc);
        }
        else if (this->GetType() == "functional_for_ee")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->FaceValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::FaceFunctional");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    double
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AlgebraicFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (this->GetType() == "cost_functional")
      {
        // state values in quadrature points
        return GetFunctional()->AlgebraicValue(param_values, domain_values);
      }
      else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[this->GetTypeNum()]->AlgebraicValue(
            param_values, domain_values);
      }
      else if (this->GetType() == "functional_for_ee")
      //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
      {
        return _aux_functionals[_functional_for_ee_num]->AlgebraicValue(
            param_values, domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "OptProblemContainer::AlgebraicFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale, double scale_ico)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellEquation(cdc, local_cell_vector, scale, scale_ico);
        }
        else if ((this->GetType() == "adjoint")
            || (this->GetType() == "adjoint_for_ee"))
        {
          // state values in quadrature points
          this->GetPDE().CellEquation_U(cdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().CellEquation_UTT(cdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          this->GetPDE().CellEquation_UT(cdc, local_cell_vector, scale,
              scale_ico);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          // control values in quadrature points
          this->GetPDE().ControlCellEquation(cdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AlgebraicResidual(
        VECTOR& residual,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (this->GetType() == "gradient")
      {
        // state values in quadrature points
        return GetFunctional()->AlgebraicGradient_Q(residual, param_values,
            domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "OptProblemContainer::AlgebraicFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellTimeEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          this->GetPDE().CellTimeEquation(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee")
        {
          this->GetPDE().CellTimeEquation_U(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          this->GetPDE().CellTimeEquation_UTT(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "tangent")
        {
          this->GetPDE().CellTimeEquation_UT(cdc, local_cell_vector, scale);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellTimeEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellTimeEquationExplicit(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          this->GetPDE().CellTimeEquationExplicit(cdc, local_cell_vector,
              scale);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee")
        {
          this->GetPDE().CellTimeEquationExplicit_U(cdc, local_cell_vector,
              scale);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          this->GetPDE().CellTimeEquationExplicit_UTT(cdc, local_cell_vector,
              scale);
        }
        else if (this->GetType() == "tangent")
        {
          this->GetPDE().CellTimeEquationExplicit_UT(cdc, local_cell_vector,
              scale);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellTimeEquationExplicit");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellTimeEquationExplicit");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation(fdc, local_cell_vector, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation_UTT(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation_UT(fdc, local_cell_vector, scale,
              scale_ico);
        }
//        else if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
//        {
//          // control values in quadrature points
//          this->GetPDE().ControlFaceEquation(fdc, local_cell_vector, scale);
//        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::FaceEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::InterfaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          this->GetPDE().InterfaceEquation(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().InterfaceEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::InterfaceEquation");
        }
      }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::BoundaryEquation(
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
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation_U(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation_UTT(fdc, local_cell_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation_UT(fdc, local_cell_vector, scale,
              scale_ico);
        }
//        else if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
//        {
//          // control values in quadrature points
//          this->GetPDE().ControlBoundaryEquation(fdc, local_cell_vector, scale);
//        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellBoundaryEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellRhs(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {
        //FIXME: In timedependent Problems one should evaluate the
        // Functional terms here only if they are time distributed,
        //otherwise the scaling should be different (e.g. +-1 for local in time
        //Functionals)
        //The same applies to all ...Rhs functions, except init_...

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().CellRightHandSide(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("domain") != std::string::npos)
            GetFunctional()->Value_U(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          _aux_functionals[_functional_for_ee_num]->Value_U(cdc,
              local_cell_vector, scale);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          this->GetPDE().CellEquation_QT(cdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("domain") != std::string::npos)
          {
            GetFunctional()->Value_UU(cdc, local_cell_vector, scale);
            GetFunctional()->Value_QU(cdc, local_cell_vector, scale);
          }
          scale *= -1;
          this->GetPDE().CellEquation_UU(cdc, local_cell_vector, scale, scale);
          this->GetPDE().CellEquation_QU(cdc, local_cell_vector, scale, scale);
          //TODO: make some example where this realy matters to check if this is right
          this->GetPDE().CellTimeEquationExplicit_UU(cdc, local_cell_vector,
              scale);
        }
        else if (this->GetType() == "gradient")
        {
          if (GetSpaceTimeHandler()->GetControlType()
              == DOpEtypes::ControlType::initial)
          {
            this->GetPDE().Init_CellRhs_Q(cdc, local_cell_vector, scale);
          }
          // state values in quadrature points
          if (GetFunctional()->GetType().find("domain") != std::string::npos)
            GetFunctional()->Value_Q(cdc, local_cell_vector, scale);

          scale *= -1;
          this->GetPDE().CellEquation_Q(cdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "hessian")
        {
          if (GetSpaceTimeHandler()->GetControlType()
              == DOpEtypes::ControlType::initial)
          {
            this->GetPDE().Init_CellRhs_QTT(cdc, local_cell_vector, scale);
            this->GetPDE().Init_CellRhs_QQ(cdc, local_cell_vector, scale);
          }

          if (GetFunctional()->GetType().find("domain") != std::string::npos)
          {
            GetFunctional()->Value_QQ(cdc, local_cell_vector, scale);
            GetFunctional()->Value_UQ(cdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().CellEquation_QTT(cdc, local_cell_vector, scale, scale);
          this->GetPDE().CellEquation_UQ(cdc, local_cell_vector, scale, scale);
          this->GetPDE().CellEquation_QQ(cdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "global_constraint_gradient")
        {
          GetConstraints()->Value_Q(cdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "global_constraint_hessian")
        {
          GetConstraints()->Value_QQ(cdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellRhs");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::PointRhs(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values,
        VECTOR& rhs_vector, double scale)
    {

      if (this->GetType() == "adjoint")
      {
        // state values in quadrature points
        if (GetFunctional()->GetType().find("point") != std::string::npos)
        {
          GetFunctional()->PointValue_U(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
        }
      }
      else if (this->GetType() == "adjoint_for_ee")
      {
        //values of the derivative of the functional for error estimation
        _aux_functionals[_functional_for_ee_num]->PointValue_U(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values, rhs_vector, scale);
      }
      else if (this->GetType() == "adjoint_hessian")
      {
        // state values in quadrature points
        if (GetFunctional()->GetType().find("point") != std::string::npos)
        {
          GetFunctional()->PointValue_UU(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
          GetFunctional()->PointValue_QU(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
        }
      }
      else if (this->GetType() == "gradient")
      {
        // state values in quadrature points
        if (GetFunctional()->GetType().find("point") != std::string::npos)
        {
          GetFunctional()->PointValue_Q(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
        }
      }
      else if (this->GetType() == "hessian")
      {
        // state values in quadrature points
        if (GetFunctional()->GetType().find("point") != std::string::npos)
        {
          GetFunctional()->PointValue_QQ(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
          GetFunctional()->PointValue_UQ(
              this->GetSpaceTimeHandler()->GetControlDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
        }
      }
      else
      {
        throw DOpEException("Not implemented", "OptProblem::CellRhs");
      }

    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().FaceRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("face") != std::string::npos)
            GetFunctional()->FaceValue_U(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (_aux_functionals[_functional_for_ee_num]->GetType().find("face")
              != std::string::npos)
            _aux_functionals[_functional_for_ee_num]->FaceValue_U(fdc,
                local_cell_vector, scale);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          this->GetPDE().FaceEquation_QT(fdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("face") != std::string::npos)
          {
            GetFunctional()->FaceValue_UU(fdc, local_cell_vector, scale);
            GetFunctional()->FaceValue_QU(fdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().FaceEquation_UU(fdc, local_cell_vector, scale, scale);

          this->GetPDE().FaceEquation_QU(fdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "gradient")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("face") != std::string::npos)
          {
            GetFunctional()->FaceValue_Q(fdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().FaceEquation_Q(fdc, local_cell_vector, scale, scale);
        }
        else if (this->GetType() == "hessian")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("face") != std::string::npos)
          {
            GetFunctional()->FaceValue_QQ(fdc, local_cell_vector, scale);
            GetFunctional()->FaceValue_UQ(fdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().FaceEquation_QTT(fdc, local_cell_vector, scale, scale);

          this->GetPDE().FaceEquation_UQ(fdc, local_cell_vector, scale, scale);

          this->GetPDE().FaceEquation_QQ(fdc, local_cell_vector, scale, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellFaceRhs");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::BoundaryRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("boundary") != std::string::npos)
            GetFunctional()->BoundaryValue_U(fdc, local_cell_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (_aux_functionals[_functional_for_ee_num]->GetType().find(
              "boundary") != std::string::npos)
            _aux_functionals[_functional_for_ee_num]->BoundaryValue_U(fdc,
                local_cell_vector, scale);
        }
        else if (this->GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          this->GetPDE().BoundaryEquation_QT(fdc, local_cell_vector, scale,
              scale);
        }
        else if (this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("boundary") != std::string::npos)
          {
            GetFunctional()->BoundaryValue_UU(fdc, local_cell_vector, scale);
            GetFunctional()->BoundaryValue_QU(fdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().BoundaryEquation_UU(fdc, local_cell_vector, scale,
              scale);
          this->GetPDE().BoundaryEquation_QU(fdc, local_cell_vector, scale,
              scale);
        }
        else if (this->GetType() == "gradient")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("boundary") != std::string::npos)
            GetFunctional()->BoundaryValue_Q(fdc, local_cell_vector, scale);

          scale *= -1;
          this->GetPDE().BoundaryEquation_Q(fdc, local_cell_vector, scale,
              scale);
        }
        else if (this->GetType() == "hessian")
        {
          // state values in quadrature points
          if (GetFunctional()->GetType().find("boundary") != std::string::npos)
          {
            GetFunctional()->BoundaryValue_QQ(fdc, local_cell_vector, scale);
            GetFunctional()->BoundaryValue_UQ(fdc, local_cell_vector, scale);
          }

          scale *= -1;
          this->GetPDE().BoundaryEquation_QTT(fdc, local_cell_vector, scale,
              scale);
          this->GetPDE().BoundaryEquation_UQ(fdc, local_cell_vector, scale,
              scale);
          this->GetPDE().BoundaryEquation_QQ(fdc, local_cell_vector, scale,
              scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::CellBoundaryRhs");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellMatrix(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {

        if (this->GetType() == "state" || this->GetType() == "tangent")
        {
          // state values in quadrature points
          this->GetPDE().CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee"
            || this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().CellMatrix_T(cdc, local_entry_matrix, scale,
              scale_ico);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          // control values in quadrature points
          this->GetPDE().ControlCellMatrix(cdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellTimeMatrix(
          const DATACONTAINER& cdc, FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state" || this->GetType() == "tangent")
        {
          // state values in quadrature points
          this->GetPDE().CellTimeMatrix(cdc, local_entry_matrix);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee"
            || this->GetType() == "adjoint_hessian")
        {
          this->GetPDE().CellTimeMatrix_T(cdc, local_entry_matrix);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellTimeMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::CellTimeMatrixExplicit(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state" || this->GetType() == "tangent")
        {
          this->GetPDE().CellTimeMatrixExplicit(cdc, local_entry_matrix);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_hessian")
        {
          this->GetPDE().CellTimeMatrixExplicit_T(cdc, local_entry_matrix);
        }
        else if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellTimeMatrixExplicit");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellTimeMatrixExplicit");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state" || this->GetType() == "tangent")
        {
          // state values in face quadrature points
          this->GetPDE().FaceMatrix(fdc, local_entry_matrix, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee"
            || this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().FaceMatrix_T(fdc, local_entry_matrix, scale,
              scale_ico);
        }
//        else if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
//        {
//          // control values in quadrature points
//          this->GetPDE().ControlFaceMatrix(fdc, local_entry_matrix);
//        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonFaceMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::InterfaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          this->GetPDE().InterfaceMatrix(fdc, local_entry_matrix, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee"
            || this->GetType() == "adjoint_hessian")
        {
          this->GetPDE().InterfaceMatrix_T(fdc, local_entry_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonInterfaceMatrix");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::BoundaryMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_cell_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state" || this->GetType() == "tangent")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryMatrix(fdc, local_cell_matrix, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint"
            || this->GetType() == "adjoint_for_ee"
            || this->GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryMatrix_T(fdc, local_cell_matrix, scale,
              scale_ico);
        }
//        else if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
//        {
//          // control values in quadrature points
//          this->GetPDE().ControlBoundaryMatrix(fdc, local_cell_matrix);
//        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblemContainer::NewtonCellBoundaryMatrix");
        }

      }

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ComputeLocalControlConstraints(
        VECTOR& constraints,
        const std::map<std::string, const dealii::Vector<double>*> &/*values*/,
        const std::map<std::string, const VECTOR*> &block_values)
    {
      if (this->GetType() == "constraints")
      {
        if (this->GetSpaceTimeHandler()->GetNLocalConstraints() != 0)
        {
          const VECTOR& control = *GetBlockVector(block_values, "control");
          this->GetConstraints()->EvaluateLocalControlConstraints(control,
              constraints);
        }
      }
      else
      {
        throw DOpEException("Wrong problem type" + this->GetType(),
            "OptProblemContainer::ComputeLocalConstraints");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    std::string
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDoFType() const
    {
      if (this->GetType() == "state" || this->GetType() == "adjoint"
          || this->GetType() == "adjoint_for_ee" || this->GetType() == "tangent"
          || this->GetType() == "adjoint_hessian")
      {
        return "state";
      }
      else if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
          || (this->GetType() == "hessian_inverse"))
      {
        return "control";
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDoFType");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const FE<dealdim, dealdim>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFESystem() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || this->GetType() == "adjoint_for_ee" || this->GetType() == "tangent"
          || this->GetType() == "adjoint_hessian")
      {
        return this->GetSpaceTimeHandler()->GetFESystem("state");
      }
      else if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
          || (this->GetType() == "global_constraint_gradient"))
      {
#if dope_dimension > 0
        if(dopedim == dealdim)
        return this->GetSpaceTimeHandler()->GetFESystem("control");
        else
        throw DOpEException("Non matching dimensions!","OptProblemContainer::GetFESystem");
#else
        return this->GetSpaceTimeHandler()->GetFESystem("state");
#endif
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetFESystem");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    UpdateFlags
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetUpdateFlags() const
    {

      UpdateFlags r;
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[this->GetTypeNum()]->GetUpdateFlags();
      }
      else if (this->GetType().find("functional") != std::string::npos)
      {
        r = this->GetFunctional()->GetUpdateFlags();
      }
      else if (this->GetType().find("constraints") != std::string::npos)
      {
        r = this->GetConstraints()->GetUpdateFlags();
      }
      else if (this->GetType() == "functional_for_ee")
      {
        r = _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetUpdateFlags();
        if (this->GetType() == "adjoint_hessian" || this->GetType() == "adjoint"
            || (this->GetType() == "hessian")
            || (this->GetType() == "gradient"))
        {
          r = r | this->GetFunctional()->GetUpdateFlags();
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          r = r | _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    UpdateFlags
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[this->GetTypeNum()]->GetFaceUpdateFlags();
      }
      else if (this->GetType().find("functional") != std::string::npos)
      {
        r = this->GetFunctional()->GetFaceUpdateFlags();
      }
      else if (this->GetType().find("constraints") != std::string::npos)
      {
        r = this->GetConstraints()->GetFaceUpdateFlags();
      }
      else if (this->GetType() == "functional_for_ee")
      {
        r = _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetFaceUpdateFlags();
        if (this->GetType() == "adjoint_hessian" || this->GetType() == "adjoint"
            || (this->GetType() == "hessian"))
        {
          r = r | this->GetFunctional()->GetFaceUpdateFlags();
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          r = r
              | _aux_functionals[_functional_for_ee_num]->GetFaceUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    std::string
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctionalType() const
    {
      if (this->GetType() == "aux_functional")
      {
        return _aux_functionals[this->GetTypeNum()]->GetType();
      }
      else if (this->GetType() == "functional_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->GetType();
      }
      return GetFunctional()->GetType();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    std::string
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctionalName() const
    {
      if (this->GetType() == "aux_functional")
      {
        return _aux_functionals[this->GetTypeNum()]->GetName();
      }
      else if (this->GetType() == "functional_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->GetName();
      }
      return GetFunctional()->GetName();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    std::string
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetConstraintType() const
    {
      return GetConstraints()->GetType();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetTime(double time,
        const TimeIterator& interval)
    {
      GetSpaceTimeHandler()->SetInterval(interval);
      //      GetSpaceTimeHandler()->SetTimeDoFNumber(time_point);

      { //Zeit an Dirichlet Werte uebermitteln
        for (unsigned int i = 0;
            i < _transposed_control_gradient_dirichlet_values.size(); i++)
          _transposed_control_gradient_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0;
            i < _transposed_control_hessian_dirichlet_values.size(); i++)
          _transposed_control_hessian_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i < _primal_dirichlet_values.size(); i++)
          _primal_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i < _tangent_dirichlet_values.size(); i++)
          _tangent_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i < _control_dirichlet_values.size(); i++)
          _control_dirichlet_values[i]->SetTime(time);
        //Functionals
        GetFunctional()->SetTime(time);
        for (unsigned int i = 0; i < _aux_functionals.size(); i++)
          _aux_functionals[i]->SetTime(time);
        //PDE
        this->GetPDE().SetTime(time);
      }
      //Update Auxiliary Control, State and Constraint Vectors
      {
        typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
            _auxiliary_controls.begin();
        for (; it != _auxiliary_controls.end(); it++)
        {
          it->second->SetTime(time, interval);
        }
      }
      {
        typename std::map<std::string, const StateVector<VECTOR> *>::iterator it =
            _auxiliary_state.begin();
        for (; it != _auxiliary_state.end(); it++)
        {
          it->second->SetTime(time, interval);
        }
      }
      {
        typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
            _auxiliary_constraints.begin();
        for (; it != _auxiliary_constraints.end(); it++)
        {
          it->second->SetTime(time, interval);
        }
      }

    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      if (this->GetType() == "state" || this->GetType() == "tangent"
          || this->GetType() == "adjoint_for_ee" || this->GetType() == "adjoint"
          || this->GetType() == "adjoint_hessian")
      {
        this->GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
      }
      else if ((this->GetType() == "gradient")
          || (this->GetType() == "hessian"))
      {
#if  dope_dimension > 0
        this->GetSpaceTimeHandler()->ComputeControlSparsityPattern(sparsity);
#else
        throw DOpEException("Wrong dimension",
            "OptProblemContainer::ComputeSparsityPattern");
#endif
      }
      else
      {
        throw DOpEException("Unknown type " + this->GetType(),
            "OptProblemContainer::ComputeSparsityPattern");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::IsFeasible(
        const ConstraintVector<VECTOR>& g) const
    {
      return this->GetConstraints()->IsFeasible(g);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::PostProcessConstraints(
        ConstraintVector<VECTOR>& g) const
    {
      return this->GetConstraints()->PostProcessConstraints(g);
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AddAuxiliaryControl(
        const ControlVector<VECTOR>* c, std::string name)
    {
      if (_auxiliary_controls.find(name) != _auxiliary_controls.end())
      {
        throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "OptProblemContainer::AddAuxiliaryControl");
      }
      _auxiliary_controls.insert(
          std::pair<std::string, const ControlVector<VECTOR>*>(name, c));
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AddAuxiliaryState(
        const StateVector<VECTOR>* c, std::string name)
    {
      if (_auxiliary_state.find(name) != _auxiliary_state.end())
      {
        throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "OptProblemContainer::AddAuxiliaryState");
      }
      _auxiliary_state.insert(
          std::pair<std::string, const StateVector<VECTOR>*>(name, c));
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AddAuxiliaryConstraint(
        const ConstraintVector<VECTOR>* c, std::string name)
    {
      if (_auxiliary_constraints.find(name) != _auxiliary_constraints.end())
      {
        throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "OptProblemContainer::AddAuxiliaryConstraint");
      }
      _auxiliary_constraints.insert(
          std::pair<std::string, const ConstraintVector<VECTOR>*>(name, c));
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const ControlVector<VECTOR>*
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetAuxiliaryControl(
        std::string name) const
    {
      typename std::map<std::string, const ControlVector<VECTOR> *>::const_iterator it =
          _auxiliary_controls.find(name);
      if (it == _auxiliary_controls.end())
      {
        throw DOpEException("Could not find Data with name " + name,
            "OptProblemContainer::GetAuxiliaryControl");
      }
      return it->second;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const StateVector<VECTOR>*
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetAuxiliaryState(
        std::string name) const
    {
      typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
          _auxiliary_state.find(name);
      if (it == _auxiliary_state.end())
      {
        throw DOpEException("Could not find Data with name " + name,
            "OptProblemContainer::GetAuxiliaryState");
      }
      return it->second;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::DeleteAuxiliaryControl(
        std::string name)
    {
      typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
          _auxiliary_controls.find(name);
      if (it == _auxiliary_controls.end())
      {
        throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "OptProblemContainer::DeleteAuxiliaryControl");
      }
      _auxiliary_controls.erase(it);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::DeleteAuxiliaryState(
        std::string name)
    {
      typename std::map<std::string, const StateVector<VECTOR> *>::iterator it =
          _auxiliary_state.find(name);
      if (it == _auxiliary_state.end())
      {
        throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "OptProblemContainer::DeleteAuxiliaryState");
      }
      _auxiliary_state.erase(it);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::DeleteAuxiliaryConstraint(
        std::string name)
    {
      typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
          _auxiliary_constraints.find(name);
      if (it == _auxiliary_constraints.end())
      {
        throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "OptProblemContainer::DeleteAuxiliaryConstraint");
      }
      _auxiliary_constraints.erase(it);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    FUNCTIONAL*
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctional()
    {
      if (this->GetType() == "aux_functional"
          || this->GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //    return _aux_functionals[this->GetTypeNum()];
      }
      return _functional;

    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const FUNCTIONAL*
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctional() const
    {
      if (this->GetType() == "aux_functional"
          || this->GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //       return _aux_functionals[this->GetTypeNum()];
      }
      return _functional;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::HasFaces() const
    {
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        return _aux_functionals[this->GetTypeNum()]->HasFaces();
      }
      else if (this->GetType().find("functional") != std::string::npos)
      {
        return this->GetFunctional()->HasFaces();
      }
      else if (this->GetType().find("constraint") != std::string::npos)
      {
        return this->GetConstraints()->HasFaces();
      }
      else
      {
        if ((this->GetType() == "state") || (this->GetType() == "tangent")
            || (this->GetType() == "gradient"))
        {
          return this->GetPDE().HasFaces();
        }
        else if ((this->GetType() == "adjoint")
            || (this->GetType() == "adjoint_hessian")
            || (this->GetType() == "hessian"))
        {
          return this->GetPDE().HasFaces() || this->GetFunctional()->HasFaces();
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          return this->GetPDE().HasFaces()
              || _aux_functionals[_functional_for_ee_num]->HasFaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
              "OptProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::HasPoints() const
    {
      if (this->GetType().find("constraint") != std::string::npos
          || (this->GetType() == "functional")
          || this->GetType() == "aux_functional" || (this->GetType() == "state")
          || (this->GetType() == "tangent"))
      {
        // We dont need PointRhs in this cases.
        return false;
      }
      else if ((this->GetType() == "adjoint")
          || (this->GetType() == "adjoint_hessian")
          || (this->GetType() == "hessian"))
      {
        return this->GetFunctional()->HasPoints();
      }
      else if (this->GetType() == "adjoint_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->HasPoints();
      }
      else if (this->GetType() == "gradient")
      {
        return this->GetFunctional()->HasPoints();
      }
      else
      {
        throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
            "OptProblem::HasPoints");
      }
    }
//    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::HasInterfaces() const
    {
      if (this->GetType().find("aux_functional") != std::string::npos)
      {
        return false;
      }
      else if (this->GetType().find("functional") != std::string::npos)
      {
        return false;
      }
      else if (this->GetType().find("constraint") != std::string::npos)
      {
        return false;
      }
      else
      {
        if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee"
            || (this->GetType() == "tangent") || (this->GetType() == "gradient")
            || (this->GetType() == "adjoint")
            || (this->GetType() == "adjoint_hessian")
            || (this->GetType() == "hessian"))
        {
          return this->GetPDE().HasInterfaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
              "OptProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetControlDirichletBoundaryColors(
        unsigned int color, const std::vector<bool>& comp_mask,
        const DOpEWrapper::Function<dealdim>* values)
    {
      assert(values);

      unsigned int comp = _control_dirichlet_colors.size();
      for (unsigned int i = 0; i < _control_dirichlet_colors.size(); ++i)
      {
        if (_control_dirichlet_colors[i] == color)
        {
          comp = i;
          break;
        }
      }
      if (comp != _control_dirichlet_colors.size())
      {
        std::stringstream s;
        s << "ControlDirichletColor" << color << " has multiple occurences !";
        throw DOpEException(s.str(),
            "OptProblemContainer::SetControlDirichletBoundary");
      }
      _control_dirichlet_colors.push_back(color);
      _control_dirichlet_comps.push_back(comp_mask);
      _control_dirichlet_values.push_back(values);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetDirichletBoundaryColors(
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
            "OptProblemContainer::SetDirichletBoundary");
      }
      _dirichlet_colors.push_back(color);
      _dirichlet_comps.push_back(comp_mask);
      PrimalDirichletData<DD, VECTOR, dopedim, dealdim>* data =
          new PrimalDirichletData<DD, VECTOR, dopedim, dealdim>(*values);
      _primal_dirichlet_values.push_back(data);
      TangentDirichletData<DD, VECTOR, dopedim, dealdim>* tdata =
          new TangentDirichletData<DD, VECTOR, dopedim, dealdim>(*values);
      _tangent_dirichlet_values.push_back(tdata);

      if (values->NeedsControl())
      {
        _control_transposed_dirichlet_colors.push_back(color);
        TransposedGradientDirichletData<DD, VECTOR, dopedim, dealdim> * gdata =
            new TransposedGradientDirichletData<DD, VECTOR, dopedim, dealdim>(
                *values);
        _transposed_control_gradient_dirichlet_values.push_back(gdata);
        TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim> * hdata =
            new TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim>(
                *values);
        _transposed_control_hessian_dirichlet_values.push_back(hdata);
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<unsigned int>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletColors() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
      {
        return _dirichlet_colors;
      }
      else if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
          || (this->GetType() == "global_constraint_gradient"))
      {
        return _control_dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDirichletColors");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<unsigned int>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletColors() const
    {
      if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
      {
        return _control_transposed_dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetTransposedDirichletColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<bool>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletCompMask(
        unsigned int color) const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
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
              "OptProblemContainer::GetDirichletCompMask");
        }
        return _dirichlet_comps[comp];
      }
      else if ((this->GetType() == "gradient")
          || (this->GetType() == "hessian"))
      {
        unsigned int comp = _control_dirichlet_colors.size();
        for (unsigned int i = 0; i < _control_dirichlet_colors.size(); ++i)
        {
          if (_control_dirichlet_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp == _control_dirichlet_colors.size())
        {
          std::stringstream s;
          s << "ControlDirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
              "OptProblemContainer::GetDirichletCompMask");
        }
        return _control_dirichlet_comps[comp];
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDirichletCompMask");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<bool>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletCompMask(
        unsigned int color) const
    {
      if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
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
              "OptProblemContainer::GetDirichletCompMask");
        }
        return _dirichlet_comps[comp];
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetTransposedDirichletCompMask");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const Function<dealdim>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletValues(
        unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {

      unsigned int col = _dirichlet_colors.size();
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
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
              "OptProblemContainer::GetDirichletValues");
        }
      }
      else if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        col = _control_dirichlet_colors.size();
        for (unsigned int i = 0; i < _control_dirichlet_colors.size(); ++i)
        {
          if (_control_dirichlet_colors[i] == color)
          {
            col = i;
            break;
          }
        }
        if (col == _control_dirichlet_colors.size())
        {
          std::stringstream s;
          s << "ControlDirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
              "OptProblemContainer::GetDirichletValues");
        }
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDirichletValues");
      }

      if (this->GetType() == "state")
      {
        _primal_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_primal_dirichlet_values[col]);
      }
      else if (this->GetType() == "tangent")
      {
        _tangent_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_tangent_dirichlet_values[col]);
      }
      else if (this->GetType() == "adjoint"
          || this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "adjoint_hessian"))
      {
        return *(_zero_dirichlet_values);
      }
      else if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        return *(_control_dirichlet_values[col]);
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDirichletValues");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const TransposedDirichletDataInterface<dopedim, dealdim>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletValues(
        unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {
      unsigned int col = _control_transposed_dirichlet_colors.size();
      if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        for (unsigned int i = 0;
            i < _control_transposed_dirichlet_colors.size(); ++i)
        {
          if (_control_transposed_dirichlet_colors[i] == color)
          {
            col = i;
            break;
          }
        }
        if (col == _control_transposed_dirichlet_colors.size())
        {
          std::stringstream s;
          s << "TransposedControlDirichletColor" << color
              << " has not been found !";
          throw DOpEException(s.str(),
              "OptProblemContainer::GetTransposedDirichletValues");
        }
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetTransposedDirichletValues");
      }

      if (this->GetType() == "gradient")
      {
        _transposed_control_gradient_dirichlet_values[col]->ReInit(param_values,
            domain_values, color);
        return *(_transposed_control_gradient_dirichlet_values[col]);
      }
      else if (this->GetType() == "hessian")
      {
        _transposed_control_hessian_dirichlet_values[col]->ReInit(param_values,
            domain_values, color);
        return *(_transposed_control_hessian_dirichlet_values[col]);
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetTransposedDirichletValues");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<unsigned int>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetBoundaryEquationColors() const
    {
      if (this->GetType() == "state" || this->GetType() == "tangent")
      {
        return _state_boundary_equation_colors;
      }
      else if (this->GetType() == "adjoint"
          || this->GetType() == "adjoint_for_ee"
          || this->GetType() == "adjoint_hessian")
      {
        return _adjoint_boundary_equation_colors;
      }
      else if (this->GetType() == "gradient" || (this->GetType() == "hessian")
          || (this->GetType() == "global_constraint_gradient"))
      {
        return _control_boundary_equation_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetBoundaryEquationColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetControlBoundaryEquationColors(
        unsigned int color)
    {
      { //Control Boundary Equation colors are simply inserted
        unsigned int comp = _control_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _control_boundary_equation_colors.size();
            ++i)
        {
          if (_control_boundary_equation_colors[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != _control_boundary_equation_colors.size())
        {
          std::stringstream s;
          s << "Boundary Equation Color" << color
              << " has multiple occurences !";
          throw DOpEException(s.str(),
              "OptProblemContainer::SetControlBoundaryEquationColors");
        }
        _control_boundary_equation_colors.push_back(color);
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetBoundaryEquationColors(
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
              "OptProblemContainer::SetBoundaryEquationColors");
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

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<unsigned int>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetBoundaryFunctionalColors() const
    {
      if (this->GetType() == "cost_functional"
          || this->GetType() == "aux_functional"
          || this->GetType() == "functional_for_ee") //fixme: what about error_evaluation?
      {
        return _boundary_functional_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetBoundaryFunctionalColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    void
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetBoundaryFunctionalColors(
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
              "OptProblemContainer::SetBoundaryFunctionalColors");
        }
        _boundary_functional_colors.push_back(color);
      }
      { //For the  adjoint they are addeed  to the boundary equation colors
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

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    unsigned int
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetControlNBlocks() const
    {
      return this->GetPDE().GetControlNBlocks();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    unsigned int
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetStateNBlocks() const
    {
      return this->GetPDE().GetStateNBlocks();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    unsigned int
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetNBlocks() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint_for_ee")
          || (this->GetType() == "adjoint") || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
      {
        return this->GetStateNBlocks();
      }
      else if ((this->GetType() == "gradient")
          || (this->GetType() == "hessian"))
      {
        return this->GetControlNBlocks();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetNBlocks");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    unsigned int
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDoFsPerBlock(
        unsigned int b) const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || (this->GetType() == "adjoint_for_ee")
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock(b);
      }
      else if ((this->GetType() == "gradient")
          || (this->GetType() == "hessian"))
      {
        return GetSpaceTimeHandler()->GetControlDoFsPerBlock(b);
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const std::vector<unsigned int>&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDoFsPerBlock() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || (this->GetType() == "adjoint_for_ee")
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      }
      else if ((this->GetType() == "gradient")
          || (this->GetType() == "hessian"))
      {
        return GetSpaceTimeHandler()->GetControlDoFsPerBlock();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    const dealii::ConstraintMatrix&
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDoFConstraints() const
    {
      if ((this->GetType() == "state") || (this->GetType() == "adjoint")
          || (this->GetType() == "adjoint_for_ee")
          || (this->GetType() == "tangent")
          || (this->GetType() == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateDoFConstraints();
      }
      else if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
          || (this->GetType() == "global_constraint_gradient"))
      {
        return GetSpaceTimeHandler()->GetControlDoFConstraints();
      }
      else
      {
        throw DOpEException("Unknown Type:" + this->GetType(),
            "OptProblemContainer::GetDoFConstraints");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::NeedTimeFunctional() const
    {
      if (this->GetType() == "cost_functional")
        return GetFunctional()->NeedTime();
      else if (this->GetType() == "aux_functional")
        return _aux_functionals[this->GetTypeNum()]->NeedTime();
      else if (this->GetType() == "functional_for_ee")
        return _aux_functionals[_functional_for_ee_num]->NeedTime();
      else
        throw DOpEException("Not implemented",
            "OptProblemContainer::NeedTimeFunctional");
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
      template<int, int> class DH>
    bool
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::HasControlInDirichletData() const
    {
      return (!_control_transposed_dirichlet_colors.empty());
    }

}
#endif
