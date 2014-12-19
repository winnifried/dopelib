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

#ifndef PDEProblemContainer_H_
#define PDEProblemContainer_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "functionalinterface.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "function_wrapper.h"
#include "statespacetimehandler.h"
#include "primaldirichletdata.h"
#include "elementdatacontainer.h"
#include "facedatacontainer.h"
#include "stateproblem.h"
#include "problemcontainer_internal.h"
#include <deal.II/multigrid/mg_dof_handler.h>
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

// Multi-level routines (step-16 in deal.II)
#include <deal.II/multigrid/mg_dof_handler.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>



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

  /**
   * Container class for all stationary PDE problems.
   * This class collects all problem depended data needed to 
   * calculate the solution to the PDE.
   *
   * @tparam PDE               The description of the PDE, see PDEInterface for details.
   * @tparam DD                The description of the Dirichlet data, see 
   *                           DirichletDataInterface for details.
   * @tparam SPARSITYPATTERN   The sparsity pattern to be used in the stiffness matrix.
   * @tparam VECTOR            The vector type in which the coordinate vector of the 
   *                           solution is to be stored.
   * @tparam dealdim           The dimension of the domain in which the PDE is considered.
   * @tparam FE                The finite element under consideration.
   * @tparam DH                The spatial DoFHandler to be used when evaluating the 
   *                           weak form.
   */
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
      /**
       * Returns a description of the PDE
       */
        StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
        GetStateProblem()
        {
          if (state_problem_ == NULL)
          {
            state_problem_ = new StateProblem<
                PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim,
                    FE, DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(*this,
                this->GetPDE());
          }
          return *state_problem_;
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
          OutputHandler_ = OH;
        }

        /******************************************************/

        void
        RegisterExceptionHandler(DOpEExceptionHandler<VECTOR>* OH)
        {
          ExceptionHandler_ = OH;
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
         * This function returns a functional value on a element.
         * Different types of functionals
         * have been implemented so far: `cost_functional', `aux_functional'
         * and 'functional_for_ee'.
         * The first one is needed for optimization problems. The second one
         * can be used for the computation of arbitrary functionals that
         * are defined on elements, e.g., drag and lift computation. Or
         * computations of deflections and deformations. The last
         * one is needed if one wants to use the goal oriented adaptive
         * grid refinement based on the DWR method.
         *
         *
         * @template DATACONTAINER    Class of the datacontainer, distinguishes
         *                            between hp- and classical case.
         *
         * @param edc                 A DataContainer holding all the needed information
         *                            of the face.
         */
        template<typename DATACONTAINER>
          double
          ElementFunctional(const DATACONTAINER& edc);

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
        /**
         * This function returns a functional value that is computed entirely
	 * out of the knowledge of the coordinate vectors of state and control.
	 *
	 * No integration routine is implemented inbetween!
         */
        double
        AlgebraicFunctional(
            const std::map<std::string, const dealii::Vector<double>*> &values,
            const std::map<std::string, const VECTOR*> &block_values);

        /******************************************************/

        /**
         * Computes the value of the element equation which corresponds
         * to the residuum in nonlinear cases. This function is the basis
         * for all stationary examples and unsteady configurations as well.
         * However, in unsteady computations one has to differentiate
         * between explicit (diffusion, convection) and implicit terms
         * (pressure, incompressibility). For that reason a second
         * function ElementEquationImplicit also exists.
         *
         * If no differentiation between explicit and implicit terms is needed
         * this function should be used.
         *
         * @template DATACONTAINER        Class of the datacontainer in use, distinguishes
         *                                between hp- and classical case.
         *
         * @param edc                     A DataContainer holding all the needed information
         *                                of the element
         * @param local_vector        This vector contains the locally computed values of the element equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<typename DATACONTAINER>
          void
          ElementEquation(const DATACONTAINER& edc,
              dealii::Vector<double> &local_vector, double scale,
              double scale_ico);

        /******************************************************/

        /******************************************************/

        /**
         * This function has the same functionality as the ElementEquation function.
         * It is needed for time derivatives when working with
         * time dependent problems.
         */
        template<typename DATACONTAINER>
          void
          ElementTimeEquation(const DATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale = 1.);

        /******************************************************/

        /**
         * This function has the same functionality as the ElementTimeEquation function.
         * It is needed for problems with nonlinearities in the time derivative, like
	 * fluid-structure interaction problems and has
         * special structure.
         */
        template<typename DATACONTAINER>
          void
          ElementTimeEquationExplicit(const DATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the right-hand side of the problem at hand.
         *
         * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
         *                                 between hp- and classical case.
         *
         * @param edc                      A DataContainer holding all the needed information
         * @param local_vector        This vector contains the locally computed values of the element equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        template<typename DATACONTAINER>
          void
          ElementRhs(const DATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale = 1.);

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
         * Computes the value of the element matrix which is derived
         * by computing the directional derivatives of the residuum equation of the PDE
         * problem under consideration.
         *
         * The differentiation between explicit and implicit terms is
         * equivalent to the ElementEquation. We refer to its documentation.
         *
         * Moreover, you find an extensive explication in the
         * time step algorithms, e.g., backward_euler_problem.h.
         *
         * @template DATACONTAINER      Class of the datacontainer in use, distinguishes
         *                              between hp- and classical case.
         *
         * @param edc                   A DataContainer holding all the needed information
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
          ElementMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
              double scale_ico = 1.);

        /******************************************************/

        /**
         * Computes the value of the element matrix which is derived
         * by computing the directional derivatives of the time residuum of the PDE
         * problem under consideration.
         *
         * The differentiation between explicit and implicit terms is
         * equivalent to the ElementTimeEquation. We refer to its documentation.
         *
         * Moreover, you find an extensive explication in the
         * time step algorithms, e.g., backward_euler_problem.h.
         */
        template<typename DATACONTAINER>
          void
          ElementTimeMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        /******************************************************/

        /**
         * Computes the value of the element matrix which is derived
         * by computing the directional derivatives of the time residuum of the PDE
         * problem under consideration.
         *
         * This function is only needed for fluid-structure interaction problems.
         * Please ask Thomas Wick WHY and HOW to use this function.
         *
         */
        template<typename DATACONTAINER>
          void
          ElementTimeMatrixExplicit(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        /******************************************************/

        /**
         * Computes the value of face on a element.
         * It has the same functionality as ElementEquation. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          FaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the product of two different finite elements
         * on a interior face. It has the same functionality as ElementEquation.
         * We refer to its documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          InterfaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the value of face on a element.
         * It has the same functionality as ElementRhs. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          FaceRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of face on a element.
         * It has the same functionality as ElementMatrix. We refer to its
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
         * ElementMatrix. We refer to its documentation.
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
         * It has the same functionality as ElementEquation. We refer to its
         * documentation.
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the value of the boundary on a element.
         * It has the same functionality as ElementRhs. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the boundary on a element.
         * It has the same functionality as ElementMatrix. We refer to its
         * documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_matrix, double scale = 1.,
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
          initial_values_ = values;
        }
        const dealii::Function<dealdim>&
        GetInitialValues() const
        {
          return *initial_values_;
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
            FunctionalInterface<ElementDataContainer, FaceDataContainer, DH,
                VECTOR, dealdim>* F)
        {
          aux_functionals_.push_back(F);
          if (functional_position_.find(F->GetName())
              != functional_position_.end())
          {
            throw DOpEException(
                "You cant add two functionals with the same name.",
                "PDEProblemContainer::AddFunctional");
          }
          functional_position_[F->GetName()] = aux_functionals_.size() - 1;
          //-1 because we are in the pde case and have therefore no cost functional
        }

        /******************************************************/

        /**
         * Through this function one sets the functional for the
         * error estimation. The name given by functional_name is
         * looked up in aux_functionals_, so the function assumes
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
            if (aux_functionals_[i]->GetName() == functional_name)
            {
              //if the names match, we have found our functional.
              found = true;
              functional_for_ee_num_ = i;
              break;
            }
          }
          //If we have not found a functional with the given name,
          //we throw an error.
          if (!found)
          {
            throw DOpEException(
                "Can't find functional " + functional_name
                    + " in aux_functionals_",
                "PDEProblemContainer::SetFunctionalForErrorEstimation");
          }
        }

        /******************************************************/

        unsigned int
        GetNFunctionals() const
        {
          return aux_functionals_.size();
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
          assert(ExceptionHandler_);
          return ExceptionHandler_;
        }

        /******************************************************/

        DOpEOutputHandler<VECTOR>*
        GetOutputHandler()
        {
          assert(OutputHandler_);
          return OutputHandler_;
        }

        /******************************************************/

        void
        SetTime(double time, 
		unsigned int time_dof_number,
		const TimeIterator& interval);

        /******************************************************/

        const StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>*
        GetSpaceTimeHandler() const
        {
          return STH_;
        }

        /******************************************************/

        StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>*
        GetSpaceTimeHandler()
        {
          return STH_;
        }

        /******************************************************/

        void
        ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

        /******************************************************/
      /*
       * Experimental status:
       * Needed for MG prec
       */
      void
        ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const;

      /******************************************************/
      /*
       * Experimental status:
       * Needed for MG prec
       */
        void
        ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const;

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
          return functional_position_;
        }

        /******************************************************/
      private:
        DOpEExceptionHandler<VECTOR>* ExceptionHandler_;
        DOpEOutputHandler<VECTOR>* OutputHandler_;

        std::string algo_type_;

        std::vector<
            FunctionalInterface<ElementDataContainer, FaceDataContainer, DH,
                VECTOR, dealdim>*> aux_functionals_;
        std::map<std::string, unsigned int> functional_position_;

        unsigned int functional_for_ee_num_;
        StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim> * STH_;

        std::vector<unsigned int> dirichlet_colors_;
        std::vector<std::vector<bool> > dirichlet_comps_;
        std::vector<PrimalDirichletData<DD, VECTOR, dealdim>*> primal_dirichlet_values_;
        const dealii::Function<dealdim>* zero_dirichlet_values_;

        const dealii::Function<dealdim>* initial_values_;

        std::vector<unsigned int> state_boundary_equation_colors_;
        std::vector<unsigned int> adjoint_boundary_equation_colors_;

        std::vector<unsigned int> boundary_functional_colors_;

        StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>* state_problem_;

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
        ProblemContainerInternal<PDE>(pde), STH_(&STH), state_problem_(NULL)
    {
      ExceptionHandler_ = NULL;
      OutputHandler_ = NULL;
      zero_dirichlet_values_ = new ZeroFunction<dealdim>(
          this->GetPDE().GetStateNComponents());
      algo_type_ = "";
      functional_for_ee_num_ = dealii::numbers::invalid_unsigned_int;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::~PDEProblemContainer()
    {
      if (zero_dirichlet_values_ != NULL)
      {
        delete zero_dirichlet_values_;
      }
      for (unsigned int i = 0; i < primal_dirichlet_values_.size(); i++)
      {
        if (primal_dirichlet_values_[i] != NULL)
          delete primal_dirichlet_values_[i];
      }
      if (state_problem_ != NULL)
      {
        delete state_problem_;
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ReInit(
        std::string algo_type)
    {
      if (state_problem_ != NULL)
      {
        delete state_problem_;
        state_problem_ = NULL;
      }

      if (algo_type_ != algo_type && algo_type_ != "")
      {
        throw DOpEException("Conflicting Algorithms!",
            "PDEProblemContainer::ReInit");
      }
      else
      {
        algo_type_ = algo_type;
        this->SetTypeInternal("");

        if (algo_type_ == "reduced")
        {
          GetSpaceTimeHandler()->ReInit(this->GetPDE().GetStateNBlocks(),
              this->GetPDE().GetStateBlockComponent());
        }
        else
        {
          throw DOpEException("Unknown Algorithm " + algo_type_,
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
        if (functional_for_ee_num_ != dealii::numbers::invalid_unsigned_int)
          aux_functionals_[functional_for_ee_num_]->SetProblemType(type);
      }
      //Nothing to do.
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementFunctional(
          const DATACONTAINER& edc)
      {

        if (this->GetType() == "cost_functional")
        {
          return 0;
        }
        else if (this->GetType() == "aux_functional")
        {
          // state values in quadrature points
          return aux_functionals_[this->GetTypeNum()]->ElementValue(edc);
        }
        else if (this->GetType() == "error_evaluation")
        {
          return aux_functionals_[functional_for_ee_num_]->ElementValue(edc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementFunctional");
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
        return aux_functionals_[this->GetTypeNum()]->PointValue(
            this->GetSpaceTimeHandler()->GetStateDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif aux_functional
      else if (this->GetType() == "error_evaluation")
      {
        return aux_functionals_[functional_for_ee_num_]->PointValue(
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
          return aux_functionals_[this->GetTypeNum()]->BoundaryValue(fdc);
        }
        else if (this->GetType() == "error_evaluation")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return aux_functionals_[functional_for_ee_num_]->BoundaryValue(fdc);
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
          return aux_functionals_[this->GetTypeNum()]->FaceValue(fdc);
        }
        else if (this->GetType() == "error_evaluation")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return aux_functionals_[functional_for_ee_num_]->FaceValue(fdc);
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
        return aux_functionals_[this->GetTypeNum()]->AlgebraicValue(
            param_values, domain_values);
      }
      else if (this->GetType() == "error_evaluation")
      //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
      {
        return aux_functionals_[functional_for_ee_num_]->AlgebraicValue(
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
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementEquation(
          const DATACONTAINER& edc, dealii::Vector<double> &local_vector,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementEquation(edc, local_vector, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().ElementEquation_U(edc, local_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementTimeEquation(
          const DATACONTAINER& edc, dealii::Vector<double> &local_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementTimeEquation(edc, local_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented",
              "OptProblem::ElementTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementTimeEquationExplicit(
          const DATACONTAINER& edc, dealii::Vector<double> &local_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementTimeEquationExplicit(edc, local_vector,
              scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented",
              "OptProblem::ElementTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation(fdc, local_vector, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().FaceEquation_U(fdc, local_vector, scale,
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
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().InterfaceEquation(fdc, local_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().InterfaceEquation_U(fdc, local_vector, scale,
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
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation(fdc, local_vector, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryEquation_U(fdc, local_vector, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementBoundaryEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementRhs(
          const DATACONTAINER& edc, dealii::Vector<double> &local_vector,
          double scale)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementRightHandSide(edc, local_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //TODO currently, pointvalue is not working for dual error estimation!
          //values of the derivative of the functional for error estimation.
          //Check, if we have to evaluate an integral over a domain.
          if (aux_functionals_[functional_for_ee_num_]->GetType().find("domain")
              != std::string::npos)
            aux_functionals_[functional_for_ee_num_]->ElementValue_U(edc,
                local_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementRhs");
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
        if (aux_functionals_[functional_for_ee_num_]->GetType().find("point")
            != std::string::npos)
          aux_functionals_[functional_for_ee_num_]->PointValue_U(
              this->GetSpaceTimeHandler()->GetStateDoFHandler(),
              this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
              domain_values, rhs_vector, scale);
      }
      else
      {
        throw DOpEException("Not implemented", "OptProblem::ElementRhs");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::FaceRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_vector, double scale)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().FaceRightHandSide(fdc, local_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (aux_functionals_[functional_for_ee_num_]->GetType().find("face")
              != std::string::npos)
            aux_functionals_[functional_for_ee_num_]->FaceValue_U(fdc,
                local_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementFaceRhs");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::BoundaryRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_vector, double scale)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryRightHandSide(fdc, local_vector, scale);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          if (aux_functionals_[functional_for_ee_num_]->GetType().find(
              "boundary") != std::string::npos)
            aux_functionals_[functional_for_ee_num_]->BoundaryValue_U(fdc,
                local_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementBoundaryRhs");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementMatrix(
          const DATACONTAINER& edc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementMatrix(edc, local_entry_matrix, scale, scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().ElementMatrix_T(edc, local_entry_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementTimeMatrix(
          const DATACONTAINER& edc, FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementTimeMatrix(edc, local_entry_matrix);
        }
        else if (this->GetType() == "dual_for_ee")
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ElementTimeMatrixExplicit(
          const DATACONTAINER& edc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {

        if (this->GetType() == "state")
        {
          // state values in quadrature points
          this->GetPDE().ElementTimeMatrixExplicit(edc, local_entry_matrix);
        }
        else if (this->GetType() == "dual_for_ee")
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementTimeMatrix");
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
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_matrix,
          double scale, double scale_ico)
      {
        if (this->GetType() == "state")
        {
          // state values in face quadrature points
          this->GetPDE().BoundaryMatrix(fdc, local_matrix, scale,
              scale_ico);
        }
        else if (this->GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          this->GetPDE().BoundaryMatrix_T(fdc, local_matrix, scale,
              scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::ElementBoundaryMatrix");
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
        r = aux_functionals_[this->GetTypeNum()]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetUpdateFlags();
        if (this->GetType() == "adjoint_for_ee"
            || this->GetType() == "error_evaluation")
        {
          if (functional_for_ee_num_ != dealii::numbers::invalid_unsigned_int)
            r = r | aux_functionals_[functional_for_ee_num_]->GetUpdateFlags();
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
        r = aux_functionals_[this->GetTypeNum()]->GetFaceUpdateFlags();
      }
      else
      {
        r = this->GetPDE().GetFaceUpdateFlags();
        if (this->GetType() == "adjoint_for_ee"
            || this->GetType() == "error_evaluation")
        {
          if (functional_for_ee_num_ != dealii::numbers::invalid_unsigned_int)
            r =
                r
                    | aux_functionals_[functional_for_ee_num_]->GetFaceUpdateFlags();
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
        return aux_functionals_[this->GetTypeNum()]->GetType();
      }
      else if (this->GetType() == "error_evaluation")
      {
        return aux_functionals_[functional_for_ee_num_]->GetType();
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
        return aux_functionals_[this->GetTypeNum()]->GetName();
      }
      else if (this->GetType() == "error_evaluation")
      {
        return aux_functionals_[functional_for_ee_num_]->GetName();
      }
      return "";
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::SetTime(
        double time,
	unsigned int time_dof_number, 
	const TimeIterator& interval)
    {
      GetSpaceTimeHandler()->SetInterval(interval);

      { //Zeit an Dirichlet Werte uebermitteln
        for (unsigned int i = 0; i < primal_dirichlet_values_.size(); i++)
          primal_dirichlet_values_[i]->SetTime(time);
        for (unsigned int i = 0; i < aux_functionals_.size(); i++)
          aux_functionals_[i]->SetTime(time);
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
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ComputeMGSparsityPattern(
        dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const
    {
      if (this->GetType() == "state")
      {
        this->GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
      }
      else
      {
        throw DOpEException("Unknown type " + this->GetType(),
            "PDEProblemContainer::ComputeMGSparsityPattern");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::ComputeMGSparsityPattern(
        dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const
    {
      if (this->GetType() == "state")
      {
        this->GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
      }
      else
      {
        throw DOpEException("Unknown type " + this->GetType(),
            "PDEProblemContainer::ComputeMGSparsityPattern");
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
        return aux_functionals_[this->GetTypeNum()]->HasFaces();
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
              || aux_functionals_[functional_for_ee_num_]->HasFaces();
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
        return aux_functionals_[functional_for_ee_num_]->HasPoints();
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

      unsigned int comp = dirichlet_colors_.size();
      for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
      {
        if (dirichlet_colors_[i] == color)
        {
          comp = i;
          break;
        }
      }
      if (comp != dirichlet_colors_.size())
      {
        std::stringstream s;
        s << "DirichletColor" << color << " has multiple occurrences !";
        throw DOpEException(s.str(),
            "PDEProblemContainer::SetDirichletBoundary");
      }
      dirichlet_colors_.push_back(color);
      dirichlet_comps_.push_back(comp_mask);
      PrimalDirichletData<DD, VECTOR, dealdim> *data = new PrimalDirichletData<
          DD, VECTOR, dealdim>(*values);
      primal_dirichlet_values_.push_back(data);
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, template<int, int> class FE, template<int, int> class DH>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>::GetDirichletColors() const
    {
      if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
      {
        return dirichlet_colors_;
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
        unsigned int comp = dirichlet_colors_.size();
        for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
        {
          if (dirichlet_colors_[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp == dirichlet_colors_.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::GetDirichletCompMask");
        }
        return dirichlet_comps_[comp];
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

      unsigned int col = dirichlet_colors_.size();
      if ((this->GetType() == "state") || this->GetType() == "adjoint_for_ee")
      {
        for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
        {
          if (dirichlet_colors_[i] == color)
          {
            col = i;
            break;
          }
        }
        if (col == dirichlet_colors_.size())
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
        primal_dirichlet_values_[col]->ReInit(param_values, domain_values,
            color);
        return *(primal_dirichlet_values_[col]);
      }
      else if (this->GetType() == "adjoint_for_ee"
          || (this->GetType() == "adjoint_hessian"))
      {
        return *(zero_dirichlet_values_);
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
        return state_boundary_equation_colors_;
      }
      else if (this->GetType() == "adjoint_for_ee")
      {
        return adjoint_boundary_equation_colors_;
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
        unsigned int comp = state_boundary_equation_colors_.size();
        for (unsigned int i = 0; i < state_boundary_equation_colors_.size();
            ++i)
        {
          if (state_boundary_equation_colors_[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != state_boundary_equation_colors_.size())
        {
          std::stringstream s;
          s << "Boundary Equation Color" << color
              << " has multiple occurences !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::SetBoundaryEquationColors");
        }
        state_boundary_equation_colors_.push_back(color);
      }
      { //For the  adjoint they are added with the boundary functional colors
        unsigned int comp = adjoint_boundary_equation_colors_.size();
        for (unsigned int i = 0; i < adjoint_boundary_equation_colors_.size();
            ++i)
        {
          if (adjoint_boundary_equation_colors_[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != adjoint_boundary_equation_colors_.size())
        {
          //Seems this color is already added, however it might have been a functional color
          //so we don't  do anything.
        }
        else
        {
          adjoint_boundary_equation_colors_.push_back(color);
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
        return boundary_functional_colors_;
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
        unsigned int comp = boundary_functional_colors_.size();
        for (unsigned int i = 0; i < boundary_functional_colors_.size(); ++i)
        {
          if (boundary_functional_colors_[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != boundary_functional_colors_.size())
        {
          std::stringstream s;
          s << "Boundary Functional Color" << color
              << " has multiple occurences !";
          throw DOpEException(s.str(),
              "PDEProblemContainer::SetBoundaryFunctionalColors");
        }
        boundary_functional_colors_.push_back(color);
      }
      { //For the  adjoint they are added  to the boundary equation colors
        unsigned int comp = adjoint_boundary_equation_colors_.size();
        for (unsigned int i = 0; i < adjoint_boundary_equation_colors_.size();
            ++i)
        {
          if (adjoint_boundary_equation_colors_[i] == color)
          {
            comp = i;
            break;
          }
        }
        if (comp != adjoint_boundary_equation_colors_.size())
        {
          //Seems this color is already added, however it might have been a equation color
          //so we don't  do anything.
        }
        else
        {
          adjoint_boundary_equation_colors_.push_back(color);
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
        return aux_functionals_[this->GetTypeNum()]->NeedTime();
      else if (this->GetType() == "error_evaluation")
        return aux_functionals_[functional_for_ee_num_]->NeedTime();
      else
        throw DOpEException("Not implemented",
            "PDEProblemContainer::NeedTimeFunctional");
    }

}
#endif
