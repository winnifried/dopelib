#ifndef _PDEProblemContainer_H_
#define _PDEProblemContainer_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "functionalinterface.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "finiteelement_wrapper.h"
#include "function_wrapper.h"
#include "statespacetimehandler.h"
#include "primaldirichletdata.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "stateproblem.h"

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

using namespace dealii;

namespace DOpE
{
  //Predeclaration necessary
  template<typename VECTOR>
    class DOpEOutputHandler;
  template<typename VECTOR>
    class DOpEExceptionHandler;
  /////////////////////////////

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE = DOpEWrapper::FiniteElement<dealdim>,
      typename DOFHANDLER = dealii::DoFHandler<dealdim> >
    class PDEProblemContainer
    {
      public:
        PDEProblemContainer(PDE& pde,
            StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
                dealdim>& STH);

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
                DOFHANDLER>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
        GetStateProblem()
        {
          if (_state_problem == NULL)
          {
            _state_problem = new StateProblem<
                PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim,
                    FE, DOFHANDLER>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
                *this, *_pde);
          }
          return *_state_problem;
        }

        //TODO This is Pfush needed to split into different subproblems and allow optproblem to
        //be substituted as any of these problems. Can be removed once the splitting is complete.
        PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
            DOFHANDLER>&
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
         */
        template<typename DATACONTAINER>
          void
          CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        /******************************************************/
        /**
         * Computes the contribution of the cell to overall error
         * in a previously specified functional. For example, this
         * could be a residual with appropriate weights.
         *
         * @template CDC                Class of the celldatacontainer in use,
         *                              distinguishes between hp- and classical case.
         * @template FDC                Class of the facedatacontainer in use,
         *                              distinguishes between hp- and classical case.
         *
         * @param cdc                   A DataContainer holding all the needed information
         *                              for the computation of the residuum on the cell.
         * @param dwrc                  A DWRDataContainer containing all the information
         *                              needed to evaluate the error on the cell (form of the residual,
         *                              the weights, etc.).
         * @param cell_contrib          Vector in which we write the contribution of the cell to the overall
         *                              error. 1st component: primal_part, 2nd component: dual_part
         * @param scale                 A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<class CDC, class FDC>
          void
          CellErrorContribution(const CDC& cdc,
              const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
              std::vector<double>& cell_contrib, double scale,
              double /*scale_ico*/);

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
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the contribution of the face to overall error
         * in a previously specified functional. This is the place
         * where for instance jump terms come into play.
         *
         * It has the same functionality
         * as CellErrorContribution, so we refer to its documentation.
         *
         */
        template<class CDC, class FDC>
          void
          FaceErrorContribution(const FDC& fdc,
              const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
              std::vector<double>& error_contrib, double scale = 1.);

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
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

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
              dealii::FullMatrix<double> &local_entry_matrix);

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
              dealii::FullMatrix<double> &local_entry_matrix);

        /******************************************************/

        /**
         * Computes the contribution of the boundary to overall error
         * in a previously specified functional.
         *
         * It has the same functionality
         * as CellErrorContribution, so we refer to its documentation.
         *
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /******************************************************/

        /**
         * Computes the value of the strong residuum ont the boundary on a cell.
         * It has the same functionality as StrongCellResidual. We refer to its
         * documentation.
         *
         */
        template<class CDC, class FDC>
          void
          BoundaryErrorContribution(const FDC& dc,
              const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
              std::vector<double>&, double scale = 1.);

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
              dealii::FullMatrix<double> &local_cell_matrix);

        /******************************************************/

        const FE&
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
            FunctionalInterface<CellDataContainer, FaceDataContainer,
                DOFHANDLER, VECTOR, dealdim>* F)
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
        GetHangingNodeConstraints() const;

        /******************************************************/

        std::string
        GetType() const
        {
          return _problem_type;
        }
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

        const StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
            dealdim>*
        GetSpaceTimeHandler() const
        {
          return _STH;
        }

        /******************************************************/

        StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dealdim>*
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
          return this->GetPDE()->GetStateNBlocks();
        }

        /******************************************************/

        std::vector<unsigned int>&
        GetStateBlockComponent()
        {
          return this->GetPDE()->GetStateBlockComponent();
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
      protected:
        PDE*
        GetPDE()
        {
          return _pde;
        }
        const PDE*
        GetPDE() const
        {
          return _pde;
        }

        /******************************************************/
      private:
        DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
        DOpEOutputHandler<VECTOR>* _OutputHandler;
        std::string _problem_type, _algo_type;

        unsigned int _problem_type_num;

        std::vector<
            FunctionalInterface<CellDataContainer, FaceDataContainer,
                DOFHANDLER, VECTOR, dealdim>*> _aux_functionals;
        std::map<std::string, unsigned int> _functional_position;

        unsigned int _functional_for_ee_num;
        PDE* _pde;
        StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dealdim> * _STH;

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
                DOFHANDLER>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>* _state_problem;

        friend class StateProblem<
            PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
                DOFHANDLER>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    };
  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::PDEProblemContainer(PDE& pde,
        StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dealdim>& STH)
        : _pde(&pde), _STH(&STH), _state_problem(NULL)
    {
      _ExceptionHandler = NULL;
      _OutputHandler = NULL;
      _zero_dirichlet_values = new ZeroFunction<dealdim>(
          this->GetPDE()->GetStateNComponents());
      _algo_type = "";
      _functional_for_ee_num = dealii::numbers::invalid_unsigned_int;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::~PDEProblemContainer()
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
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::ReInit(std::string algo_type)
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
        _problem_type = "";

        if (_algo_type == "reduced")
        {
          GetSpaceTimeHandler()->ReInit(this->GetPDE()->GetStateNBlocks(),
              this->GetPDE()->GetStateBlockComponent());
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
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::SetType(std::string type, unsigned int num)
    {
      if (_problem_type != type || _problem_type_num != num)
      {
        _problem_type_num = num;
        _problem_type = type;
        this->GetPDE()->SetProblemType(_problem_type);
      }
      //Nothing to do.
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellFunctional(const DATACONTAINER& cdc)
      {

        if (GetType() == "cost_functional")
        {
          return 0;
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->Value(cdc);
        }
        else if (GetType() == "error_evaluation")
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
      int dealdim, typename FE, typename DOFHANDLER>
    double
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::PointFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (GetType() == "cost_functional")
      {
        return 0.;
      } //endif cost_functional
      else if (GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[_problem_type_num]->PointValue(
            this->GetSpaceTimeHandler()->GetStateDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif aux_functional
      else if (GetType() == "error_evaluation")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::BoundaryFunctional(const FACEDATACONTAINER& fdc)
      {
        if (GetType() == "cost_functional")
        {
          // state values in quadrature points
          return 0.;
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->BoundaryValue(fdc);
        }
        else if (GetType() == "error_evaluation")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      double
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::FaceFunctional(const FACEDATACONTAINER& fdc)
      {
        if (GetType() == "cost_functional")
        {
          // state values in quadrature points
          return 0.;
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->FaceValue(fdc);
        }
        else if (GetType() == "error_evaluation")
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
      int dealdim, typename FE, typename DOFHANDLER>
    double
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::AlgebraicFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (GetType() == "cost_functional")
      {
        // state values in quadrature points
        return 0.;
      }
      else if (GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[_problem_type_num]->AlgebraicValue(param_values,
            domain_values);
      }
      else if (GetType() == "error_evaluation")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellEquation(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellEquation(cdc, local_cell_vector, scale, scale_ico);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->CellEquation_U(cdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<class CDC, class FDC>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellErrorContribution(const CDC& cdc,
          const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
          std::vector<double>& error, double scale, double scale_ico)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());

        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {
          switch (dwrc.GetEETerms())
          {
            case DOpEtypes::primal_only:
              GetPDE()->StrongCellResidual(cdc, dwrc.GetCellWeight(), error[0],
                  scale, scale_ico);
              break;
            case DOpEtypes::dual_only:
              GetPDE()->StrongCellResidual_U(cdc, dwrc.GetCellWeight(),
                  error[1], scale);
              break;
            case DOpEtypes::mixed:
              GetPDE()->StrongCellResidual(cdc, dwrc.GetCellWeight(), error[0],
                  scale, scale_ico);
              GetPDE()->StrongCellResidual_U(cdc, dwrc.GetCellWeight(),
                  error[1], scale);
              break;
            default:
              throw DOpEException("Not implemented for this EETerm.",
                  "PDEProblemContainer::CellErrorContribution");
              break;
          }
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::CellErrorContribution");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<class CDC, class FDC>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::FaceErrorContribution(const FDC& fdc,
          const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
          std::vector<double>& error, double scale)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());

        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {
          switch (dwrc.GetEETerms())
          {
            case DOpEtypes::primal_only:
              GetPDE()->StrongFaceResidual(fdc, dwrc.GetFaceWeight(), error[0],
                  scale);
              break;
            case DOpEtypes::dual_only:
              GetPDE()->StrongFaceResidual_U(fdc, dwrc.GetFaceWeight(),
                  error[1], scale);
              break;
            case DOpEtypes::mixed:
              GetPDE()->StrongFaceResidual(fdc, dwrc.GetFaceWeight(), error[0],
                  scale);
              GetPDE()->StrongFaceResidual_U(fdc, dwrc.GetFaceWeight(),
                  error[1], scale);
              break;
            default:
              throw DOpEException("Not implemented for this EETerm.",
                  "PDEProblemContainer::CellErrorContribution");
              break;
          }
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::CellErrorContribution");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<class CDC, class FDC>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::BoundaryErrorContribution(const FDC& fdc,
          const DWRDataContainer<CDC, FDC, VECTOR>& dwrc,
          std::vector<double>& error, double scale)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());
        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {
          // state values in quadrature points
          GetPDE()->StrongBoundaryResidual(fdc, dwrc.GetFaceWeight(), error[0],
              scale);
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::CellErrorContribution");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellTimeEquation(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeEquation(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellTimeEquationExplicit(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeEquationExplicit(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::FaceEquation(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation_U(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::FaceEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::InterfaceEquation(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->InterfaceEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->InterfaceEquation_U(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::InterfaceEquation");
        }
      }
  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::BoundaryEquation(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation_U(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::CellBoundaryEquation");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellRhs(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellRightHandSide(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::FaceRhs(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->FaceRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::BoundaryRhs(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->BoundaryRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellMatrix(const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->CellMatrix_T(cdc, local_entry_matrix, scale, scale_ico);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellTimeMatrix(const DATACONTAINER& cdc,
          FullMatrix<double> &local_entry_matrix)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeMatrix(cdc, local_entry_matrix);
        }
        else if (GetType() == "dual_for_ee")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::CellTimeMatrixExplicit(const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeMatrixExplicit(cdc, local_entry_matrix);
        }
        else if (GetType() == "dual_for_ee")
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
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::FaceMatrix(const FACEDATACONTAINER& fdc,
          FullMatrix<double> &local_entry_matrix)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->FaceMatrix(fdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          GetPDE()->FaceMatrix_T(fdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonFaceMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::InterfaceMatrix(const FACEDATACONTAINER& fdc,
          FullMatrix<double> &local_entry_matrix)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->InterfaceMatrix(fdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          GetPDE()->InterfaceMatrix_T(fdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonInterfaceMatrix");
        }
      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
          DOFHANDLER>::BoundaryMatrix(const FACEDATACONTAINER& fdc,
          FullMatrix<double> &local_cell_matrix)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->BoundaryMatrix(fdc, local_cell_matrix);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->BoundaryMatrix_T(fdc, local_cell_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "PDEProblemContainer::NewtonCellBoundaryMatrix");
        }

      }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDoFType() const
    {
      if (GetType() == "state" || GetType() == "adjoint_for_ee")
      {
        return "state";
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDoFType");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const FE&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetFESystem() const
    {
      if ((GetType() == "state") || GetType() == "adjoint_for_ee")
      {
        return this->GetSpaceTimeHandler()->GetFESystem("state");
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetFESystem");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    UpdateFlags
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetUpdateFlags() const
    {

      UpdateFlags r;
      if (GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[_problem_type_num]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE()->GetUpdateFlags();
        if (GetType() == "adjoint_for_ee" || GetType() == "error_evaluation")
        {
          r = r | _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    UpdateFlags
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      if (GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[_problem_type_num]->GetFaceUpdateFlags();
      }
      else
      {
        r = this->GetPDE()->GetFaceUpdateFlags();
        if (GetType() == "adjoint_for_ee" || GetType() == "error_evaluation")
        {
          r = r
              | _aux_functionals[_functional_for_ee_num]->GetFaceUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetFunctionalType() const
    {
      if (GetType() == "aux_functional")
      {
        return _aux_functionals[_problem_type_num]->GetType();
      }
      else if (GetType() == "error_evaluation")
      {
        return _aux_functionals[_functional_for_ee_num]->GetType();
      }
      return "none";
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    std::string
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetFunctionalName() const
    {
      if (GetType() == "aux_functional")
      {
        return _aux_functionals[_problem_type_num]->GetName();
      }
      else if (GetType() == "error_evaluation")
      {
        return _aux_functionals[_functional_for_ee_num]->GetName();
      }
      return "";
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::SetTime(double time, const TimeIterator& interval)
    {
      GetSpaceTimeHandler()->SetInterval(interval);

      { //Zeit an Dirichlet Werte uebermitteln
        for (unsigned int i = 0; i < _primal_dirichlet_values.size(); i++)
          _primal_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i < _aux_functionals.size(); i++)
          _aux_functionals[i]->SetTime(time);
        //PDE
        GetPDE()->SetTime(time);
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const
    {
      if (GetType() == "state" || GetType() == "adjoint_for_ee")
      {
        this->GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
      }
      else
      {
        throw DOpEException("Unknown type " + GetType(),
            "PDEProblemContainer::ComputeSparsityPattern");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::HasFaces() const
    {
      if (GetType().find("aux_functional") != std::string::npos)
      {
        return _aux_functionals[_problem_type_num]->HasFaces();
      }
      else
      {
        if ((GetType() == "state"))
        {
          return this->GetPDE()->HasFaces();
        }
        else if (GetType() == "adjoint_for_ee")
        {
          return this->GetPDE()->HasFaces()
              || _aux_functionals[_functional_for_ee_num]->HasFaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + GetType() + "'!",
              "PDEProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::HasInterfaces() const
    {
      if (GetType().find("aux_functional") != std::string::npos)
      {
        return false;
      }
      else
      {
        if ((GetType() == "state") || GetType() == "adjoint_for_ee")
        {
          return this->GetPDE()->HasInterfaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + GetType() + "'!",
              "PDEProblemContainer::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::SetDirichletBoundaryColors(unsigned int color,
        const std::vector<bool>& comp_mask, const DD* values)
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
      PrimalDirichletData<DD, VECTOR, dealdim>* data = new PrimalDirichletData<
          DD, VECTOR, dealdim>(*values);
      _primal_dirichlet_values.push_back(data);
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDirichletColors() const
    {
      if ((GetType() == "state") || GetType() == "adjoint_for_ee")
      {
        return _dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDirichletColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const std::vector<bool>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDirichletCompMask(unsigned int color) const
    {
      if ((GetType() == "state" || GetType() == "adjoint_for_ee"))
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
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDirichletCompMask");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const Function<dealdim>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDirichletValues(unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {

      unsigned int col = _dirichlet_colors.size();
      if ((GetType() == "state") || GetType() == "adjoint_for_ee")
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
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDirichletValues");
      }

      if (GetType() == "state")
      {
        _primal_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_primal_dirichlet_values[col]);
      }
      else if (GetType() == "adjoint_for_ee"
          || (GetType() == "adjoint_hessian"))
      {
        return *(_zero_dirichlet_values);
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDirichletValues");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetBoundaryEquationColors() const
    {
      if (GetType() == "state")
      {
        return _state_boundary_equation_colors;
      }
      else if (GetType() == "adjoint_for_ee")
      {
        return _adjoint_boundary_equation_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetBoundaryEquationColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::SetBoundaryEquationColors(unsigned int color)
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
      int dealdim, typename FE, typename DOFHANDLER>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetBoundaryFunctionalColors() const
    {
      //FIXME cost_functional?? This is pdeproblemcontainer, we should not have a cost functional! ~cg
      if (GetType() == "cost_functional" || GetType() == "aux_functional"
          || GetType() == "error_evaluation")
      {
        return _boundary_functional_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetBoundaryFunctionalColors");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    void
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::SetBoundaryFunctionalColors(unsigned int color)
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
      int dealdim, typename FE, typename DOFHANDLER>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetStateNBlocks() const
    {
      return this->GetPDE()->GetStateNBlocks();
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetNBlocks() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint_for_ee"))
      {
        return this->GetStateNBlocks();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetNBlocks");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    unsigned int
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDoFsPerBlock(unsigned int b) const
    {
      if ((GetType() == "state") || (GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock(b);
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const std::vector<unsigned int>&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetDoFsPerBlock() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    const dealii::ConstraintMatrix&
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::GetHangingNodeConstraints() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint_for_ee"))
      {
        return GetSpaceTimeHandler()->GetStateHangingNodeConstraints();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "PDEProblemContainer::GetHangingNodeConstraints");
      }
    }

  /******************************************************/

  template<typename PDE, typename DD, typename SPARSITYPATTERN, typename VECTOR,
      int dealdim, typename FE, typename DOFHANDLER>
    bool
    PDEProblemContainer<PDE, DD, SPARSITYPATTERN, VECTOR, dealdim, FE,
        DOFHANDLER>::NeedTimeFunctional() const
    {
      if (GetType() == "cost_functional")
        return false;
      else if (GetType() == "aux_functional")
        return _aux_functionals[_problem_type_num]->NeedTime();
      else if (GetType() == "error_evaluation")
        return _aux_functionals[_functional_for_ee_num]->NeedTime();
      else
        throw DOpEException("Not implemented",
            "PDEProblemContainer::NeedTimeFunctional");
    }

}
#endif
