#ifndef _OptProblem_H_
#define _OptProblem_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "functionalinterface.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "finiteelement_wrapper.h"
#include "function_wrapper.h"
#include "spacetimehandler.h"
#include "primaldirichletdata.h"
#include "tangentdirichletdata.h"
#include "transposeddirichletdatainterface.h"
#include "transposedgradientdirichletdata.h"
#include "transposedhessiandirichletdata.h"
#include "constraintvector.h"
#include "controlvector.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

#include "stateproblem.h"

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

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim,
      typename FE = DOpEWrapper::FiniteElement<dealdim>,
      typename DOFHANDLER = dealii::DoFHandler<dealdim> >
    class OptProblem
    {
      public:
        OptProblem(FUNCTIONAL& functional, PDE& pde, CONSTRAINTS& constraints,
            SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
                dealdim>& STH);

        /******************************************************/

        ~OptProblem();

        /******************************************************/

        virtual std::string
        GetName() const
        {
          return "OptProblem";
        }

        /******************************************************/
        StateProblem<OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
            CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
            DOFHANDLER> , PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>&
        GetStateProblem()
        {
          if (_state_problem == NULL)
          {
            _state_problem = new StateProblem<OptProblem<FUNCTIONAL_INTERFACE,
                FUNCTIONAL, PDE, DD, CONSTRAINTS, SPARSITYPATTERN, VECTOR,
                dopedim, dealdim, FE, DOFHANDLER> , PDE, DD, SPARSITYPATTERN,
                VECTOR, dopedim, dealdim> (*this, *_pde);
          }
          return *_state_problem;
        }

        //TODO This is Pfush needed to split into different subproblems and allow optproblem to
        //be substituted as any of these problems. Can be removed once the splitting is complete.
        OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
            SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>&
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
        PointFunctional(const std::map<std::string,
            const dealii::Vector<double>*> &param_values, const std::map<
            std::string, const VECTOR*> &domain_values);

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
        AlgebraicFunctional(const std::map<std::string, const dealii::Vector<
            double>*> &values,
            const std::map<std::string, const VECTOR*> &block_values);

        void
        AlgebraicResidual(VECTOR& residual, const std::map<std::string,
            const dealii::Vector<double>*> &values, const std::map<std::string,
            const VECTOR*> &block_values);

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
          CellTimeEquationExplicit(const DATACONTAINER& dc, dealii::Vector<
              double> &local_cell_vector, double scale = 1.);

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
         * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
         *                                 between hp- and classical case.
         *
         * @param cdc                      A DataContainer holding all the needed information
         *

         * @param cell_diameter            Diameter of the cell.
         * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
         *                                 filled by the locally computed values. For more information of its functionality, please
         *                                 search for the keyword `FullMatrix' in the deal.ii manual.
         */
        template<typename DATACONTAINER>
          void
          CellMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix,
              double scale = 1., double scale_ico = 1.);

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
          CellTimeMatrixExplicit(const DATACONTAINER& dc, dealii::FullMatrix<
              double> &local_entry_matrix);

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
          InterfaceMatrix(const FACEDATACONTAINER& dc, dealii::FullMatrix<
              double> &local_entry_matrix);

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
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

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
        //
        //      /**
        //       * This is to evaluate local constraints on the control and the state variable.
        //       *
        //       * @param control         The values of the control variable
        //       * @param state           The values of the state variable
        //       * @param constraints     The vector that upon completion should contain the values of the local constraint
        //       *                        mappings.
        //       */
        //      void
        //      ComputeLocalConstraints(const VECTOR& control, const VECTOR& state,
        //          VECTOR& constraints);

        //      /******************************************************/
        //
        //      /**
        //       * This is to evaluate local constraints on the control variable.
        //       *
        //       * @param control         The values of the control variable
        //       * @param constraints     The vector that upon completion should contain the values of the local constraint
        //       *                        mappings.
        //       */
        //      void
        //          ComputeLocalControlConstraints(const VECTOR& control,
        //              VECTOR& constraints);
        //
        void
        ComputeLocalControlConstraints(VECTOR& constraints, const std::map<
            std::string, const dealii::Vector<double>*> &values,
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

        const dealii::SmartPointer<const FE>
        GetFESystem() const;

        /******************************************************/

        //			const dealii::SmartPointer<const DOpEWrapper::FECollection<dealdim> > GetFECollection() const;

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
        SetControlDirichletBoundaryColors(unsigned int color,
            const std::vector<bool>& comp_mask, const DOpEWrapper::Function<
                dealdim>* values);

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
        GetDirichletValues(unsigned int color, const std::map<std::string,
            const dealii::Vector<double>*> &param_values, const std::map<
            std::string, const VECTOR*> &domain_values) const;

        /******************************************************/

        const TransposedDirichletDataInterface<dopedim, dealdim> &
        GetTransposedDirichletValues(unsigned int color, const std::map<
            std::string, const dealii::Vector<double>*> &param_values,
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

        void
        AddFunctional(FUNCTIONAL_INTERFACE* F)
        {
          _aux_functionals.push_back(F);
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
            throw DOpEException("Can't find functional " + functional_name
                + " in _aux_functionals",
                "Optproblem::SetFunctionalForErrorEstimation");
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
        GetHangingNodeConstraints() const;

        /******************************************************/

        unsigned int
        GetTypeNum() const
        {
          return _problem_type_num;
        }

        std::string
        GetType() const
        {
          if (_problem_type == "state")
          {
            //            std::cout << "_problem_type : " << _problem_type
            //                << " should no longer be used! Use the corresponding Problems, e.g. StateProblem..."
            //                << std::endl;
            //abort();
          }
          return _problem_type;
        }
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

        const SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
            dopedim, dealdim>*
        GetSpaceTimeHandler() const
        {
          return _STH;
        }

        /******************************************************/

        SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>*
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
        FeasibilityShift(const ControlVector<VECTOR>& g_hat, ControlVector<
            VECTOR>& g, double lambda) const
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
        AddAuxiliaryConstraint(const ConstraintVector<VECTOR>* c,
            std::string name);

        /******************************************************/

        const ControlVector<VECTOR>*
        GetAuxiliaryControl(std::string name) const;

        /******************************************************/

        void
        DeleteAuxiliaryControl(std::string name);

        /******************************************************/

        void
        DeleteAuxiliaryConstraint(std::string name);

        /*****************************************************************/

        const ConstraintVector<VECTOR>*
        GetAuxiliaryConstraint(std::string name)
        {
          typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator
              it = _auxiliary_constraints.find(name);
          if (it == _auxiliary_constraints.end())
          {
            throw DOpEException("Could not find data" + name,
                "OptProblem::GetAuxiliaryConstraint");
          }
          return it->second;
        }
        /*****************************************************************/

        template<typename INTEGRATOR>
          void
          AddAuxiliaryToIntegrator(INTEGRATOR& integrator)
          {
            {
              typename std::map<std::string, const ControlVector<VECTOR> *>::iterator
                  it = _auxiliary_controls.begin();
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
                      "OptProblem::AddAuxiliaryToIntegrator");
                }
              }
            }
            {
              typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator
                  it = _auxiliary_constraints.begin();
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
              typename std::map<std::string, const ControlVector<VECTOR> *>::iterator
                  it = _auxiliary_controls.begin();
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
                      "OptProblem::AddAuxiliaryToIntegrator");
                }
              }
            }
            {
              typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator
                  it = _auxiliary_constraints.begin();
              for (; it != _auxiliary_constraints.end(); it++)
              {
                integrator.DeleteDomainData(it->first + "_local");
                integrator.DeleteParamData(it->first + "_global");
              }
            }
          }

        /******************************************************/

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
                "OptProblem::GetBlockVector");
          }
          return it->second;
        }
        const dealii::Vector<double>*
        GetVector(const std::map<std::string, const Vector<double>*>& values,
            std::string name)
        {
          typename std::map<std::string, const Vector<double>*>::const_iterator
              it = values.find(name);
          if (it == values.end())
          {
            throw DOpEException("Did not find " + name, "OptProblem::GetVector");
          }
          return it->second;
        }
        /******************************************************/
      private:
        DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
        DOpEOutputHandler<VECTOR>* _OutputHandler;
        std::string _problem_type, _algo_type;

        unsigned int _problem_type_num;
        unsigned int _functional_for_ee_num;
        std::vector<FUNCTIONAL_INTERFACE*> _aux_functionals;
        FUNCTIONAL* _functional;
        PDE* _pde;
        CONSTRAINTS* _constraints;
        SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>* _STH;

        std::vector<unsigned int> _control_dirichlet_colors;
        std::vector<unsigned int> _control_transposed_dirichlet_colors;
        std::vector<std::vector<bool> > _control_dirichlet_comps;
        std::vector<const DOpEWrapper::Function<dealdim>*>
            _control_dirichlet_values;
        std::vector<TransposedGradientDirichletData<DD, VECTOR, dopedim,
            dealdim>*> _transposed_control_gradient_dirichlet_values;
        std::vector<
            TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim>*>
            _transposed_control_hessian_dirichlet_values;

        std::vector<unsigned int> _dirichlet_colors;
        std::vector<std::vector<bool> > _dirichlet_comps;
        std::vector<PrimalDirichletData<DD, VECTOR, dopedim, dealdim>*>
            _primal_dirichlet_values;
        std::vector<TangentDirichletData<DD, VECTOR, dopedim, dealdim>*>
            _tangent_dirichlet_values;
        const dealii::Function<dealdim>* _zero_dirichlet_values;

        const dealii::Function<dealdim>* _initial_values;

        std::vector<unsigned int> _control_boundary_equation_colors;
        std::vector<unsigned int> _state_boundary_equation_colors;
        std::vector<unsigned int> _adjoint_boundary_equation_colors;

        std::vector<unsigned int> _boundary_functional_colors;

        std::map<std::string, const ControlVector<VECTOR>*> _auxiliary_controls;
        std::map<std::string, const ConstraintVector<VECTOR>*>
            _auxiliary_constraints;

        StateProblem<OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
            CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
            DOFHANDLER> , PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>
            * _state_problem;

        friend class StateProblem<OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL,
            PDE, DD, CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
            FE, DOFHANDLER> , PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim> ;
    };
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::OptProblem(
        FUNCTIONAL& functional, PDE& pde, CONSTRAINTS& constraints,
        SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>& STH) :
      _functional(&functional), _pde(&pde), _constraints(&constraints), _STH(
          &STH), _state_problem(NULL)
    {
      _ExceptionHandler = NULL;
      _OutputHandler = NULL;
      _zero_dirichlet_values = new ZeroFunction<dealdim> (
          this->GetPDE()->GetStateNComponents());
      _algo_type = "";
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::~OptProblem()
    {
      if (_zero_dirichlet_values != NULL)
      {
        delete _zero_dirichlet_values;
      }

      for (unsigned int i = 0; i
          < _transposed_control_gradient_dirichlet_values.size(); i++)
      {
        if (_transposed_control_gradient_dirichlet_values[i] != NULL)
          delete _transposed_control_gradient_dirichlet_values[i];
      }
      for (unsigned int i = 0; i
          < _transposed_control_hessian_dirichlet_values.size(); i++)
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
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::ReInit(
        std::string algo_type)
    {
      if (_state_problem != NULL)
      {
        delete _state_problem;
        _state_problem = NULL;
      }

      if (_algo_type != algo_type && _algo_type != "")
      {
        throw DOpEException("Conflicting Algorithms!", "OptProblem::ReInit");
      }
      else
      {
        _algo_type = algo_type;
        _problem_type = "";

        if (_algo_type == "reduced")
        {
          GetSpaceTimeHandler()->ReInit(this->GetPDE()->GetControlNBlocks(),
              this->GetPDE()->GetControlBlockComponent(),
              this->GetPDE()->GetStateNBlocks(),
              this->GetPDE()->GetStateBlockComponent());
        }
        else
        {
          throw DOpEException("Unknown Algorithm " + _algo_type,
              "OptProblem::ReInit");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetType(
        std::string type, unsigned int num)
    {
      if (_problem_type != type || _problem_type_num != num)
      {
        _problem_type_num = num;
        _problem_type = type;
        this->GetPDE()->SetProblemType(_problem_type);
        this->GetConstraints()->SetProblemType(_problem_type, _problem_type_num);

#if dope_dimension > 0
        if(dealdim == dopedim)
        {
          //Prepare DoFHandlerPointer

          {
            if(_problem_type == "state" ||_problem_type == "adjoint"
                || _problem_type == "adjoint_for_ee" || _problem_type == "cost_functional"
                || _problem_type == "aux_functional" || _problem_type == "functional_for_ee"
                || _problem_type == "tangent" || _problem_type == "adjoint_hessian"
                || _problem_type.find("constraints") != std::string::npos)
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(1,0);
            }
            else if (_problem_type == "gradient"||_problem_type == "hessian"||_problem_type == "hessian_inverse" || _problem_type == "global_constraint_gradient"|| _problem_type == "global_constraint_hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0,1);
            }
            else
            {
              throw DOpEException("_problem_type : "+_problem_type+" not implemented!", "OptProblem::SetType");
            }
          }
        }
        else
        {
          throw DOpEException("dopedim not implemented", "OptProblem::SetType");
        }
#else
        //dopedim ==0
        {
          //Prepare DoFHandlerPointer
          {

            if (_problem_type == "state" || _problem_type == "adjoint"
                || _problem_type == "adjoint_for_ee" || _problem_type
                == "functional_for_ee" || _problem_type == "cost_functional"
                || _problem_type == "aux_functional" || _problem_type
                == "tangent" || _problem_type == "adjoint_hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0, 0);
            }
            else if (_problem_type == "gradient" || _problem_type
                == "hessian_inverse" || _problem_type == "hessian")
            {
              GetSpaceTimeHandler()->SetDoFHandlerOrdering(0, 0);
            }
            else
            {
              throw DOpEException("_problem_type : " + _problem_type
                  + " not implemented!", "OptProblem::SetType");
            }
          }
        }
#endif
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      double
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellFunctional(
          const DATACONTAINER& cdc)
      {

        if (GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->Value(cdc);
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->Value(cdc);
        }
        else if (GetType() == "functional_for_ee")
        {//TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
          return _aux_functionals[_functional_for_ee_num]->Value(cdc);
        }
        else if (GetType().find("constraints") != std::string::npos)
        {
          return GetConstraints()->Value(cdc);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellFunctional");
        }
      }

  /******************************************************/
  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    double
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::PointFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (GetType() == "cost_functional")
      {
        // state values in quadrature points
        return GetFunctional()->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif cost_functional
      else if (GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[_problem_type_num]->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif aux_functional
      else if (GetType() == "functional_for_ee")
      {
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        return _aux_functionals[_functional_for_ee_num]->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);
      }//endif functional_for_ee
      else if (GetType().find("constraints") != std::string::npos)
      {
        return GetConstraints()->PointValue(
            this->GetSpaceTimeHandler()->GetControlDoFHandler(),
            this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
            domain_values);

      } //endif constraints
      else
      {
        throw DOpEException("Not implemented", "OptProblem::PointFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      double
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::BoundaryFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->BoundaryValue(fdc);
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->BoundaryValue(fdc);
        }
        else if (GetType() == "functional_for_ee")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->BoundaryValue(fdc);
        }
        else if (GetType().find("constraints") != std::string::npos)
        {
          return GetConstraints()->BoundaryValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::BoundaryFunctional");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      double
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::FaceFunctional(
          const FACEDATACONTAINER& fdc)
      {
        if (GetType() == "cost_functional")
        {
          // state values in quadrature points
          return GetFunctional()->FaceValue(fdc);
        }
        else if (GetType() == "aux_functional")
        {
          // state values in quadrature points
          return _aux_functionals[_problem_type_num]->FaceValue(fdc);
        }
        else if (GetType() == "functional_for_ee")
        //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
        {
          return _aux_functionals[_functional_for_ee_num]->FaceValue(fdc);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::FaceFunctional");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    double
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::AlgebraicFunctional(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (GetType() == "cost_functional")
      {
        // state values in quadrature points
        return GetFunctional()->AlgebraicValue(param_values, domain_values);
      }
      else if (GetType() == "aux_functional")
      {
        // state values in quadrature points
        return _aux_functionals[_problem_type_num]->AlgebraicValue(
            param_values, domain_values);
      }
      else if (GetType() == "functional_for_ee")
      //TODO ist das hier korrekt? Sollten wir eigentlich nicht benoetigen.
      {
        return _aux_functionals[_functional_for_ee_num]->AlgebraicValue(
            param_values, domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "OptProblem::AlgebraicFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale, double scale_ico)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellEquation(cdc, local_cell_vector, scale, scale_ico);
        }
        else if ((GetType() == "adjoint") || (GetType() == "adjoint_for_ee"))
        {
          // state values in quadrature points
          GetPDE()->CellEquation_U(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->CellEquation_UTT(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->CellEquation_UT(cdc, local_cell_vector, scale);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlCellEquation(cdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::AlgebraicResidual(
        VECTOR& residual, const std::map<std::string, const dealii::Vector<
            double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values)
    {
      if (GetType() == "gradient")
      {
        // state values in quadrature points
        return GetFunctional()->AlgebraicGradient_Q(residual, param_values,
            domain_values);
      }
      else
      {
        throw DOpEException("Not implemented",
            "OptProblem::AlgebraicFunctional");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellTimeEquation(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeEquation(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if (GetType() == "adjoint_hessian")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if (GetType() == "tangent")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellTimeEquationExplicit(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellTimeEquationExplicit(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if (GetType() == "adjoint_hessian")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if (GetType() == "tangent")
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellTimeEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::FaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation_U(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation_UTT(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->FaceEquation_UT(fdc, local_cell_vector, scale);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlFaceEquation(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::FaceEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::InterfaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          GetPDE()->InterfaceEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->InterfaceEquation_U(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::InterfaceEquation");
        }
      }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::BoundaryEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation_U(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation_UTT(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->BoundaryEquation_UT(fdc, local_cell_vector, scale);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlBoundaryEquation(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::CellBoundaryEquation");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellRhs(
          const DATACONTAINER& cdc, dealii::Vector<double> &local_cell_vector,
          double scale)
      {

        if (GetType() == "state")
        {
          // state values in quadrature points
          GetPDE()->CellRightHandSide(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint")
        {
          // state values in quadrature points
          GetFunctional()->Value_U(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          _aux_functionals[_functional_for_ee_num]->Value_U(cdc,
              local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          GetPDE()->CellEquation_QT(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetFunctional()->Value_UU(cdc, local_cell_vector, scale);
          GetFunctional()->Value_QU(cdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->CellEquation_UU(cdc, local_cell_vector, scale);
          GetPDE()->CellEquation_QU(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "gradient")
        {
          // state values in quadrature points
          GetFunctional()->Value_Q(cdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->CellEquation_Q(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "hessian")
        {
          // state values in quadrature points
          GetFunctional()->Value_QQ(cdc, local_cell_vector, scale);
          GetFunctional()->Value_UQ(cdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->CellEquation_QTT(cdc, local_cell_vector, scale);
          GetPDE()->CellEquation_UQ(cdc, local_cell_vector, scale);
          GetPDE()->CellEquation_QQ(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "global_constraint_gradient")
        {
          //    GetConstraints()->Value_Q(this->GetSpaceTimeHandler()->GetControlFEValues(),
          //			      this->GetSpaceTimeHandler()->GetStateFEValues(),
          //			      param_values,
          //			      domain_values,
          //			      n_dofs_per_cell,
          //			      n_q_points,
          //			      material_id,
          //			      cell_diameter,
          //			      local_cell_vector,
          //			      scale);
          GetConstraints()->Value_Q(cdc, local_cell_vector, scale);
        }
        else if (GetType() == "global_constraint_hessian")
        {
          //    GetConstraints()->Value_QQ(this->GetSpaceTimeHandler()->GetControlFEValues(),
          //			       this->GetSpaceTimeHandler()->GetStateFEValues(),
          //			       param_values,
          //			       domain_values,
          //			       n_dofs_per_cell,
          //			       n_q_points,
          //			       material_id,
          //			       cell_diameter,
          //			       local_cell_vector,
          //			       scale);
          GetConstraints()->Value_QQ(cdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellRhs");
        }
      }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::FaceRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->FaceRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint")
        {
          // state values in quadrature points
          GetFunctional()->FaceValue_U(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          _aux_functionals[_functional_for_ee_num]->FaceValue_U(fdc,
              local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          GetPDE()->FaceEquation_QT(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetFunctional()->FaceValue_UU(fdc, local_cell_vector, scale);

          GetFunctional()->FaceValue_QU(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->FaceEquation_UU(fdc, local_cell_vector, scale);

          GetPDE()->FaceEquation_QU(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "gradient")
        {
          // state values in quadrature points
          GetFunctional()->FaceValue_Q(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->FaceEquation_Q(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "hessian")
        {
          // state values in quadrature points
          GetFunctional()->FaceValue_QQ(fdc, local_cell_vector, scale);

          GetFunctional()->FaceValue_UQ(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->FaceEquation_QTT(fdc, local_cell_vector, scale);

          GetPDE()->FaceEquation_UQ(fdc, local_cell_vector, scale);

          GetPDE()->FaceEquation_QQ(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellFaceRhs");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::BoundaryRhs(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        if (GetType() == "state")
        {
          // state values in face quadrature points
          GetPDE()->BoundaryRightHandSide(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint")
        {
          // state values in quadrature points
          GetFunctional()->BoundaryValue_U(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          //values of the derivative of the functional for error estimation
          _aux_functionals[_functional_for_ee_num]->BoundaryValue_U(fdc,
              local_cell_vector, scale);
        }
        else if (GetType() == "tangent")
        {
          // state values in quadrature points
          scale *= -1;
          GetPDE()->BoundaryEquation_QT(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetFunctional()->BoundaryValue_UU(fdc, local_cell_vector, scale);
          GetFunctional()->BoundaryValue_QU(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->BoundaryEquation_UU(fdc, local_cell_vector, scale);
          GetPDE()->BoundaryEquation_QU(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "gradient")
        {
          // state values in quadrature points
          GetFunctional()->BoundaryValue_Q(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->BoundaryEquation_Q(fdc, local_cell_vector, scale);
        }
        else if (GetType() == "hessian")
        {
          // state values in quadrature points
          GetFunctional()->BoundaryValue_QQ(fdc, local_cell_vector, scale);
          GetFunctional()->BoundaryValue_UQ(fdc, local_cell_vector, scale);
          scale *= -1;
          GetPDE()->BoundaryEquation_QTT(fdc, local_cell_vector, scale);
          GetPDE()->BoundaryEquation_UQ(fdc, local_cell_vector, scale);
          GetPDE()->BoundaryEquation_QQ(fdc, local_cell_vector, scale);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::CellBoundaryRhs");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellMatrix(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {

        if (GetType() == "state" || GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
            || GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->CellMatrix_T(cdc, local_entry_matrix, scale, scale_ico);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlCellMatrix(cdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::NewtonCellMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellTimeMatrix(
          const DATACONTAINER& cdc, FullMatrix<double> &local_entry_matrix)
      {

        if (GetType() == "state" || GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->CellTimeMatrix(cdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
            || GetType() == "adjoint_hessian")
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename DATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::CellTimeMatrixExplicit(
          const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {

        if (GetType() == "state" || GetType() == "tangent")
        {
          // state values in quadrature points
          GetPDE()->CellTimeMatrixExplicit(cdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_hessian")
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellTimeMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::FaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix)
      {
        if (GetType() == "state" || GetType() == "tangent")
        {
          // state values in face quadrature points
          GetPDE()->FaceMatrix(fdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
            || GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->FaceMatrix_T(fdc, local_entry_matrix);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlFaceMatrix(fdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented", "OptProblem::NewtonFaceMatrix");
        }

      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::InterfaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix)
      {
        if (GetType() == "state")
        {
          GetPDE()->InterfaceMatrix(fdc, local_entry_matrix);
        }
        else if (GetType() == "adjoint_for_ee")
        {
          GetPDE()->InterfaceMatrix_T(fdc, local_entry_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonInterfaceMatrix");
        }
      }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    template<typename FACEDATACONTAINER>
      void
      OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
          SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::BoundaryMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_cell_matrix)
      {
        if (GetType() == "state" || GetType() == "tangent")
        {
          // state values in face quadrature points
          GetPDE()->BoundaryMatrix(fdc, local_cell_matrix);
        }
        else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
            || GetType() == "adjoint_hessian")
        {
          // state values in quadrature points
          GetPDE()->BoundaryMatrix_T(fdc, local_cell_matrix);
        }
        else if ((GetType() == "gradient") || (GetType() == "hessian"))
        {
          // control values in quadrature points
          GetPDE()->ControlBoundaryMatrix(fdc, local_cell_matrix);
        }
        else
        {
          throw DOpEException("Not implemented",
              "OptProblem::NewtonCellBoundaryMatrix");
        }

      }

  //FIXME Kann das weg?
  /******************************************************/
  //
  //  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE, typename DD,
  //      typename CONSTRAINTS, typename SPARSITYPATTERN, typename VECTOR,
  //      int dopedim, int dealdim, typename FE, typename DOFHANDLER>
  //    void
  //    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS, SPARSITYPATTERN, VECTOR,
  //        dopedim, dealdim,FE, DOFHANDLER>::ComputeLocalConstraints(
  //        const VECTOR& control, const VECTOR& state, VECTOR& constraints)
  //    {
  //      if (GetType() == "constraints")
  //        {
  //          if (this->GetSpaceTimeHandler()->GetNLocalConstraints() != 0)
  //            this->GetConstraints()->EvaluateLocalConstraints(control, state,
  //                constraints);
  //        }
  //      else
  //        {
  //          throw DOpEException("Wrong problem type" + GetType(),
  //              "OptProblem::ComputeLocalConstraints");
  //        }
  //    }
  /******************************************************/

  //  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE, typename DD,
  //      typename CONSTRAINTS, typename SPARSITYPATTERN, typename VECTOR,
  //      int dopedim, int dealdim, typename FE, typename DOFHANDLER>
  //    void
  //    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS, SPARSITYPATTERN, VECTOR,
  //        dopedim, dealdim,FE, DOFHANDLER>::ComputeLocalControlConstraints(
  //        const VECTOR& control, VECTOR& constraints)
  //    {
  //      if (GetType() == "constraints")
  //        {
  //          if (this->GetSpaceTimeHandler()->GetNLocalConstraints() != 0)
  //            this->GetConstraints()->EvaluateLocalControlConstraints(control,
  //                constraints);
  //        }
  //      else
  //        {
  //          throw DOpEException("Wrong problem type" + GetType(),
  //              "OptProblem::ComputeLocalConstraints");
  //        }
  //    }
  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::ComputeLocalControlConstraints(
        VECTOR& constraints, const std::map<std::string, const dealii::Vector<
            double>*> &/*values*/,
        const std::map<std::string, const VECTOR*> &block_values)
    {
      if (GetType() == "constraints")
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
        throw DOpEException("Wrong problem type" + GetType(),
            "OptProblem::ComputeLocalConstraints");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    std::string
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDoFType() const
    {
      if (GetType() == "state" || GetType() == "adjoint" || GetType()
          == "adjoint_for_ee" || GetType() == "tangent" || GetType()
          == "adjoint_hessian")
      {
        return "state";
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian")
          || (GetType() == "hessian_inverse"))
      {
        return "control";
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDoFType");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const SmartPointer<const FE>
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFESystem() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || GetType()
          == "adjoint_for_ee" || GetType() == "tangent" || GetType()
          == "adjoint_hessian")
      {
        return this->GetSpaceTimeHandler()->GetFESystem("state");
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian")
          || (GetType() == "global_constraint_gradient"))
      {
#if dope_dimension > 0
        if(dopedim == dealdim)
        return this->GetSpaceTimeHandler()->GetFESystem("control");
        else
        throw DOpEException("Non matching dimensions!","OptProblem::GetFESystem");
#else
        return this->GetSpaceTimeHandler()->GetFESystem("state");
#endif
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetFESystem");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    UpdateFlags
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetUpdateFlags() const
    {

      UpdateFlags r;
      if (GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[_problem_type_num]->GetUpdateFlags();
      }
      else if (GetType().find("functional") != std::string::npos)
      {
        r = this->GetFunctional()->GetUpdateFlags();
      }
      else if (GetType().find("constraints") != std::string::npos)
      {
        r = this->GetConstraints()->GetUpdateFlags();
      }
      else if (GetType() == "functional_for_ee")
      {
        r = _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE()->GetUpdateFlags();
        if (GetType() == "adjoint_hessian" || GetType() == "adjoint"
            || (GetType() == "hessian") || (GetType() == "gradient"))
        {
          r = r | this->GetFunctional()->GetUpdateFlags();
        }
        else if (GetType() == "adjoint_for_ee")
        {
          r = r | _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
        }
      }
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    UpdateFlags
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      if (GetType().find("aux_functional") != std::string::npos)
      {
        r = _aux_functionals[_problem_type_num]->GetFaceUpdateFlags();
      }
      else if (GetType().find("functional") != std::string::npos)
      {
        r = this->GetFunctional()->GetFaceUpdateFlags();
      }
      else if (GetType().find("constraints") != std::string::npos)
      {
        r = this->GetConstraints()->GetFaceUpdateFlags();
      }
      else if (GetType() == "functional_for_ee")
      {
        r = _aux_functionals[_functional_for_ee_num]->GetUpdateFlags();
      }
      else
      {
        r = this->GetPDE()->GetFaceUpdateFlags();
        if (GetType() == "adjoint_hessian" || GetType() == "adjoint"
            || (GetType() == "hessian"))
        {
          r = r | this->GetFunctional()->GetFaceUpdateFlags();
        }
        else if (GetType() == "adjoint_for_ee")
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
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    std::string
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFunctionalType() const
    {
      if (GetType() == "aux_functional")
      {
        return _aux_functionals[_problem_type_num]->GetType();
      }
      else if (GetType() == "functional_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->GetType();
      }
      return GetFunctional()->GetType();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    std::string
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFunctionalName() const
    {
      if (GetType() == "aux_functional")
      {
        return _aux_functionals[_problem_type_num]->GetName();
      }
      else if (GetType() == "functional_for_ee")
      {
        return _aux_functionals[_functional_for_ee_num]->GetName();
      }
      return GetFunctional()->GetName();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    std::string
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetConstraintType() const
    {
      return GetConstraints()->GetType();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetTime(
        double time, const TimeIterator& interval)
    {
      GetSpaceTimeHandler()->SetInterval(interval);
      //      GetSpaceTimeHandler()->SetTimeDoFNumber(time_point);

      { //Zeit an Dirichlet Werte uebermitteln
        for (unsigned int i = 0; i
            < _transposed_control_gradient_dirichlet_values.size(); i++)
          _transposed_control_gradient_dirichlet_values[i]->SetTime(time);
        for (unsigned int i = 0; i
            < _transposed_control_hessian_dirichlet_values.size(); i++)
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
        GetPDE()->SetTime(time);
      }
      //Update Auxiliary Control and Constraint Vectors
      {
        typename std::map<std::string, const ControlVector<VECTOR> *>::iterator
            it = _auxiliary_controls.begin();
        for (; it != _auxiliary_controls.end(); it++)
        {
          it->second->SetTime(time, interval);
        }
      }
      {
        typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator
            it = _auxiliary_constraints.begin();
        for (; it != _auxiliary_constraints.end(); it++)
        {
          it->second->SetTime(time, interval);
        }
      }

    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      if (GetType() == "state" || GetType() == "tangent" || GetType()
          == "adjoint_for_ee" || GetType() == "adjoint" || GetType()
          == "adjoint_hessian")
      {
        this->GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian"))
      {
#if  dope_dimension > 0
        this->GetSpaceTimeHandler()->ComputeControlSparsityPattern(sparsity);
#else
        throw DOpEException("Wrong dimension",
            "OptProblem::ComputeSparsityPattern");
#endif
      }
      else
      {
        throw DOpEException("Unknown type " + GetType(),
            "OptProblem::ComputeSparsityPattern");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    bool
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::IsFeasible(
        const ConstraintVector<VECTOR>& g) const
    {
      return this->GetConstraints()->IsFeasible(g);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::PostProcessConstraints(
        ConstraintVector<VECTOR>& g) const
    {
      return this->GetConstraints()->PostProcessConstraints(g);
    }
  /******************************************************/

  //FIXME kann das weg?
  //  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE, typename DD,
  //      typename CONSTRAINTS, typename SPARSITYPATTERN, typename VECTOR,
  //      int dopedim, int dealdim, typename FE, typename DOFHANDLER>
  //    void
  //    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS, SPARSITYPATTERN, VECTOR,
  //        dopedim, dealdim,FE, DOFHANDLER>::PostProcessConstraints(
  //        ConstraintVector<VECTOR>& g, bool process_global_in_time_constraints) const
  //    {
  //      return this->GetConstraints()->PostProcessConstraints(g,
  //          process_global_in_time_constraints);
  //    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::AddAuxiliaryControl(
        const ControlVector<VECTOR>* c, std::string name)
    {
      if (_auxiliary_controls.find(name) != _auxiliary_controls.end())
      {
        throw DOpEException("Adding multiple Data with name " + name
            + " is prohibited!", "OptProblem::AddAuxiliaryControl");
      }
      _auxiliary_controls.insert(std::pair<std::string, const ControlVector<
          VECTOR>*>(name, c));
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::AddAuxiliaryConstraint(
        const ConstraintVector<VECTOR>* c, std::string name)
    {
      if (_auxiliary_constraints.find(name) != _auxiliary_constraints.end())
      {
        throw DOpEException("Adding multiple Data with name " + name
            + " is prohibited!", "OptProblem::AddAuxiliaryConstraint");
      }
      _auxiliary_constraints.insert(std::pair<std::string,
          const ConstraintVector<VECTOR>*>(name, c));
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const ControlVector<VECTOR>*
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetAuxiliaryControl(
        std::string name) const
    {
      typename std::map<std::string, const ControlVector<VECTOR> *>::const_iterator
          it = _auxiliary_controls.find(name);
      if (it == _auxiliary_controls.end())
      {
        throw DOpEException("Could not find Data with name " + name,
            "OptProblem::GetAuxiliaryControl");
      }
      return it->second;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::DeleteAuxiliaryControl(
        std::string name)
    {
      typename std::map<std::string, const ControlVector<VECTOR> *>::iterator
          it = _auxiliary_controls.find(name);
      if (it == _auxiliary_controls.end())
      {
        throw DOpEException("Deleting Data " + name
            + " is impossible! Data not found",
            "OptProblem::DeleteAuxiliaryControl");
      }
      _auxiliary_controls.erase(it);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::DeleteAuxiliaryConstraint(
        std::string name)
    {
      typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator
          it = _auxiliary_constraints.find(name);
      if (it == _auxiliary_constraints.end())
      {
        throw DOpEException("Deleting Data " + name
            + " is impossible! Data not found",
            "OptProblem::DeleteAuxiliaryConstraint");
      }
      _auxiliary_constraints.erase(it);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    FUNCTIONAL*
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFunctional()
    {
      if (GetType() == "aux_functional" || GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //    return _aux_functionals[_problem_type_num];
      }
      return _functional;

    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const FUNCTIONAL*
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetFunctional() const
    {
      if (GetType() == "aux_functional" || GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //       return _aux_functionals[_problem_type_num];
      }
      return _functional;
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    bool
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::HasFaces() const
    {
      if (GetType().find("aux_functional") != std::string::npos)
      {
        return _aux_functionals[_problem_type_num]->HasFaces();
      }
      else if (GetType().find("functional") != std::string::npos)
      {
        return this->GetFunctional()->HasFaces();
      }
      else if (GetType().find("constraint") != std::string::npos)
      {
        return this->GetConstraints()->HasFaces();
      }
      else
      {
        if ((GetType() == "state") || (GetType() == "tangent") || (GetType()
            == "gradient"))
        {
          return this->GetPDE()->HasFaces();
        }
        else if ((GetType() == "adjoint") || (GetType() == "adjoint_hessian")
            || (GetType() == "hessian"))
        {
          return this->GetPDE()->HasFaces()
              || this->GetFunctional()->HasFaces();
        }
        else if (GetType() == "adjoint_for_ee")
        {
          return this->GetPDE()->HasFaces()
              || _aux_functionals[_functional_for_ee_num]->HasFaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + GetType() + "'!",
              "OptProblem::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    bool
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::HasInterfaces() const
    {
      if (GetType().find("aux_functional") != std::string::npos)
      {
        return false;
      }
      else if (GetType().find("functional") != std::string::npos)
      {
        return false;
      }
      else if (GetType().find("constraint") != std::string::npos)
      {
        return false;
      }
      else
      {
        if ((GetType() == "state") || GetType() == "adjoint_for_ee"
            || (GetType() == "tangent") || (GetType() == "gradient")
            || (GetType() == "adjoint") || (GetType() == "adjoint_hessian")
            || (GetType() == "hessian"))
        {
          return this->GetPDE()->HasInterfaces();
        }
        else
        {
          throw DOpEException("Unknown Type: '" + GetType() + "'!",
              "OptProblem::HasFaces");
        }
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetControlDirichletBoundaryColors(
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
        throw DOpEException(s.str(), "OptProblem::SetControlDirichletBoundary");
      }
      _control_dirichlet_colors.push_back(color);
      _control_dirichlet_comps.push_back(comp_mask);
      _control_dirichlet_values.push_back(values);
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetDirichletBoundaryColors(
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
        throw DOpEException(s.str(), "OptProblem::SetDirichletBoundary");
      }
      _dirichlet_colors.push_back(color);
      _dirichlet_comps.push_back(comp_mask);
      PrimalDirichletData<DD, VECTOR, dopedim, dealdim>* data =
          new PrimalDirichletData<DD, VECTOR, dopedim, dealdim> (*values);
      _primal_dirichlet_values.push_back(data);
      TangentDirichletData<DD, VECTOR, dopedim, dealdim>* tdata =
          new TangentDirichletData<DD, VECTOR, dopedim, dealdim> (*values);
      _tangent_dirichlet_values.push_back(tdata);

      if (values->NeedsControl())
      {
        _control_transposed_dirichlet_colors.push_back(color);
        TransposedGradientDirichletData<DD, VECTOR, dopedim, dealdim> * gdata =
            new TransposedGradientDirichletData<DD, VECTOR, dopedim, dealdim> (
                *values);
        _transposed_control_gradient_dirichlet_values.push_back(gdata);
        TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim> * hdata =
            new TransposedHessianDirichletData<DD, VECTOR, dopedim, dealdim> (
                *values);
        _transposed_control_hessian_dirichlet_values.push_back(hdata);
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<unsigned int>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDirichletColors() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || GetType()
          == "adjoint_for_ee" || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
      {
        return _dirichlet_colors;
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian")
          || (GetType() == "global_constraint_gradient"))
      {
        return _control_dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDirichletColors");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<unsigned int>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetTransposedDirichletColors() const
    {
      if ((GetType() == "gradient") || (GetType() == "hessian"))
      {
        return _control_transposed_dirichlet_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetTransposedDirichletColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<bool>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDirichletCompMask(
        unsigned int color) const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || GetType()
          == "adjoint_for_ee" || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
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
          throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
        }
        return _dirichlet_comps[comp];
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian"))
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
          throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
        }
        return _control_dirichlet_comps[comp];
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDirichletCompMask");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<bool>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetTransposedDirichletCompMask(
        unsigned int color) const
    {
      if ((GetType() == "gradient") || (GetType() == "hessian"))
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
          throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
        }
        return _dirichlet_comps[comp];
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetTransposedDirichletCompMask");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const Function<dealdim>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDirichletValues(
        unsigned int color, const std::map<std::string, const dealii::Vector<
            double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {

      unsigned int col = _dirichlet_colors.size();
      if ((GetType() == "state") || (GetType() == "adjoint") || GetType()
          == "adjoint_for_ee" || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
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
          throw DOpEException(s.str(), "OptProblem::GetDirichletValues");
        }
      }
      else if (GetType() == "gradient" || (GetType() == "hessian"))
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
          throw DOpEException(s.str(), "OptProblem::GetDirichletValues");
        }
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDirichletValues");
      }

      if (GetType() == "state")
      {
        _primal_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_primal_dirichlet_values[col]);
      }
      else if (GetType() == "tangent")
      {
        _tangent_dirichlet_values[col]->ReInit(param_values, domain_values,
            color);
        return *(_tangent_dirichlet_values[col]);
      }
      else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
          || (GetType() == "adjoint_hessian"))
      {
        return *(_zero_dirichlet_values);
      }
      else if (GetType() == "gradient" || (GetType() == "hessian"))
      {
        return *(_control_dirichlet_values[col]);
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDirichletValues");
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const TransposedDirichletDataInterface<dopedim, dealdim>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetTransposedDirichletValues(
        unsigned int color, const std::map<std::string, const dealii::Vector<
            double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {
      unsigned int col = _control_transposed_dirichlet_colors.size();
      if (GetType() == "gradient" || (GetType() == "hessian"))
      {
        for (unsigned int i = 0; i
            < _control_transposed_dirichlet_colors.size(); ++i)
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
              "OptProblem::GetTransposedDirichletValues");
        }
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetTransposedDirichletValues");
      }

      if (GetType() == "gradient")
      {
        _transposed_control_gradient_dirichlet_values[col]->ReInit(
            param_values, domain_values, color);
        return *(_transposed_control_gradient_dirichlet_values[col]);
      }
      else if (GetType() == "hessian")
      {
        _transposed_control_hessian_dirichlet_values[col]->ReInit(param_values,
            domain_values, color);
        return *(_transposed_control_hessian_dirichlet_values[col]);
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetTransposedDirichletValues");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<unsigned int>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetBoundaryEquationColors() const
    {
      if (GetType() == "state" || GetType() == "tangent")
      {
        return _state_boundary_equation_colors;
      }
      else if (GetType() == "adjoint" || GetType() == "adjoint_for_ee"
          || GetType() == "adjoint_hessian")
      {
        return _adjoint_boundary_equation_colors;
      }
      else if (GetType() == "gradient" || (GetType() == "hessian")
          || (GetType() == "global_constraint_gradient"))
      {
        return _control_boundary_equation_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetBoundaryEquationColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetControlBoundaryEquationColors(
        unsigned int color)
    {
      { //Control Boundary Equation colors are simply inserted
        unsigned int comp = _control_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _control_boundary_equation_colors.size(); ++i)
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
              "OptProblem::SetControlBoundaryEquationColors");
        }
        _control_boundary_equation_colors.push_back(color);
      }
    }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetBoundaryEquationColors(
        unsigned int color)
    {
      { //State Boundary Equation colors are simply inserted
        unsigned int comp = _state_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _state_boundary_equation_colors.size(); ++i)
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
          throw DOpEException(s.str(), "OptProblem::SetBoundaryEquationColors");
        }
        _state_boundary_equation_colors.push_back(color);
      }
      { //For the  adjoint they are added with the boundary functional colors
        unsigned int comp = _adjoint_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _adjoint_boundary_equation_colors.size(); ++i)
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
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<unsigned int>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetBoundaryFunctionalColors() const
    {
      if (GetType() == "cost_functional" || GetType() == "aux_functional"
          || GetType() == "functional_for_ee")
      {
        return _boundary_functional_colors;
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetBoundaryFunctionalColors");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    void
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::SetBoundaryFunctionalColors(
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
              "OptProblem::SetBoundaryFunctionalColors");
        }
        _boundary_functional_colors.push_back(color);
      }
      { //For the  adjoint they are addeed  to the boundary equation colors
        unsigned int comp = _adjoint_boundary_equation_colors.size();
        for (unsigned int i = 0; i < _adjoint_boundary_equation_colors.size(); ++i)
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
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    unsigned int
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetControlNBlocks() const
    {
      return this->GetPDE()->GetControlNBlocks();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    unsigned int
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetStateNBlocks() const
    {
      return this->GetPDE()->GetStateNBlocks();
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    unsigned int
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetNBlocks() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint_for_ee")
          || (GetType() == "adjoint") || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
      {
        return this->GetStateNBlocks();
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian"))
      {
        return this->GetControlNBlocks();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetNBlocks");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    unsigned int
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDoFsPerBlock(
        unsigned int b) const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || (GetType()
          == "adjoint_for_ee") || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock(b);
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian"))
      {
        return GetSpaceTimeHandler()->GetControlDoFsPerBlock(b);
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const std::vector<unsigned int>&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetDoFsPerBlock() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || (GetType()
          == "adjoint_for_ee") || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian"))
      {
        return GetSpaceTimeHandler()->GetControlDoFsPerBlock();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetDoFsPerBlock");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    const dealii::ConstraintMatrix&
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::GetHangingNodeConstraints() const
    {
      if ((GetType() == "state") || (GetType() == "adjoint") || (GetType()
          == "adjoint_for_ee") || (GetType() == "tangent") || (GetType()
          == "adjoint_hessian"))
      {
        return GetSpaceTimeHandler()->GetStateHangingNodeConstraints();
      }
      else if ((GetType() == "gradient") || (GetType() == "hessian")
          || (GetType() == "global_constraint_gradient"))
      {
        return GetSpaceTimeHandler()->GetControlHangingNodeConstraints();
      }
      else
      {
        throw DOpEException("Unknown Type:" + GetType(),
            "OptProblem::GetHangingNodeConstraints");
      }
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    bool
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::NeedTimeFunctional() const
    {
      if (GetType() == "cost_functional")
        return GetFunctional()->NeedTime();
      else if (GetType() == "aux_functional")
        return _aux_functionals[_problem_type_num]->NeedTime();
      else if (GetType() == "functional_for_ee")
        return _aux_functionals[_functional_for_ee_num]->NeedTime();
      else
        throw DOpEException("Not implemented", "OptProblem::NeedTimeFunctional");
    }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
      typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
      typename VECTOR, int dopedim, int dealdim, typename FE,
      typename DOFHANDLER>
    bool
    OptProblem<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>::HasControlInDirichletData() const
    {
      return (!_control_transposed_dirichlet_colors.empty());
    }

}
#endif
