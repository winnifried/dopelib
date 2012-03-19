#ifndef _SOLVER_INTERFACE_H_
#define _SOLVER_INTERFACE_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "controlvector.h"
#include "constraintvector.h"
#include "dopetypes.h"

#include <assert.h>

#include <lac/vector.h>

namespace DOpE
{
  //Predeclaration necessary
  template<typename VECTOR>
    class DOpEOutputHandler;
  template<typename VECTOR>
    class DOpEExceptionHandler;
  /////////////////////////////
  /**
   * The base class for all solvers.
   * Defines the non dimension dependent interface for the output handling
   */
  template<typename VECTOR>
    class ReducedProblemInterface_Base
    {
      public:
        ReducedProblemInterface_Base()
        {
          _ExceptionHandler = NULL;
          _OutputHandler = NULL;
        }
        /**
         * Basic function to get information of the state size.
         *
         * @param out         The output stream.
         */
        virtual void
        StateSizeInfo(std::stringstream& out)=0;

        /******************************************************/

        /**
         * Basic function to write vectors in files.
         *
         *  @param v           The BlockVector to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
         */
        virtual void
        WriteToFile(const VECTOR &v, std::string name, std::string outfile,
            std::string dof_type, std::string filetype)=0;

        /******************************************************/

        /**
         * Basic function to write vectors in files.
         *
         *  @param v           The ControlVector to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
         */
        virtual void
        WriteToFile(const ControlVector<VECTOR> &v, std::string name,
            std::string outfile, std::string dof_type, std::string filetype)=0;

        /******************************************************/
        //
        //    /**
        //     * Basic function to write vectors in files.
        //     *
        //     *  @param v           The ControlVector to write to a file.
        //     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
        //     *  @param outfile     The basic name for the output file to print.
        //     *  @param dof_type    Has the DoF type: state or control.
        //     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
        //     */
        //    virtual void WriteToFile(const ControlVector<dealii::Vector<double> > &v,
        //           std::string name,
        //           std::string outfile,
        //           std::string dof_type,
        //           std::string filetype)=0;
        /******************************************************/

        /**
         * Basic function to write a std::vector to a file.
         *
         *  @param v           A std::vector to write to a file.
         *  @param outfile     The basic name for the output file to print.
         */
        virtual void
        WriteToFile(const std::vector<double> &v, std::string outfile) =0;

        void
        RegisterOutputHandler(DOpEOutputHandler<VECTOR>* OH)
        {
          _OutputHandler = OH;
        }
        void
        RegisterExceptionHandler(DOpEExceptionHandler<VECTOR>* OH)
        {
          _ExceptionHandler = OH;
        }

        DOpEExceptionHandler<VECTOR>*
        GetExceptionHandler()
        {
          assert(_ExceptionHandler);
          return _ExceptionHandler;
        }
        DOpEOutputHandler<VECTOR>*
        GetOutputHandler()
        {
          assert(_OutputHandler);
          return _OutputHandler;
        }
      protected:

      private:
        DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
        DOpEOutputHandler<VECTOR>* _OutputHandler;

    };

  /**
   * A template for different solver types to be used for solving
   * PDE- as well as optimization problems.
   */
  template<typename PROBLEM, typename VECTOR, int dopedim, int dealdim>
    class ReducedProblemInterface : public ReducedProblemInterface_Base<VECTOR>
    {
      public:
        ReducedProblemInterface(PROBLEM *OP, int base_priority = 0) :
          ReducedProblemInterface_Base<VECTOR> ()
        {
          _OP = OP;
          _base_priority = base_priority;
          _post_index = "_" + this->GetProblem()->GetName();
        }
        ~ReducedProblemInterface()
        {
        }

        /******************************************************/

        /**
         * Basic function which is given to instatsolver.h and statsolver.h, respectively,
         * and reinitializes vectors, matrices, etc.
         *
         */
        virtual void
        ReInit()
        {
          this->GetProblem()->ReInit("reduced");
        }

        /******************************************************/

        /**
         * Basic function which is given to instatsolver.h and statsolver.h, respectively,
         * It computes the value of the constraint mapping and returns a boolean indicating
         * whether the point is feasible.
         *
         * @param q            The ControlVector is given to this function.
         * @param g            The ConstraintVector that contains the value of the constraint mapping after completion.
         *
         * @return             True if feasible, false otherwise.
         */
        virtual bool
        ComputeReducedConstraints(const ControlVector<VECTOR>& q,
            ConstraintVector<VECTOR>& g) = 0;

        /******************************************************/

        /**
         * Basic function which is given to instatsolver.h and statsolver.h, respectively,
         * It fills the values of the lower and upper box constraints on the control variable in a vector
         *
         * @param lb           The ControlVector to store the lower bounds
         * @param ub           The ControlVector to store the upper bounds
         */
        virtual void
        GetControlBoxConstraints(ControlVector<VECTOR>& lb, ControlVector<
            VECTOR>& ub)= 0;

        /******************************************************/

        /**
         * Basic function to compute the reduced gradient solution.
         * We assume that state u(q) is already computed.
         * However the adjoint is not assumed to be computed.
         *
         * @param q                    The ControlVector is given to this function.
         * @param gradient             The gradient vector.
         * @param gradient_transposed  The transposed version of the gradient vector.
         */
        virtual void
        ComputeReducedGradient(const ControlVector<VECTOR>& q, ControlVector<
            VECTOR>& gradient, ControlVector<VECTOR>& gradient_transposed)=0;

        /******************************************************/

        /**
         * Basic function to return the computed value of the reduced cost functional.
         *
         * @param q            The ControlVector is given to this function.
         */
        virtual double
        ComputeReducedCostFunctional(const ControlVector<VECTOR>& q)=0;

        /******************************************************/

        /**
         * Basic function to compute reduced functionals.
         * We assume that state u(q) is already computed.
         *
         * @param q            The ControlVector is given to this function.
         */
        virtual void
        ComputeReducedFunctionals(const ControlVector<VECTOR>& q)=0;

        /******************************************************/

        /**
         * Basic function to compute the error indicators with
         * the DWR method and higher order interpolation to gain the weights.
         * We assume that the state is already computed,
         * whereas the dual solution will be computed inside this function.
         *
         * @param q                 The ControlVector is given to this function.
         * @param ref_ind           The Vector in which the function writes
         *                          the error indicators on the different cells.
         *                          This vector is resized to the number of cells
         *                          of the actual grid.
         * @param ee_state          Which terms of the error Identity should get computed
         *                          (i.e. primal-term, dual-term, both)?
         * @param weight_comp       How to compute the weights?
         *
         * @return                  The error in the previously specified functional.
         *
         */
        virtual float
        ComputeRefinementIndicators(const ControlVector<VECTOR>& q, Vector<
            float>& ref_ind, DOpEtypes::EE_state ee_state =
            DOpEtypes::EE_state::mixed,
            DOpEtypes::WeightComputation weight_comp =
                DOpEtypes::WeightComputation::higher_order_interpolation) = 0;

        /******************************************************/

        /**
         * Basic function to compute the reduced gradient solution.
         * We assume that adjoint state z(u(q)) is already computed.
         *
         * @param q                             The ControlVector is given to this function.
         * @param direction                     Documentation will follow later.
         * @param hessian_direction             Documentation will follow later.
         * @param hessian_direction_transposed  Documentation will follow later.
         */
        virtual void
        ComputeReducedHessianVector(const ControlVector<VECTOR>& q,
            const ControlVector<VECTOR>& direction,
            ControlVector<VECTOR>& hessian_direction,
            ControlVector<VECTOR>& hessian_direction_transposed)=0;

        virtual void
        ComputeReducedHessianInverseVector(const ControlVector<VECTOR>& q __attribute__((unused)),
            const ControlVector<VECTOR>& direction __attribute__((unused)),
            ControlVector<VECTOR>& hessian_direction __attribute__((unused)))
        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::ComputeReducedHessianInverseVector");
        }

        /**
         * We assume that the constraints g have been evaluated at the corresponding
         * point q. This comutes the reduced gradient of the global constraint num
         * with respect to the control variable.
         *
         * @param num                           Number of the global constraint to which we want to
         *                                      compute the gradient.
         * @param q                             The ControlVector<VECTOR> is given to this function.
         * @param g                             The ConstraintVector<VECTOR> which contains the
         *                                      value of the constraints at q.
         * @param gradient                      The vector where the gradient will be stored in.
         * @param gradient_transposed           The transposed version of the gradient vector.
         */
        virtual void
        ComputeReducedGradientOfGlobalConstraints(unsigned int /*num*/,
            const ControlVector<VECTOR>& /*q*/,
            const ConstraintVector<VECTOR>& /*g*/,
            ControlVector<VECTOR>& /*gradient*/, ControlVector<VECTOR>& /*gradient_transposed*/)

        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::ComputeReducedGradientOfGlobalConstraints");
        }
        virtual bool
        IsEpsilonFeasible(const ConstraintVector<VECTOR>& g __attribute__((unused)), double p __attribute__((unused)))
        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::IsEpsilonFeasible");
        }

        virtual double
        GetMaxViolation(const ConstraintVector<VECTOR>& /*g*/)
        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::GetMaxViolation");
        }
        virtual void
        FeasibilityShift(const ControlVector<VECTOR>& /*g_hat*/, ControlVector<
            VECTOR>& /*g*/, double /*lambda*/)
        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::FeasibilityShift");
        }
        virtual double
        Complementarity(const ConstraintVector<VECTOR>& /*f*/,
            const ConstraintVector<VECTOR>& /*g*/)
        {
          throw DOpEException("Method not implemented",
              "ReducedProblemInterface::Complementarity");
        }

        /*****************************************************/
        /**
         * Sets the type of the Problem _OP. This function secures the proper initialization of the
         * FEValues after the type has changed. See also the documentation of SetType in optproblem.h
         */
        void
        SetProblemType(std::string type, unsigned int num = 0)
        {
          this->GetProblem()->SetType(type, num);
        }
        PROBLEM*
        GetProblem()
        {
          return _OP;
        }
      protected:

        std::string
        GetPostIndex()
        {
          return _post_index;
        }
        int
        GetBasePriority()
        {
          return _base_priority;
        }
      private:
        PROBLEM* _OP;
        int _base_priority;
        std::string _post_index;
    };

}
#endif
