/*
 * TSBase.h
 *
 *  Created on: 04.05.2012
 *      Author: cgoll
 */

#ifndef _TSBase_H_
#define _TSBase_H_

#include "lac/vector.h"

namespace DOpE
{

  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim,
      typename FE = DOpEWrapper::FiniteElement<dealdim>,
      typename DOFHANDLER = dealii::DoFHandler<dealdim> >
    class TSBase
    {
      public:
        TSBase(OPTPROBLEM& OP) :
            _OP(OP)
        {
        }
        ;
        ~TSBase()
        {
        }
        ;

        /******************************************************/
        /**
         * Sets the step part which should actually computed, e.g.,
         * previous solution within the NewtonStepSolver or
         * last time step solutions.
         * @param s    Name of the step part
         */
        void
        SetStepPart(std::string s)
        {
          _part = s;
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
        SetTime(double time, const TimeIterator& interval)
        {
          _OP.SetTime(time, interval);
        }

        /******************************************************/

        /**
         * Returns just _OP.CellFunctional(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename DATACONTAINER>
          double
          CellFunctional(const DATACONTAINER& dc)
          {
            return _OP.CellFunctional(dc);
          }

        /******************************************************/

        /**
         *  Returns just _OP.PointFunctional(...). For more information we refer to
         * the file optproblemcontainer.h
         */

        double
        PointFunctional(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values)
        {
          return _OP.PointFunctional(param_values, domain_values);
        }

        /******************************************************/

        /**
         * Not implemented so far. Returns just _OP.BoundaryFunctional(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          double
          BoundaryFunctional(const FACEDATACONTAINER& fdc)
          {
            return _OP.BoundaryFunctional(fdc);
          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just _OP.FaceFunctional(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          double
          FaceFunctional(const FACEDATACONTAINER& fdc)
          {
            return _OP.FaceFunctional(fdc);
          }

        /******************************************************/

        /**
         * A pointer to the whole FESystem
         *
         * @return A const pointer to the FESystem()
         */
        const dealii::SmartPointer<const DOpEWrapper::FiniteElement<dealdim> >
        GetFESystem() const
        {
          return _OP.GetFESystem();
        }

        /******************************************************/

        /**
         * This function determines whether a loop over all faces is required or not.
         *
         * @return Returns whether or not this functional has components on faces between elements.
         *         The default value is false.
         */
        bool
        HasFaces() const
        {
          return _OP.HasFaces();
        }

        /******************************************************/
        /**
         * See optproblem.h.
         */
        bool
        HasPoints() const
        {
          return _OP.HasPoints();
        }

        /******************************************************/
        /**
         * This function determines whether a loop over all faces is required or not.
         *
         * @return Returns whether or not this functional has components on faces between elements.
         *         The default value is false.
         */
        bool
        HasInterfaces() const
        {
          return _OP.HasInterfaces();
        }

        /******************************************************/

        /**
         * This function returns the update flags for domain values
         * for the computation of shape values, gradients, etc.
         * For detailed explication, please visit `Finite element access/FEValues classes' in
         * the deal.ii manual.
         *
         * @return Returns the update flags to use in a computation.
         */
        dealii::UpdateFlags
        GetUpdateFlags() const
        {
          return _OP.GetUpdateFlags();
        }

        /******************************************************/

        /**
         * This function returns the update flags for face values
         * for the computation of shape values, gradients, etc.
         * For detailed explication, please visit
         * `FEFaceValues< dim, spacedim > Class Template Reference' in
         * the deal.ii manual.
         *
         * @return Returns the update flags for faces to use in a computation.
         */
        dealii::UpdateFlags
        GetFaceUpdateFlags() const
        {
          return _OP.GetFaceUpdateFlags();
        }

        /******************************************************/

        /**
         * A std::vector of integer values which contains the colors of Dirichlet boundaries.
         *
         * @return Returns the Dirichlet Colors.
         */
        const std::vector<unsigned int>&
        GetDirichletColors() const
        {
          return _OP.GetDirichletColors();
        }

        /******************************************************/

        /**
         * A std::vector of boolean values to decide at which parts of the boundary and solutions variables
         * Dirichlet values should be applied.
         *
         * @return Returns a component mask for each boundary color.
         */
        const std::vector<bool>&
        GetDirichletCompMask(unsigned int color) const
        {
          return _OP.GetDirichletCompMask(color);
        }

        /******************************************************/

        /**
         * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply
         * on each boundary part with color 'color'.
         *
         * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
         */
        const dealii::Function<dealdim>&
        GetDirichletValues(unsigned int color,
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values) const
        {
          return _OP.GetDirichletValues(color,/*control_dof_handler,state_dof_handler,*/
          param_values, domain_values);
        }

        /******************************************************/

        /**
         * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
         * problem, respectively.
         *
         * @return Returns a dealii::Function of initial values.
         */
        const dealii::Function<dealdim>&
        GetInitialValues() const
        {
          return _OP.GetInitialValues();
        }

        /******************************************************/

        /**
         * A std::vector of integer values which contains the colors of the boundary equation.
         *
         * @return Returns colors for the boundary equation.
         */
        const std::vector<unsigned int>&
        GetBoundaryEquationColors() const
        {
          return _OP.GetBoundaryEquationColors();
        }

        /******************************************************/

        /**
         * A std::vector of integer values which contains the colors of the boundary functionals.
         *
         * @return Returns colors for the boundary functionals.
         */
        const std::vector<unsigned int>&
        GetBoundaryFunctionalColors() const
        {
          return _OP.GetBoundaryFunctionalColors();
        }

        /******************************************************/

        /**
         * This function returns the number of functionals to be considered in the problem.
         *
         * @return Returns the number of functionals.
         */
        unsigned int
        GetNFunctionals() const
        {
          return _OP.GetNFunctionals();
        }

        /******************************************************/

        /**
         * This function gets the number of blocks considered in the PDE problem.
         * Example 1: in fluid problems we have to find velocities and pressure
         * --> number of blocks is 2.
         * Example 2: in FSI problems we have to find velocities, displacements, and pressure.
         *  --> number of blocks is 3.
         *
         * @return Returns the number of blocks.
         */
        unsigned int
        GetNBlocks() const
        {
          return _OP.GetNBlocks();
        }

        /******************************************************/

        /**
         * A function which has the number of degrees of freedom for the block `b'.
         *
         * @return Returns the number of DoFs for block `b'.
         */
        unsigned int
        GetDoFsPerBlock(unsigned int b) const
        {
          return _OP.GetDoFsPerBlock(b);
        }

        /******************************************************/

        /**
         * A std::vector which contains the number of degrees of freedom per block.
         *
         * @return Returns a vector with DoFs.
         */
        const std::vector<unsigned int>&
        GetDoFsPerBlock() const
        {
          return _OP.GetDoFsPerBlock();
        }

        /******************************************************/

        /**
         * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
         *
         * @return Returns a matrix with hanging node constraints.
         */
        const dealii::ConstraintMatrix&
        GetDoFConstraints() const
        {
          return _OP.GetDoFConstraints();
        }

        std::string
        GetType() const
        {
          return _OP.GetType();
        }
        std::string
        GetDoFType() const
        {
          return _OP.GetDoFType();
        }

        /******************************************************/

        /**
         * This function describes what type of Functional is considered
         *
         * @return A string describing the functional, feasible values are "domain", "boundary", "point" or "face"
         *         if it contains domain, or boundary ... parts all combinations of these keywords are feasible.
         *         In time dependent problems use "timelocal" to indicate that
         *         it should only be evaluated at a certain time_point, or "timedistributed" to consider \int_0^T J(t,q(t),u(t))  \dt
         *         only one of the words "timelocal" and "timedistributed" should be considered if not it will be considered to be
         *         "timelocal"
         *
         */
        std::string
        GetFunctionalType() const
        {
          return _OP.GetFunctionalType();
        }

        /******************************************************/

        /**
         * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
         *
         * @return A string. This is the name beeing displayed next to the computed values.
         */
        std::string
        GetFunctionalName() const
        {
          return _OP.GetFunctionalName();
        }

        /******************************************************/

        /**
         * A pointer to the OutputHandler() object.
         *
         * @return The OutputHandler() object.
         */
        DOpEOutputHandler<VECTOR>*
        GetOutputHandler()
        {
          return _OP.GetOutputHandler();
        }

        /******************************************************/

        /**
         * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
         *
         * @return The SpaceTimeHandler() object.
         */
        const SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>*
        GetSpaceTimeHandler() const
        {
          return _OP.GetBaseProblem().GetSpaceTimeHandler();
        }
        SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>*
        GetSpaceTimeHandler()
        {
          return _OP.GetBaseProblem().GetSpaceTimeHandler();
        }

        /******************************************************/

        void
        ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const
        {
          _OP.ComputeSparsityPattern(sparsity);
        }
      protected:
        /******************************************************/
        /**
         * Return the problem.
         */
        OPTPROBLEM&
        GetProblem()
        {
          return _OP;
        }
        /******************************************************/

        /**
         * Sets the step part which should actually computed, e.g.,
         * previous solution within the NewtonStepSolver or
         * last time step solutions.
         * @param s    Name of the step part
         */
        std::string
        GetPart() const
        {
          return _part;
        }

      private:
        OPTPROBLEM& _OP;
        std::string _part;
    };
}
#endif
