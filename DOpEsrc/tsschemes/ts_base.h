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

#ifndef TSBase_H_
#define TSBase_H_

#include <deal.II/lac/vector.h>

namespace DOpE
{
  /**
   * This class contains the methods which all time stepping schemes share.
   *
   * @tparam <OPTPROBLEM>       The problem to deal with.
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state.
   * @tparam <VECTOR>           The vector type for control & state
   *                            (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <dealdim>          The dimension of the state variable.
   * @tparam <FE>               The type of finite elements in use, must be compatible with the DH.
   * @tparam <DH>               The type of the DoFHandler in use
   *                            (to be more precise: The type of the dealii-DoFhandler which forms
   *                            the base class of the DOpEWrapper::DoFHandler in use.)
   */

  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
           int dealdim,
           template <int, int> class FE = dealii::FESystem,
           template <int, int> class DH = dealii::DoFHandler>
  class TSBase
  {
  public:
    TSBase(OPTPROBLEM &OP) :
      OP_(OP)
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
      part_ = s;
    }

    /******************************************************/

    /**
     * Sets the actual time.
     *
     * @param time      The actual time.
    * @param time_dof_number The dofnumber in time associated to the vectors
     * @param interval  The actual interval. Make sure that time
     *                  lies in interval!
     * @param initial   Do we solve at the initial time?
     */

    void
    SetTime(double time,
            unsigned int time_dof_number,
            const TimeIterator &interval, bool initial = false)
    {
      OP_.SetTime(time, time_dof_number, interval,initial);
    }

    /******************************************************/

    /**
     * Returns just OP_.ElementFunctional(...). For more information we refer to
     * the file optproblemcontainer.h
     */
    template<typename DATACONTAINER>
    double
    ElementFunctional(const DATACONTAINER &dc)
    {
      return OP_.ElementFunctional(dc);
    }

    /******************************************************/

    /**
     *  Returns just OP_.PointFunctional(...). For more information we refer to
     * the file optproblemcontainer.h
     */

    double
    PointFunctional(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values)
    {
      return OP_.PointFunctional(param_values, domain_values);
    }

    /******************************************************/

    /**
     * Not implemented so far. Returns just OP_.BoundaryFunctional(...). For more information we refer to
     * the file optproblemcontainer.h
     */
    template<typename FACEDATACONTAINER>
    double
    BoundaryFunctional(const FACEDATACONTAINER &fdc)
    {
      return OP_.BoundaryFunctional(fdc);
    }

    /******************************************************/

    /**
     * Not implemented so far. Returns just OP_.FaceFunctional(...). For more information we refer to
     * the file optproblemcontainer.h
     */
    template<typename FACEDATACONTAINER>
    double
    FaceFunctional(const FACEDATACONTAINER &fdc)
    {
      return OP_.FaceFunctional(fdc);
    }

    /******************************************************/

    /**
     * A pointer to the whole FESystem
     *
     * @return A const pointer to the FESystem()
     */
    const dealii::SmartPointer<const dealii::FESystem<dealdim> >
    GetFESystem() const
    {
      return OP_.GetFESystem();
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
      return OP_.HasFaces();
    }

    /******************************************************/
    /**
     * See optproblem.h.
     */
    bool
    HasPoints() const
    {
      return OP_.HasPoints();
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
      return OP_.HasInterfaces();
    }

    /******************************************************/
    /**
    * see pdeinterface.h
    */
    template<typename ELEMENTITERATOR>
    bool AtInterface(ELEMENTITERATOR &element, unsigned int face) const
    {
      return OP_.AtInterface(element,face);
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
      return OP_.GetUpdateFlags();
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
      return OP_.GetFaceUpdateFlags();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of Dirichlet boundaries.
     *
     * @return Returns the Dirichlet Colors.
     */
    const std::vector<unsigned int> &
    GetDirichletColors() const
    {
      return OP_.GetDirichletColors();
    }

    /******************************************************/

    /**
     * A std::vector of boolean values to decide at which parts of the boundary and solutions variables
     * Dirichlet values should be applied.
     *
     * @return Returns a component mask for each boundary color.
     */
    const std::vector<bool> &
    GetDirichletCompMask(unsigned int color) const
    {
      return OP_.GetDirichletCompMask(color);
    }

    /******************************************************/

    /**
     * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply
     * on each boundary part with color 'color'.
     *
     * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
     */
    const dealii::Function<dealdim> &
    GetDirichletValues(unsigned int color,
                       const std::map<std::string, const dealii::Vector<double>*> &param_values,
                       const std::map<std::string, const VECTOR *> &domain_values) const
    {
      return OP_.GetDirichletValues(color, param_values, domain_values);
    }

    /******************************************************/

    /**
     * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
     * problem, respectively.
     *
     * @return Returns a dealii::Function of initial values.
     */
    const dealii::Function<dealdim> &
    GetInitialValues() const
    {
      return OP_.GetInitialValues();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of the boundary equation.
     *
     * @return Returns colors for the boundary equation.
     */
    const std::vector<unsigned int> &
    GetBoundaryEquationColors() const
    {
      return OP_.GetBoundaryEquationColors();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of the boundary functionals.
     *
     * @return Returns colors for the boundary functionals.
     */
    const std::vector<unsigned int> &
    GetBoundaryFunctionalColors() const
    {
      return OP_.GetBoundaryFunctionalColors();
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
      return OP_.GetNFunctionals();
    }

    /******************************************************/

    /**
     * This function gets the number of blocks considered in the PDE problem.
     *
     * @return Returns the number of blocks.
     */
    unsigned int
    GetNBlocks() const
    {
      return OP_.GetNBlocks();
    }

    /******************************************************/

    /**
     * A std::vector which contains the number of degrees of freedom per block.
     *
     * @return Returns a vector with DoFs.
     */
    const std::vector<unsigned int> &
    GetDoFsPerBlock() const
    {
      return OP_.GetDoFsPerBlock();
    }

    /******************************************************/

    /**
     * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
     *
     * @return Returns a matrix with hanging node constraints.
     */
    const dealii::ConstraintMatrix &
    GetDoFConstraints() const
    {
      return OP_.GetDoFConstraints();
    }

    std::string
    GetType() const
    {
      return OP_.GetType();
    }
    std::string
    GetDoFType() const
    {
      return OP_.GetDoFType();
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
      return OP_.GetFunctionalType();
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
      return OP_.GetFunctionalName();
    }

    /******************************************************/

    /**
     * A pointer to the OutputHandler() object.
     *
     * @return The OutputHandler() object.
     */
    DOpEOutputHandler<VECTOR> *
    GetOutputHandler()
    {
      return OP_.GetOutputHandler();
    }

    /******************************************************/

    /**
     * A pointer to the SpaceTimeHandler  object.
     *
     * @return The SpaceTimeHandler() object.
     */
    const auto *
    GetSpaceTimeHandler() const
    {
      return OP_.GetBaseProblem().GetSpaceTimeHandler();
    }
    auto *
    GetSpaceTimeHandler()
    {
      return OP_.GetBaseProblem().GetSpaceTimeHandler();
    }

    /******************************************************/

    void
    ComputeSparsityPattern(SPARSITYPATTERN &sparsity) const
    {
      OP_.ComputeSparsityPattern(sparsity);
    }
  protected:
    /******************************************************/
    /**
     * Return the problem.
     */
    OPTPROBLEM &
    GetProblem()
    {
      return OP_;
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
      return part_;
    }

  private:
    OPTPROBLEM &OP_;
    std::string part_;
  };
}
#endif
