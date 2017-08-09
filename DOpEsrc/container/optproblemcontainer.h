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

#ifndef OptProblemContainer_H_
#define OptProblemContainer_H_

#include <include/dopeexceptionhandler.h>
#include <include/outputhandler.h>
#include <interfaces/functionalinterface.h>
#include <wrapper/dofhandler_wrapper.h>
#include <wrapper/fevalues_wrapper.h>
#include <wrapper/function_wrapper.h>
#include <basic/spacetimehandler.h>
#include <problemdata/primaldirichletdata.h>
#include <problemdata/tangentdirichletdata.h>
#include <interfaces/transposeddirichletdatainterface.h>
#include <problemdata/transposedgradientdirichletdata.h>
#include <problemdata/transposedhessiandirichletdata.h>
#include <include/constraintvector.h>
#include <include/controlvector.h>
#include <include/statevector.h>
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <problemdata/stateproblem.h>
#include <problemdata/tangentproblem.h>
#include <problemdata/adjointproblem.h>
#include <problemdata/adjoint_hessianproblem.h>
#include <problemdata/opt_adjoint_for_eeproblem.h>
#include <basic/dopetypes.h>
#include <container/dwrdatacontainer.h>
#include <container/problemcontainer_internal.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#if DEAL_II_VERSION_GTE(8,5,0)
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#else
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#endif

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
   * Container class for all stationary Optimization problems.
   * This class collects all problem depended data needed to
   * calculate the solution to the optimization problem
   *
   * @tparam FUNCTIONAL_INTERFACE   A generic interface to arbitrary functionals to be evaluated.
   * @tparam FUNCTIONAL             The cost functional, see FunctionalInterface for details.
   * @tparam PDE                    The description of the PDE, see PDEInterface for details.
   * @tparam DD                     The description of the Dirichlet data, see
   *                                DirichletDataInterface for details.
   * @tparam CONSTRAINTS            The description of, possible, additional constraints for the
   *                                optimization problem, see ConstraintInterface for details.
   * @tparam SPARSITYPATTERN        The sparsity pattern to be used in the stiffness matrix.
   * @tparam VECTOR                 The vector type in which the coordinate vector of the
   *                                solution is to be stored.
   * @tparam dopedim                The dimension of the domain in which the control is considered.
   * @tparam dealdim                The dimension of the domain in which the PDE is considered.
   * @tparam FE                     The finite element under consideration.
   * @tparam DH                     The spatial DoFHandler to be used when evaluating the
   *                                weak form.
   */
  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
           typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
           typename VECTOR, int dopedim, int dealdim,
           template<int, int> class FE = dealii::FESystem,
           template<int, int> class DH = dealii::DoFHandler>
  class OptProblemContainer : public ProblemContainerInternal<PDE>
  {
  public:
    OptProblemContainer(FUNCTIONAL &functional, PDE &pde,
                        CONSTRAINTS &constraints,
                        SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim> &STH);

    /******************************************************/

    virtual ~OptProblemContainer();

    /******************************************************/

    virtual std::string
    GetName() const
    {
      return "OptProblem";
    }

    /******************************************************/
    /**
     * Returns a description of the PDE
     */
    StateProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
                        GetStateProblem()
    {
      if (state_problem_ == NULL)
        {
          state_problem_ = new StateProblem<
          OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
          DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
            *this, this->GetPDE());
        }
      return *state_problem_;
    }

    /**
     * Returns a description of the Tangent PDE
     */
    TangentProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
    CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
    PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
    GetTangentProblem()
    {
      if (tangent_problem_ == NULL)
        {
          tangent_problem_ = new TangentProblem<
          OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
          DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
            *this, this->GetPDE());
        }
      return *tangent_problem_;
    }
    /**
     * Returns a description of the Adjoint PDE for error estimation
     */
    AdjointProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
    CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
    PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
    GetAdjointProblem()
    {
      if (adjoint_problem_ == NULL)
        {
          adjoint_problem_ = new AdjointProblem<
          OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
          DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
            *this, this->GetPDE());
        }
      return *adjoint_problem_;
    }
    /**
     * Returns a description of the Adjoint for Hessian PDE for error estimation
     */
    Adjoint_HessianProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
    CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
    PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
    GetAdjoint_HessianProblem()
    {
      if (adjoint_hessian_problem_ == NULL)
        {
          adjoint_hessian_problem_ = new Adjoint_HessianProblem<
          OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
          DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
            *this, this->GetPDE());
        }
      return *adjoint_hessian_problem_;
    }

    /**
     * Returns a description of the Adjoint PDE for error estimation
     */
    OPT_Adjoint_For_EEProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
    CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
    PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>&
    GetAdjoint_For_EEProblem()
    {
      if (adjoint_for_ee_problem_ == NULL)
        {
          adjoint_for_ee_problem_ = new OPT_Adjoint_For_EEProblem<
          OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
          CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
          DH>, PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>(
            *this, this->GetPDE());
        }
      return *adjoint_for_ee_problem_;
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
    RegisterOutputHandler(DOpEOutputHandler<VECTOR> *OH)
    {
      OutputHandler_ = OH;
    }

    /******************************************************/

    void
    RegisterExceptionHandler(DOpEExceptionHandler<VECTOR> *OH)
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
     * have been implemented so far: `cost_functional' and `aux_functional'.
     * The first one is needed for optimization problems. The second one
     * can be used for the computation of arbitrary functionals that
     * are defined on elements, e.g., drag and lift computation.
     * Or computations of deflections and deformations.
     *
     * @template DATACONTAINER    Class of the datacontainer, distinguishes
     *                                between hp- and classical case.
     *
     * @param edc                     A DataContainer holding all the needed information
     *                                of the element.
     */
    template<typename DATACONTAINER>
    double
    ElementFunctional(const DATACONTAINER &edc);

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
      const std::map<std::string, const VECTOR *> &domain_values);

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
    BoundaryFunctional(const FACEDATACONTAINER &fdc);

    /******************************************************/

    /**
     * This function returns a functional value of quantities that
     * are defined on faces. This function is very similar to the
     * BoundaryFunctional and has the same functionality.
     */

    template<typename FACEDATACONTAINER>
    double
    FaceFunctional(const FACEDATACONTAINER &fdc);

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
      const std::map<std::string, const VECTOR *> &block_values);

    /**
     * This function returns a residual to an equation that is depending
    * directly on the knowledge of the coordinate vectors of state and control.
    *
    * No integration routine is implemented inbetween!
     */
    void
    AlgebraicResidual(VECTOR &residual,
                      const std::map<std::string, const dealii::Vector<double>*> &values,
                      const std::map<std::string, const VECTOR *> &block_values);

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
     *                                of the element.
     * @param local_vector       This vector contains the locally computed values of the element equation. For more information
     *                                on dealii::Vector, please visit, the deal.ii manual pages.
     * @param scale                   A scaling factor which is -1 or 1 depending on the subroutine to compute.
     * @param scale_ico             A scaling factor for terms which will be treated fully implicit
     *                              in an instationary equation.
     */
    template<typename DATACONTAINER>
    void
    ElementEquation(const DATACONTAINER &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double scale_ico);

    /******************************************************/

    /**
     * This function has the same functionality as the ElementEquation function.
     * It is needed for time derivatives when working with
     * time dependent problems.
     */
    template<typename DATACONTAINER>
    void
    ElementTimeEquation(const DATACONTAINER &dc,
                        dealii::Vector<double> &local_vector, double scale = 1.);

    /******************************************************/

    /**
     * This function has the same functionality as the ElementTimeEquation function.
     * It is mainly needed for fluid-structure interaction problems and should
     * be used when the term of the time derivative contains
     * nonlinear terms, i.e. $\partial_t u v + ...$
     * in which u and v denote solution variables.
     * Secondly, this function should be used when the densities
     * are not constant: $\partial_t \rho v + ...$
     */
    template<typename DATACONTAINER>
    void
    ElementTimeEquationExplicit(const DATACONTAINER &dc,
                                dealii::Vector<double> &local_vector, double scale = 1.);

    /******************************************************/

    /**
     * Computes the value of the right-hand side of the problem at hand.
     *
     * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
     *                                 between hp- and classical case.
     *
     * @param edc                      A DataContainer holding all the needed information of the element.
     * @param local_vector        This vector contains the locally computed values of the element equation. For more information
     *                                 on dealii::Vector, please visit, the deal.ii manual pages.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
     */
    template<typename DATACONTAINER>
    void
    ElementRhs(const DATACONTAINER &dc,
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
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs_vector, double scale = 1.);

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
     * @template DATACONTAINER         Class of the datacontainer in use, distinguishes
     *                                 between hp- and classical case.
     *
     * @param edc                      A DataContainer holding all the needed information
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
    ElementMatrix(const DATACONTAINER &dc,
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
    ElementTimeMatrix(const DATACONTAINER &dc,
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
    ElementTimeMatrixExplicit(const DATACONTAINER &dc,
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
    FaceEquation(const FACEDATACONTAINER &dc,
                 dealii::Vector<double> &local_vector, double scale,
                 double scale_ico);

    /******************************************************/
    /**
     * Computes the product of two different finite elements
     * on a interior face. It has the same functionality as ElementEquation.
     * We refer to its documentation.
     *
     */
    //FIXME maybe InterfaceEquation and InterfaceMatrix could get
    //integrated into FaceEquation and FaceMatrix?
    template<typename FACEDATACONTAINER>
    void
    InterfaceEquation(const FACEDATACONTAINER &dc,
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
    FaceRhs(const FACEDATACONTAINER &dc,
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
    FaceMatrix(const FACEDATACONTAINER &dc,
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
    InterfaceMatrix(const FACEDATACONTAINER &dc,
                    dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
                    double scale_ico = 1.);

    /******************************************************/

    /**
     * Computes the value of the boundary on a element.
     * It has the same functionality as ElementEquation. We refer to its
     * documentation.
     *
     */
    template<typename FACEDATACONTAINER>
    void
    BoundaryEquation(const FACEDATACONTAINER &dc,
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
    BoundaryRhs(const FACEDATACONTAINER &dc,
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
    BoundaryMatrix(const FACEDATACONTAINER &dc,
                   dealii::FullMatrix<double> &local_matrix, double scale = 1.,
                   double scale_ico = 1.);
    /******************************************************/
    void
    ComputeLocalControlConstraints(VECTOR &constraints,
                                   const std::map<std::string, const dealii::Vector<double>*> &values,
                                   const std::map<std::string, const VECTOR *> &block_values);
    /******************************************************/

    /**
     * This is to get the lower and upper control box constraints
     *

     */
    void
    GetControlBoxConstraints(VECTOR &lb, VECTOR &ub) const
    {
      this->GetConstraints()->GetControlBoxConstraints(lb, ub);
    }

    /******************************************************/

    const FE<dealdim, dealdim> &
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
                                      const std::vector<bool> &comp_mask,
                                      const DOpEWrapper::Function<dealdim> *values);

    /******************************************************/

    void
    SetDirichletBoundaryColors(unsigned int color,
                               const std::vector<bool> &comp_mask, const DD *values);

    /******************************************************/

    const std::vector<unsigned int> &
    GetDirichletColors() const;
    const std::vector<unsigned int> &
    GetTransposedDirichletColors() const;
    const std::vector<bool> &
    GetDirichletCompMask(unsigned int color) const;
    const std::vector<bool> &
    GetTransposedDirichletCompMask(unsigned int color) const;

    /******************************************************/

    const dealii::Function<dealdim> &
    GetDirichletValues(unsigned int color,
                       const std::map<std::string, const dealii::Vector<double>*> &param_values,
                       const std::map<std::string, const VECTOR *> &domain_values) const;

    /******************************************************/

    const TransposedDirichletDataInterface<dealdim> &
    GetTransposedDirichletValues(unsigned int color,
                                 const std::map<std::string, const dealii::Vector<double>*> &param_values,
                                 const std::map<std::string, const VECTOR *> &domain_values) const;

    /******************************************************/

    void
    SetInitialValues(const dealii::Function<dealdim> *values)
    {
      assert(values->n_components==this->GetPDE().GetStateNComponents());
      initial_values_ = values;
    }
    const dealii::Function<dealdim> &
    GetInitialValues() const
    {
      return *initial_values_;
    }

    /******************************************************/

    void
    SetControlBoundaryEquationColors(unsigned int color);
    void
    SetBoundaryEquationColors(unsigned int color);
    const std::vector<unsigned int> &
    GetBoundaryEquationColors() const;
    void
    SetBoundaryFunctionalColors(unsigned int color);
    const std::vector<unsigned int> &
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
    AddFunctional(FUNCTIONAL_INTERFACE *F)
    {
      aux_functionals_.push_back(F);
      if (functional_position_.find(F->GetName())
          != functional_position_.end())
        {
          throw DOpEException(
            "You cant add two functionals with the same name.",
            "OPTProblemContainer::AddFunctional");
        }
      functional_position_[F->GetName()] = aux_functionals_.size();
      //remember! At functional_values_[0] we store always the cost functional!
    }

    /******************************************************/

    /**
     * Through this function one sets the functional for the
     * error estimation. The name given by functional_name is
     * looked up in aux_functionals_, so the function assumes
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
          functional_for_ee_is_cost_ = true;
          functional_for_ee_num_ = dealii::numbers::invalid_unsigned_int;
        }
      else
        {
          functional_for_ee_is_cost_ = false;
          bool found = false;
          //we go through all aux functionals.
          for (unsigned int i = 0; i < this->GetNFunctionals(); i++)
            {
              if (aux_functionals_[i]->GetName() == functional_name)
                {
                  //if the names match, we have found our functional.
                  found = true;
                  functional_for_ee_num_ = i;
                }
            }
          //If we have not found a functional with the given name,
          //we throw an error.
          if (!found)
            {
              throw DOpEException(
                "Can't find functional " + functional_name
                + " in aux_functionals_",
                "Optproblem::SetFunctionalForErrorEstimation");
            }
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

    const std::vector<unsigned int> &
    GetDoFsPerBlock() const;

    /******************************************************/

    const dealii::ConstraintMatrix &
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
    unsigned int FunctionalNeedPrecomputations() const;

    bool FunctionalNeedFinalValue() const;

    /******************************************************/

    bool
    NeedTimeFunctional() const;

    /******************************************************/

    bool
    HasControlInDirichletData() const;

    /******************************************************/

    DOpEExceptionHandler<VECTOR> *
    GetExceptionHandler()
    {
      assert(ExceptionHandler_);
      return ExceptionHandler_;
    }

    /******************************************************/

    DOpEOutputHandler<VECTOR> *
    GetOutputHandler()
    {
      assert(OutputHandler_);
      return OutputHandler_;
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
            const TimeIterator &interval, bool initial = false);

    /******************************************************/

    const SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim> *
    GetSpaceTimeHandler() const
    {
      return STH_;
    }

    /******************************************************/

    SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim> *
    GetSpaceTimeHandler()
    {
      return STH_;
    }

    /******************************************************/

    void
    ComputeSparsityPattern(SPARSITYPATTERN &sparsity) const;

    /******************************************************/

    void
    PostProcessConstraints(ConstraintVector<VECTOR> &g) const;

    /******************************************************/

    void
    AddAuxiliaryControl(const ControlVector<VECTOR> *c, std::string name);

    /******************************************************/

    void
    AddAuxiliaryState(const StateVector<VECTOR> *c, std::string name);

    /******************************************************/

    void
    AddAuxiliaryConstraint(const ConstraintVector<VECTOR> *c,
                           std::string name);

    /******************************************************/

    const ControlVector<VECTOR> *
    GetAuxiliaryControl(std::string name) const;

    /******************************************************/

    const StateVector<VECTOR> *
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

    const ConstraintVector<VECTOR> *
    GetAuxiliaryConstraint(std::string name)
    {
      typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
        auxiliary_constraints_.find(name);
      if (it == auxiliary_constraints_.end())
        {
          throw DOpEException("Could not find data" + name,
                              "OptProblemContainer::GetAuxiliaryConstraint");
        }
      return it->second;
    }
    /*****************************************************************/
    /**
    * Adds the auxiliary Vectors from the integrator, so that their values are
    * available for the integrated object.
    *
    * @param integrator         The integrator in which the vecors should be available
    *
    */

    template<typename INTEGRATOR>
    void
    AddAuxiliaryToIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
          auxiliary_controls_.begin();
        for (; it != auxiliary_controls_.end(); it++)
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
          auxiliary_state_.begin();
        for (; it != auxiliary_state_.end(); it++)
          {
            integrator.AddDomainData(it->first,
                                     &(it->second->GetSpacialVector()));
          }
      }
      {
        typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
          auxiliary_constraints_.begin();
        for (; it != auxiliary_constraints_.end(); it++)
          {
            integrator.AddDomainData(it->first + "_local",
                                     &(it->second->GetSpacialVector("local")));
            integrator.AddParamData(it->first + "_global",
                                    &(it->second->GetGlobalConstraints()));
          }
      }
    }

    /*****************************************************************/
    /**
     * Adds the auxiliary Vectors from the integrator, so that their values are
     * available for the integrated object. Takes the values from the next (in
     * natural time direction)
     * time step to the integrator.
     *
     * This adds only the vector named "state" and no other vectors!
     *
     * @param integrator         The integrator in which the vecors should be available
     *
     */

    template<typename INTEGRATOR>
    void
    AddNextAuxiliaryToIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
          auxiliary_state_.find("state");
        if (it != auxiliary_state_.end())
          {
            integrator.AddDomainData("state_i+1",
                                     &(it->second->GetNextSpacialVector()));
          }
      }
    }
    /*****************************************************************/
    /**
     * Adds the auxiliary Vectors from the integrator, so that their values are
     * available for the integrated object. Takes the values from the previous (in
     * natural time direction)
     * time step to the integrator.
     *
     * This adds only the vector named "state" and no other vectors!
     *
     * @param integrator         The integrator in which the vecors should be available
     *
     */

    template<typename INTEGRATOR>
    void
    AddPreviousAuxiliaryToIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
          auxiliary_state_.find("state");
        if (it != auxiliary_state_.end())
          {
            integrator.AddDomainData("state_i-1",
                                     &(it->second->GetPreviousSpacialVector()));
          }
      }
    }

    /*****************************************************************/
    /**
     * Deletes the auxiliary Vectors from the integrator, so that their values are
     * available for the integrated object. Takes the values from the next (in
     * natural time direction)
     * time step to the integrator.
     *
     * This adds only the vector named "state" and no other vectors!
     *
     * @param integrator         The integrator in which the vecors should be available
     *
     */

    template<typename INTEGRATOR>
    void
    DeleteNextAuxiliaryFromIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
          auxiliary_state_.find("state");
        if (it != auxiliary_state_.end())
          {
            integrator.DeleteDomainData("state_i+1");
          }
      }
    }
    /*****************************************************************/
    /**
     * Adds the auxiliary Vectors from the integrator, so that their values are
     * available for the integrated object. Takes the values from the previous (in
     * natural time direction)
     * time step to the integrator.
     *
     * This adds only the vector named "state" and no other vectors!
     *
     * @param integrator         The integrator in which the vecors should be available
     *
     */

    template<typename INTEGRATOR>
    void
    DeletePreviousAuxiliaryFromIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
          auxiliary_state_.find("state");
        if (it != auxiliary_state_.end())
          {
            integrator.DeleteDomainData("state_i-1");
          }
      }
    }


    /******************************************************/
    /**
    * Deletes the auxiliary Vectors from the integrator.
    * This is required to add vecors of the same name but possibly
    * at a different point in time.
    *
    * @param integrator         The integrator in which the vecors should be available
    */

    template<typename INTEGRATOR>
    void
    DeleteAuxiliaryFromIntegrator(INTEGRATOR &integrator)
    {
      {
        typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
          auxiliary_controls_.begin();
        for (; it != auxiliary_controls_.end(); it++)
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
          auxiliary_state_.begin();
        for (; it != auxiliary_state_.end(); it++)
          {
            integrator.DeleteDomainData(it->first);
          }
      }
      {
        typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
          auxiliary_constraints_.begin();
        for (; it != auxiliary_constraints_.end(); it++)
          {
            integrator.DeleteDomainData(it->first + "_local");
            integrator.DeleteParamData(it->first + "_global");
          }
      }
    }

    /******************************************************/
    const std::map<std::string, unsigned int> &
    GetFunctionalPosition() const
    {
      return functional_position_;
    }

    unsigned int
    GetStateNBlocks()
    {
      return this->GetPDE().GetStateNBlocks();
    }

    /******************************************************/

    std::vector<unsigned int> &
    GetControlBlockComponent()
    {
      return this->GetPDE().GetControlBlockComponent();
    }
    /******************************************************/

    std::vector<unsigned int> &
    GetStateBlockComponent()
    {
      return this->GetPDE().GetStateBlockComponent();
    }

    /******************************************************/
    /******************************************************/

    /**
     * Returns whether the functional for error estimation
     * is the costfunctional
     */
    bool
    EEFunctionalIsCost() const
    {
      return functional_for_ee_is_cost_;
    }

    template<typename ELEMENTITERATOR>
    bool AtInterface(ELEMENTITERATOR &element, unsigned int face) const;

    /**
     * Initializes the HigherOrderDWRDataContainer
     */
    template<class DWRC>
    void
    InitializeDWRC(DWRC &dwrc)
    {
      dwrc.Initialize(GetSpaceTimeHandler(),
                      GetControlNBlocks(),
                      GetControlBlockComponent(),
                      GetStateNBlocks(),
                      GetStateBlockComponent(),
                      &control_dirichlet_colors_,
                      &control_dirichlet_comps_,
                      &dirichlet_colors_,
                      &dirichlet_comps_);
    }


  protected:
    FUNCTIONAL *
    GetFunctional();
    const FUNCTIONAL *
    GetFunctional() const;
    CONSTRAINTS *
    GetConstraints()
    {
      return constraints_;
    }
    const CONSTRAINTS *
    GetConstraints() const
    {
      return constraints_;
    }

    const VECTOR *
    GetBlockVector(const std::map<std::string, const VECTOR *> &values,
                   std::string name)
    {
      typename std::map<std::string, const VECTOR *>::const_iterator it =
        values.find(name);
      if (it == values.end())
        {
          throw DOpEException("Did not find " + name,
                              "OptProblemContainer::GetBlockVector");
        }
      return it->second;
    }
    const dealii::Vector<double> *
    GetVector(const std::map<std::string, const Vector<double>*> &values,
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
    DOpEExceptionHandler<VECTOR> *ExceptionHandler_;
    DOpEOutputHandler<VECTOR> *OutputHandler_;
    std::string algo_type_;

    bool functional_for_ee_is_cost_;
    double c_interval_length_, interval_length_;
    unsigned int functional_for_ee_num_;
    std::vector<FUNCTIONAL_INTERFACE *> aux_functionals_;
    std::map<std::string, unsigned int> functional_position_;
    FUNCTIONAL *functional_;
    CONSTRAINTS *constraints_;
    SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim> *STH_;

    std::vector<unsigned int> control_dirichlet_colors_;
    std::vector<unsigned int> control_transposed_dirichlet_colors_;
    std::vector<std::vector<bool> > control_dirichlet_comps_;
    std::vector<const DOpEWrapper::Function<dealdim>*> control_dirichlet_values_;
    std::vector<
    TransposedGradientDirichletData<DD, VECTOR, dealdim> *> transposed_control_gradient_dirichlet_values_;
    std::vector<
    TransposedHessianDirichletData<DD, VECTOR, dealdim> *> transposed_control_hessian_dirichlet_values_;

    std::vector<unsigned int> dirichlet_colors_;
    std::vector<std::vector<bool> > dirichlet_comps_;
    std::vector<PrimalDirichletData<DD, VECTOR, dealdim>*> primal_dirichlet_values_;
    std::vector<TangentDirichletData<DD, VECTOR, dealdim>*> tangent_dirichlet_values_;
    const dealii::Function<dealdim> *zero_dirichlet_values_;

    const dealii::Function<dealdim> *initial_values_;

    std::vector<unsigned int> control_boundary_equation_colors_;
    std::vector<unsigned int> state_boundary_equation_colors_;
    std::vector<unsigned int> adjoint_boundary_equation_colors_;

    std::vector<unsigned int> boundary_functional_colors_;

    std::map<std::string, const ControlVector<VECTOR>*> auxiliary_controls_;
    std::map<std::string, const StateVector<VECTOR>*> auxiliary_state_;
    std::map<std::string, const ConstraintVector<VECTOR>*> auxiliary_constraints_;

    bool initial_; //Do we solve the problem at initial time?

    StateProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> * state_problem_;
    TangentProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> * tangent_problem_;
    AdjointProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> * adjoint_problem_;
    Adjoint_HessianProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> * adjoint_hessian_problem_;
    OPT_Adjoint_For_EEProblem<
    OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                        CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> * adjoint_for_ee_problem_;

    friend class StateProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    friend class TangentProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    friend class AdjointProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    friend class Adjoint_HessianProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
    friend class OPT_Adjoint_For_EEProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim> ;
  };
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
           typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
           typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
           template<int, int> class DH>
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::OptProblemContainer(
                        FUNCTIONAL &functional, PDE &pde, CONSTRAINTS &constraints,
                        SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim, dealdim> &STH) :
                        ProblemContainerInternal<PDE>(pde), functional_(&functional), constraints_(
                          &constraints), STH_(&STH), state_problem_(NULL),
                        tangent_problem_(NULL),  adjoint_problem_(NULL),
                        adjoint_hessian_problem_(NULL), adjoint_for_ee_problem_(NULL)
  {
    ExceptionHandler_ = NULL;
    OutputHandler_ = NULL;
    zero_dirichlet_values_ = new ZeroFunction<dealdim>(
      this->GetPDE().GetStateNComponents());
    algo_type_ = "";
    functional_position_[functional_->GetName()] = 0;
    //remember! At functional_values_[0] we store always the cost functional!
    functional_for_ee_num_ = dealii::numbers::invalid_unsigned_int;
    c_interval_length_ = 1.;
    interval_length_ = 1.;
  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
           typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
           typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
           template<int, int> class DH>
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::~OptProblemContainer()
  {
    if (zero_dirichlet_values_ != NULL)
      {
        delete zero_dirichlet_values_;
      }

    for (unsigned int i = 0;
         i < transposed_control_gradient_dirichlet_values_.size(); i++)
      {
        if (transposed_control_gradient_dirichlet_values_[i] != NULL)
          delete transposed_control_gradient_dirichlet_values_[i];
      }
    for (unsigned int i = 0;
         i < transposed_control_hessian_dirichlet_values_.size(); i++)
      {
        if (transposed_control_hessian_dirichlet_values_[i] != NULL)
          delete transposed_control_hessian_dirichlet_values_[i];
      }
    for (unsigned int i = 0; i < primal_dirichlet_values_.size(); i++)
      {
        if (primal_dirichlet_values_[i] != NULL)
          delete primal_dirichlet_values_[i];
      }
    for (unsigned int i = 0; i < tangent_dirichlet_values_.size(); i++)
      {
        if (tangent_dirichlet_values_[i] != NULL)
          delete tangent_dirichlet_values_[i];
      }
    if (state_problem_ != NULL)
      {
        delete state_problem_;
      }
    if (tangent_problem_ != NULL)
      {
        delete tangent_problem_;
      }
    if (adjoint_problem_ != NULL)
      {
        delete adjoint_problem_;
      }
    if (adjoint_hessian_problem_ != NULL)
      {
        delete adjoint_hessian_problem_;
      }
    if (adjoint_for_ee_problem_ != NULL)
      {
        delete adjoint_for_ee_problem_;
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
    if (state_problem_ != NULL)
      {
        delete state_problem_;
        state_problem_ = NULL;
      }
    if (tangent_problem_ != NULL)
      {
        delete tangent_problem_;
        tangent_problem_ = NULL;
      }
    if (adjoint_problem_ != NULL)
      {
        delete adjoint_problem_;
        adjoint_problem_ = NULL;
      }
    if (adjoint_hessian_problem_ != NULL)
      {
        delete adjoint_hessian_problem_;
        adjoint_hessian_problem_ = NULL;
      }
    if (adjoint_for_ee_problem_ != NULL)
      {
        delete adjoint_for_ee_problem_;
        adjoint_for_ee_problem_ = NULL;
      }

    if (algo_type_ != algo_type && algo_type_ != "")
      {
        throw DOpEException("Conflicting Algorithms!",
                            "OptProblemContainer::ReInit");
      }
    else
      {
        algo_type_ = algo_type;
        this->SetTypeInternal("");

        if (algo_type_ == "reduced")
          {
            GetSpaceTimeHandler()->ReInit(this->GetPDE().GetControlNBlocks(),
                                          this->GetPDE().GetControlBlockComponent(),
                                          DirichletDescriptor(control_dirichlet_colors_,control_dirichlet_comps_),
                                          this->GetPDE().GetStateNBlocks(),
                                          this->GetPDE().GetStateBlockComponent(),
                                          DirichletDescriptor(dirichlet_colors_,dirichlet_comps_));
          }
        else
          {
            throw DOpEException("Unknown Algorithm " + algo_type_,
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
        if (functional_for_ee_num_ != dealii::numbers::invalid_unsigned_int)
          aux_functionals_[functional_for_ee_num_]->SetProblemType(type,num);
        this->GetConstraints()->SetProblemType(type, num);
        functional_->SetProblemType(type, num);

#if dope_dimension > 0
        if (dealdim == dopedim)
          {
            //Prepare DoFHandlerPointer

            {
              if (this->GetType() == "state" || this->GetType() == "adjoint"
                  || this->GetType() == "adjoint_for_ee" || this->GetType() == "cost_functional"
                  || this->GetType() == "cost_functional_pre"
                  || this->GetType() == "cost_functional_pre_tangent"
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
                  throw DOpEException("problem_type_ : "+this->GetType()+" not implemented!", "OptProblemContainer::SetType");
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
                || this->GetType() == "cost_functional_pre"
                || this->GetType() == "cost_functional_pre_tangent"
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
                  "problem_type_ : " + this->GetType() + " not implemented!",
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementFunctional(
                        const DATACONTAINER &edc)
  {

    if ((this->GetType() == "cost_functional") || (this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre_tangent"))
      {
        // state values in quadrature points
        return GetFunctional()->ElementValue(edc);
      }
    else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return aux_functionals_[this->GetTypeNum()]->ElementValue(edc);
      }
    else if (this->GetType() == "functional_for_ee")
      {
        // TODO is this correct? Should not be needed.
        return aux_functionals_[functional_for_ee_num_]->ElementValue(edc);
      }
    else if (this->GetType().find("constraints") != std::string::npos)
      {
        return GetConstraints()->ElementValue(edc);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementFunctional");
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
                        const std::map<std::string, const VECTOR *> &domain_values)
  {
    if ((this->GetType() == "cost_functional")||(this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre_tangent"))
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
        return aux_functionals_[this->GetTypeNum()]->PointValue(
                 this->GetSpaceTimeHandler()->GetControlDoFHandler(),
                 this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                 domain_values);

      } //endif aux_functional
    else if (this->GetType() == "functional_for_ee")
      {
        // TODO is this correct? Should not be needed.
        return aux_functionals_[functional_for_ee_num_]->PointValue(
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
                        const FACEDATACONTAINER &fdc)
  {
    if ((this->GetType() == "cost_functional")||(this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre_tangent"))
      {
        // state values in quadrature points
        return GetFunctional()->BoundaryValue(fdc);
      }
    else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return aux_functionals_[this->GetTypeNum()]->BoundaryValue(fdc);
      }
    else if (this->GetType() == "functional_for_ee")
      // TODO is this correct? Should not be needed.
      {
        return aux_functionals_[functional_for_ee_num_]->BoundaryValue(fdc);
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
                        const FACEDATACONTAINER &fdc)
  {
    if ((this->GetType() == "cost_functional")||(this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre_tangent"))
      {
        // state values in quadrature points
        return GetFunctional()->FaceValue(fdc);
      }
    else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return aux_functionals_[this->GetTypeNum()]->FaceValue(fdc);
      }
    else if (this->GetType() == "functional_for_ee")
      // TODO is this correct? Should not be needed.
      {
        return aux_functionals_[functional_for_ee_num_]->FaceValue(fdc);
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
                        const std::map<std::string, const VECTOR *> &domain_values)
  {
    if ((this->GetType() == "cost_functional")||(this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre_tangent"))
      {
        // state values in quadrature points
        return GetFunctional()->AlgebraicValue(param_values, domain_values);
      }
    else if (this->GetType() == "aux_functional")
      {
        // state values in quadrature points
        return aux_functionals_[this->GetTypeNum()]->AlgebraicValue(
                 param_values, domain_values);
      }
    else if (this->GetType() == "functional_for_ee")
      // TODO is this correct? Should not be needed.
      {
        return aux_functionals_[functional_for_ee_num_]->AlgebraicValue(
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementEquation(
                        const DATACONTAINER &edc, dealii::Vector<double> &local_vector,
                        double scale, double /*scale_ico*/)
  {
    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        // control values in quadrature points
        this->GetPDE().ControlElementEquation(edc, local_vector, scale*c_interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementEquation");
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
                        VECTOR &residual,
                        const std::map<std::string, const dealii::Vector<double>*> &param_values,
                        const std::map<std::string, const VECTOR *> &domain_values)
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementTimeEquation(
                        const DATACONTAINER &edc, dealii::Vector<double> &local_vector,
                        double scale)
  {

    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeEquation");
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeEquation");
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementTimeEquationExplicit(
                        const DATACONTAINER &edc, dealii::Vector<double> &local_vector,
                        double scale)
  {

    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeEquationExplicit");
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeEquationExplicit");
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
                        const FACEDATACONTAINER &fdc,
                        dealii::Vector<double> &local_vector, double scale,
                        double /*scale_ico*/)
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
      {
        // control values in quadrature points
        this->GetPDE().ControlBoundaryEquation(fdc, local_vector, scale*c_interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementBoundaryEquation");
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementRhs(
                        const DATACONTAINER &edc, dealii::Vector<double> &local_vector,
                        double scale)
  {
    if (this->GetType() == "gradient")
      {
        if (GetSpaceTimeHandler()->GetControlType()
            == DOpEtypes::ControlType::initial && initial_)
          {
            this->GetPDE().Init_ElementRhs_Q(edc, local_vector, scale);
          }
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("domain") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->ElementValue_Q(edc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->ElementValue_Q(edc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::ElementRhs");
                  }
              }
          }
        scale *= -1;
        this->GetPDE().ElementEquation_Q(edc, local_vector, scale*interval_length_, scale*interval_length_);
      }
    else if (this->GetType() == "hessian")
      {
        if (GetSpaceTimeHandler()->GetControlType()
            == DOpEtypes::ControlType::initial && initial_)
          {
            this->GetPDE().Init_ElementRhs_QTT(edc, local_vector, scale);
            this->GetPDE().Init_ElementRhs_QQ(edc, local_vector, scale);
          }
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("domain") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->ElementValue_QQ(edc, local_vector, scale*interval_length_);
                    GetFunctional()->ElementValue_UQ(edc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->ElementValue_QQ(edc, local_vector, scale);
                    GetFunctional()->ElementValue_UQ(edc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::ElementRhs");
                  }
              }
          }

        scale *= -1;
        this->GetPDE().ElementEquation_QTT(edc, local_vector, scale*interval_length_, scale*interval_length_);
        this->GetPDE().ElementEquation_UQ(edc, local_vector, scale*interval_length_, scale*interval_length_);
        this->GetPDE().ElementEquation_QQ(edc, local_vector, scale*interval_length_, scale*interval_length_);
      }
    else if (this->GetType() == "global_constraint_gradient")
      {
        assert(interval_length_==1.);
        GetConstraints()->ElementValue_Q(edc, local_vector, scale);
      }
    else if (this->GetType() == "global_constraint_hessian")
      {
        assert(interval_length_==1.);
        GetConstraints()->ElementValue_QQ(edc, local_vector, scale);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementRhs");
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
                        const std::map<std::string, const VECTOR *> &domain_values,
                        VECTOR &rhs_vector, double scale)
  {
    if (this->GetType() == "gradient")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("point") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->PointValue_Q(
                      this->GetSpaceTimeHandler()->GetControlDoFHandler(),
                      this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                      domain_values, rhs_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->PointValue_Q(
                      this->GetSpaceTimeHandler()->GetControlDoFHandler(),
                      this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                      domain_values, rhs_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::PointRhs");
                  }
              }
          }
      }
    else if (this->GetType() == "hessian")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("point") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->PointValue_QQ(
                      this->GetSpaceTimeHandler()->GetControlDoFHandler(),
                      this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                      domain_values, rhs_vector, scale*interval_length_);
                    GetFunctional()->PointValue_UQ(
                      this->GetSpaceTimeHandler()->GetControlDoFHandler(),
                      this->GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                      domain_values, rhs_vector, scale*interval_length_);
                  }
                else
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
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::PointRhs");
                  }
              }
          }
      }
    else
      {
        throw DOpEException("Not implemented", "OptProblem::PointRhs");
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
                        const FACEDATACONTAINER &fdc,
                        dealii::Vector<double> &local_vector, double scale)
  {
    if (this->GetType() == "gradient")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("face") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->FaceValue_Q(fdc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->FaceValue_Q(fdc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::FaceRhs");
                  }
              }
          }

        scale *= -1;
        this->GetPDE().FaceEquation_Q(fdc, local_vector, scale*interval_length_, scale*interval_length_);
      }
    else if (this->GetType() == "hessian")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("face") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->FaceValue_QQ(fdc, local_vector, scale*interval_length_);
                    GetFunctional()->FaceValue_UQ(fdc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->FaceValue_QQ(fdc, local_vector, scale);
                    GetFunctional()->FaceValue_UQ(fdc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::FaceRhs");
                  }
              }
          }

        scale *= -1;
        this->GetPDE().FaceEquation_QTT(fdc, local_vector, scale*interval_length_, scale*interval_length_);
        this->GetPDE().FaceEquation_UQ(fdc, local_vector, scale*interval_length_, scale*interval_length_);
        this->GetPDE().FaceEquation_QQ(fdc, local_vector, scale*interval_length_, scale*interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::FaceRhs");
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
                        const FACEDATACONTAINER &fdc,
                        dealii::Vector<double> &local_vector, double scale)
  {
    if (this->GetType() == "gradient")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("boundary") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->BoundaryValue_Q(fdc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->BoundaryValue_Q(fdc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::BoundaryRhs");
                  }
              }
          }

        scale *= -1;
        this->GetPDE().BoundaryEquation_Q(fdc, local_vector, scale*interval_length_,
                                          scale*interval_length_);
      }
    else if (this->GetType() == "hessian")
      {
        // state values in quadrature points
        if (GetFunctional()->NeedTime())
          {
            if (GetFunctional()->GetType().find("boundary") != std::string::npos)
              {
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos)
                  {
                    GetFunctional()->BoundaryValue_QQ(fdc, local_vector, scale*interval_length_);
                    GetFunctional()->BoundaryValue_UQ(fdc, local_vector, scale*interval_length_);
                  }
                else
                  {
                    GetFunctional()->BoundaryValue_QQ(fdc, local_vector, scale);
                    GetFunctional()->BoundaryValue_UQ(fdc, local_vector, scale);
                  }
                if (GetFunctional()->GetType().find("timedistributed") != std::string::npos && GetFunctional()->GetType().find("timelocal") != std::string::npos)
                  {
                    throw DOpEException("Conflicting functional types: "+ GetFunctional()->GetType(),
                                        "OptProblemContainer::BoundaryRhs");
                  }
              }
          }

        scale *= -1;
        this->GetPDE().BoundaryEquation_QTT(fdc, local_vector, scale*interval_length_,
                                            scale*interval_length_);
        this->GetPDE().BoundaryEquation_UQ(fdc, local_vector, scale*interval_length_,
                                           scale*interval_length_);
        this->GetPDE().BoundaryEquation_QQ(fdc, local_vector, scale*interval_length_,
                                           scale*interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::BoundaryRhs");
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementMatrix(
                        const DATACONTAINER &edc,
                        dealii::FullMatrix<double> &local_entry_matrix, double scale,
                        double /*scale_ico*/)
  {

    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        // control values in quadrature points
        this->GetPDE().ControlElementMatrix(edc, local_entry_matrix, scale*c_interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementMatrix");
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementTimeMatrix(
                        const DATACONTAINER &edc, FullMatrix<double> &local_entry_matrix)
  {
    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeMatrix");
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeMatrix");
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ElementTimeMatrixExplicit(
                        const DATACONTAINER &edc,
                        dealii::FullMatrix<double> &local_entry_matrix)
  {
    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeMatrixExplicit");
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementTimeMatrixExplicit");
      }

  }

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  template<typename FACEDATACONTAINER>
  void
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD,
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceEquation(
                        const FACEDATACONTAINER & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                        double /*scale_ico*/)
  {
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
                        const FACEDATACONTAINER & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                        double /*scale_ico*/)
  {
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
                      CONSTRAINTS, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FaceMatrix(
                        const FACEDATACONTAINER & /*fdc*/, FullMatrix<double> &/*local_entry_matrix*/,
                        double /*scale*/, double /*scale_ico*/)
  {
//        else if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
//        {
//          // control values in quadrature points
//          this->GetPDE().ControlFaceMatrix(fdc, local_entry_matrix);
//        }
//        else
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
                        const FACEDATACONTAINER & /*fdc*/, FullMatrix<double> &/*local_entry_matrix*/,
                        double /*scale*/, double /*scale_ico*/)
  {
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
                        const FACEDATACONTAINER &fdc, FullMatrix<double> &local_matrix,
                        double scale, double /*scale_ico*/)
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
      {
        // control values in quadrature points
        this->GetPDE().ControlBoundaryMatrix(fdc, local_matrix, scale*c_interval_length_);
      }
    else
      {
        throw DOpEException("Not implemented",
                            "OptProblemContainer::ElementBoundaryMatrix");
      }

  }

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  void
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::ComputeLocalControlConstraints(
                        VECTOR &constraints,
                        const std::map<std::string, const dealii::Vector<double>*> &/*values*/,
                        const std::map<std::string, const VECTOR *> &block_values)
  {
    if (this->GetType() == "constraints")
      {
        if (this->GetSpaceTimeHandler()->GetNLocalConstraints() != 0)
          {
            const VECTOR &control = *GetBlockVector(block_values, "control");
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
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
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
  const FE<dealdim, dealdim> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFESystem() const
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
        || (this->GetType() == "global_constraint_gradient"))
      {
#if dope_dimension > 0
        if (dopedim == dealdim)
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
        r = aux_functionals_[this->GetTypeNum()]->GetUpdateFlags();
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
        r = aux_functionals_[functional_for_ee_num_]->GetUpdateFlags();
      }
    else
      {
        r = this->GetPDE().GetUpdateFlags();
        if ((this->GetType() == "hessian")
            || (this->GetType() == "gradient"))
          {
            r = r | this->GetFunctional()->GetUpdateFlags();
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
        r = aux_functionals_[this->GetTypeNum()]->GetFaceUpdateFlags();
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
        r = aux_functionals_[functional_for_ee_num_]->GetUpdateFlags();
      }
    else
      {
        r = this->GetPDE().GetFaceUpdateFlags();
        if (this->GetType() == "gradient"
            || (this->GetType() == "hessian"))
          {
            r = r | this->GetFunctional()->GetFaceUpdateFlags();
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
        return aux_functionals_[this->GetTypeNum()]->GetType();
      }
    else if (this->GetType() == "functional_for_ee")
      {
        return aux_functionals_[functional_for_ee_num_]->GetType();
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
        return aux_functionals_[this->GetTypeNum()]->GetName();
      }
    else if (this->GetType() == "functional_for_ee")
      {
        return aux_functionals_[functional_for_ee_num_]->GetName();
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
  unsigned int
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FunctionalNeedPrecomputations() const
  {
    if (this->GetType() == "aux_functional")
      {
        return aux_functionals_[this->GetTypeNum()]->NeedPrecomputations();
      }
    else if (this->GetType() == "functional_for_ee")
      {
        return aux_functionals_[functional_for_ee_num_]->NeedPrecomputations();
      }
    return GetFunctional()->NeedPrecomputations();
  }


  /******************************************************/
  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  bool
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::FunctionalNeedFinalValue() const
  {
    return GetFunctional()->NeedFinalValue();
  }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  void
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetTime(double time,
                          unsigned int time_dof_number,
                          const TimeIterator &interval, bool initial)
  {
    GetSpaceTimeHandler()->SetInterval(interval);
    initial_ = initial;
    //      GetSpaceTimeHandler()->SetTimeDoFNumber(time_point);
    interval_length_ = GetSpaceTimeHandler()->GetStepSize();

    if (GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial || GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary )
      {
        c_interval_length_ = 1.;
      }
    else
      {
        c_interval_length_ = GetSpaceTimeHandler()->GetStepSize();
      }

    {
      //Zeit an Dirichlet Werte uebermitteln
      for (unsigned int i = 0;
           i < transposed_control_gradient_dirichlet_values_.size(); i++)
        transposed_control_gradient_dirichlet_values_[i]->SetTime(time);
      for (unsigned int i = 0;
           i < transposed_control_hessian_dirichlet_values_.size(); i++)
        transposed_control_hessian_dirichlet_values_[i]->SetTime(time);
      for (unsigned int i = 0; i < primal_dirichlet_values_.size(); i++)
        primal_dirichlet_values_[i]->SetTime(time);
      for (unsigned int i = 0; i < tangent_dirichlet_values_.size(); i++)
        tangent_dirichlet_values_[i]->SetTime(time);
      for (unsigned int i = 0; i < control_dirichlet_values_.size(); i++)
        control_dirichlet_values_[i]->SetTime(time);
      //Functionals
      GetFunctional()->SetTime(time,interval_length_);
      for (unsigned int i = 0; i < aux_functionals_.size(); i++)
        aux_functionals_[i]->SetTime(time,interval_length_);
      //PDE
      this->GetPDE().SetTime(time,interval_length_);
    }
    //Update Auxiliary Control, State and Constraint Vectors
    {
      typename std::map<std::string, const ControlVector<VECTOR> *>::iterator it =
        auxiliary_controls_.begin();
      for (; it != auxiliary_controls_.end(); it++)
        {
          it->second->SetTimeDoFNumber(time_dof_number);
        }
    }
    {
      typename std::map<std::string, const StateVector<VECTOR> *>::iterator it =
        auxiliary_state_.begin();
      for (; it != auxiliary_state_.end(); it++)
        {
          it->second->SetTimeDoFNumber(time_dof_number, interval);
        }
    }
    {
      typename std::map<std::string, const ConstraintVector<VECTOR> *>::iterator it =
        auxiliary_constraints_.begin();
      for (; it != auxiliary_constraints_.end(); it++)
        {
          it->second->SetTimeDoFNumber(time_dof_number);
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
                        SPARSITYPATTERN &sparsity) const
  {
    if ((this->GetType() == "gradient")
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
  void
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::PostProcessConstraints(
                        ConstraintVector<VECTOR> &g) const
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
                        const ControlVector<VECTOR> *c, std::string name)
  {
    if (auxiliary_controls_.find(name) != auxiliary_controls_.end())
      {
        throw DOpEException(
          "Adding multiple Data with name " + name + " is prohibited!",
          "OptProblemContainer::AddAuxiliaryControl");
      }
    auxiliary_controls_.insert(
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
                        const StateVector<VECTOR> *c, std::string name)
  {
    if (auxiliary_state_.find(name) != auxiliary_state_.end())
      {
        throw DOpEException(
          "Adding multiple Data with name " + name + " is prohibited!",
          "OptProblemContainer::AddAuxiliaryState");
      }
    auxiliary_state_.insert(
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
                        const ConstraintVector<VECTOR> *c, std::string name)
  {
    if (auxiliary_constraints_.find(name) != auxiliary_constraints_.end())
      {
        throw DOpEException(
          "Adding multiple Data with name " + name + " is prohibited!",
          "OptProblemContainer::AddAuxiliaryConstraint");
      }
    auxiliary_constraints_.insert(
      std::pair<std::string, const ConstraintVector<VECTOR>*>(name, c));
  }
  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  const ControlVector<VECTOR> *
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetAuxiliaryControl(
                        std::string name) const
  {
    typename std::map<std::string, const ControlVector<VECTOR> *>::const_iterator it =
      auxiliary_controls_.find(name);
    if (it == auxiliary_controls_.end())
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
  const StateVector<VECTOR> *
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetAuxiliaryState(
                        std::string name) const
  {
    typename std::map<std::string, const StateVector<VECTOR> *>::const_iterator it =
      auxiliary_state_.find(name);
    if (it == auxiliary_state_.end())
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
      auxiliary_controls_.find(name);
    if (it == auxiliary_controls_.end())
      {
        throw DOpEException(
          "Deleting Data " + name + " is impossible! Data not found",
          "OptProblemContainer::DeleteAuxiliaryControl");
      }
    auxiliary_controls_.erase(it);
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
      auxiliary_state_.find(name);
    if (it == auxiliary_state_.end())
      {
        throw DOpEException(
          "Deleting Data " + name + " is impossible! Data not found",
          "OptProblemContainer::DeleteAuxiliaryState");
      }
    auxiliary_state_.erase(it);
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
      auxiliary_constraints_.find(name);
    if (it == auxiliary_constraints_.end())
      {
        throw DOpEException(
          "Deleting Data " + name + " is impossible! Data not found",
          "OptProblemContainer::DeleteAuxiliaryConstraint");
      }
    auxiliary_constraints_.erase(it);
  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  FUNCTIONAL *
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctional()
  {
    if (this->GetType() == "aux_functional"
        || this->GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //    return aux_functionals_[this->GetTypeNum()];
      }
    return functional_;

  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  const FUNCTIONAL *
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetFunctional() const
  {
    if (this->GetType() == "aux_functional"
        || this->GetType() == "functional_for_ee")
      {
        //This may no longer happen!
        abort();
        //       return aux_functionals_[this->GetTypeNum()];
      }
    return functional_;
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
        return aux_functionals_[this->GetTypeNum()]->HasFaces();
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
        if ((this->GetType() == "gradient"))
          {
            return this->GetPDE().HasFaces();
          }
        else if ((this->GetType() == "hessian"))
          {
            return this->GetPDE().HasFaces() || this->GetFunctional()->HasFaces();
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
        || this->GetType() == "aux_functional")
      {
        // We dont need PointRhs in this cases.
        return false;
      }
    else if ((this->GetType() == "hessian"))
      {
        return this->GetFunctional()->HasPoints();
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
        if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
          {
            return this->GetPDE().HasInterfaces();
          }
        else
          {
            throw DOpEException("Unknown Type: '" + this->GetType() + "'!",
                                "OptProblemContainer::HasInterfaces");
          }
      }
  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  template<typename ELEMENTITERATOR>
  bool
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::AtInterface(ELEMENTITERATOR &element, unsigned int face) const
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
        if ((this->GetType() == "gradient")
            || (this->GetType() == "hessian"))
          {
            return this->GetPDE().AtInterface(element,face);
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
                        unsigned int color, const std::vector<bool> &comp_mask,
                        const DOpEWrapper::Function<dealdim> *values)
  {
    assert(values);

    unsigned int comp = control_dirichlet_colors_.size();
    for (unsigned int i = 0; i < control_dirichlet_colors_.size(); ++i)
      {
        if (control_dirichlet_colors_[i] == color)
          {
            comp = i;
            break;
          }
      }
    if (comp != control_dirichlet_colors_.size())
      {
        std::stringstream s;
        s << "ControlDirichletColor" << color << " has multiple occurences !";
        throw DOpEException(s.str(),
                            "OptProblemContainer::SetControlDirichletBoundary");
      }
    control_dirichlet_colors_.push_back(color);
    control_dirichlet_comps_.push_back(comp_mask);
    control_dirichlet_values_.push_back(values);
  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  void
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::SetDirichletBoundaryColors(
                        unsigned int color, const std::vector<bool> &comp_mask,
                        const DD *values)
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
                            "OptProblemContainer::SetDirichletBoundary");
      }
    dirichlet_colors_.push_back(color);
    dirichlet_comps_.push_back(comp_mask);
    PrimalDirichletData<DD, VECTOR, dealdim> *data =
      new PrimalDirichletData<DD, VECTOR, dealdim>(*values);
    primal_dirichlet_values_.push_back(data);
    TangentDirichletData<DD, VECTOR, dealdim> *tdata =
      new TangentDirichletData<DD, VECTOR, dealdim>(*values);
    tangent_dirichlet_values_.push_back(tdata);

    if (values->NeedsControl())
      {
        control_transposed_dirichlet_colors_.push_back(color);
        TransposedGradientDirichletData<DD, VECTOR, dealdim> *gdata =
          new TransposedGradientDirichletData<DD, VECTOR, dealdim>(
          *values);
        transposed_control_gradient_dirichlet_values_.push_back(gdata);
        TransposedHessianDirichletData<DD, VECTOR, dealdim> *hdata =
          new TransposedHessianDirichletData<DD, VECTOR, dealdim>(
          *values);
        transposed_control_hessian_dirichlet_values_.push_back(hdata);
      }
  }

  /******************************************************/

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  const std::vector<unsigned int> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletColors() const
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
        || (this->GetType() == "global_constraint_gradient"))
      {
        return control_dirichlet_colors_;
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
  const std::vector<unsigned int> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletColors() const
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
      {
        return control_transposed_dirichlet_colors_;
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
  const std::vector<bool> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletCompMask(
                        unsigned int color) const
  {
    if ((this->GetType() == "gradient")
        || (this->GetType() == "hessian"))
      {
        unsigned int comp = control_dirichlet_colors_.size();
        for (unsigned int i = 0; i < control_dirichlet_colors_.size(); ++i)
          {
            if (control_dirichlet_colors_[i] == color)
              {
                comp = i;
                break;
              }
          }
        if (comp == control_dirichlet_colors_.size())
          {
            std::stringstream s;
            s << "ControlDirichletColor" << color << " has not been found !";
            throw DOpEException(s.str(),
                                "OptProblemContainer::GetDirichletCompMask");
          }
        return control_dirichlet_comps_[comp];
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
  const std::vector<bool> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletCompMask(
                        unsigned int color) const
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian"))
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
                                "OptProblemContainer::GetDirichletCompMask");
          }
        return dirichlet_comps_[comp];
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
  const Function<dealdim> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDirichletValues(
                        unsigned int color,
                        const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
                        const std::map<std::string, const VECTOR *> &/*domain_values*/) const
  {

    unsigned int col = dirichlet_colors_.size();
    if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        col = control_dirichlet_colors_.size();
        for (unsigned int i = 0; i < control_dirichlet_colors_.size(); ++i)
          {
            if (control_dirichlet_colors_[i] == color)
              {
                col = i;
                break;
              }
          }
        if (col == control_dirichlet_colors_.size())
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

    if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        return *(control_dirichlet_values_[col]);
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
  const TransposedDirichletDataInterface<dealdim> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetTransposedDirichletValues(
                        unsigned int color,
                        const std::map<std::string, const dealii::Vector<double>*> &param_values,
                        const std::map<std::string, const VECTOR *> &domain_values) const
  {
    unsigned int col = control_transposed_dirichlet_colors_.size();
    if (this->GetType() == "gradient" || (this->GetType() == "hessian"))
      {
        for (unsigned int i = 0;
             i < control_transposed_dirichlet_colors_.size(); ++i)
          {
            if (control_transposed_dirichlet_colors_[i] == color)
              {
                col = i;
                break;
              }
          }
        if (col == control_transposed_dirichlet_colors_.size())
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
        transposed_control_gradient_dirichlet_values_[col]->ReInit(param_values,
                                                                   domain_values, color);
        return *(transposed_control_gradient_dirichlet_values_[col]);
      }
    else if (this->GetType() == "hessian")
      {
        transposed_control_hessian_dirichlet_values_[col]->ReInit(param_values,
                                                                  domain_values, color);
        return *(transposed_control_hessian_dirichlet_values_[col]);
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
  const std::vector<unsigned int> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetBoundaryEquationColors() const
  {
    if (this->GetType() == "gradient" || (this->GetType() == "hessian")
        || (this->GetType() == "global_constraint_gradient"))
      {
        return control_boundary_equation_colors_;
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
    {
      //Control Boundary Equation colors are simply inserted
      unsigned int comp = control_boundary_equation_colors_.size();
      for (unsigned int i = 0; i < control_boundary_equation_colors_.size();
           ++i)
        {
          if (control_boundary_equation_colors_[i] == color)
            {
              comp = i;
              break;
            }
        }
      if (comp != control_boundary_equation_colors_.size())
        {
          std::stringstream s;
          s << "Boundary Equation Color" << color
            << " has multiple occurences !";
          throw DOpEException(s.str(),
                              "OptProblemContainer::SetControlBoundaryEquationColors");
        }
      control_boundary_equation_colors_.push_back(color);
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
    {
      //State Boundary Equation colors are simply inserted
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
                              "OptProblemContainer::SetBoundaryEquationColors");
        }
      state_boundary_equation_colors_.push_back(color);
    }
    {
      //For the  adjoint they are added with the boundary functional colors
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

  template<typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
  typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
  typename VECTOR, int dopedim, int dealdim, template<int, int> class FE,
  template<int, int> class DH>
  const std::vector<unsigned int> &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetBoundaryFunctionalColors() const
  {
    if (this->GetType() == "cost_functional"
        || this->GetType() == "cost_functional_pre"
        || this->GetType() == "cost_functional_pre_tangent"
        || this->GetType() == "aux_functional"
        || this->GetType() == "functional_for_ee") //fixme: what about error_evaluation?
      {
        return boundary_functional_colors_;
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
    {
      //Boundary Functional colors are simply inserted
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
                              "OptProblemContainer::SetBoundaryFunctionalColors");
        }
      boundary_functional_colors_.push_back(color);
    }
    {
      //For the  adjoint they are addeed  to the boundary equation colors
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
  const std::vector<unsigned int> &
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
  const dealii::ConstraintMatrix &
  OptProblemContainer<FUNCTIONAL_INTERFACE, FUNCTIONAL, PDE, DD, CONSTRAINTS,
                      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>::GetDoFConstraints() const
  {
    if ((this->GetType() == "gradient") || (this->GetType() == "hessian")
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
    if ((this->GetType() == "cost_functional")||(this->GetType() == "cost_functional_pre")
        || (this->GetType() == "cost_functional_pre"))
      return GetFunctional()->NeedTime();
    else if (this->GetType() == "aux_functional")
      return aux_functionals_[this->GetTypeNum()]->NeedTime();
    else if (this->GetType() == "functional_for_ee")
      return aux_functionals_[functional_for_ee_num_]->NeedTime();
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
    return (!control_transposed_dirichlet_colors_.empty());
  }

}
#endif
