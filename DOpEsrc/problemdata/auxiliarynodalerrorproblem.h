/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef AUXILIARYNODALERROR_PROBLEM_H_
#define AUXILIARYNODALERROR_PROBLEM_H_

#include <basic/spacetimehandler.h>
#include <problemdata/initialnewtonproblem.h>

using namespace dealii;

namespace DOpE
{
  /**
   * This is a problem used in the solution of the
   * primal pde problem, i.e., the auxiliarynodalerror-equation.
   *
   * @tparam OPTPROBLEM     The container with the OPT-Problem description
   * @tparam PDE            The container with the PDE-description
   *                        note the PDE is the one we use for all
   *                        things related to the PDE. This is so to allow
   *                        switching between timesteps.
   * @tparam DD             Dirichlet datan
   * @tparam VECTOR         The vector class
   * @tparam dim            The dimension of the domain.
   */
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  class AuxiliaryNodalErrorProblem
  {
  public:
    AuxiliaryNodalErrorProblem(OPTPROBLEM &OP, PDE &pde) :
      pde_(pde), opt_problem_(OP)
    {
      dirichlet_colors_ = opt_problem_.dirichlet_colors_;
      dirichlet_comps_ = opt_problem_.dirichlet_comps_;
      primal_dirichlet_values_ = opt_problem_.primal_dirichlet_values_;
      state_boundary_equation_colors_
        = opt_problem_.state_boundary_equation_colors_;
      interval_length_ = 1.;
    }

    ~AuxiliaryNodalErrorProblem()
    {
    }

    std::string
    GetName() const
    {
      return "AuxiliaryNodalErrorProblem";
    }
    std::string
    GetType() const
    {
      return "state";
    }

    /**
     * Function returning whether the PDE needs to set a special initial
     * value for the newton solver to work.
     * Relevant for stationary PDEs only when the constant zero is not a
     * feasible starting point.
     */
    bool
    NeedInitialState() const
    {
      return false;
    }

    InitialNewtonProblem<OPTPROBLEM,AuxiliaryNodalErrorProblem<OPTPROBLEM,PDE,DD,SPARSITYPATTERN,VECTOR,dim>, VECTOR, dim> &
    GetNewtonInitialProblem()
    {
      throw DOpEException("GetNewtonInitialProblem has no meaning in case NeedInitialState() == false","AuxiliaryNodalErrorProblem::GetNewtonInitialProblem");

    }

    /******************************************************/
    /****For the initial values ***************/
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    void Init_ElementEquation(const EDC &edc,
                              dealii::Vector<double> &local_vector, double scale,
                              double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_ElementEquation");
    }

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    void
    Init_ElementRhs(const EDC &edc,
                    dealii::Vector<double> &local_vector, double scale)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_ElementRhs");
    }

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    void
    Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
      const std::map<std::string, const VECTOR *> &/*domain_values*/,
      VECTOR & /*rhs_vector*/, double /*scale=1.*/)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_PointRhs");
    }

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    void Init_ElementMatrix(const EDC &edc,
                            dealii::FullMatrix<double> &local_entry_matrix, double scale,
                            double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_ElementMatrix");
    }

    template<typename FDC>
    inline void Init_OutflowValues(const  FDC &fdc,
                                   std::vector<bool> &present_in_outflow,
                                   dealii::Vector<double> &local_vector,
                                   double scale,
                                   double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_OutflowValues");
    }
    template<typename FDC>
    inline void
    Init_OutflowMatrix(const FDC &fdc,
                       std::vector<bool> &present_in_outflow,
                       dealii::FullMatrix<double> &local_matrix,
                       double scale,
                       double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_OutflowMatrix");
    }
    inline void Init_PipeCouplingResidual(dealii::Vector<double> &res,
                                          const dealii::Vector<double> &u,
                                          const std::vector<bool> &present_in_outflow)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_PipeCouplingResidual");
    }

    inline void Init_CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                                    const std::vector<bool> &present_in_outflow)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem::Init_CouplingMatrix");
    }
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
    template<typename EDC>
    inline void
    ElementEquation(const EDC &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double scale_ico);

    /**
     * This function has the same functionality as the ElementEquation function.
     * It is needed for time derivatives when working with
     * time dependent problems.
     */
    template<typename EDC>
    inline void
    ElementTimeEquation(const EDC &edc,
                        dealii::Vector<double> &local_vector, double scale = 1.);

    /**
     * This function has the same functionality as the ElementTimeEquation function.
     * It is needed for problems with nonlinearities in the time derivative, like
    * fluid-structure interaction problems and has
     * special structure.
     */
    template<typename EDC>
    inline void
    ElementTimeEquationExplicit(const EDC &edc,
                                dealii::Vector<double> &local_vector, double scale = 1.);

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
    template<typename EDC>
    inline void
    ElementRhs(const EDC &edc,
               dealii::Vector<double> &local_vector, double scale = 1.);

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
      VECTOR &rhs_vector, double scale);

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
    template<typename EDC>
    inline void
    ElementMatrix(const EDC &edc,
                  dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
                  double scale_ico = 1.);

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
    template<typename EDC>
    inline void
    ElementTimeMatrix(const EDC &edc,
                      dealii::FullMatrix<double> &local_entry_matrix);

    /**
     * Computes the value of the element matrix which is derived
     * by computing the directional derivatives of the time residuum of the PDE
     * problem under consideration.
     *
     * This function is only needed for fluid-structure interaction problems.
     * Please ask Thomas Wick WHY and HOW to use this function.
     *
     */
    template<typename EDC>
    inline void
    ElementTimeMatrixExplicit(const EDC &edc,
                              dealii::FullMatrix<double> &local_entry_matrix);

    /**
     * Computes the value of face on a element.
     * It has the same functionality as ElementEquation. We refer to its
     * documentation.
     *
     */
    template<typename FDC>
    inline void
    FaceEquation(const FDC &fdc,
                 dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
     * Computes the product of two different finite elements
     * on a interior face. It has the same functionality as ElementEquation.
     * We refer to its documentation.
     *
     */
    template<typename FDC>
    inline void
    InterfaceEquation(const FDC &fdc,
                      dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
     * Computes the value of face on a element.
     * It has the same functionality as ElementRhs. We refer to its
     * documentation.
     *
     */
    template<typename FDC>
    inline void
    FaceRhs(const FDC &fdc,
            dealii::Vector<double> &local_vector, double scale = 1.);

    /**
     * Computes the value of face on a element.
     * It has the same functionality as ElementMatrix. We refer to its
     * documentation.
     *
     */
    template<typename FDC>
    inline void
    FaceMatrix(const FDC &fdc,
               dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

    /**
     * Computes the product of two different finite elements
     * on an interior face. It has the same functionality as
     * ElementMatrix. We refer to its documentation.
     *
     */
    template<typename FDC>
    inline void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

    /**
     * Computes the value of face on a boundary.
     * It has the same functionality as ElementEquation. We refer to its
     * documentation.
     */
    template<typename FDC>
    inline void
    BoundaryEquation(const FDC &fdc,
                     dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
     * Computes the value of the boundary on a element.
     * It has the same functionality as ElementRhs. We refer to its
     * documentation.
     *
     */
    template<typename FDC>
    inline void
    BoundaryRhs(const FDC &fdc,
                dealii::Vector<double> &local_vector, double scale = 1.);

    /**
     * Computes the value of the boundary on a element.
     * It has the same functionality as ElementMatrix. We refer to its
     * documentation.
     *
     */
    template<typename FDC>
    inline void
    BoundaryMatrix(const FDC &fdc,
                   dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const dealii::SmartPointer<const dealii::FESystem<dim> >
    GetFESystem() const;

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const dealii::SmartPointer<
    const dealii::hp::FECollection<dim> >
    GetFECollection() const;

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline std::string
    GetDoFType() const;

    /**
     * This function determines whether a loop over all faces is required or not.
     *
     * @return Returns whether or not this functional has components on faces between elements.
     *         The default value is false.
     */
    inline bool
    HasFaces() const;

    /**
      * Do we need the evaluation of PointRhs?
      */
    inline bool
    HasPoints() const;
    /**
    * This function determines whether a loop over all faces is required or not.
    *
    * @return Returns whether or not this functional needs to compute the product
    * of two different finite element functions across an internal face.
    */

    inline bool
    HasInterfaces() const;

    /**
      * Do we need evaluation at the vertices?
      */
    inline bool
    HasVertices() const;

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline dealii::UpdateFlags
    GetUpdateFlags() const;

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline dealii::UpdateFlags
    GetFaceUpdateFlags() const;

    /******************************************************/
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline void
    SetTime(double time,
            unsigned int time_dof_number,
            const TimeIterator &interval, bool initial = false);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline void
    ComputeSparsityPattern(SPARSITYPATTERN &sparsity) const;


    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const std::vector<unsigned int> &
    GetDirichletColors() const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const std::vector<bool> &
    GetDirichletCompMask(unsigned int color) const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const Function<dim>
    &
    GetDirichletValues(
      unsigned int color,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values) const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline const std::vector<unsigned int> &
    GetBoundaryEquationColors() const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
#if DEAL_II_VERSION_GTE(9,1,1)
    inline const dealii::AffineConstraints<double> &
    GetDoFConstraints() const;
#else
    inline const dealii::ConstraintMatrix &
    GetDoFConstraints() const;
#endif
#if DEAL_II_VERSION_GTE(9,1,1)
    inline const dealii::AffineConstraints<double> &
    GetHNConstraints() const;
#else
    inline const dealii::ConstraintMatrix &
    GetHNConstraints() const;
#endif
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    const dealii::Function<dim> &
    GetInitialValues() const;
    /******************************************************/
    DOpEOutputHandler<VECTOR> *
    GetOutputHandler()
    {
      return opt_problem_.GetOutputHandler();
    }
    OPTPROBLEM &
    GetBaseProblem()
    {
      return opt_problem_;
    }

    template<typename ELEMENTITERATOR>
    bool AtInterface(ELEMENTITERATOR &element, unsigned int face) const;

    /********************Functions on Networks********************/
    template<typename FDC>
    inline void BoundaryEquation_BV(const FDC &fdc,
                                    dealii::Vector<double> &local_vector,
                                    double scale,
                                    double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }

    template<typename FDC>
    inline void BoundaryMatrix_BV(const FDC &fdc,
                                  std::vector<bool> &present_in_outflow,
                                  dealii::FullMatrix<double> &local_entry_matrix,
                                  double scale,
                                  double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }

    template<typename FDC>
    inline void OutflowValues(const  FDC &fdc,
                              std::vector<bool> &present_in_outflow,
                              dealii::Vector<double> &local_vector,
                              double scale,
                              double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }

    template<typename FDC>
    inline void
    OutflowMatrix(const FDC &fdc,
                  std::vector<bool> &present_in_outflow,
                  dealii::FullMatrix<double> &local_entry_matrix,
                  double scale,
                  double scale_ico)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }
    inline void PipeCouplingResidual(dealii::Vector<double> &res,
                                     const dealii::Vector<double> &u,
                                     const std::vector<bool> &present_in_outflow)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }
    inline void CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                               const std::vector<bool> &present_in_outflow)
    {
      throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
    }

  protected:

  private:
    PDE &pde_;
    OPTPROBLEM &opt_problem_;

    std::vector<unsigned int> dirichlet_colors_;
    std::vector<std::vector<bool> > dirichlet_comps_;
    std::vector<PrimalDirichletData<DD, VECTOR, dim>*>
    primal_dirichlet_values_;
    std::vector<unsigned int> state_boundary_equation_colors_;
    double interval_length_;
  };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementEquation(const EDC &edc,
                                                   dealii::Vector<double> &local_vector, double scale,
                                                   double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementTimeEquation(const EDC &edc,
                                                       dealii::Vector<double> &local_vector, double scale)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementTimeEquationExplicit(const EDC &edc,
                                 dealii::Vector<double> &local_vector, double scale)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::FaceEquation(const FDC &fdc,
                                                dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::InterfaceEquation(const FDC &fdc,
                                                     dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::BoundaryEquation(const FDC &fdc,
                                                    dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementRhs(const EDC &edc,
                                              dealii::Vector<double> &local_vector, double scale)
  {
    pde_.ElementAuxRhs(edc, local_vector, scale*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::PointRhs(
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs_vector*/, double /*scale*/)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::FaceRhs(const FDC &fdc,
                                           dealii::Vector<double> &local_vector, double scale)
  {
    pde_.FaceAuxRhs(fdc, local_vector, scale*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::BoundaryRhs(const FDC &fdc,
                                               dealii::Vector<double> &local_vector, double scale)
  {
    pde_.BoundaryAuxRhs(fdc, local_vector, scale*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementMatrix(const EDC &edc,
                                                 dealii::FullMatrix<double> &local_entry_matrix, double scale,
                                                 double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementTimeMatrix(const EDC &edc,
                                                     FullMatrix<double> &local_entry_matrix)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::ElementTimeMatrixExplicit(const EDC &edc,
                                                             dealii::FullMatrix<double> &local_entry_matrix)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::FaceMatrix(const FDC &fdc,
                                              FullMatrix<double> &local_entry_matrix, double scale,
                                              double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::InterfaceMatrix(const FDC &fdc,
                                                   FullMatrix<double> &local_entry_matrix, double scale,
                                                   double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                             dim>::BoundaryMatrix(const FDC &fdc,
                                                  FullMatrix<double> &local_matrix, double scale,
                                                  double scale_ico)
  {
    throw DOpEException("This should never be called!","AuxiliaryNodalErrorProblem");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  std::string
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFType() const
  {
    return "state";
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const SmartPointer<const dealii::FESystem<dim> >
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFESystem() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetFESystem("state");
  }

  /******************************************************/
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const SmartPointer<const dealii::hp::FECollection<dim> >
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFECollection() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetFECollection("state");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  UpdateFlags
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  UpdateFlags
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFaceUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetFaceUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::SetTime(
    double time,
    unsigned int time_dof_number, const TimeIterator &interval, bool initial)
  {
    opt_problem_.SetTime(time, time_dof_number, interval, initial);
    interval_length_ = opt_problem_.GetSpaceTimeHandler()->GetStepSize();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeSparsityPattern(
    SPARSITYPATTERN &sparsity) const
  {
    opt_problem_.GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasFaces() const
  {
    return pde_.HasFaces();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasPoints() const
  {
    return false;//We have no PointRhs in normal stateproblems at the moment.
  }


  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasInterfaces() const
  {
    return pde_.HasInterfaces();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasVertices() const
  {
    return pde_.HasVertices();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletColors() const
  {
    return dirichlet_colors_;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<bool> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletCompMask(
    unsigned int color) const
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
        throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
      }
    return dirichlet_comps_[comp];
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const Function<dim> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletValues(
    unsigned int color,
    const std::map<std::string, const dealii::Vector<double>*> &param_values,
    const std::map<std::string, const VECTOR *> &domain_values) const
  {
    unsigned int col = dirichlet_colors_.size();
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
        throw DOpEException(s.str(), "OptProblem::GetDirichletValues");
      }
    primal_dirichlet_values_[col]->ReInit(param_values, domain_values, color);
    return *(primal_dirichlet_values_[col]);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetBoundaryEquationColors() const
  {
    return state_boundary_equation_colors_;
  }

  /******************************************************/

#if DEAL_II_VERSION_GTE(9,1,1)
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::AffineConstraints<double> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateDoFConstraints();
  }
#else
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::ConstraintMatrix &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateDoFConstraints();
  }
#endif
  /******************************************************/

#if DEAL_II_VERSION_GTE(9,1,1)
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::AffineConstraints<double> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetHNConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateHNConstraints();
  }
#else
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::ConstraintMatrix &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetHNConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateHNConstraints();
  }
#endif
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>  const dealii::Function<dim> &
  AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetInitialValues() const
  {
    return opt_problem_.GetInitialValues();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename ELEMENTITERATOR>
  bool AuxiliaryNodalErrorProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>
  ::AtInterface(ELEMENTITERATOR &element, unsigned int face) const
  {
    return pde_.AtInterface(element,face);
  }


///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
