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

#ifndef ADJOINT_PROBLEM_H_
#define ADJOINT_PROBLEM_H_

#include <basic/spacetimehandler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * This is a problem used in the solution of the
   * adjoint pde problem associated to the state-pde
   *
   * @tparam OPTPROBLEM     The container with the OPT-Problem description
   * @tparam PDE            The container with the PDE-description
   *                        note the PDE is the one we use for all
   *                        things related to the PDE. This is so to allow
   *                        switching between timesteps.
   * @tparam DD             Dirichlet data
   * @tparam VECTOR         The vector class
   * @tparam dim            The dimension of the domain.
   */
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  class AdjointProblem
  {
  public:
    AdjointProblem(OPTPROBLEM &OP, PDE &pde) :
      pde_(pde), opt_problem_(OP)
    {
      dirichlet_colors_ = opt_problem_.dirichlet_colors_;
      dirichlet_comps_ = opt_problem_.dirichlet_comps_;
      adjoint_dirichlet_values_ = opt_problem_.zero_dirichlet_values_;
      adjoint_boundary_equation_colors_
        = opt_problem_.adjoint_boundary_equation_colors_;
      interval_length_ = 1.;
    }

    ~AdjointProblem()
    {
    }

    std::string
    GetName() const
    {
      return "AdjointProblem";
    }
    std::string
    GetType() const
    {
      return "adjoint";
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
      pde_.Init_ElementEquation(edc, local_vector, scale,
                                scale_ico);
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
      if (opt_problem_.GetFunctional()->NeedTime())
        {
          if (opt_problem_.GetFunctional()->GetType().find("timelocal")
              != std::string::npos)
            {
              if (opt_problem_.GetFunctional()->GetType().find("domain")
                  != std::string::npos)
                {
                  opt_problem_.GetFunctional()->ElementValue_U(edc, local_vector, scale);
                }
            }
        }
    }

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    void
    Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs_vector, double scale=1.)
    {
      if (opt_problem_.GetFunctional()->NeedTime())
        {
          if (opt_problem_.GetFunctional()->GetType().find("timelocal")
              != std::string::npos)
            {
              if (opt_problem_.GetFunctional()->GetType().find("point")
                  != std::string::npos)
                {
                  opt_problem_.GetFunctional()->PointValue_U(
                    opt_problem_.GetSpaceTimeHandler()->GetControlDoFHandler(),
                    opt_problem_.GetSpaceTimeHandler()->GetStateDoFHandler(),
                    param_values, domain_values,
                    rhs_vector, scale);
                }
            }
        }
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
      pde_.Init_ElementMatrix(edc, local_entry_matrix, scale,
                              scale_ico);
    }

    /******************************************************/
    /* Functions as in OptProblem */
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementEquation(const EDC &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double scale_ico);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementTimeEquation(const EDC &edc,
                        dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementTimeEquationExplicit(const EDC &edc,
                                dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementRhs(const EDC &edc,
               dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    void
    PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs_vector, double scale);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementMatrix(const EDC &edc,
                  dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
                  double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementTimeMatrix(const EDC &edc,
                      dealii::FullMatrix<double> &local_entry_matrix);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementTimeMatrixExplicit(const EDC &edc,
                              dealii::FullMatrix<double> &local_entry_matrix);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    FaceEquation(const FDC &fdc,
                 dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    InterfaceEquation(const FDC &fdc,
                      dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    FaceRhs(const FDC &fdc,
            dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    FaceMatrix(const FDC &fdc,
               dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

    template<typename FDC>
    inline void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    BoundaryEquation(const FDC &fdc,
                     dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    BoundaryRhs(const FDC &fdc,
                dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
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
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline bool
    HasFaces() const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline bool
    HasPoints() const;
    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
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
     *  Experimental status: Needed for MG prec.
     */
//      inline void
//      ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
//              unsigned int n_levels) const;
//
//      inline void
//      ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
//              unsigned int n_levels) const;

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
    inline void BoundaryEquation_BV(const FDC & /*fdc*/,
                                    dealii::Vector<double> &/*local_vector*/,
                                    double /*scale*/,
                                    double /*scale_ico*/)
    {
      abort();
    }
    template<typename FDC>
    inline void BoundaryMatrix_BV(const FDC & /*fdc*/,
                                  std::vector<bool> & /*present_in_outflow*/,
                                  dealii::FullMatrix<double> &/*local_entry_matrix*/,
                                  double /*scale*/,
                                  double /*scale_ico*/)
    {
      abort();
    }
    template<typename FDC>
    inline void OutflowValues(const  FDC & /*fdc*/,
                              std::vector<bool> & /*present_in_outflow*/,
                              dealii::Vector<double> &/*local_vector*/,
                              double /*scale*/,
                              double /*scale_ico*/)
    {
      abort();
    }
    template<typename FDC>
    inline void
    OutflowMatrix(const FDC & /*fdc*/,
                  std::vector<bool> & /*present_in_outflow*/,
                  dealii::FullMatrix<double> &/*local_entry_matrix*/,
                  double /*scale*/,
                  double /*scale_ico*/)
    {
      abort();
    }
    inline void PipeCouplingResidual(dealii::Vector<double> & /*res*/,
                                     const dealii::Vector<double> & /*u*/,
                                     const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }
    inline void CouplingMatrix(dealii::SparseMatrix<double> & /*matrix*/,
                               const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }

  protected:

  private:
    PDE &pde_;
    OPTPROBLEM &opt_problem_;

    std::vector<unsigned int> dirichlet_colors_;
    std::vector<std::vector<bool> > dirichlet_comps_;
    const dealii::Function<dim> *adjoint_dirichlet_values_;
    std::vector<unsigned int> adjoint_boundary_equation_colors_;
    double interval_length_;
  };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementEquation(const EDC &edc,
                                       dealii::Vector<double> &local_vector, double scale,
                                       double scale_ico)
  {
    pde_.ElementEquation_U(edc, local_vector, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementTimeEquation(const EDC &edc,
                                           dealii::Vector<double> &local_vector, double scale)
  {
    pde_.ElementTimeEquation_U(edc, local_vector, scale);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementTimeEquationExplicit(const EDC &edc,
                                                   dealii::Vector<double> &local_vector, double scale)
  {
    pde_.ElementTimeEquationExplicit_U(edc, local_vector,
                                       scale);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::FaceEquation(const FDC &fdc,
                                    dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    pde_.FaceEquation_U(fdc, local_vector, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::InterfaceEquation(const FDC &fdc,
                                         dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    pde_.InterfaceEquation_U(fdc,  local_vector, scale*interval_length_, scale_ico*interval_length_);
  }
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::BoundaryEquation(const FDC &fdc,
                                        dealii::Vector<double> &local_vector, double scale, double scale_ico)
  {
    pde_.BoundaryEquation_U(fdc, local_vector, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementRhs(const EDC &edc,
                                  dealii::Vector<double> &local_vector, double scale)
  {
    //values of the derivative of the functional for error estimation

    if (opt_problem_.GetFunctional()->NeedTime())
      {
        if (opt_problem_.GetFunctional()->GetType().find("domain") != std::string::npos)
          {
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos)
              {
                opt_problem_.GetFunctional()->ElementValue_U(edc, local_vector, scale*interval_length_);
              }
            else // Otherwise always local if(opt_problem_.GetFunctional()->GetType().find("timelocal") != std::string::npos)
              {
                opt_problem_.GetFunctional()->ElementValue_U(edc, local_vector, scale);
              }

            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos && opt_problem_.GetFunctional()->GetType().find("timelocal") != std::string::npos)
              {
                throw DOpEException("Conflicting functional types: "+ opt_problem_.GetFunctional()->GetType(),
                                    "AdjointProblem::ElementRhs");
              }
          }
      }
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::PointRhs(
    const std::map<std::string, const dealii::Vector<double>*> &param_values,
    const std::map<std::string, const VECTOR *> &domain_values,
    VECTOR &rhs_vector, double scale)
  {
    //values of the derivative of the functional for error estimation
    if (opt_problem_.GetFunctional()->NeedTime())
      {
        if (opt_problem_.GetFunctional()->GetType().find("point") != std::string::npos)
          {
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos)
              {
                opt_problem_.GetFunctional()->PointValue_U(
                  opt_problem_.GetSpaceTimeHandler()->GetControlDoFHandler(),
                  opt_problem_.GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                  domain_values, rhs_vector, scale*interval_length_);
              }
            else
              {
                opt_problem_.GetFunctional()->PointValue_U(
                  opt_problem_.GetSpaceTimeHandler()->GetControlDoFHandler(),
                  opt_problem_.GetSpaceTimeHandler()->GetStateDoFHandler(), param_values,
                  domain_values, rhs_vector, scale);
              }
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos && opt_problem_.GetFunctional()->GetType().find("timelocal") != std::string::npos)
              {
                throw DOpEException("Conflicting functional types: "+ opt_problem_.GetFunctional()->GetType(),
                                    "AdjointProblem::PointRhs");
              }
          }
      }
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::FaceRhs(const FDC &fdc,
                               dealii::Vector<double> &local_vector, double scale)
  {
    //values of the derivative of the functional for error estimation
    if (opt_problem_.GetFunctional()->NeedTime())
      {
        if (opt_problem_.GetFunctional()->GetType().find("face") != std::string::npos)
          {
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos)
              {
                opt_problem_.GetFunctional()->FaceValue_U(fdc, local_vector, scale*interval_length_);
              }
            else
              {
                opt_problem_.GetFunctional()->FaceValue_U(fdc, local_vector, scale);
              }

            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos && opt_problem_.GetFunctional()->GetType().find("timelocal") != std::string::npos)
              {
                throw DOpEException("Conflicting functional types: "+ opt_problem_.GetFunctional()->GetType(),
                                    "AdjointProblem::FaceRhs");
              }
          }
      }
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::BoundaryRhs(const FDC &fdc,
                                   dealii::Vector<double> &local_vector, double scale)
  {
    //values of the derivative of the functional for error estimation
    if (opt_problem_.GetFunctional()->NeedTime())
      {
        if (opt_problem_.GetFunctional()->GetType().find("boundary") != std::string::npos)
          {
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos)
              {
                opt_problem_.GetFunctional()->BoundaryValue_U(fdc, local_vector, scale*interval_length_);
              }
            else
              {
                opt_problem_.GetFunctional()->BoundaryValue_U(fdc, local_vector, scale);
              }
            if (opt_problem_.GetFunctional()->GetType().find("timedistributed") != std::string::npos && opt_problem_.GetFunctional()->GetType().find("timelocal") != std::string::npos)
              {
                throw DOpEException("Conflicting functional types: "+ opt_problem_.GetFunctional()->GetType(),
                                    "AdjointProblem::BoundaryRhs");
              }
          }
      }
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementMatrix(const EDC &edc,
                                     dealii::FullMatrix<double> &local_entry_matrix, double scale,
                                     double scale_ico)
  {
    pde_.ElementMatrix_T(edc, local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementTimeMatrix(const EDC &edc,
                                         FullMatrix<double> &local_entry_matrix)
  {
    pde_.ElementTimeMatrix_T(edc, local_entry_matrix);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename EDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::ElementTimeMatrixExplicit(const EDC &edc,
                                                 dealii::FullMatrix<double> &local_entry_matrix)
  {
    pde_.ElementTimeMatrixExplicit_T(edc, local_entry_matrix);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::FaceMatrix(const FDC &fdc,
                                  FullMatrix<double> &local_entry_matrix, double scale,
                                  double scale_ico)
  {
    pde_.FaceMatrix_T(fdc, local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::InterfaceMatrix(const FDC &fdc,
                                       FullMatrix<double> &local_entry_matrix, double scale,
                                       double scale_ico)
  {
    pde_.InterfaceMatrix_T(fdc,  local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename FDC>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
                 dim>::BoundaryMatrix(const FDC &fdc,
                                      FullMatrix<double> &local_matrix, double scale,
                                      double scale_ico)
  {
    pde_.BoundaryMatrix_T(fdc, local_matrix, scale*interval_length_, scale_ico*interval_length_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  std::string
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFType() const
  {
    return "state";
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const SmartPointer<const dealii::FESystem<dim> >
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFESystem() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetFESystem("state");
  }

  /******************************************************/
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const SmartPointer<const dealii::hp::FECollection<dim> >
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFECollection() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetFECollection("state");
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  UpdateFlags
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetUpdateFlags();
    r = r | opt_problem_.GetFunctional()->GetUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  UpdateFlags
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFaceUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetFaceUpdateFlags();
    r = r | opt_problem_.GetFunctional()->GetFaceUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  void
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::SetTime(
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
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeSparsityPattern(
    SPARSITYPATTERN &sparsity) const
  {
    opt_problem_.GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
  }

  /******************************************************/

//  template<typename OPTPROBLEM, typename PDE, typename DD,
//      typename SPARSITYPATTERN, typename VECTOR, int dim>
//    void
//    AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeMGSparsityPattern(
//        dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
//              unsigned int n_levels) const
//    {
//      opt_problem_.GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
//    }
//
///******************************************************/
//
//  template<typename OPTPROBLEM, typename PDE, typename DD,
//      typename SPARSITYPATTERN, typename VECTOR, int dim>
//    void
//    AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeMGSparsityPattern(
//        dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
//              unsigned int n_levels) const
//    {
//      opt_problem_.GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
//    }



  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasFaces() const
  {
    return pde_.HasFaces()
           || opt_problem_.GetFunctional()->HasFaces();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasPoints() const
  {
    return opt_problem_.GetFunctional()->HasPoints();
  }


  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasInterfaces() const
  {
    return pde_.HasInterfaces();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  bool
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasVertices() const
  {
    return pde_.HasVertices();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletColors() const
  {
    return dirichlet_colors_;
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<bool> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletCompMask(
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
        throw DOpEException(s.str(), "AdjointProblem::GetDirichletCompMask");
      }
    return dirichlet_comps_[comp];
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const Function<dim> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletValues(
    unsigned int color,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/) const
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

    return *(adjoint_dirichlet_values_);
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetBoundaryEquationColors() const
  {
    return adjoint_boundary_equation_colors_;
  }

  /******************************************************/
#if DEAL_II_VERSION_GTE(9,1,1)
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::AffineConstraints<double> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateDoFConstraints();
  }
#else
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::ConstraintMatrix &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateDoFConstraints();
  }
#endif
  /******************************************************/
#if DEAL_II_VERSION_GTE(9,1,1)
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::AffineConstraints<double> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetHNConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateHNConstraints();
  }
#else
  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  const dealii::ConstraintMatrix &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetHNConstraints() const
  {
    return opt_problem_.GetSpaceTimeHandler()->GetStateHNConstraints();
  }
#endif
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>  const dealii::Function<dim> &
  AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetInitialValues() const
  {
    return opt_problem_.GetInitialValues();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
           typename SPARSITYPATTERN, typename VECTOR, int dim>
  template<typename ELEMENTITERATOR>
  bool AdjointProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>
  ::AtInterface(ELEMENTITERATOR &element, unsigned int face) const
  {
    return pde_.AtInterface(element,face);
  }


///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
