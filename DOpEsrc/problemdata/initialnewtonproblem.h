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

#ifndef INITIAL_NEWTON_PROBLEM_H_
#define INITIAL_NEWTON_PROBLEM_H_

#include <basic/spacetimehandler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * This is a problem used in the solution of the
   * initial value for nonstationary problems
   *
   * @tparam PDE     The container with the Optimization problem description
   * @tparam PDE     The container with the PDE description
   * @tparam VECTOR  The vector class
   * @tparam dim     The dimension of the domain.
   */
  template<typename OPTPROBLEM, typename PDE, typename VECTOR, int dim>
  class InitialNewtonProblem
  {
  public:
    InitialNewtonProblem(PDE &pde) :
      pde_(pde)
    {
    }

    std::string
    GetName() const
    {
      return "InitialNewtonProblem";
    }
    std::string
    GetType() const
    {
      return "initial_state";
    }

    /******************************************************/
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
    ElementRhs(const EDC &edc,
               dealii::Vector<double> &local_vector, double scale = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    inline void
    PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs_vector, double scale = 1);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename EDC>
    inline void
    ElementMatrix(const EDC &edc,
                  dealii::FullMatrix<double> &local_matrix, double scale = 1.,
                  double scale_ico = 1.);

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
               dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename FDC>
    inline void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

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
    inline const dealii::SmartPointer<const dealii::hp::FECollection<dim> >
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
    SetTime(double time, const TimeIterator &interval);

    /**
    * Functions providing the required information for the integrator.
    * see OptProblemContainer for details.
    */
    template<typename SPARSITYPATTERN>
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
    inline const Function<dim> &
    GetDirichletValues(unsigned int color,
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
    inline const dealii::ConstraintMatrix &
    GetDoFConstraints() const;
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
      return pde_.GetOutputHandler();
    }
    OPTPROBLEM &
    GetBaseProblem()
    {
      return pde_.GetBaseProblem();
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
      //No coupling via the boundary should happen.
    }
    template<typename FDC>
    inline void OutflowValues(const  FDC &fdc,
                              std::vector<bool> &present_in_outflow,
                              dealii::Vector<double> &local_vector,
                              double scale,
                              double scale_ico)
    {
      pde_.Init_OutflowValues(fdc,present_in_outflow,local_vector,scale,scale_ico);
    }
    template<typename FDC>
    inline void
    OutflowMatrix(const FDC &fdc,
                  std::vector<bool> &present_in_outflow,
                  dealii::FullMatrix<double> &local_matrix,
                  double scale,
                  double scale_ico)
    {
      //No coupling to outflow values.
      pde_.Init_OutflowMatrix(fdc,present_in_outflow,local_matrix,scale,scale_ico);
    }
    inline void PipeCouplingResidual(dealii::Vector<double> &res,
                                     const dealii::Vector<double> &u,
                                     const std::vector<bool> &present_in_outflow)
    {
      pde_.Init_PipeCouplingResidual(res, u, present_in_outflow);
    }
    inline void CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                               const std::vector<bool> &present_in_outflow)
    {
      pde_.Init_CouplingMatrix(matrix,present_in_outflow);
    }


  protected:

  private:
    PDE &pde_;
  };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename EDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::ElementEquation(const EDC &edc,
      dealii::Vector<double> &local_vector, double scale,
      double scale_ico)
  {
    pde_.Init_ElementEquation(edc, local_vector, scale, scale_ico);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::FaceEquation(
    const FDC & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    //N.A.
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::InterfaceEquation(
    const FDC & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    //
  }
  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::BoundaryEquation(
    const FDC & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    //
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename EDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::ElementRhs(const EDC &edc,
      dealii::Vector<double> &local_vector, double scale)
  {
    pde_.Init_ElementRhs(edc, local_vector, scale);
  }

  /******************************************************/
  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::PointRhs(
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs_vector*/, double /*scale*/)
  {
    //
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::FaceRhs(
    const FDC & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::BoundaryRhs(
    const FDC & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename EDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::ElementMatrix(const EDC &edc,
      dealii::FullMatrix<double> &local_matrix, double scale, double scale_ico)
  {
    pde_.Init_ElementMatrix(edc, local_matrix, scale, scale_ico);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::FaceMatrix(const FDC & /*fdc*/,
      FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
  {
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::InterfaceMatrix(
    const FDC & /*fdc*/, FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
  {

  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename FDC>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::BoundaryMatrix(
    const FDC & /*fdc*/, FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
  {

  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  std::string
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetDoFType() const
  {
    return pde_.GetDoFType();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const SmartPointer<const dealii::FESystem<dim> >
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetFESystem() const
  {
    return pde_.GetFESystem();
  }

  /******************************************************/
  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const SmartPointer<const dealii::hp::FECollection<dim> >
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetFECollection() const
  {
    return pde_.GetFECollection();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  UpdateFlags
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  UpdateFlags
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetFaceUpdateFlags() const
  {
    UpdateFlags r;
    r = pde_.GetFaceUpdateFlags();
    return r | update_JxW_values;
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::SetTime(double time,
                                                             const TimeIterator &interval)
  {
    pde_.SetTime(time, interval);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename SPARSITYPATTERN>
  void
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::ComputeSparsityPattern(
    SPARSITYPATTERN &sparsity) const
  {
    pde_.ComputeStateSparsityPattern(sparsity);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  bool
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::HasFaces() const
  {
    return pde_.HasFaces();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  bool
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::HasPoints() const
  {
    return pde_.HasPoints();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  bool
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::HasInterfaces() const
  {
    return pde_.HasInterfaces();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetDirichletColors() const
  {
    return pde_.GetDirichletColors();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const std::vector<bool> &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetDirichletCompMask(
    unsigned int color) const
  {
    return pde_.GetDirichletCompMask(color);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const Function<dim> &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetDirichletValues(unsigned int color,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values) const
  {
    return pde_.GetDirichletValues(color, param_values, domain_values);
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const std::vector<unsigned int> &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetBoundaryEquationColors() const
  {
    return pde_.GetBoundaryEquationColors();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const dealii::ConstraintMatrix &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetDoFConstraints() const
  {
    return pde_.GetDoFConstraints();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  const dealii::Function<dim> &
  InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>::GetInitialValues() const
  {
    return pde_.GetInitialValues();
  }

  /******************************************************/

  template<typename OPTPROBEM, typename PDE, typename VECTOR, int dim>
  template<typename ELEMENTITERATOR>
  bool InitialNewtonProblem<OPTPROBEM, PDE, VECTOR, dim>
  ::AtInterface(ELEMENTITERATOR &element, unsigned int face) const
  {
    return pde_.AtInterface(element,face);
  }

///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
