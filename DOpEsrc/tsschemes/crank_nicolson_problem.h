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

#ifndef CRANKNICOLSONProblem_H_
#define CRANKNICOLSONProblem_H_

#include <problemdata/initialproblem.h>
#include <tsschemes/primal_ts_base.h>

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the Crank-Nicolson
   * time stepping scheme.
   *
   * All member functions have a corresponding function in BackwardEulerProblem.
   * For a detailed documentation please consult the corresponding documentation of
   * BackwardEulerProblem
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
   *
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
           int dealdim,
           template <int, int> class FE = dealii::FESystem,
           template <int, int> class DH = dealii::DoFHandler>
  class CrankNicolsonProblem : public PrimalTSBase<OPTPROBLEM,
    SPARSITYPATTERN, VECTOR, dealdim, FE, DH>
  {
  public:
    CrankNicolsonProblem(OPTPROBLEM &OP) :
      PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dealdim,
      FE, DH>(OP)
    {
      initial_problem_ = NULL;
    }
    ~CrankNicolsonProblem()
    {
      if (initial_problem_ != NULL)
        delete initial_problem_;
    }

    /******************************************************/

    std::string
    GetName()
    {
      return "Crank-Nicolson";
    }
    /******************************************************/

    InitialProblem<
    CrankNicolsonProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                         dealdim, FE, DH>, VECTOR, dealdim>&
                         GetInitialProblem()
    {
      if (initial_problem_ == NULL)
        {
          initial_problem_ = new InitialProblem<
          CrankNicolsonProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
          dealdim, FE, DH>, VECTOR, dealdim>(*this);
        }
      return *initial_problem_;
    }

    /******************************************************/
    CrankNicolsonProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
    dealdim, FE, DH>&
    GetBaseProblem()
    {
      return *this;
    }

    /******************************************************/

    template<typename EDC>
    void
    ElementEquation(const EDC &edc,
                    dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          dealii::Vector<double> tmp(local_vector);

          tmp = 0.0;
          // The remaining parts; e.g. for fluid problems: laplace, convection, etc.
          // Multiplication by 1/2 due to CN discretization
          this->GetProblem().ElementEquation(edc, tmp, 0.5 * scale, scale);
          local_vector += tmp;

          tmp = 0.0;
          this->GetProblem().ElementTimeEquation(edc, tmp,
                                                 scale);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquationExplicit(edc, local_vector,
                                                         scale);

        }
      else if (this->GetPart() == "Old")
        {
          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;

          // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
          // Multiplication by 1/2 due to CN discretization
          this->GetProblem().ElementEquation(edc, tmp, 0.5 * scale, 0.);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquation(
            edc,
            local_vector,
            (-1) * scale);
        }
      else
        {
          abort();
        }
    }

    /******************************************************/

    template<typename EDC>
    void
    ElementRhs(const EDC &edc,
               dealii::Vector<double> &local_vector, double scale)
    {
      if (this->GetPart() == "New")
        {
          this->GetProblem().ElementRhs(edc, local_vector, 0.5 * scale);
        }
      else if (this->GetPart() == "Old")
        {
          this->GetProblem().ElementRhs(edc, local_vector, 0.5 * scale);
        }
      else
        {
          abort();
        }
    }

    /******************************************************/

    void
    PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs_vector, double scale)
    {
      if (this->GetPart() == "New")
        {
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      0.5 * scale);
        }
      else if (this->GetPart() == "Old")
        {
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      (0.5) * scale);
        }
      else
        {
          abort();
        }
    }

    /******************************************************/

    template<typename EDC>
    void
    ElementMatrix(const EDC &edc,
                  dealii::FullMatrix<double> &local_matrix)
    {
      assert(this->GetPart() == "New");
      dealii::FullMatrix<double> m(local_matrix);

      // multiplication with 1/2 for scale due to CN discretization,
      //no multiplication with 1/2 for scale_ico due to implicit treatment of pressure, etc. (in the case of fluid problems)
      this->GetProblem().ElementMatrix(edc, local_matrix, 0.5, 1.);

      m = 0.;
      this->GetProblem().ElementTimeMatrix(edc, m);
      local_matrix.add(
        1.0 , m);

      m = 0.;
      this->GetProblem().ElementTimeMatrixExplicit(edc, m);
      local_matrix.add(
        1.0 , m);
    }

    /******************************************************/

    template<typename FDC>
    void
    FaceEquation(const FDC &fdc,
                 dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          this->GetProblem().FaceEquation(fdc, local_vector, 0.5 * scale, scale);
        }
      else if (this->GetPart() == "Old")
        {
          this->GetProblem().FaceEquation(fdc, local_vector, 0.5 * scale,0.);
        }
      else
        {
          abort();
        }

    }

    /******************************************************/

    template<typename FDC>
    void
    InterfaceEquation(const FDC &fdc,
                      dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               0.5 * scale, scale);
        }
      else if (this->GetPart() == "Old")
        {
          this->GetProblem().InterfaceEquation(fdc,  local_vector,
                                               0.5 * scale, 0.);
        }
      else
        {
          abort();
        }

    }

    /******************************************************/

    template<typename FDC>
    void
    FaceRhs(const FDC &fdc,
            dealii::Vector<double> &local_vector, double scale = 1.)
    {
      this->GetProblem().FaceRhs(fdc, local_vector, scale);
    }

    /******************************************************/

    template<typename FDC>
    void
    FaceMatrix(const FDC &fdc,
               dealii::FullMatrix<double> &local_matrix)
    {
      assert(this->GetPart() == "New");
      // Hier nicht mit this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_matrix schon skaliert ist
      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 due to CN time discretization
      this->GetProblem().FaceMatrix(fdc, m, 0.5, 1.);

      local_matrix.add(1., m);
    }

    /******************************************************/

    template<typename FDC>
    void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_matrix)
    {
      assert(this->GetPart() == "New");
      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 due to CN time discretization
      this->GetProblem().InterfaceMatrix(fdc,  m, 0.5, 1.);

      local_matrix.add(1., m);
    }

    /******************************************************/

    template<typename FDC>
    void
    BoundaryEquation(const FDC &fdc,
                     dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector, 0.5 * scale, scale);
        }
      else if (this->GetPart() == "Old")
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector, 0.5 * scale, 0.);
        }
      else
        {
          abort();
        }

    }

    /******************************************************/

    template<typename FDC>
    void
    BoundaryRhs(const FDC &fdc,
                dealii::Vector<double> &local_vector, double scale)
    {
      this->GetProblem().BoundaryRhs(fdc, local_vector, scale);
    }

    /******************************************************/

    template<typename FDC>
    void
    BoundaryMatrix(const FDC &fdc,
                   dealii::FullMatrix<double> &local_matrix)
    {
      assert(this->GetPart() == "New");
      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 due to CN time discretization
      this->GetProblem().BoundaryMatrix(fdc, m, 0.5, 1.);
      local_matrix.add(1., m);

    }
  private:
    InitialProblem<CrankNicolsonProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dealdim, FE, DH>, VECTOR, dealdim> *initial_problem_;
  };
}

#endif
