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

#ifndef SHIFTEDCRANKNICOLSONProblem_H_
#define SHIFTEDCRANKNICOLSONProblem_H_

#include <problemdata/initialproblem.h>
#include <tsschemes/primal_ts_base.h>

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the shifted
   * Crank-Nicolson time-stepping scheme. The parameter \theta
   * is given by 1/2 + k, where k describes the time step size.
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
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
           int dealdim,
           template<int, int> class  FE = dealii::FESystem,
           template<int, int> class  DH = dealii::DoFHandler>
  class ShiftedCrankNicolsonProblem : public PrimalTSBase<OPTPROBLEM,
    SPARSITYPATTERN, VECTOR, dealdim, FE, DH>
  {
  public:
    ShiftedCrankNicolsonProblem(OPTPROBLEM &OP) :
      PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dealdim,
      FE, DH>(OP)
    {
      initial_problem_ = NULL;
    }
    ~ShiftedCrankNicolsonProblem()
    {
      if (initial_problem_ != NULL)
        delete initial_problem_;
    }

    /******************************************************/

    std::string
    GetName()
    {
      return "shifted Crank-Nicolson";
    }
    /******************************************************/

    InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim> &
    GetInitialProblem()
    {
      if (initial_problem_ == NULL)
        {
          initial_problem_ = new InitialProblem<ShiftedCrankNicolsonProblem,
          VECTOR, dealdim>(*this);
        }
      return *initial_problem_;
    }

    /******************************************************/
    ShiftedCrankNicolsonProblem &
    GetBaseProblem()
    {
      return *this;
    }

    /******************************************************/

    template<typename EDC>
    void
    ElementEquation(const EDC &dc,
                    dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;
          // implicit parts; e.g. for fluid problems: pressure and incompressibilty of v, get scaled with scale
          // The remaining parts; e.g. for fluid problems: laplace, convection, etc.:
          // Multiplication by 1/2 + k due to CN discretization

          this->GetProblem().ElementEquation(dc, tmp, damped_cn_theta * scale , scale);
          local_vector += tmp;

          tmp = 0.0;
          this->GetProblem().ElementTimeEquation(dc, tmp,
                                                 scale);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquationExplicit(dc, local_vector,
                                                         scale);
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;

          // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
          // Multiplication by 1/2 + k due to CN discretization
          this->GetProblem().ElementEquation(dc, tmp, (1.0 - damped_cn_theta) * scale, 0.);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquation(dc, local_vector,
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
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().ElementRhs(edc, local_vector, damped_cn_theta * scale);
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().ElementRhs(edc, local_vector,
                                        (1 - damped_cn_theta) * scale);
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
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      damped_cn_theta * scale);
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      (1 - damped_cn_theta) * scale);
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
      damped_cn_theta = 0.5
                        + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
      dealii::FullMatrix<double> m(local_matrix);

      // multiplication with 1/2 + k due to CN discretization for the 'normal' parts
      // no multiplication with 1/2 + k for the implicit parts
      //due to implicit treatment of pressure, etc. (in the case of fluid problems)
      this->GetProblem().ElementMatrix(edc, local_matrix, damped_cn_theta, 1.);

      m = 0.;
      this->GetProblem().ElementTimeMatrix(edc, m);
      local_matrix.add(
        1.0,
        m);

      m = 0.;
      this->GetProblem().ElementTimeMatrixExplicit(edc, m);
      local_matrix.add(
        1.0,
        m);
    }

    /******************************************************/

    template<typename FDC>
    void
    FaceEquation(const FDC &fdc,
                 dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
    {
      if (this->GetPart() == "New")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().FaceEquation(fdc, local_vector, damped_cn_theta * scale, scale);
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().FaceEquation(fdc, local_vector, (1.0 - damped_cn_theta) * scale, 0.);
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
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               damped_cn_theta * scale,scale );
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().InterfaceEquation(fdc, local_vector, (1.0 - damped_cn_theta) * scale, 0.);
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
      damped_cn_theta = 0.5
                        + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 + k due to CN time discretization
      this->GetProblem().FaceMatrix(fdc, m, damped_cn_theta, 1.);

      local_matrix.add(1., m);

    }

    /******************************************************/

    template<typename FDC>
    void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_matrix)
    {
      assert(this->GetPart() == "New");
      damped_cn_theta = 0.5
                        + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 + k due to CN time discretization
      this->GetProblem().InterfaceMatrix(fdc, m, damped_cn_theta, 1.);

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
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              damped_cn_theta * scale,scale);
        }
      else if (this->GetPart() == "Old")
        {
          damped_cn_theta = 0.5
                            + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              (1.0 - damped_cn_theta) * scale, 0.);
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
      damped_cn_theta = 0.5
                        + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
      dealii::FullMatrix<double> m(local_matrix);

      m = 0.;
      // Multiplication with 1/2 + k due to CN time discretization
      this->GetProblem().BoundaryMatrix(fdc, m, damped_cn_theta, 1.);
      local_matrix.add(1., m);

    }

  private:
    double damped_cn_theta;
    InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim> *initial_problem_;
  };
}

#endif
