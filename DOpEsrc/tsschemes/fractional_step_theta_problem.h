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

#ifndef FRACTIONALSTEPTHETAPROBLEM_H_
#define FRACTIONALSTEPTHETAPROBLEM_H_

#include <problemdata/initialproblem.h>
#include <tsschemes/primal_ts_base.h>

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the Fractional-Step-Theta
   * time stepping scheme.
   *
   * This time stepping scheme is divided into three subroutines, i.e.,
   * normally six parts have to be computed. Since two parts coincide
   * the function is split into five parts.
   *
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
           int dealdim, template<int, int> class FE = dealii::FESystem,
           template<int, int> class DH = dealii::DoFHandler>
  class FractionalStepThetaProblem : public PrimalTSBase<OPTPROBLEM,
    SPARSITYPATTERN, VECTOR, dealdim, FE, DH>
  {
  public:
    FractionalStepThetaProblem(OPTPROBLEM &OP) :
      PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dealdim,
      FE, DH>(OP)
    {
      fs_theta_ = 1.0 - std::sqrt(2.0) / 2.0;
      fs_theta_prime_ = 1.0 - 2.0 * fs_theta_;
      fs_alpha_ = (1.0 - 2.0 * fs_theta_) / (1.0 - fs_theta_);
      fs_beta_ = 1.0 - fs_alpha_;
      initial_problem_ = NULL;
    }

    ~FractionalStepThetaProblem()
    {
      if (initial_problem_ != NULL)
        delete initial_problem_;
    }

    /******************************************************/

    std::string
    GetName()
    {
      return "Fractional-Step-Theta";
    }
    /******************************************************/

    InitialProblem<
    FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                               dealdim, FE, DH>, VECTOR, dealdim>&
                               GetInitialProblem()
    {
      if (initial_problem_ == NULL)
        {
          initial_problem_ = new InitialProblem<
          FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
          dealdim, FE, DH>, VECTOR, dealdim>(*this);
        }
      return *initial_problem_;
    }

    /******************************************************/
    FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
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
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          dealii::Vector<double> tmp(local_vector);

          tmp = 0.0;
          this->GetProblem().ElementEquation(edc, tmp,
                                             scale * fs_alpha_,
                                             scale);
          local_vector += tmp;

          tmp = 0.0;
          this->GetProblem().ElementTimeEquation(edc, tmp, scale / (fs_theta_));
          local_vector += tmp;

          this->GetProblem().ElementTimeEquationExplicit(edc, local_vector,
                                                         scale / (fs_theta_));
        }
      else if (this->GetPart() == "Old_for_1st_cycle")
        {
          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;

          // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
          this->GetProblem().ElementEquation(edc, tmp,
                                             scale * fs_beta_,
                                             0.);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquation(edc, local_vector,
                                                 (-1) * scale / (fs_theta_));
        }
      else if (this->GetPart() == "Old_for_3rd_cycle")
        {
          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;

          // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
          this->GetProblem().ElementEquation(edc, tmp,
                                             scale * fs_beta_,
                                             0.);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquation(edc, local_vector,
                                                 (-1) * scale / (fs_theta_));
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          dealii::Vector<double> tmp(local_vector);

          tmp = 0.0;
          this->GetProblem().ElementEquation(edc, tmp,
                                             scale * fs_beta_,
                                             scale);
          local_vector += tmp;

          tmp = 0.0;
          this->GetProblem().ElementTimeEquation(edc, tmp,
                                                 scale / (fs_theta_prime_));
          local_vector += tmp;

          this->GetProblem().ElementTimeEquationExplicit(edc, local_vector,
                                                         scale / (fs_theta_prime_));
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
          dealii::Vector<double> tmp(local_vector);
          tmp = 0.0;

          // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
          this->GetProblem().ElementEquation(edc, tmp,
                                             scale * fs_alpha_,
                                             0.);
          local_vector += tmp;

          this->GetProblem().ElementTimeEquation(edc, local_vector,
                                                 (-1) * scale / (fs_theta_prime_));
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
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {

        }
      else if (this->GetPart() == "Old_for_1st_cycle"
               || this->GetPart() == "Old_for_3rd_cycle")
        {
          this->GetProblem().ElementRhs(edc, local_vector,
                                        scale);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          this->GetProblem().ElementRhs(edc, local_vector,
                                        scale);
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
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
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {

        }
      else if (this->GetPart() == "Old_for_1st_cycle"
               || this->GetPart() == "Old_for_3rd_cycle")
        {
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      scale);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                                      scale);
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
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
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().ElementMatrix(edc, m,
                                           fs_alpha_,
                                           1.);
          local_matrix.add(1.0, m);

          m = 0.;
          this->GetProblem().ElementTimeMatrix(edc, m);
          local_matrix.add(1.0 / (fs_theta_), m);

          m = 0.;
          this->GetProblem().ElementTimeMatrixExplicit(edc, m);
          local_matrix.add(1.0 / (fs_theta_), m);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().ElementMatrix(edc, local_matrix,
                                           fs_beta_,
                                           1.);
          local_matrix.add(1.0, m);

          m = 0.;
          this->GetProblem().ElementTimeMatrix(edc, m);
          local_matrix.add(1.0 / (fs_theta_prime_), m);

          m = 0.;
          this->GetProblem().ElementTimeMatrixExplicit(edc, m);
          local_matrix.add(1.0 / (fs_theta_prime_), m);
        }
    }

    /******************************************************/

    template<typename FDC>
    void
    FaceEquation(const FDC &fdc,
                 dealii::Vector<double> &local_vector, double scale, double)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          this->GetProblem().FaceEquation(fdc, local_vector,
                                          scale * fs_alpha_,
                                          scale);

        }
      else if ((this->GetPart() == "Old_for_1st_cycle")
               || (this->GetPart() == "Old_for_3rd_cycle"))
        {
          this->GetProblem().FaceEquation(fdc, local_vector,
                                          scale * fs_beta_,
                                          0);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          this->GetProblem().FaceEquation(fdc, local_vector,
                                          scale * fs_beta_,
                                          scale);
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
          this->GetProblem().FaceEquation(fdc, local_vector,
                                          scale * fs_alpha_,
                                          0.);
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
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               scale * fs_alpha_,
                                               scale);

        }
      else if ((this->GetPart() == "Old_for_1st_cycle")
               || (this->GetPart() == "Old_for_3rd_cycle"))
        {
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               scale * fs_beta_,
                                               0);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               scale * fs_beta_,
                                               scale);
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
          this->GetProblem().InterfaceEquation(fdc, local_vector,
                                               scale * fs_alpha_,
                                               0.);
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
      this->GetProblem().FaceRhs(fdc, local_vector,
                                 scale);
    }

    /******************************************************/

    template<typename FDC>
    void
    FaceMatrix(const FDC &fdc,
               dealii::FullMatrix<double> &local_matrix)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().FaceMatrix(fdc, m,
                                        fs_alpha_,
                                        1.);
          local_matrix.add(1., m);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().FaceMatrix(fdc, m,
                                        fs_beta_,
                                        1.);
          local_matrix.add(1.0, m);
        }

    }

    /******************************************************/

    template<typename FDC>
    void
    InterfaceMatrix(const FDC &fdc,
                    dealii::FullMatrix<double> &local_matrix)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().InterfaceMatrix(fdc, m,
                                             fs_alpha_,
                                             1.);
          local_matrix.add(1.0, m);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().InterfaceMatrix(fdc, m,
                                             fs_beta_,
                                             1.);
          local_matrix.add(1.0, m);
        }

    }

    /******************************************************/

    template<typename FDC>
    void
    BoundaryEquation(const FDC &fdc,
                     dealii::Vector<double> &local_vector, double scale, double)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              scale * fs_alpha_, scale);
        }
      else if ((this->GetPart() == "Old_for_1st_cycle")
               || (this->GetPart() == "Old_for_3rd_cycle"))
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              scale * fs_beta_,
                                              0.);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              scale * fs_beta_,
                                              scale);
        }
      else if (this->GetPart() == "Old_for_2nd_cycle")
        {
          this->GetProblem().BoundaryEquation(fdc, local_vector,
                                              scale * fs_alpha_,
                                              0);
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
      this->GetProblem().BoundaryRhs(fdc, local_vector,
                                     scale);
    }

    /******************************************************/

    template<typename FDC>
    void
    BoundaryMatrix(const FDC &fdc,
                   dealii::FullMatrix<double> &local_matrix)
    {
      if (this->GetPart() == "New_for_1st_and_3rd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().BoundaryMatrix(fdc, m,
                                            fs_alpha_,
                                            1.);
          local_matrix.add(1.0, m);
        }
      else if (this->GetPart() == "New_for_2nd_cycle")
        {
          dealii::FullMatrix<double> m(local_matrix);
          m = 0.;
          this->GetProblem().BoundaryMatrix(fdc, m,
                                            fs_beta_,
                                            1.);
          local_matrix.add(1.0, m);
        }
    }
  private:
    // parameters for FS scheme
    double fs_theta_;
    double fs_theta_prime_;
    double fs_alpha_;
    double fs_beta_;

    InitialProblem<
    FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                               dealdim, FE, DH>, VECTOR, dealdim> * initial_problem_;
  };
}

#endif
