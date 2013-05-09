/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _FRACTIONALSTEPTHETAPROBLEM_H_
#define _FRACTIONALSTEPTHETAPROBLEM_H_

#include "initialproblem.h" 
#include "primal_ts_base.h"

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
   * @tparam <dopedim>          The dimension for the control variable.
   * @tparam <dealdim>          The dimension of the state variable.
   * @tparam <FE>               The type of finite elements in use, must be compatible with the DH.
   * @tparam <DH>               The type of the DoFHandler in use 
   *                            (to be more precise: The type of the dealii-DoFhandler which forms
   *                            the base class of the DOpEWrapper::DoFHandler in use.)
   *
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim, template<int, int> class FE = dealii::FESystem,
      template<int, int> class DH = dealii::DoFHandler>
    class FractionalStepThetaProblem : public PrimalTSBase<OPTPROBLEM,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>
    {
      public:
        FractionalStepThetaProblem(OPTPROBLEM& OP) :
            PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
                FE, DH>(OP)
        {
          _fs_theta = 1.0 - std::sqrt(2.0) / 2.0;
          _fs_theta_prime = 1.0 - 2.0 * _fs_theta;
          _fs_alpha = (1.0 - 2.0 * _fs_theta) / (1.0 - _fs_theta);
          _fs_beta = 1.0 - _fs_alpha;
          _initial_problem = NULL;
        }

        ~FractionalStepThetaProblem()
        {
          if (_initial_problem != NULL)
            delete _initial_problem;
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
                dopedim, dealdim, FE, DH>, VECTOR, dealdim>&
        GetInitialProblem()
        {
          if (_initial_problem == NULL)
          {
            _initial_problem = new InitialProblem<
                FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                    dopedim, dealdim, FE, DH>, VECTOR, dealdim>(*this);
          }
          return *_initial_problem;
        }

        /******************************************************/
        FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim, FE, DH>&
        GetBaseProblem()
        {
          return *this;
        }
        /******************************************************/

         template<typename CDC>
          void
          CellEquation(const CDC& cdc,
		       dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);

              tmp = 0.0;
              this->GetProblem().CellEquation(cdc, tmp,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(cdc, tmp, scale / (_fs_theta));
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(cdc, local_cell_vector,
                  scale / (_fs_theta));
            }
            else if (this->GetPart() == "Old_for_1st_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(cdc, tmp,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(cdc, local_cell_vector,
                  (-1) * scale / (_fs_theta));
            }
            else if (this->GetPart() == "Old_for_3rd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(cdc, tmp,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(cdc, local_cell_vector,
                  (-1) * scale / (_fs_theta));
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);

              tmp = 0.0;
              this->GetProblem().CellEquation(cdc, tmp,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(cdc, tmp,
                  scale / (_fs_theta_prime));
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(cdc, local_cell_vector,
                  scale / (_fs_theta_prime));
            }
            else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(cdc, tmp,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(cdc, local_cell_vector,
                  (-1) * scale / (_fs_theta_prime));
            }
            else
            {
              abort();
            }
          }

        /******************************************************/

        template<typename CDC>
          void
          CellRhs(const CDC& cdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {

            }
            else if (this->GetPart() == "Old_for_1st_cycle"
                || this->GetPart() == "Old_for_3rd_cycle")
            {
              this->GetProblem().CellRhs(cdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().CellRhs(cdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
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
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
          {

          }
          else if (this->GetPart() == "Old_for_1st_cycle"
              || this->GetPart() == "Old_for_3rd_cycle")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }
          else if (this->GetPart() == "New_for_2nd_cycle")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
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

        template<typename CDC>
          void
          CellMatrix(const CDC& cdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().CellMatrix(cdc, m,
                  _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1.0, m);

              m = 0.;
              this->GetProblem().CellTimeMatrix(cdc, m);
              local_entry_matrix.add(1.0 / (_fs_theta), m);

              m = 0.;
              this->GetProblem().CellTimeMatrixExplicit(cdc, m);
              local_entry_matrix.add(1.0 / (_fs_theta), m);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().CellMatrix(cdc, local_entry_matrix,
                  _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1.0, m);

              m = 0.;
              this->GetProblem().CellTimeMatrix(cdc, m);
              local_entry_matrix.add(1.0 / (_fs_theta_prime), m);

              m = 0.;
              this->GetProblem().CellTimeMatrixExplicit(cdc, m);
              local_entry_matrix.add(1.0 / (_fs_theta_prime), m);
            }
          }

        /******************************************************/

        template<typename FDC>
          void
          FaceEquation(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale, double)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

            }
            else if ((this->GetPart() == "Old_for_1st_cycle")
                || (this->GetPart() == "Old_for_3rd_cycle"))
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
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
          InterfaceEquation(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double /*scale_ico*/)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

            }
            else if ((this->GetPart() == "Old_for_1st_cycle")
                || (this->GetPart() == "Old_for_3rd_cycle"))
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
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
          FaceRhs(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.)
          {
            this->GetProblem().FaceRhs(fdc, local_cell_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

        template<typename FDC>
          void
          FaceMatrix(const FDC& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().FaceMatrix(fdc, m,
                  _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1., m);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().FaceMatrix(fdc, m,
                  _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1.0, m);
            }

          }

        /******************************************************/

        template<typename FDC>
          void
          InterfaceMatrix(const FDC& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().InterfaceMatrix(fdc, m,
                  _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1.0, m);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().InterfaceMatrix(fdc, m,
                  _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_entry_matrix.add(1.0, m);
            }

          }

        /******************************************************/

        template<typename FDC>
          void
          BoundaryEquation(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale, double)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if ((this->GetPart() == "Old_for_1st_cycle")
                || (this->GetPart() == "Old_for_3rd_cycle"))
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  0.);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
                  scale * _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
                  scale * _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
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
          BoundaryRhs(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            this->GetProblem().BoundaryRhs(fdc, local_cell_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

        template<typename FDC>
          void
          BoundaryMatrix(const FDC& fdc,
              dealii::FullMatrix<double> &local_cell_matrix)
          {
            if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_cell_matrix);
              m = 0.;
              this->GetProblem().BoundaryMatrix(fdc, m,
                  _fs_alpha
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_matrix.add(1.0, m);
            }
            else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_cell_matrix);
              m = 0.;
              this->GetProblem().BoundaryMatrix(fdc, m,
                  _fs_beta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  1.
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_matrix.add(1.0, m);
            }
          }
      private:
        // parameters for FS scheme
        double _fs_theta;
        double _fs_theta_prime;
        double _fs_alpha;
        double _fs_beta;

        InitialProblem<
            FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                dopedim, dealdim, FE, DH>, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
