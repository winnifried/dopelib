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

#ifndef _ForwardEulerProblem_H_
#define _ForwardEulerProblem_H_

#include "initialproblem.h" 
#include "primal_ts_base.h"

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the forward Euler
   * time stepping scheme which is an explicit scheme.
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
      int dopedim, int dealdim,
      template <int, int> class FE = dealii::FESystem,
      template <int, int> class DH = dealii::DoFHandler>
    class ForwardEulerProblem : public PrimalTSBase<OPTPROBLEM,
    SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>
    {
      public:
        ForwardEulerProblem(OPTPROBLEM& OP) :
            PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
                FE, DH>(OP)
        {
          _initial_problem = NULL;
        }
        ~ForwardEulerProblem()
        {
          if (_initial_problem != NULL)
            delete _initial_problem;
        }

      /******************************************************/

      std::string
      GetName()
      {
        return "forward Euler";
      }
      
      /******************************************************/

      InitialProblem<ForwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>, VECTOR, dealdim>&
      GetInitialProblem()
      {
	if (_initial_problem == NULL)
	{
	  _initial_problem = new InitialProblem<ForwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>, VECTOR, dealdim>
	  (*this);
	}
	return *_initial_problem;
      }

      /******************************************************/
      ForwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim, FE, DH>&
      GetBaseProblem()
      {
        return *this;
      }
      /******************************************************/
      
      template<typename CDC>
        void
        ElementEquation(const CDC& cdc,
		     dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
        {
          if (this->GetPart() == "New")
            {
              dealii::Vector<double> tmp(local_vector);
              tmp = 0.0;
              this->GetProblem().ElementEquation(cdc, tmp, 0., scale * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_vector += tmp;

              tmp = 0.0;
              this->GetProblem().ElementTimeEquation(cdc, tmp,
                  scale );
              local_vector += tmp;

              this->GetProblem().ElementTimeEquationExplicit(cdc, local_vector,
                  scale );

            }
          else if (this->GetPart() == "Old")
            {
              dealii::Vector<double> tmp(local_vector);
              tmp = 0.0;
              this->GetProblem().ElementEquation(cdc, tmp, scale * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 0.);
              local_vector += tmp;

              this->GetProblem().ElementTimeEquation(
                  cdc,
                  local_vector,
                  (-1) * scale);
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

     template<typename CDC>
        void
        ElementRhs(const CDC& cdc,
            dealii::Vector<double> &local_vector, double scale)
        {
          if (this->GetPart() == "New")
            {

            }
          else if (this->GetPart() == "Old")
            {
              this->GetProblem().ElementRhs(cdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else
            {
              abort();
            }
        }


        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale)
        {
          if (this->GetPart() == "New")
          {
          }
          else if (this->GetPart() == "Old")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }
          else
          {
            abort();
          }
        }

      /******************************************************/

      template<typename CDC>
        void
        ElementMatrix(const CDC& cdc,
		   dealii::FullMatrix<double> &local_matrix)
        {
          assert(this->GetPart() == "New");
          dealii::FullMatrix<double> m(local_matrix);

          this->GetProblem().ElementMatrix(cdc, local_matrix, 0., 1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

          m = 0.;
          this->GetProblem().ElementTimeMatrix(cdc, m);
          local_matrix.add(
              1.0, m);

          m = 0.;
          this->GetProblem().ElementTimeMatrixExplicit(cdc, m);
          local_matrix.add(
              1.0 , m);

        }

      /******************************************************/

     template<typename FDC>
        void
        FaceEquation(const FDC& fdc,
		     dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
        {
          if (this->GetPart() == "New")
            {
	      this->GetProblem().FaceEquation(fdc, local_vector, 0., scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else if (this->GetPart() == "Old")
            {
              this->GetProblem().FaceEquation(fdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),0.);
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
			  dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
        {
          if (this->GetPart() == "New")
            {
	      this->GetProblem().InterfaceEquation(fdc, local_vector, 0., scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else if (this->GetPart() == "Old")
            {
              this->GetProblem().InterfaceEquation(fdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),0.);
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
            dealii::Vector<double> &local_vector, double scale = 1.)
        {
          this->GetProblem().FaceRhs(fdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
        }

      /******************************************************/

      template<typename FDC>
        void
        FaceMatrix(const FDC& fdc,
		   dealii::FullMatrix<double> &local_matrix)
        {
          assert(this->GetPart() == "New");
	  this->GetProblem().FaceMatrix(fdc, local_matrix, 0., 1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          
        }

      /******************************************************/

      template<typename FDC>
        void
        InterfaceMatrix(const FDC& fdc,
            dealii::FullMatrix<double> &local_matrix)
        {
          assert(this->GetPart() == "New");
          this->GetProblem().InterfaceMatrix(fdc, local_matrix, 0., 1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          
        }

      /******************************************************/

      template<typename FDC>
        void
        BoundaryEquation(const FDC& fdc,
			 dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/)
        {
          if (this->GetPart() == "New")
            {
	      this->GetProblem().BoundaryEquation(fdc, local_vector, 0., scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else if (this->GetPart() == "Old")
            {
              this->GetProblem().BoundaryEquation(fdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),0.);
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
            dealii::Vector<double> &local_vector, double scale)
        {
          this->GetProblem().BoundaryRhs(fdc, local_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
        }

      /******************************************************/

      template<typename FDC>
        void
        BoundaryMatrix(const FDC& fdc,
		       dealii::FullMatrix<double> &local_matrix)
        {
          assert(this->GetPart() == "New");
          this->GetProblem().BoundaryMatrix(fdc, local_matrix, 0., 1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
        }
    private:
      InitialProblem<ForwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
