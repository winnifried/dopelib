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

#ifndef _BackwardEulerProblem_H_
#define _BackwardEulerProblem_H_

#include "initialproblem.h" 
#include "primal_ts_base.h"

namespace DOpE
{
  /**
   * @class BackwardEulerProblem 
   *
   * Class to compute time dependent problems with the backward Euler
   * time stepping scheme.
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
    class BackwardEulerProblem : public PrimalTSBase<OPTPROBLEM,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>
    {
      public:
        BackwardEulerProblem(OPTPROBLEM& OP) :
            PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
                FE, DH>(OP)
        {
          _initial_problem = NULL;
        }
        ~BackwardEulerProblem()
        {
          if (_initial_problem != NULL)
            delete _initial_problem;
        }

        /******************************************************/

        /**
         * Returns the name of the time stepping scheme.
         *
         * @return A string containing the name of the time stepping scheme.
         */
        std::string
        GetName()
        {
          return "backward Euler";
        }
        /******************************************************/
      
        /**
	 * Returns a pointer to the problem used to calculate
	 * the initial values used for this scheme.
	 */
        InitialProblem<
            BackwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
                dealdim, FE, DH>, VECTOR, dealdim>&
        GetInitialProblem()
        {
          if (_initial_problem == NULL)
          {
            _initial_problem = new InitialProblem<
                BackwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
                    dopedim, dealdim, FE, DH>, VECTOR, dealdim>(*this);
          }
          return *_initial_problem;
        }

        /******************************************************/
        
        /**
         * Returns a pointer to the base problem, here `this'.
	 * This behavior is temporary to allow use of the BackwardEulerProblem
	 * until all subproblems (i.e., Primal, Dual, Tangent,...)
	 * have their own description.
	 */
        BackwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim, FE, DH> &
        GetBaseProblem()
        {
          return *this;
        }

        /******************************************************/

        /**
         * Computes the value of the cell equation for the time-step problem.
         * This is build from the three functions 
	 * CellEquation, CellTimeEquation, CellTimeEquationExplicit
	 * provided by the PDE:
	 * CellEquation: The spatial integrals
         * CellTimeEquation: The time derivative part in the equation
	 *                   if \partial_t can be approximated with 
	 *                   difference quotient between t_n and t_{n+1}.
	 * TimeEquationExplicit: Explicit calculation of the time derivative if
	 *                       a simple difference quotient is not sufficient
	 *                       as it may happen for timederivatives of nonlinear terms.
         *
         * The function is divided into two parts `old' and `new' which  are given
         * to the Newton solver. Then, the computation is done in two steps: first
         * computation of the old Newton- or time step equation parts. After,
         * computation of the actual parts.
         *
	 * @tparam <CDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., CellDataContainer
         *
	 * @param cdc                      The CDC object.
         * @param local_cell_vector        This vector contains the locally computed values 
         *                                 of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
	 * @param scale_ico                Given for compatibility reasons with the CellEquation 
	 *                                 in PDEInterface. Should not be used here!
         */
        template<typename CDC>
          void
          CellEquation(const CDC & cdc,
		       dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
          {
            if (this->GetPart() == "New")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;
              this->GetProblem().CellEquation(cdc, tmp,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(cdc, tmp, scale);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(cdc, local_cell_vector,
                  scale);
            }
            else if (this->GetPart() == "Old")
            {
              this->GetProblem().CellTimeEquation(cdc, local_cell_vector,
                  (-1) * scale);
            }
            else
            {
              abort();
            }
          }

        /******************************************************/

        /**
         * Computes the value of the right-hand side.
         * The function is divided into two parts `old' and `new' which  are given
         * the Newton solver. Then, the computation is done in two steps: first
         * computation of the old Newton- or time step equation parts. After,
         * computation of the actual parts.
         *
	 * @tparam <CDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., CellDataContainer
         *
         * @param cdc                      A DataContainer holding all the needed information
         *                                 of the cell
         * @param local_cell_vector        This vector contains the locally computed values of 
	 *                                 the CellRhs. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
         */
        template<typename CDC>
          void
          CellRhs(const CDC & cdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            if (this->GetPart() == "New")
            {
              this->GetProblem().CellRhs(cdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
            }
            else
            {
              abort();
            }
          }

        /******************************************************/

        /**
         * Computes the value of the right-hand side of the problem at hand, if it
         * contains pointevaluations.
         * The function is divided into two parts `old' and `new' which  are given
         * the Newton solver. Then, the computation is done in two steps: first
         * computation of the old Newton- or time step equation parts. After,
         * computation of the actual parts.
         *
         *
         * @param param_values             A std::map containing parameter data 
	 *                                 (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map 
	 *                                 at the position "control".
         * @param domain_values            A std::map containing domain data 
	 *                                 (i.e., nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the
	 *                                 position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param local_cell_vector        This vector contains the locally computed values 
	 *                                 of the PointRhs. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
         */
        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale)
        {
          if (this->GetPart() == "New")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }
          else if (this->GetPart() == "Old")
          {
          }
          else
          {
            abort();
          }
        }

        /******************************************************/

        /**
         * Computes the value of the cell matrix which is derived
         * by computing the directional derivatives of the residuum equation of the PDE
         * under consideration.
         * This function itself contains a maximum of four subroutines of matrix equations:
         * CellMatrix, CellMatrixExplicit, CellTimeMatrix, CellTimeMatrixExplicit.
         * So far, all three types are needed for fluid-structure interaction problems:
         * CellMatrix:           implicit terms, like pressure.
         * CellMatrixExplicit:   stress tensors, fluid convection, etc.
         * TimeMatrixExplicit:   time derivatives of certain variables which are
         *                       combined with transformations, etc.
         *
         * In fluid problems, the CellMatrix terms coincide. However the
         * TimeMatrix terms differ:
         * CellTimeMatrix: time derivatives, e.g., dt v in direction \partial v
         *
         * This function is just considered in the `new' part. This is due to that directional
         * derivatives vanish if they are applied to old values which are, of course,
         * already computed and therefore constant.
         *
	 * @tparam <CDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., CellDataContainer
         * @param cdc                      A DataContainer holding all the needed information
         *                                 of the cell
         * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs 
 	 *                                 times local DoFs and is
         *                                 filled by the locally computed values. For more information 
	 *                                 of its functionality, please
         *                                 search for the keyword `FullMatrix' in the deal.ii manual.
         */
        template<typename CDC>
          void
          CellMatrix(const CDC & cdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            dealii::FullMatrix<double> m(local_entry_matrix);

            this->GetProblem().CellMatrix(cdc, local_entry_matrix,
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

            m = 0.;
            this->GetProblem().CellTimeMatrix(cdc, m);
            local_entry_matrix.add(1.0, m);

            m = 0.;
            this->GetProblem().CellTimeMatrixExplicit(cdc, m);
            local_entry_matrix.add(1.0, m);

          }

        /******************************************************/

        /**
         * Same functionality as for the CellEquation, but on Faces.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_cell_vector        This vector contains the locally computed values 
         *                                 of the Facequation. 
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
	 * @param scale_ico                Given for compatibility reasons with the CellEquation 
	 *                                 in PDEInterface. Should not be used here!
          */
        template<typename FDC>
          void
          FaceEquation(const FDC & fdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double /*scale_ico*/)
          {
            if (this->GetPart() == "New")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
            }
            else
            {
              abort();
            }
          }

        /******************************************************/

       /**
         * Same functionality as for the CellEquation, but on Interfaces, i.e. the same as 
	 * FaceEquation but with access to the FEValues on both sides.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_cell_vector        This vector contains the locally computed values 
         *                                 of the InterfaceEquation. 
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
	 * @param scale_ico                Given for compatibility reasons with the CellEquation 
	 *                                 in PDEInterface. Should not be used here!
	 */
         template<typename FDC>
          void
          InterfaceEquation(const FDC& fdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double /*scale_ico*/)
          {
            if (this->GetPart() == "New")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
            }
            else
            {
              abort();
            }
          }

        /******************************************************/

         /**
         * Same functionality as for the CellRhs, but on Faces.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_cell_vector        This vector contains the locally computed values 
         *                                 of the FaceRhs. 
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
	 */
 
        template<typename FDC>
          void
          FaceRhs(const FDC & fdc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.)
          {
            this->GetProblem().FaceRhs(fdc, local_cell_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

         /**
         * Same functionality as for the CellMatrix, but on Faces.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_entry_matrix       This matrix contains the locally computed values 
         *                                 of the FaceMatrix. 
 	 */
        template<typename FDC>
          void
          FaceMatrix(const FDC & fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            this->GetProblem().FaceMatrix(fdc, local_entry_matrix,
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

          }

        /******************************************************/

      /**
         * Same functionality as for the CellMatrix, but on Interfaces.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_entry_matrix       This matrix contains the locally computed values 
         *                                 of the InterfaceMatrix. 
 	 */
         template<typename FDC>
          void
          InterfaceMatrix(const FDC& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            this->GetProblem().InterfaceMatrix(fdc, local_entry_matrix,
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

          }

        /******************************************************/

        /**
         * Same functionality as for the CellEquation, but on Boundaries.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_cell_vector        This vector contains the locally computed values 
         *                                 of the Facequation. 
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
	 *                                 to compute.
	 * @param scale_ico                Given for compatibility reasons with the CellEquation 
	 *                                 in PDEInterface. Should not be used here!
          */
        template<typename FDC>
          void
          BoundaryEquation(const FDC & fdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double /*scale_ico*/)
          {
            if (this->GetPart() == "New")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                  scale
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
            }
            else
            {
              abort();
            }

          }

        /******************************************************/

      /**
       * Same functionality as for the CellRhs, but on Boundaries.
       * Note that no time derivatives may occure on faces of the domain at present!
       * @tparam <FDC>                   A container that contains all relevant data
       *                                 needed on the element, e.g., element size, finite element values;
       *                                 see, e.g., FaceDataContainer
       *
       * @param fdc                      The FDC object.
       * @param local_cell_vector        This vector contains the locally computed values 
       *                                 of the FaceRhs. 
       * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine 
       *                                 to compute.
       */
        template<typename FDC>
          void
          BoundaryRhs(const FDC & fdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            this->GetProblem().BoundaryRhs(fdc, local_cell_vector,
                scale
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

        /**
         * Same functionality as for the CellMatrix, but on Boundaries.
	 * Note that no time derivatives may occure on faces of the domain at present!
	 * @tparam <FDC>                   A container that contains all relevant data
	 *                                 needed on the element, e.g., element size, finite element values;
	 *                                 see, e.g., FaceDataContainer
         *
	 * @param fdc                      The FDC object.
         * @param local_entry_matrix       This matrix contains the locally computed values 
         *                                 of the FaceMatrix. 
 	 */
 
        template<typename FDC>
          void
          BoundaryMatrix(const FDC & fdc,
              dealii::FullMatrix<double> &local_cell_matrix)
          {
            assert(this->GetPart() == "New");
            this->GetProblem().BoundaryMatrix(fdc, local_cell_matrix,
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),
                1.
                    * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

          }
      private:
        InitialProblem<
            BackwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
                dealdim, FE, DH>, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
