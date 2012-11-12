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

#ifndef _SHIFTEDCRANKNICOLSONProblem_H_
#define _SHIFTEDCRANKNICOLSONProblem_H_

#include "initialproblem.h" 
#include "primal_ts_base.h"

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the shifted
   * Crank-Nicolson time-stepping scheme. The parameter \theta
   * is given by 1/2 + k, where k describes the time step size.
   *
   * @tparam <OPTPROBLEM>       The problem to deal with.
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <FE>               The type of finite elements in use, must be compatible with the DOFHANDLER.
   * @tparam <DOFHANDLER>       The type of the DoFHandler in use (to be more precise: The type of the dealii-DoFhandler which forms
   *                            the base class of the DOpEWrapper::DoFHandler in use.)
   * @tparam <dopedim>          The dimension for the control variable.
   * @tparam <dealdim>          The dimension of the state variable.
   *
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim,
      typename FE = dealii::FESystem<dealdim>,
      typename DOFHANDLER = dealii::DoFHandler<dealdim>>
    class ShiftedCrankNicolsonProblem : public PrimalTSBase<OPTPROBLEM,
        SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>
    {
      public:
        ShiftedCrankNicolsonProblem(OPTPROBLEM& OP) :
            PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
                FE, DOFHANDLER>(OP)
        {
          _initial_problem = NULL;
        }
        ~ShiftedCrankNicolsonProblem()
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
          return "shifted Crank-Nicolson";
        }
        /******************************************************/

        InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim>&
        GetInitialProblem()
        {
          if (_initial_problem == NULL)
          {
            _initial_problem = new InitialProblem<ShiftedCrankNicolsonProblem,
                VECTOR, dealdim>(*this);
          }
          return *_initial_problem;
        }

        /******************************************************/
        ShiftedCrankNicolsonProblem&
        GetBaseProblem()
        {
          return *this;
        }

        /******************************************************/

        /**
         * Computes the value of the cell equation which corresponds
         * to the residuum in nonlinear cases. This function
         * itself contains a maximum of four subroutines of cell equations:
         * CellEquation, CellEquationExplicit, TimeEquation, TimeEquationExplicit.
         * So far, three types are needed for fluid-structure interaction problems:
         * CellEquation: implicit terms, like pressure.
         * CellEquationExplicit: stress tensors, fluid convection, etc.
         * TimeEquationExplicit: time derivatives of certain variables which are
         *                       combined with transformations, etc.
         *
         * In fluid problems, the CellEquations terms coincide. However the
         * TimeEquations terms differ:
         * CellTimeEquation: time derivatives, e.g., dt v
         *
         * The function is divided into two parts `old' and `new' which  are given
         * the Newton solver. Then, the computation is done in two steps: first
         * computation of the old Newton- or time step equation parts. After,
         * computation of the actual parts.
         *
         *
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
         * @param n_q_points               Number of quadrature points on a cell.
         * @param material_id              Material Id of the cell.
         * @param cell_diameter            Diameter of the cell.
         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        template<typename DATACONTAINER>
          void
          CellEquation(const DATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale, double)
          {
            if (this->GetPart() == "New")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;
              // implicit parts; e.g. for fluid problems: pressure and incompressibilty of v, get scaled with scale
              // The remaining parts; e.g. for fluid problems: laplace, convection, etc.:
              // Multiplication by 1/2 + k due to CN discretization

              this->GetProblem().CellEquation(dc, tmp, damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(dc, tmp,
                  scale);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(dc, local_cell_vector,
                  scale);
            }
            else if (this->GetPart() == "Old")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              // Multiplication by 1/2 + k due to CN discretization
              this->GetProblem().CellEquation(dc, tmp, (1.0 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(dc, local_cell_vector,
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
         *
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
         * @param n_q_points               Number of quadrature points on a cell.
         * @param material_id              Material Id of the cell.
         * @param cell_diameter            Diameter of the cell.
         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        template<typename FACEDATACONTAINER>
          void
          CellRhs(const FACEDATACONTAINER& fdc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.)
          {
            if (this->GetPart() == "New")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().CellRhs(fdc, local_cell_vector, damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().CellRhs(fdc, local_cell_vector,
                  (1 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
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
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"

         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1.)
        {
          if (this->GetPart() == "New")
          {
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }
          else if (this->GetPart() == "Old")
          {
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector,
                (1 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
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
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
         * @param n_q_points               Number of quadrature points on a cell.
         * @param material_id              Material Id of the cell.
         * @param cell_diameter            Diameter of the cell.
         * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
         *                                 filled by the locally computed values. For more information of its functionality, please
         *                                 search for the keyword `FullMatrix' in the deal.ii manual.
         */
        template<typename FACEDATACONTAINER>
          void
          CellMatrix(const FACEDATACONTAINER& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            dealii::FullMatrix<double> m(local_entry_matrix);

            // multiplication with 1/2 + k due to CN discretization for the 'normal' parts
            // no multiplication with 1/2 + k for the implicit parts
            //due to implicit treatment of pressure, etc. (in the case of fluid problems)
            this->GetProblem().CellMatrix(fdc, local_entry_matrix, damped_cn_theta* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());

            m = 0.;
            this->GetProblem().CellTimeMatrix(fdc, m);
            local_entry_matrix.add(
                1.0,
                m);

            m = 0.;
            this->GetProblem().CellTimeMatrixExplicit(fdc, m);
            local_entry_matrix.add(
                1.0,
                m);
          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().FaceEquation(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          FaceEquation(const FACEDATACONTAINER& fdc,
		       dealii::Vector<double> &local_cell_vector, double scale = 1., double /*scale_ico*/ = 1.)
          {
            if (this->GetPart() == "New")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().FaceEquation(fdc, local_cell_vector, damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
	      damped_cn_theta = 0.5
	              + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().FaceEquation(fdc, local_cell_vector, (1.0 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 0.);
            }
            else
            {
              abort();
            }

          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().InterfaceEquation(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          InterfaceEquation(const FACEDATACONTAINER& fdc,
			    dealii::Vector<double> &local_cell_vector, double scale = 1., double /*scale_ico*/ = 1.)
          {
            if (this->GetPart() == "New")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
						   damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() );
            }
            else if (this->GetPart() == "Old")
            {
              damped_cn_theta = 0.5
	              + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector, (1.0 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 0.);
            }
            else
            {
              abort();
            }

          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().FaceRhs(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          FaceRhs(const FACEDATACONTAINER& fdc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.)
          {
            this->GetProblem().FaceRhs(fdc, local_cell_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().FaceMatrix(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          FaceMatrix(const FACEDATACONTAINER& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
           
            dealii::FullMatrix<double> m(local_entry_matrix);

            m = 0.;
            // Multiplication with 1/2 + k due to CN time discretization
            this->GetProblem().FaceMatrix(fdc, m, damped_cn_theta, 1.);

            local_entry_matrix.add(1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().FaceMatrix(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          InterfaceMatrix(const FACEDATACONTAINER& fdc,
              dealii::FullMatrix<double> &local_entry_matrix)
          {
            assert(this->GetPart() == "New");
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            
	    dealii::FullMatrix<double> m(local_entry_matrix);

            m = 0.;
            // Multiplication with 1/2 + k due to CN time discretization
            this->GetProblem().InterfaceMatrix(fdc, m, damped_cn_theta, 1.);

            local_entry_matrix.add(1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

          }

        /******************************************************/

        /**
         * Computes the value of boundary equations. In the actual
         * implementation we just consider values for the actual time step.
         * Therefore, no implementation is necessary of older time steps, etc.
         * An example of a boundary equations is the `do-nothing' outflow
         * condition of fluid flows in a channel when using the symmetric stress tensor.
         * In order to have correct outflow conditions the transposed part of the
         * stress tensor should be subtracted at the outflow boundary.
         *
         *
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
         * @param n_q_points               Number of quadrature points on a cell.
         * @param material_id              Material Id of the cell.
         * @param cell_diameter            Diameter of the cell.
         * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
         *                                 on dealii::Vector, please visit, the deal.ii manual pages.
         * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryEquation(const FACEDATACONTAINER& fdc,
			   dealii::Vector<double> &local_cell_vector, double scale = 1., double /*scale_ico*/ = 1.)
          {
            if (this->GetPart() == "New")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
						  damped_cn_theta * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(),scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
            else if (this->GetPart() == "Old")
            {
              damped_cn_theta = 0.5
                  + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector,
						  (1.0 - damped_cn_theta) * scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), 0.);
            }
            else
            {
              abort();
            }

          }

        /******************************************************/

        /**
         * Not implemented so far. Returns just this->GetProblem().FaceMatrix(...). For more information we refer to
         * the file optproblemcontainer.h
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryRhs(const FACEDATACONTAINER& fdc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.)
          {
            this->GetProblem().BoundaryRhs(fdc, local_cell_vector, scale* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
          }

        /******************************************************/

        /**
         * Computes the matrix entries of boundary equations.
         * This function is just considered in the `new' part. This is due to that directional
         * derivatives vanish if they are applied to old values which are, of course,
         * already computed and therefore constant.
         * An example of a boundary equations is the `do-nothing' outflow
         * condition of fluid flows in a channel when using the symmetric stress tensor.
         * In order to have correct outflow conditions the transposed part of the
         * stress tensor should be subtracted at the outflow boundary.
         *
         *
         * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                 is done by parameters, it is contained in this map at the position "control".
         * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                 is distributed, it is contained in this map at the position "control". The state may always
         *                                 be found in this map at the position "state"
         * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
         * @param n_q_points               Number of quadrature points on a cell.
         * @param material_id              Material Id of the cell.
         * @param cell_diameter            Diameter of the cell.
         * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
         *                                 filled by the locally computed values. For more information of its functionality, please
         *                                 search for the keyword `FullMatrix' in the deal.ii manual.
         */
        template<typename FACEDATACONTAINER>
          void
          BoundaryMatrix(const FACEDATACONTAINER& fdc,
              dealii::FullMatrix<double> &local_cell_matrix)
          {
            assert(this->GetPart() == "New");
            damped_cn_theta = 0.5
                + this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            dealii::FullMatrix<double> m(local_cell_matrix);

            m = 0.;
            // Multiplication with 1/2 + k due to CN time discretization
            this->GetProblem().BoundaryMatrix(fdc, m, damped_cn_theta, 1.);
            local_cell_matrix.add(1.* this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

          }

      private:
        double damped_cn_theta;
        InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
