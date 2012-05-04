#ifndef _SHIFTEDCRANKNICOLSONProblem_H_
#define _SHIFTEDCRANKNICOLSONProblem_H_

#include "initialproblem.h" 

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
      int dopedim, int dealdim, typename FE = DOpEWrapper::FiniteElement<
          dealdim>, typename DOFHANDLER = dealii::DoFHandler<dealdim>>
    class ShiftedCrankNicolsonProblem
    {
    public:
      ShiftedCrankNicolsonProblem(OPTPROBLEM& OP) :
        _OP(OP)
      {
	_initial_problem = NULL;
      }
      ~ShiftedCrankNicolsonProblem()
      {
	if(_initial_problem != NULL)
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
	  _initial_problem = new InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim>
	  (*this);
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
       * Sets the step part which should actually computed, e.g.,
       * previous solution within the NewtonStepSolver or
       * last time step solutions.
       * @param s    Name of the step part
       */
      void
      SetStepPart(std::string s)
      {
        _part = s;
      }
      /******************************************************/

      /**
       * Sets the actual time.
       *
       * @param time      The actual time.
       * @param interval  The actual interval. Make sure that time
       *                  lies in interval!
       */

      void SetTime(double time, const TimeIterator& interval)
      {
        _OP.SetTime(time, interval);
      }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.CellFunctional(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename DATACONTAINER>
        double
        CellFunctional(const DATACONTAINER& dc)
        {
          return _OP.CellFunctional(dc);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.PointFunctional(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      double
      PointFunctional(
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values)
      {
        return _OP.PointFunctional(param_values, domain_values);
      }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.BoundaryFunctional(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        double
        BoundaryFunctional(const FACEDATACONTAINER& fdc)
        {
          return _OP.BoundaryFunctional(fdc);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceFunctional(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        double
        FaceFunctional(const FACEDATACONTAINER& fdc)
        {
          return _OP.FaceFunctional(fdc);
        }
     /******************************************************/
      /****For the initial values ***************/
      template<typename DATACONTAINER>
      void Init_CellEquation(const DATACONTAINER& cdc,
			     dealii::Vector<double> &local_cell_vector, double scale,
			     double scale_ico)
      {
        _OP.Init_CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

      template<typename DATACONTAINER>
      void
      Init_CellRhs(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _OP.Init_CellRhs(cdc, local_cell_vector, scale);
      }

      template<typename DATACONTAINER>
      void Init_CellMatrix(const DATACONTAINER& cdc,
			   dealii::FullMatrix<double> &local_entry_matrix, double scale,
			   double scale_ico)
      {
        _OP.Init_CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

      void
      Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
      const std::map<std::string, const VECTOR*> &/*domain_values*/,
      VECTOR& /*rhs_vector*/, double /*scale=1.*/)
      {
      }
 
      template<typename FACEDATACONTAINER>
      void Init_FaceEquation(const FACEDATACONTAINER& /*fdc*/,
			     dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      template<typename FACEDATACONTAINER>
      void Init_InterfaceEquation(const FACEDATACONTAINER& /*fdc*/,
				  dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      template<typename FACEDATACONTAINER>
      void Init_BoundaryEquation(const FACEDATACONTAINER& /*fdc*/,
				 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }
     
      template<typename FACEDATACONTAINER>
      void Init_FaceMatrix(const FACEDATACONTAINER& /*fdc*/,
			   FullMatrix<double> &/*local_entry_matrix*/)
      {
      }

      template<typename FACEDATACONTAINER>
      void Init_InterfaceMatrix(const FACEDATACONTAINER& /*fdc*/,
				FullMatrix<double> &/*local_entry_matrix*/)
      {
      }
      
      template<typename FACEDATACONTAINER>
      void Init_BoundaryMatrix(const FACEDATACONTAINER& /*fdc*/,
			       FullMatrix<double> &/*local_cell_matrix*/)
      {
      }

      /****End the initial values ***************/
      /******************************************************/

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
          if (_part == "New")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;
              // implicit parts; e.g. for fluid problems: pressure and incompressibilty of v, get scaled with scale
              // The remaining parts; e.g. for fluid problems: laplace, convection, etc.:
              // Multiplication by 1/2 + k due to CN discretization

              _OP.CellEquation(dc, tmp, damped_cn_theta * scale, scale);
              local_cell_vector += tmp;

              tmp = 0.0;
              _OP.CellTimeEquation(dc, tmp,
                  scale / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              _OP.CellTimeEquationExplicit(dc, local_cell_vector,
                  scale / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else if (_part == "Old")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();

              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              // Multiplication by 1/2 + k due to CN discretization
              _OP.CellEquation(dc, tmp, (1.0 - damped_cn_theta) * scale, 0.);
              local_cell_vector += tmp;

              _OP.CellTimeEquation(
                  dc,
                  local_cell_vector,
                  (-1) * scale
                      / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
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
          if (_part == "New")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.CellRhs(fdc, local_cell_vector, damped_cn_theta * scale);
            }
          else if (_part == "Old")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.CellRhs(fdc, local_cell_vector, (1 - damped_cn_theta) * scale);
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
          if (_part == "New")
          {
            damped_cn_theta = 0.5
                + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            _OP.PointRhs(param_values, domain_values, rhs_vector,
                damped_cn_theta * scale);
          }
          else if (_part == "Old")
          {
            damped_cn_theta = 0.5
                + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
            _OP.PointRhs(param_values, domain_values, rhs_vector,
                (1 - damped_cn_theta) * scale);
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
          assert(_part == "New");
          damped_cn_theta = 0.5
              + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          dealii::FullMatrix<double> m(local_entry_matrix);

          // multiplication with 1/2 + k due to CN discretization for the 'normal' parts
          // no multiplication with 1/2 + k for the implicit parts
          //due to implicit treatment of pressure, etc. (in the case of fluid problems)
          _OP.CellMatrix(fdc, local_entry_matrix, damped_cn_theta, 1.);

          m = 0.;
          _OP.CellTimeMatrix(fdc, m);
          local_entry_matrix.add(
              1.0 / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

          m = 0.;
          _OP.CellTimeMatrixExplicit(fdc, m);
          local_entry_matrix.add(
              1.0 / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceEquation(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceEquation(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.FaceEquation(fdc, local_cell_vector, damped_cn_theta * scale);
            }
          else if (_part == "Old")
            {
              _OP.FaceEquation(fdc, local_cell_vector, damped_cn_theta * scale);
            }
          else
            {
              abort();
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.InterfaceEquation(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceEquation(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.InterfaceEquation(fdc,  local_cell_vector,
                  damped_cn_theta * scale);
            }
          else if (_part == "Old")
            {
              _OP.InterfaceEquation(fdc,  local_cell_vector,
                  damped_cn_theta * scale);
            }
          else
            {
              abort();
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceRhs(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceRhs(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          _OP.FaceRhs(fdc, local_cell_vector, scale);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceMatrix(const FACEDATACONTAINER& fdc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          assert(_part == "New");
          damped_cn_theta = 0.5
              + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren,
          // da local_cell_matrix schon skaliert ist
          dealii::FullMatrix<double> m(local_entry_matrix);

          m = 0.;
          // Multiplication with 1/2 + k due to CN time discretization
          _OP.FaceMatrix(fdc, m);

          local_entry_matrix.add(damped_cn_theta, m);

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceMatrix(const FACEDATACONTAINER& fdc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          assert(_part == "New");
          damped_cn_theta = 0.5
              + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren,
          // da local_cell_matrix schon skaliert ist
          dealii::FullMatrix<double> m(local_entry_matrix);

          m = 0.;
          // Multiplication with 1/2 + k due to CN time discretization
          _OP.InterfaceMatrix(fdc,  m);

          local_entry_matrix.add(damped_cn_theta, m);

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
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.BoundaryEquation(fdc, local_cell_vector,
                  damped_cn_theta * scale);
            }
          else if (_part == "Old")
            {
              damped_cn_theta = 0.5
                  + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
              _OP.BoundaryEquation(fdc, local_cell_vector,
                  (1.0 - damped_cn_theta) * scale);
            }
          else
            {
              abort();
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        BoundaryRhs(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          _OP.BoundaryRhs(fdc, local_cell_vector, scale);
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
          assert(_part == "New");
          damped_cn_theta = 0.5
              + _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize();
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
          dealii::FullMatrix<double> m(local_cell_matrix);

          m = 0.;
          // Multiplication with 1/2 + k due to CN time discretization
          _OP.BoundaryMatrix(fdc, m);
          local_cell_matrix.add(damped_cn_theta, m);

        }

      /******************************************************/

      /**
       * A pointer to the whole FESystem
       *
       * @return A const pointer to the FESystem()
       */
      const dealii::SmartPointer<const DOpEWrapper::FiniteElement<dealdim> >
      GetFESystem() const
      {
        return _OP.GetFESystem();
      }

      /******************************************************/

      /**
       * This function determines whether a loop over all faces is required or not.
       *
       * @return Returns whether or not this functional has components on faces between elements.
       *         The default value is false.
       */
      bool
      HasFaces() const
      {
        return _OP.HasFaces();
      }

      /******************************************************/
      /**
       * This function determines whether point evaluations are required or not.
       *
       * @return Returns whether or not this functional needs evaluations of
       *         point values.
       */
      bool
      HasPoints() const
      {
        return _OP.HasPoints();
      }

      /******************************************************/
      /**
       * This function determines whether a loop over all faces is required or not.
       *
       * @return Returns whether or not this functional has components on faces between elements.
       *         The default value is false.
       */
      bool
      HasInterfaces() const
      {
        return _OP.HasInterfaces();
      }

      /******************************************************/

      /**
       * This function returns the update flags for domain values
       * for the computation of shape values, gradients, etc.
       * For detailed explication, please visit `Finite element access/FEValues classes' in
       * the deal.ii manual.
       *
       * @return Returns the update flags to use in a computation.
       */
      dealii::UpdateFlags
      GetUpdateFlags() const
      {
        return _OP.GetUpdateFlags();
      }

      /******************************************************/

      /**
       * This function returns the update flags for face values
       * for the computation of shape values, gradients, etc.
       * For detailed explication, please visit
       * `FEFaceValues< dim, spacedim > Class Template Reference' in
       * the deal.ii manual.
       *
       * @return Returns the update flags for faces to use in a computation.
       */
      dealii::UpdateFlags
      GetFaceUpdateFlags() const
      {
        return _OP.GetFaceUpdateFlags();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of Dirichlet boundaries.
       *
       * @return Returns the Dirichlet Colors.
       */
      const std::vector<unsigned int>&
      GetDirichletColors() const
      {
        return _OP.GetDirichletColors();
      }

      /******************************************************/

      /**
       * A std::vector of boolean values to decide at which parts of the boundary and solutions variables
       * Dirichlet values should be applied.
       *
       * @return Returns a component mask for each boundary color.
       */
      const std::vector<bool>&
      GetDirichletCompMask(unsigned int color) const
      {
        return _OP.GetDirichletCompMask(color);
      }

      /******************************************************/

      /**
       * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply
       * on each boundary part with color 'color'.
       *
       * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
       */
      const dealii::Function<dealdim>&
      GetDirichletValues(
          unsigned int color,
          //							const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler,
          //							const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values) const
      {
        return _OP.GetDirichletValues(color,/* control_dof_handler,state_dof_handler,*/
        param_values, domain_values);
      }

      /******************************************************/

      /**
       * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
       * problem, respectively.
       *
       * @return Returns a dealii::Function of initial values.
       */
      const dealii::Function<dealdim>&
      GetInitialValues() const
      {
        return _OP.GetInitialValues();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of the boundary equation.
       *
       * @return Returns colors for the boundary equation.
       */
      const std::vector<unsigned int>&
      GetBoundaryEquationColors() const
      {
        return _OP.GetBoundaryEquationColors();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of the boundary functionals.
       *
       * @return Returns colors for the boundary functionals.
       */
      const std::vector<unsigned int>&
      GetBoundaryFunctionalColors() const
      {
        return _OP.GetBoundaryFunctionalColors();
      }

      /******************************************************/

      /**
       * This function returns the number of functionals to be considered in the problem.
       *
       * @return Returns the number of functionals.
       */
      unsigned int
      GetNFunctionals() const
      {
        return _OP.GetNFunctionals();
      }

      /******************************************************/

      /**
       * This function gets the number of blocks considered in the PDE problem.
       * Example 1: in fluid problems we have to find velocities and pressure
       * --> number of blocks is 2.
       * Example 2: in FSI problems we have to find velocities, displacements, and pressure.
       *  --> number of blocks is 3.
       *
       * @return Returns the number of blocks.
       */
      unsigned int
      GetNBlocks() const
      {
        return _OP.GetNBlocks();
      }

      /******************************************************/

      /**
       * A function which has the number of degrees of freedom for the block `b'.
       *
       * @return Returns the number of DoFs for block `b'.
       */
      unsigned int
      GetDoFsPerBlock(unsigned int b) const
      {
        return _OP.GetDoFsPerBlock(b);
      }

      /******************************************************/

      /**
       * A std::vector which contains the number of degrees of freedom per block.
       *
       * @return Returns a vector with DoFs.
       */
      const std::vector<unsigned int>&
      GetDoFsPerBlock() const
      {
        return _OP.GetDoFsPerBlock();
      }

      /******************************************************/

      /**
       * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
       *
       * @return Returns a matrix with hanging node constraints.
       */
      const dealii::ConstraintMatrix&
      GetDoFConstraints() const
      {
        return _OP.GetDoFConstraints();
      }

      std::string
      GetType() const
      {
        return _OP.GetType();
      }
      std::string
      GetDoFType() const
      {
        return _OP.GetDoFType();
      }

      /******************************************************/

      /**
       * This function describes what type of Functional is considered
       *
       * @return A string describing the functional, feasible values are "domain", "boundary", "point" or "face"
       *         if it contains domain, or boundary ... parts all combinations of these keywords are feasible.
       *         In time dependent problems use "timelocal" to indicate that
       *         it should only be evaluated at a certain time_point, or "timedistributed" to consider \int_0^T J(t,q(t),u(t))  \dt
       *         only one of the words "timelocal" and "timedistributed" should be considered if not it will be considered to be
       *         "timelocal"
       *
       */
      std::string
      GetFunctionalType() const
      {
        return _OP.GetFunctionalType();
      }

      /******************************************************/

      /**
       * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
       *
       * @return A string. This is the name beeing displayed next to the computed values.
       */
      std::string
      GetFunctionalName() const
      {
        return _OP.GetFunctionalName();
      }

      /******************************************************/

      /**
       * A pointer to the OutputHandler() object.
       *
       * @return The OutputHandler() object.
       */
      DOpEOutputHandler<VECTOR>*
      GetOutputHandler()
      {
        return _OP.GetOutputHandler();
      }

      /******************************************************/

      /**
       * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
       *
       * @return The SpaceTimeHandler() object.
       */
      const SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
          dopedim, dealdim>*
      GetSpaceTimeHandler() const
      {
        return _OP.GetBaseProblem().GetSpaceTimeHandler();
      }
      SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>*
      GetSpaceTimeHandler()
      {
        return _OP.GetBaseProblem().GetSpaceTimeHandler();
      }

      /******************************************************/

      void
      ComputeSparsityPattern(SPARSITYPATTERN& sparsity) const
      {
        _OP.ComputeSparsityPattern(sparsity);
      }

      bool
      L2ProjectionInitialDataWithDeal() const
      {
        return true;
      }

      /******************************************************/
      /**
       * Access to the private membervariable _fe_values_needed_to_be_initialized which
       * points out the necessity to initialize the fevalues. This is needed in the
       * Solverclass (i.e. statsolver or instatsolver), which holds coordinates the
       * initialization of the fevalues, which are stored in the SpaceTimeHandler.
       *
       */
      bool
      GetFEValuesNeededToBeInitialized() const
      {
        return _OP.GetFEValuesNeededToBeInitialized();
      }

      void
      SetFEValuesAreInitialized()
      {
        _OP.SetFEValuesAreInitialized();
      }

      /******************************************************/
    private:
      OPTPROBLEM& _OP;
      std::string _part;

      double damped_cn_theta;
    InitialProblem<ShiftedCrankNicolsonProblem, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
