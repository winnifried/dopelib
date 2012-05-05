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
      int dopedim, int dealdim, typename FE = dealii::FESystem<
          dealdim>, typename DOFHANDLER = dealii::DoFHandler<dealdim>>
    class FractionalStepThetaProblem : public PrimalTSBase<OPTPROBLEM,
    SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>
    {
      public:
        /**
         * Constructor which gets the Problem `OP' to compute and
         * sets the parameters theta, theta prime, alpha, and beta
       * for the Fractional-Step-Theta scheme.
       *
       * @param OP     Problem is given to the time stepping scheme.
       */
      FractionalStepThetaProblem (OPTPROBLEM& OP)  :
        PrimalTSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim,
            FE, DOFHANDLER>(OP)
      {
        _fs_theta = 1.0 - std::sqrt(2.0) / 2.0;
        _fs_theta_prime = 1.0 - 2.0 * _fs_theta;
        _fs_alpha = (1.0 - 2.0 * _fs_theta) / (1.0 - _fs_theta);
        _fs_beta = 1.0 - _fs_alpha;
        _initial_problem = NULL;
      }

      ~FractionalStepThetaProblem ()
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
        return "Fractional-Step-Theta";
      }
      /******************************************************/

      InitialProblem<FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
      dealdim, FE, DOFHANDLER>, VECTOR, dealdim>&
      GetInitialProblem()
      {
        if (_initial_problem == NULL)
        {
          _initial_problem = new InitialProblem<FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
              dealdim, FE, DOFHANDLER>, VECTOR, dealdim>
          (*this);
        }
        return *_initial_problem;
      }

       /******************************************************/
      FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim, FE, DOFHANDLER>&
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
       * This time stepping scheme is diveded into three subroutines, i.e.,
       * normally six parts have to be computed. Since two parts coincide
       * the function is split into five parts.
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
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);

              tmp = 0.0;
              this->GetProblem().CellEquation(dc, tmp, scale * _fs_alpha, scale);
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(
                  dc,
                  tmp,
                  scale / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(
                  dc,
                  local_cell_vector,
                  scale / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
            }
          else if (this->GetPart() == "Old_for_1st_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(dc, tmp, scale * _fs_beta, 0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(
                  dc,
                  local_cell_vector,
                  (-1) * scale / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
            }
          else if (this->GetPart() == "Old_for_3rd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(dc, tmp, scale * _fs_beta, 0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(
                  dc,
                  local_cell_vector,
                  (-1) * scale / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);

              tmp = 0.0;
              this->GetProblem().CellEquation(dc, tmp, scale * _fs_beta, scale);
              local_cell_vector += tmp;

              tmp = 0.0;
              this->GetProblem().CellTimeEquation(
                  dc,
                  tmp,
                  scale / (_fs_theta_prime
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquationExplicit(
                  dc,
                  local_cell_vector,
                  scale / (_fs_theta_prime
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
            }
          else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;

              // The explicit parts with old_time_values; e.g. for fluid problems: laplace, convection, etc.
              this->GetProblem().CellEquation(dc, tmp, scale * _fs_alpha, 0.);
              local_cell_vector += tmp;

              this->GetProblem().CellTimeEquation(
                  dc,
                  local_cell_vector,
                  (-1) * scale / (_fs_theta_prime
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()));
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

      /**
       * Computes the value of the right-hand side.
       * The function is divided into five parts which  are given
       * the Newton solver. For detailed discussion, please visit
       * the documentation of the CellEquation.
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
        CellRhs(const DATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {

            }
          else if (this->GetPart() == "Old_for_1st_cycle" || this->GetPart() == "Old_for_3rd_cycle")
            {
              this->GetProblem().CellRhs(dc, local_cell_vector, scale);
            }
          //      else if ()
          //	{
          //      	this->GetProblem().CellRhs(param_values, domain_values, n_dofs_per_cell, n_q_points, material_id, cell_diameter, local_cell_vector,
          //      			      scale);
          //	}
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().CellRhs(dc, local_cell_vector, scale);
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

        /**
         * Computes the value of the right-hand side which requires
         * pointevaluations.
         * The function is divided into five parts which  are given
         * the Newton solver. For detailed discussion, please visit
         * the documentation of the CellEquation.
         */
        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1.)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
          {

          }
          else if (this->GetPart() == "Old_for_1st_cycle" || this->GetPart() == "Old_for_3rd_cycle")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector, scale);
          }
          else if (this->GetPart() == "New_for_2nd_cycle")
          {
            this->GetProblem().PointRhs(param_values, domain_values, rhs_vector, scale);
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
       * This function just contains two `new' parts since the first and third
       * cycle coincide in the Fractional-Step-Theta scheme.
       * Older parts must not be computed due to, that directional
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
      template<typename DATACONTAINER>
        void
        CellMatrix(const DATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().CellMatrix(dc, m, _fs_alpha, 1.);
              local_entry_matrix.add(1.0, m);


              m = 0.;
              this->GetProblem().CellTimeMatrix(dc, m);
              local_entry_matrix.add(
                  1.0 / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()), m);

              m = 0.;
              this->GetProblem().CellTimeMatrixExplicit(dc, m);
              local_entry_matrix.add(
                  1.0 / (_fs_theta
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()), m);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().CellMatrix(dc, local_entry_matrix, _fs_beta, 1.);
              local_entry_matrix.add(1.0, m);

              m = 0.;
              this->GetProblem().CellTimeMatrix(dc, m);
              local_entry_matrix.add(
                  1.0 / (_fs_theta_prime
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()), m);

              m = 0.;
              this->GetProblem().CellTimeMatrixExplicit(dc, m);
              local_entry_matrix.add(
                  1.0 / (_fs_theta_prime
                      * this->GetProblem().GetBaseProblem().GetSpaceTimeHandler()->GetStepSize()), m);
            }
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just this->GetProblem().FaceEquation(...). For more information we refer to
       * the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceEquation(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector, scale * _fs_alpha);

            }
          else if ((this->GetPart() == "Old_for_1st_cycle") || (this->GetPart()
              == "Old_for_3rd_cycle"))
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector, scale * _fs_beta);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha * _fs_theta / _fs_theta_prime);
            }
          else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().FaceEquation(fdc, local_cell_vector, scale * _fs_alpha);
            }
          else
            {
              abort();
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just this->GetProblem().InterfaceEquation(...).
       * For more information we refer to the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceEquation(const FACEDATACONTAINER& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha);

            }
          else if ((this->GetPart() == "Old_for_1st_cycle") || (this->GetPart()
              == "Old_for_3rd_cycle"))
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_beta);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha * _fs_theta / _fs_theta_prime);
            }
          else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().InterfaceEquation(fdc, local_cell_vector,
                  scale * _fs_alpha);
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
          this->GetProblem().FaceRhs(fdc, local_cell_vector, scale);
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
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().FaceMatrix(fdc, m);
              local_entry_matrix.add(_fs_alpha, m);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().FaceMatrix(fdc, m);
              local_entry_matrix.add(_fs_alpha * _fs_theta / _fs_theta_prime, m);
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just this->GetProblem().InterfaceMatrix(...).
       *  For more information we refer to the file optproblemcontainer.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceMatrix(const FACEDATACONTAINER& fdc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().InterfaceMatrix(fdc,  m);
              local_entry_matrix.add(_fs_alpha, m);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_entry_matrix);
              m = 0.;
              this->GetProblem().InterfaceMatrix(fdc,  m);
              local_entry_matrix.add(_fs_alpha * _fs_theta / _fs_theta_prime, m);
            }

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
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector, scale * _fs_alpha);
            }
          else if ((this->GetPart() == "Old_for_1st_cycle") || (this->GetPart()
              == "Old_for_3rd_cycle"))
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector, scale * _fs_beta);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector, scale * _fs_beta);
            }
          else if (this->GetPart() == "Old_for_2nd_cycle")
            {
              this->GetProblem().BoundaryEquation(fdc, local_cell_vector, scale * _fs_alpha);
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
          this->GetProblem().BoundaryRhs(fdc, local_cell_vector, scale);
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
          if (this->GetPart() == "New_for_1st_and_3rd_cycle")
            {
              dealii::FullMatrix<double> m(local_cell_matrix);
              m = 0.;
              this->GetProblem().BoundaryMatrix(fdc, m);
              local_cell_matrix.add(_fs_alpha, m);
            }
          else if (this->GetPart() == "New_for_2nd_cycle")
            {
              dealii::FullMatrix<double> m(local_cell_matrix);
              m = 0.;
              this->GetProblem().BoundaryMatrix(fdc, m);
              local_cell_matrix.add(_fs_beta, m);
            }
        }
    private:
      // parameters for FS scheme
      double _fs_theta;
      double _fs_theta_prime;
      double _fs_alpha;
      double _fs_beta;

      InitialProblem<FractionalStepThetaProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
      dealdim, FE, DOFHANDLER>, VECTOR, dealdim> * _initial_problem;
    };
}

#endif
