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

#ifndef _STATE_PROBLEM_H_
#define _STATE_PROBLEM_H_

#include "spacetimehandler.h"

using namespace dealii;

namespace DOpE
{
  /**
   * This is a problem used in the solution of the 
   * primal pde problem, i.e., the state-equation.
   * 
   * @tparam OPTPROBLEM     The container with the OPT-Problem description
   * @tparam PDE            The container with the PDE-description
   *                        note the PDE is the one we use for all
   *                        things related to the PDE. This is so to allow
   *                        switching between timesteps.
   * @tparam DD             Dirichlet datan
   * @tparam VECTOR         The vector class
   * @tparam dim            The dimension of the domain.
   */
  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    class StateProblem
    {
    public:
      StateProblem(OPTPROBLEM& OP, PDE& pde) :
        _pde(pde), _opt_problem(OP)
      {
        _dirichlet_colors = _opt_problem._dirichlet_colors;
        _dirichlet_comps = _opt_problem._dirichlet_comps;
        _primal_dirichlet_values = _opt_problem._primal_dirichlet_values;
        _state_boundary_equation_colors
            = _opt_problem._state_boundary_equation_colors;
      }

      std::string
      GetName() const
      {
        return "StateProblem";
      }
      std::string
      GetType() const
      {
        return "state";
      }

      /******************************************************/
      /****For the initial values ***************/
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
      void Init_CellEquation(const CDC& cdc,
			     dealii::Vector<double> &local_cell_vector, double scale,
			     double scale_ico)
      {
        _pde.Init_CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
      void
      Init_CellRhs(const CDC& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_CellRhs(& GetInitialValues(), cdc, local_cell_vector, scale);
      }

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      void
      Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
      const std::map<std::string, const VECTOR*> &/*domain_values*/,
      VECTOR& /*rhs_vector*/, double /*scale=1.*/)
      {
        //Note if this is implemented one needs to update Init_PointRhs in the
        // OptProblem container in the tangent case.
      }

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
      void Init_CellMatrix(const CDC& cdc,
			   dealii::FullMatrix<double> &local_entry_matrix, double scale,
			   double scale_ico)
      {
        _pde.Init_CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

      /******************************************************/
      /* Functions as in OptProblem */
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellEquation(const CDC& cdc,
            dealii::Vector<double> &local_cell_vector, double scale,
            double scale_ico);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellTimeEquation(const CDC& cdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellTimeEquationExplicit(const CDC& cdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellRhs(const CDC& cdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellMatrix(const CDC& cdc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
            double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellTimeMatrix(const CDC& cdc,
            dealii::FullMatrix<double> &local_entry_matrix);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename CDC>
        inline void
        CellTimeMatrixExplicit(const CDC& cdc,
            dealii::FullMatrix<double> &local_entry_matrix);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        FaceEquation(const FDC& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
     template<typename FDC>
        inline void
        InterfaceEquation(const FDC& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        FaceRhs(const FDC& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        FaceMatrix(const FDC& fdc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

      template<typename FDC>
        inline void
        InterfaceMatrix(const FDC& fdc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        BoundaryEquation(const FDC& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        BoundaryRhs(const FDC& fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        BoundaryMatrix(const FDC& fdc,
            dealii::FullMatrix<double> &local_cell_matrix, double scale = 1., double scale_ico = 1.);

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
      inline const dealii::SmartPointer<
    const dealii::hp::FECollection<dim> >
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
      SetTime(double time, const TimeIterator& interval);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline void
      ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline const std::vector<unsigned int>&
      GetDirichletColors() const;
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline const std::vector<bool>&
      GetDirichletCompMask(unsigned int color) const;
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline const Function<dim>
          &
          GetDirichletValues(
              unsigned int color,
              const std::map<std::string, const dealii::Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values) const;
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline const std::vector<unsigned int>&
      GetBoundaryEquationColors() const;
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline const dealii::ConstraintMatrix&
      GetDoFConstraints() const;
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
    const dealii::Function<dim>&
    GetInitialValues() const;
      /******************************************************/
      DOpEOutputHandler<VECTOR>*
      GetOutputHandler()
      {
        return _opt_problem.GetOutputHandler();
      } 
      OPTPROBLEM&
      GetBaseProblem()
      {
        return _opt_problem;
      }
    protected:

    private:
      PDE& _pde;
      OPTPROBLEM& _opt_problem;

      std::vector<unsigned int> _dirichlet_colors;
      std::vector<std::vector<bool> > _dirichlet_comps;
      std::vector<PrimalDirichletData<DD, VECTOR, dim>*>
          _primal_dirichlet_values;
      std::vector<unsigned int> _state_boundary_equation_colors;

    };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, 
          dim>::CellEquation(const CDC& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        _pde.CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellTimeEquation(const CDC& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellTimeEquation(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellTimeEquationExplicit(const CDC& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellTimeEquationExplicit(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::FaceEquation(const FDC& fdc,
				 dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.FaceEquation(fdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::InterfaceEquation(const FDC& fdc,
          dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.InterfaceEquation(fdc,  local_cell_vector, scale, scale_ico);
      }
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryEquation(const FDC& fdc,
          dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.BoundaryEquation(fdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellRhs(const CDC& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellRightHandSide(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::PointRhs(
        const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
        const std::map<std::string, const VECTOR*> &/*domain_values*/,
        VECTOR& /*rhs_vector*/, double /*scale*/)
    {

    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::FaceRhs(const FDC& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.FaceRightHandSide(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryRhs(const FDC& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.BoundaryRightHandSide(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellMatrix(const CDC& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        _pde.CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellTimeMatrix(const CDC& cdc,
          FullMatrix<double> &local_entry_matrix)
      {
        _pde.CellTimeMatrix(cdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename CDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::CellTimeMatrixExplicit(const CDC& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {
        _pde.CellTimeMatrixExplicit(cdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::FaceMatrix(const FDC& fdc,
			       FullMatrix<double> &local_entry_matrix, double scale,
			       double scale_ico)
      {
        _pde.FaceMatrix(fdc, local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::InterfaceMatrix(const FDC& fdc,
				    FullMatrix<double> &local_entry_matrix, double scale,
				    double scale_ico)
      {
        _pde.InterfaceMatrix(fdc,  local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryMatrix(const FDC& fdc,
				   FullMatrix<double> &local_cell_matrix, double scale,
				   double scale_ico)
      {
        _pde.BoundaryMatrix(fdc, local_cell_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    std::string
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFType() const
    {
      return "state";
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const SmartPointer<const dealii::FESystem<dim> >
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFESystem() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetFESystem("state");
    }

  /******************************************************/
  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const SmartPointer<const dealii::hp::FECollection<dim> >
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFECollection() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetFECollection("state");
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetFaceUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::SetTime(
        double time, const TimeIterator& interval)
    {
      _opt_problem.SetTime(time, interval);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      _opt_problem.GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasFaces() const
    {
      return _pde.HasFaces();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasPoints() const
    {
      return false;//We have no PointRhs in normal stateproblems at the moment.
    }


  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasInterfaces() const
    {
      return _pde.HasInterfaces();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletColors() const
    {
      return _dirichlet_colors;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<bool>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletCompMask(
        unsigned int color) const
    {
      unsigned int comp = _dirichlet_colors.size();
      for (unsigned int i = 0; i < _dirichlet_colors.size(); ++i)
        {
          if (_dirichlet_colors[i] == color)
            {
              comp = i;
              break;
            }
        }
      if (comp == _dirichlet_colors.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
        }
      return _dirichlet_comps[comp];
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const Function<dim>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletValues(
        unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {
      unsigned int col = _dirichlet_colors.size();
      for (unsigned int i = 0; i < _dirichlet_colors.size(); ++i)
        {
          if (_dirichlet_colors[i] == color)
            {
              col = i;
              break;
            }
        }
      if (col == _dirichlet_colors.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(), "OptProblem::GetDirichletValues");
        }
      _primal_dirichlet_values[col]->ReInit(param_values, domain_values, color);
      return *(_primal_dirichlet_values[col]);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetBoundaryEquationColors() const
    {
      return _state_boundary_equation_colors;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const dealii::ConstraintMatrix&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetStateDoFConstraints();
    }
  template<typename OPTPROBLEM, typename PDE, typename DD,
    typename SPARSITYPATTERN, typename VECTOR, int dim>  const dealii::Function<dim>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetInitialValues() const
  {
    return _opt_problem.GetInitialValues();
  }
///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
