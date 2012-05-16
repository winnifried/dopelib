#ifndef _STATE_PROBLEM_H_
#define _STATE_PROBLEM_H_

#include "spacetimehandler.h"

using namespace dealii;

namespace DOpE
{
  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim=dopedim>
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
      template<typename DATACONTAINER>
      void Init_CellEquation(const DATACONTAINER& cdc,
			     dealii::Vector<double> &local_cell_vector, double scale,
			     double scale_ico)
      {
        _pde.Init_CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

      template<typename DATACONTAINER>
      void
      Init_CellRhs(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_CellRhs(& GetInitialValues(), cdc, local_cell_vector, scale);
      }

      void
      Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
      const std::map<std::string, const VECTOR*> &/*domain_values*/,
      VECTOR& /*rhs_vector*/, double /*scale=1.*/)
      {
	//Note if this is implemented one needs to update Init_PointRhs in the 
	// OptProblem container in the tangent case.
      }

      template<typename DATACONTAINER>
      void Init_CellMatrix(const DATACONTAINER& cdc,
			   dealii::FullMatrix<double> &local_entry_matrix, double scale,
			   double scale_ico)
      {
        _pde.Init_CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

      /******************************************************/
      /* Functions as in OptProblem */
      template<typename DATACONTAINER>
        inline void
        CellEquation(const DATACONTAINER& cdc,
            dealii::Vector<double> &local_cell_vector, double scale,
            double scale_ico);

      template<typename DATACONTAINER>
        inline void
        CellTimeEquation(const DATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

      template<typename DATACONTAINER>
        inline void
        CellTimeEquationExplicit(const DATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

      template<typename DATACONTAINER>
        inline void
        CellRhs(const DATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

        void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale);

      template<typename DATACONTAINER>
        inline void
        CellMatrix(const DATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
            double scale_ico = 1.);

      template<typename DATACONTAINER>
        inline void
        CellTimeMatrix(const DATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix);

      template<typename DATACONTAINER>
        inline void
        CellTimeMatrixExplicit(const DATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix);

      template<typename FACEDATACONTAINER>
        inline void
        FaceEquation(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        InterfaceEquation(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        FaceRhs(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        FaceMatrix(const FACEDATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        InterfaceMatrix(const FACEDATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,double scale_ico = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        BoundaryEquation(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1., double scale_ico = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        BoundaryRhs(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.);

      template<typename FACEDATACONTAINER>
        inline void
        BoundaryMatrix(const FACEDATACONTAINER& dc,
            dealii::FullMatrix<double> &local_cell_matrix, double scale = 1., double scale_ico = 1.);

      inline const dealii::SmartPointer<const dealii::FESystem<dealdim> >
      GetFESystem() const;

      inline const dealii::SmartPointer<
    const dealii::hp::FECollection<dealdim> >
      GetFECollection() const;

      inline std::string
      GetDoFType() const;

      inline bool
      HasFaces() const;
      inline bool
      HasPoints() const;
      inline bool
      HasInterfaces() const;

      inline dealii::UpdateFlags
      GetUpdateFlags() const;

      inline dealii::UpdateFlags
      GetFaceUpdateFlags() const;

      /******************************************************/
      inline void
      SetTime(double time, const TimeIterator& interval);

      inline void
      ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

      inline const std::vector<unsigned int>&
      GetDirichletColors() const;
      inline const std::vector<bool>&
      GetDirichletCompMask(unsigned int color) const;
      inline const Function<dealdim>
          &
          GetDirichletValues(
              unsigned int color,
              const std::map<std::string, const dealii::Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values) const;
      inline const std::vector<unsigned int>&
      GetBoundaryEquationColors() const;
      inline const dealii::ConstraintMatrix&
      GetDoFConstraints() const;
    const dealii::Function<dealdim>&
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
      std::vector<PrimalDirichletData<DD, VECTOR, dopedim, dealdim>*>
          _primal_dirichlet_values;
      std::vector<unsigned int> _state_boundary_equation_colors;

    };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellEquation(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        _pde.CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellTimeEquation(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellTimeEquation(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellTimeEquationExplicit(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellTimeEquationExplicit(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::FaceEquation(const FACEDATACONTAINER& fdc,
				 dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.FaceEquation(fdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::InterfaceEquation(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.InterfaceEquation(fdc,  local_cell_vector, scale, scale_ico);
      }
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::BoundaryEquation(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale, double scale_ico)
      {
        _pde.BoundaryEquation(fdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellRhs(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.CellRightHandSide(cdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::PointRhs(
        const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
        const std::map<std::string, const VECTOR*> &/*domain_values*/,
        VECTOR& /*rhs_vector*/, double /*scale*/)
    {

    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::FaceRhs(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.FaceRightHandSide(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::BoundaryRhs(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.BoundaryRightHandSide(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellMatrix(const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        _pde.CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellTimeMatrix(const DATACONTAINER& cdc,
          FullMatrix<double> &local_entry_matrix)
      {
        _pde.CellTimeMatrix(cdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename DATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::CellTimeMatrixExplicit(const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {
        _pde.CellTimeMatrixExplicit(cdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::FaceMatrix(const FACEDATACONTAINER& fdc,
			       FullMatrix<double> &local_entry_matrix, double scale,
			       double scale_ico)
      {
        _pde.FaceMatrix(fdc, local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::InterfaceMatrix(const FACEDATACONTAINER& fdc,
				    FullMatrix<double> &local_entry_matrix, double scale,
				    double scale_ico)
      {
        _pde.InterfaceMatrix(fdc,  local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    template<typename FACEDATACONTAINER>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim>::BoundaryMatrix(const FACEDATACONTAINER& fdc,
				   FullMatrix<double> &local_cell_matrix, double scale,
				   double scale_ico)
      {
        _pde.BoundaryMatrix(fdc, local_cell_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    std::string
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetDoFType() const
    {
      return "state";
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const SmartPointer<const dealii::FESystem<dealdim> >
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetFESystem() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetFESystem("state");
    }

  /******************************************************/
  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const SmartPointer<const dealii::hp::FECollection<dealdim> >
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetFECollection() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetFECollection("state");
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetFaceUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::SetTime(
        double time, const TimeIterator& interval)
    {
      _opt_problem.SetTime(time, interval);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      _opt_problem.GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::HasFaces() const
    {
      return _pde.HasFaces();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::HasPoints() const
    {
      return false;//We have no PointRhs in normal stateproblems at the moment.
    }


  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::HasInterfaces() const
    {
      return _pde.HasInterfaces();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetDirichletColors() const
    {
      return _dirichlet_colors;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const std::vector<bool>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetDirichletCompMask(
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
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const Function<dealdim>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetDirichletValues(
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
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetBoundaryEquationColors() const
    {
      return _state_boundary_equation_colors;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>
    const dealii::ConstraintMatrix&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetDoFConstraints() const
    {
      return _opt_problem.GetSpaceTimeHandler()->GetStateDoFConstraints();
    }
  template<typename OPTPROBLEM, typename PDE, typename DD,
    typename SPARSITYPATTERN, typename VECTOR, int dopedim, int dealdim>  const dealii::Function<dealdim>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>::GetInitialValues() const
  {
    return _opt_problem.GetInitialValues();
  }
///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
