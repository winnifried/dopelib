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

#ifndef STATE_PROBLEM_H_
#define STATE_PROBLEM_H_

#include <basic/spacetimehandler.h>
#include <problemdata/initialnewtonproblem.h> 

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
      pde_(pde), opt_problem_(OP), initial_newton_problem_(NULL)
      {
        dirichlet_colors_ = opt_problem_.dirichlet_colors_;
        dirichlet_comps_ = opt_problem_.dirichlet_comps_;
        primal_dirichlet_values_ = opt_problem_.primal_dirichlet_values_;
        state_boundary_equation_colors_
            = opt_problem_.state_boundary_equation_colors_;
	interval_length_ = 1.;
      }

	~StateProblem()
	{
	  if (initial_newton_problem_ != NULL)
            delete initial_newton_problem_;
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

      /**
       * Function returning whether the PDE needs to set a special initial 
       * value for the newton solver to work.
       * Relevant for stationary PDEs only when the constant zero is not a
       * feasible starting point.
       */
      bool
	NeedInitialState() const
      {
	return pde_.NeedInitialState();
      }

      InitialNewtonProblem<OPTPROBLEM,StateProblem<OPTPROBLEM,PDE,DD,SPARSITYPATTERN,VECTOR,dim>, VECTOR, dim>&
	GetNewtonInitialProblem()
      {
	if(NeedInitialState())
	{
	  if ( initial_newton_problem_ == NULL)
	  {
	    initial_newton_problem_ = new InitialNewtonProblem<OPTPROBLEM,StateProblem<OPTPROBLEM,PDE,DD,SPARSITYPATTERN,VECTOR,dim>, VECTOR, dim>
	  (*this);
	  }
	  return *initial_newton_problem_;
	}
	else
	{
	  throw DOpEException("GetNewtonInitialProblem has no meaning in case NeedInitialState() == false","StateProblem::GetNewtonInitialProblem");
	}
      }

      /******************************************************/
      /****For the initial values ***************/
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
      void Init_ElementEquation(const EDC& edc,
			     dealii::Vector<double> &local_vector, double scale,
			     double scale_ico)
      {
        pde_.Init_ElementEquation(edc, local_vector, scale, scale_ico);
      }

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
      void
      Init_ElementRhs(const EDC& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.Init_ElementRhs(& GetInitialValues(), edc, local_vector, scale);
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
      template<typename EDC>
      void Init_ElementMatrix(const EDC& edc,
			   dealii::FullMatrix<double> &local_entry_matrix, double scale,
			   double scale_ico)
      {
        pde_.Init_ElementMatrix(edc, local_entry_matrix, scale, scale_ico);
      }

      /******************************************************/
      /* Functions as in OptProblem */
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementEquation(const EDC& edc,
            dealii::Vector<double> &local_vector, double scale,
            double scale_ico);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementTimeEquation(const EDC& edc,
            dealii::Vector<double> &local_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementTimeEquationExplicit(const EDC& edc,
            dealii::Vector<double> &local_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementRhs(const EDC& edc,
            dealii::Vector<double> &local_vector, double scale = 1.);

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
      template<typename EDC>
        inline void
        ElementMatrix(const EDC& edc,
            dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
            double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementTimeMatrix(const EDC& edc,
            dealii::FullMatrix<double> &local_entry_matrix);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename EDC>
        inline void
        ElementTimeMatrixExplicit(const EDC& edc,
            dealii::FullMatrix<double> &local_entry_matrix);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        FaceEquation(const FDC& fdc,
            dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
     template<typename FDC>
        inline void
        InterfaceEquation(const FDC& fdc,
            dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        FaceRhs(const FDC& fdc,
            dealii::Vector<double> &local_vector, double scale = 1.);

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
            dealii::Vector<double> &local_vector, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        BoundaryRhs(const FDC& fdc,
            dealii::Vector<double> &local_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      template<typename FDC>
        inline void
        BoundaryMatrix(const FDC& fdc,
            dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

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
      SetTime(double time,
	      unsigned int time_dof_number, 
	      const TimeIterator& interval, bool initial = false);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
      inline void
      ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;
      /**      
       *  Experimental status: Needed for MG prec.
       */
      inline void
      ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const;

      inline void
      ComputeMGSparsityPattern(dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const;

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
        return opt_problem_.GetOutputHandler();
      } 
      OPTPROBLEM&
      GetBaseProblem()
      {
        return opt_problem_;
      }

      template<typename ELEMENTITERATOR>
      bool AtInterface(ELEMENTITERATOR& element, unsigned int face) const;

    protected:

    private:
      PDE& pde_;
      OPTPROBLEM& opt_problem_;
      InitialNewtonProblem<OPTPROBLEM,StateProblem<OPTPROBLEM,PDE,DD,SPARSITYPATTERN,VECTOR,dim>, VECTOR, dim> * initial_newton_problem_;

      std::vector<unsigned int> dirichlet_colors_;
      std::vector<std::vector<bool> > dirichlet_comps_;
      std::vector<PrimalDirichletData<DD, VECTOR, dim>*>
          primal_dirichlet_values_;
      std::vector<unsigned int> state_boundary_equation_colors_;
      double interval_length_;
    };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, 
          dim>::ElementEquation(const EDC& edc,
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        pde_.ElementEquation(edc, local_vector, scale*interval_length_, scale_ico*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementTimeEquation(const EDC& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.ElementTimeEquation(edc, local_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementTimeEquationExplicit(const EDC& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.ElementTimeEquationExplicit(edc, local_vector, scale);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::FaceEquation(const FDC& fdc,
				 dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        pde_.FaceEquation(fdc, local_vector, scale*interval_length_, scale_ico*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::InterfaceEquation(const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        pde_.InterfaceEquation(fdc,  local_vector, scale*interval_length_, scale_ico*interval_length_);
      }
  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryEquation(const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        pde_.BoundaryEquation(fdc, local_vector, scale*interval_length_, scale_ico*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementRhs(const EDC& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.ElementRightHandSide(edc, local_vector, scale*interval_length_);
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
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.FaceRightHandSide(fdc, local_vector, scale*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryRhs(const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale)
      {
        pde_.BoundaryRightHandSide(fdc, local_vector, scale*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementMatrix(const EDC& edc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        pde_.ElementMatrix(edc, local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementTimeMatrix(const EDC& edc,
          FullMatrix<double> &local_entry_matrix)
      {
        pde_.ElementTimeMatrix(edc, local_entry_matrix);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename EDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::ElementTimeMatrixExplicit(const EDC& edc,
          dealii::FullMatrix<double> &local_entry_matrix)
      {
        pde_.ElementTimeMatrixExplicit(edc, local_entry_matrix);
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
        pde_.FaceMatrix(fdc, local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
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
        pde_.InterfaceMatrix(fdc,  local_entry_matrix, scale*interval_length_, scale_ico*interval_length_);
      }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename FDC>
      void
      StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR,
          dim>::BoundaryMatrix(const FDC& fdc,
				   FullMatrix<double> &local_matrix, double scale,
				   double scale_ico)
      {
        pde_.BoundaryMatrix(fdc, local_matrix, scale*interval_length_, scale_ico*interval_length_);
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
      return opt_problem_.GetSpaceTimeHandler()->GetFESystem("state");
    }

  /******************************************************/
  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const SmartPointer<const dealii::hp::FECollection<dim> >
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFECollection() const
    {
      return opt_problem_.GetSpaceTimeHandler()->GetFECollection("state");
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetUpdateFlags() const
    {
      UpdateFlags r;
      r = pde_.GetUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    UpdateFlags
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      r = pde_.GetFaceUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::SetTime(
      double time,
      unsigned int time_dof_number, const TimeIterator& interval, bool initial)
    {
      opt_problem_.SetTime(time, time_dof_number, interval, initial);
      interval_length_ = opt_problem_.GetSpaceTimeHandler()->GetStepSize();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeSparsityPattern(
        SPARSITYPATTERN & sparsity) const
    {
      opt_problem_.GetSpaceTimeHandler()->ComputeStateSparsityPattern(sparsity);
    }

 /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeMGSparsityPattern(
        dealii::MGLevelObject<dealii::BlockSparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const
    {
      opt_problem_.GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
    }

/******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    void
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::ComputeMGSparsityPattern(
        dealii::MGLevelObject<dealii::SparsityPattern> & mg_sparsity_patterns,
				      unsigned int n_levels) const
    {
      opt_problem_.GetSpaceTimeHandler()->ComputeMGStateSparsityPattern(mg_sparsity_patterns, n_levels);
    }



  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    bool
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::HasFaces() const
    {
      return pde_.HasFaces();
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
      return pde_.HasInterfaces();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletColors() const
    {
      return dirichlet_colors_;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<bool>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDirichletCompMask(
        unsigned int color) const
    {
      unsigned int comp = dirichlet_colors_.size();
      for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
        {
          if (dirichlet_colors_[i] == color)
            {
              comp = i;
              break;
            }
        }
      if (comp == dirichlet_colors_.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(), "OptProblem::GetDirichletCompMask");
        }
      return dirichlet_comps_[comp];
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
      unsigned int col = dirichlet_colors_.size();
      for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
        {
          if (dirichlet_colors_[i] == color)
            {
              col = i;
              break;
            }
        }
      if (col == dirichlet_colors_.size())
        {
          std::stringstream s;
          s << "DirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(), "OptProblem::GetDirichletValues");
        }
      primal_dirichlet_values_[col]->ReInit(param_values, domain_values, color);
      return *(primal_dirichlet_values_[col]);
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetBoundaryEquationColors() const
    {
      return state_boundary_equation_colors_;
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    const dealii::ConstraintMatrix&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetDoFConstraints() const
    {
      return opt_problem_.GetSpaceTimeHandler()->GetStateDoFConstraints();
    }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
    typename SPARSITYPATTERN, typename VECTOR, int dim>  const dealii::Function<dim>&
    StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>::GetInitialValues() const
  {
    return opt_problem_.GetInitialValues();
  }

  /******************************************************/

  template<typename OPTPROBLEM, typename PDE, typename DD,
      typename SPARSITYPATTERN, typename VECTOR, int dim>
    template<typename ELEMENTITERATOR>
    bool StateProblem<OPTPROBLEM, PDE, DD, SPARSITYPATTERN, VECTOR, dim>
                      ::AtInterface(ELEMENTITERATOR& element, unsigned int face) const
  {
    return pde_.AtInterface(element,face);
  }


///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
