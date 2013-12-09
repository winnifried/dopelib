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

#ifndef _INITIAL_PROBLEM_H_
#define _INITIAL_PROBLEM_H_

#include "spacetimehandler.h"

using namespace dealii;

namespace DOpE
{
  /**
   * This is a problem used in the solution of the 
   * initial value for nonstationary problems
   * 
   * @tparam PDE     The container with the PDE description
   * @tparam VECTOR  The vector class
   * @tparam dim     The dimension of the domain.
   */
  template<typename PDE, typename VECTOR, int dim>
    class InitialProblem
    {
      public:
        InitialProblem(PDE& pde) :
            _pde(pde)
        {
        }

        std::string
        GetName() const
        {
          return "InitialProblem";
        }
        std::string
        GetType() const
        {
          return "initial_state";
        }

        /******************************************************/
        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        template<typename CDC>
          inline void
          ElementEquation(const CDC& cdc,
              dealii::Vector<double> &local_vector, double scale,
              double scale_ico);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        template<typename CDC>
          inline void
          ElementRhs(const CDC& cdc,
              dealii::Vector<double> &local_vector, double scale = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        inline void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        template<typename CDC>
          inline void
          ElementMatrix(const CDC& cdc,
              dealii::FullMatrix<double> &local_matrix, double scale = 1.,
              double scale_ico = 1.);

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
              dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

        /**
	 * Functions providing the required information for the integrator.
	 * see OptProblemContainer for details.
	 */
        template<typename FDC>
          inline void
          InterfaceMatrix(const FDC& fdc,
              dealii::FullMatrix<double> &local_matrix, double scale = 1., double scale_ico = 1.);

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
        inline const dealii::SmartPointer<const dealii::hp::FECollection<dim> >
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
        template<typename SPARSITYPATTERN>
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
        inline const Function<dim>&
        GetDirichletValues(unsigned int color,
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
          return _pde.GetOutputHandler();
        }
        PDE&
        GetBaseProblem()
        {
          return _pde;
        }
      protected:

      private:
        PDE& _pde;
    };

  /*****************************************************************************************/
  /********************************IMPLEMENTATION*******************************************/
  /*****************************************************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename CDC>
      void
      InitialProblem<PDE, VECTOR, dim>::ElementEquation(const CDC& cdc,
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        _pde.Init_ElementEquation(cdc, local_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceEquation(
          const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        _pde.Init_FaceEquation(fdc, local_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::InterfaceEquation(
          const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        _pde.Init_InterfaceEquation(fdc, local_vector, scale, scale_ico);
      }
  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryEquation(
          const FDC& fdc,
          dealii::Vector<double> &local_vector, double scale, double scale_ico)
      {
        _pde.Init_BoundaryEquation(fdc, local_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename CDC>
      void
      InitialProblem<PDE, VECTOR, dim>::ElementRhs(const CDC& cdc,
          dealii::Vector<double> &local_vector, double scale)
      {
        _pde.Init_ElementRhs(cdc, local_vector, scale);
      }

  /******************************************************/
  template<typename PDE, typename VECTOR, int dim>
    void
    InitialProblem<PDE, VECTOR, dim>::PointRhs(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values,
        VECTOR& rhs_vector, double scale)
    {
      _pde.Init_PointRhs(param_values, domain_values, rhs_vector, scale);
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceRhs(
          const FDC& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryRhs(
          const FDC& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename CDC>
      void
      InitialProblem<PDE, VECTOR, dim>::ElementMatrix(const CDC& cdc,
          dealii::FullMatrix<double> &local_matrix, double scale, double scale_ico)
      {
        _pde.Init_ElementMatrix(cdc, local_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceMatrix(const FDC& fdc,
          FullMatrix<double> &local_matrix, double scale, double scale_ico)
      {
        _pde.Init_FaceMatrix(fdc, local_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::InterfaceMatrix(
          const FDC& fdc, FullMatrix<double> &local_matrix, double scale, double scale_ico)
      {
        _pde.Init_InterfaceMatrix(fdc, local_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FDC>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryMatrix(
          const FDC& fdc, FullMatrix<double> &local_matrix, double scale, double scale_ico)
      {
        _pde.Init_BoundaryMatrix(fdc, local_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    std::string
    InitialProblem<PDE, VECTOR, dim>::GetDoFType() const
    {
      return _pde.GetDoFType();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const SmartPointer<const dealii::FESystem<dim> >
    InitialProblem<PDE, VECTOR, dim>::GetFESystem() const
    {
      return _pde.GetFESystem();
    }

  /******************************************************/
  template<typename PDE, typename VECTOR, int dim>
    const SmartPointer<const dealii::hp::FECollection<dim> >
    InitialProblem<PDE, VECTOR, dim>::GetFECollection() const
    {
      return _pde.GetFECollection();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    UpdateFlags
    InitialProblem<PDE, VECTOR, dim>::GetUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    UpdateFlags
    InitialProblem<PDE, VECTOR, dim>::GetFaceUpdateFlags() const
    {
      UpdateFlags r;
      r = _pde.GetFaceUpdateFlags();
      return r | update_JxW_values;
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    void
    InitialProblem<PDE, VECTOR, dim>::SetTime(double time,
        const TimeIterator& interval)
    {
      _pde.SetTime(time, interval);
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename SPARSITYPATTERN>
      void
      InitialProblem<PDE, VECTOR, dim>::ComputeSparsityPattern(
          SPARSITYPATTERN & sparsity) const
      {
        _pde.ComputeStateSparsityPattern(sparsity);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    bool
    InitialProblem<PDE, VECTOR, dim>::HasFaces() const
    {
      return _pde.HasFaces();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    bool
    InitialProblem<PDE, VECTOR, dim>::HasPoints() const
    {
      return _pde.HasPoints();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    bool
    InitialProblem<PDE, VECTOR, dim>::HasInterfaces() const
    {
      return _pde.HasInterfaces();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    InitialProblem<PDE, VECTOR, dim>::GetDirichletColors() const
    {
      return _pde.GetDirichletColors();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const std::vector<bool>&
    InitialProblem<PDE, VECTOR, dim>::GetDirichletCompMask(
        unsigned int color) const
    {
      return _pde.GetDirichletCompMask(color);
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const Function<dim>&
    InitialProblem<PDE, VECTOR, dim>::GetDirichletValues(unsigned int color,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR*> &domain_values) const
    {
      return _pde.GetDirichletValues(color, param_values, domain_values);
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const std::vector<unsigned int>&
    InitialProblem<PDE, VECTOR, dim>::GetBoundaryEquationColors() const
    {
      return _pde.GetBoundaryEquationColors();
    }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    const dealii::ConstraintMatrix&
    InitialProblem<PDE, VECTOR, dim>::GetDoFConstraints() const
    {
      return _pde.GetDoFConstraints();
    }
  template<typename PDE, typename VECTOR, int dim>
    const dealii::Function<dim>&
    InitialProblem<PDE, VECTOR, dim>::GetInitialValues() const
    {
      return _pde.GetInitialValues();
    }
///////////////ENDOF NAMESPACE DOPE///////////////////////////
}
#endif
