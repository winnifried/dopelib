#ifndef _INITIAL_PROBLEM_H_
#define _INITIAL_PROBLEM_H_

#include "spacetimehandler.h"

using namespace dealii;

namespace DOpE
{
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
        /* Functions as in OptProblem */
        template<typename DATACONTAINER>
          inline void
          CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico);

        template<typename DATACONTAINER>
          inline void
          CellRhs(const DATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        inline void
        PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values,
            VECTOR& rhs_vector, double scale = 1);

        template<typename DATACONTAINER>
          inline void
          CellMatrix(const DATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
              double scale_ico = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          FaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          InterfaceEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          FaceRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          FaceMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        template<typename FACEDATACONTAINER>
          inline void
          InterfaceMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_entry_matrix);

        template<typename FACEDATACONTAINER>
          inline void
          BoundaryEquation(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          BoundaryRhs(const FACEDATACONTAINER& dc,
              dealii::Vector<double> &local_cell_vector, double scale = 1.);

        template<typename FACEDATACONTAINER>
          inline void
          BoundaryMatrix(const FACEDATACONTAINER& dc,
              dealii::FullMatrix<double> &local_cell_matrix);

        inline const dealii::SmartPointer<const dealii::FESystem<dim> >
        GetFESystem() const;

        inline const dealii::SmartPointer<const dealii::hp::FECollection<dim> >
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

        template<typename SPARSITYPATTERN>
          inline void
          ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const;

        inline const std::vector<unsigned int>&
        GetDirichletColors() const;
        inline const std::vector<bool>&
        GetDirichletCompMask(unsigned int color) const;
        inline const Function<dim>&
        GetDirichletValues(unsigned int color,
            const std::map<std::string, const dealii::Vector<double>*> &param_values,
            const std::map<std::string, const VECTOR*> &domain_values) const;
        inline const std::vector<unsigned int>&
        GetBoundaryEquationColors() const;
        inline const dealii::ConstraintMatrix&
        GetDoFConstraints() const;
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
    template<typename DATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::CellEquation(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double scale_ico)
      {
        _pde.Init_CellEquation(cdc, local_cell_vector, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_FaceEquation(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::InterfaceEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_InterfaceEquation(fdc, local_cell_vector, scale);
      }
  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryEquation(
          const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_BoundaryEquation(fdc, local_cell_vector, scale);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename DATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::CellRhs(const DATACONTAINER& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        _pde.Init_CellRhs(cdc, local_cell_vector, scale);
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
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceRhs(
          const FACEDATACONTAINER& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryRhs(
          const FACEDATACONTAINER& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename DATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::CellMatrix(const DATACONTAINER& cdc,
          dealii::FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        _pde.Init_CellMatrix(cdc, local_entry_matrix, scale, scale_ico);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::FaceMatrix(const FACEDATACONTAINER& fdc,
          FullMatrix<double> &local_entry_matrix)
      {
        _pde.Init_FaceMatrix(fdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::InterfaceMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_entry_matrix)
      {
        _pde.Init_InterfaceMatrix(fdc, local_entry_matrix);
      }

  /******************************************************/

  template<typename PDE, typename VECTOR, int dim>
    template<typename FACEDATACONTAINER>
      void
      InitialProblem<PDE, VECTOR, dim>::BoundaryMatrix(
          const FACEDATACONTAINER& fdc, FullMatrix<double> &local_cell_matrix)
      {
        _pde.Init_BoundaryMatrix(fdc, local_cell_matrix);
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
