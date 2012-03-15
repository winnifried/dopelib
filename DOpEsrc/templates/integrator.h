#ifndef _Integrator_H_
#define _Integrator_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include <numerics/vectors.h>
#include <numerics/matrices.h>

#include <base/function.h>

#include <vector>

#include "celldatacontainer.h"
#include "facedatacontainer.h"

namespace DOpE
{
  /**
   * This class is used to integrate the righthand side, matrix and so on
   * This class assumes that one uses the same triangulation for the control and state variable.
   */
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    class Integrator
    {
      public:
        Integrator(INTEGRATORDATACONT& idc);

        ~Integrator();

        /**
         This Function should be called once after grid refinement, or changes in boundary values
         to  recompute sparsity patterns, and constraint matrices.
         */
        void
        ReInit();

        template<typename PROBLEM>
          void
          ComputeNonlinearResidual(PROBLEM& pde, VECTOR &residual,
              bool apply_boundary_values = true);
        template<typename PROBLEM>
          void
          ComputeNonlinearLhs(PROBLEM& pde, VECTOR &residual,
              bool apply_boundary_values = true);
        template<typename PROBLEM>
          void
          ComputeNonlinearRhs(PROBLEM& pde, VECTOR &residual,
              bool apply_boundary_values = true);
        template<typename PROBLEM, typename MATRIX>
          void
          ComputeMatrix(PROBLEM& pde, MATRIX &matrix);

        template<typename PROBLEM>
          void
          ComputeNonlinearAlgebraicResidual(PROBLEM& pde, VECTOR &residual);
        template<typename PROBLEM>
          void
          ComputeLocalControlConstraints(PROBLEM& pde, VECTOR &constraints);

        template<typename PROBLEM>
          SCALAR
          ComputeDomainScalar(PROBLEM& pde);
        template<typename PROBLEM>
          SCALAR
          ComputePointScalar(PROBLEM& pde);
        template<typename PROBLEM>
          SCALAR
          ComputeBoundaryScalar(PROBLEM& pde);
        template<typename PROBLEM>
          SCALAR
          ComputeFaceScalar(PROBLEM& pde);
        template<typename PROBLEM>
          SCALAR
          ComputeAlgebraicScalar(PROBLEM& pde);

        template<typename PROBLEM>
          void
          ApplyInitialBoundaryValues(PROBLEM& pde, VECTOR &u);
        template<typename PROBLEM>
          void
          ApplyTransposedInitialBoundaryValues(PROBLEM& pde, VECTOR &u);
        template<typename PROBLEM>
          void
          ApplyNewtonBoundaryValues(PROBLEM& pde, VECTOR &u);
        template<typename PROBLEM, typename MATRIX>
          void
          ApplyNewtonBoundaryValues(PROBLEM& pde, MATRIX &matrix, VECTOR &rhs,
              VECTOR &sol);

        inline void
        AddDomainData(std::string name, const VECTOR* new_data);
        inline void
        DeleteDomainData(std::string name);
        inline const std::map<std::string, const VECTOR*>&
        GetDomainData() const;

        inline void
        AddParamData(std::string name, const dealii::Vector<SCALAR>* new_data);
        inline void
        DeleteParamData(std::string name);
        inline const std::map<std::string, const dealii::Vector<SCALAR>*>&
        GetParamData() const;

        inline INTEGRATORDATACONT&
        GetIntegratorDataContainer() const;

      private:
        template<typename DOFHANDLER>
          void
              InterpolateBoundaryValues(const DOpEWrapper::DoFHandler<dim,
                  DOFHANDLER>* dof_handler, const unsigned int color,
                  const dealii::Function<dim>& function, std::map<unsigned int,
                      SCALAR>& boundary_values,
                  const std::vector<bool>& comp_mask) const;

        // given a vector of active cell iterators and a facenumber, checks if the face
        // belongs to an 'interface' (i.e. the adjoining cells have different
        // material ids)
        template<typename CELLITERATOR>
        bool AtInterface(CELLITERATOR& cell, unsigned int face)
        {
          if (cell[0]->neighbor_index(face) != -1)
            if(cell[0]->material_id() != cell[0]->neighbor(face)->material_id())
              return true;
          return false;
        }
        INTEGRATORDATACONT & _idc;

        std::map<std::string, const VECTOR*> _domain_data;
        std::map<std::string, const dealii::Vector<SCALAR>*> _param_data;
    };

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::Integrator(
        INTEGRATORDATACONT& idc) :
      _idc(idc)
    {
    }

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::~Integrator()
    {

    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ReInit()
    {

    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearResidual(
          PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {
        residual = 0.;
        // Begin integration
        unsigned int dofs_per_cell;

        dealii::Vector<SCALAR> local_cell_vector;

        std::vector<unsigned int> local_dof_indices;

        const auto& dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto
        cell =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        // Generate the data containers.
        _idc.InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = _idc.GetCellDataContainer();

#if deal_II_dimension == 2 || deal_II_dimension == 3
        bool need_faces = pde.HasFaces();
        bool need_interfaces = pde.HasInterfaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
        _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            need_interfaces);
        auto & fdc = _idc.GetFaceDataContainer();

#endif

        for (; cell[0] != endc[0]; cell[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            if (cell[dh] == endc[dh])
            {
              throw DOpEException(
                  "Cellnumbers in DoFHandlers are not matching!",
                  "Integrator::ComputeNonlinearResidual");
            }
          }

          cdc.ReInit();

          dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

          local_cell_vector.reinit(dofs_per_cell);
          local_cell_vector = 0;

          local_dof_indices.resize(0);
          local_dof_indices.resize(dofs_per_cell, 0);

          //the second '1' plays only a role in the stationary case. In the non-stationary
          //case, scale_ico is set by the time-stepping-scheme
          pde.CellEquation(cdc, local_cell_vector, 1., 1.);
          pde.CellRhs(cdc, local_cell_vector, -1.);

#if deal_II_dimension == 2 || deal_II_dimension == 3
          if(need_boundary_integrals && cell[0]->at_boundary())
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell[0]->face(face)->at_boundary()
                  &&
                  (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
              {
                fdc.ReInit(face);
                pde.BoundaryEquation(fdc,local_cell_vector);
                pde.BoundaryRhs(fdc,local_cell_vector,-1.);
              }
            }
          }
          if(need_faces)
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell[0]->neighbor_index(face) != -1)
              {
                fdc.ReInit(face);
                pde.FaceEquation(fdc, local_cell_vector);
                pde.FaceRhs(fdc, local_cell_vector,-1.);
              }
            }
          }
          if( need_interfaces)
          {

            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              auto face_it = cell[0]->face(face);
              // first, check if we are at an interface, i.e. not the neighbour exists and
              // it has a different material_id than the actual cell
              if(AtInterface(cell, face))
              {
                //now we distinguish between 3 cases:

                //note: neighbours->level is at most the actual level, i.e. they may or may not be refined!
                if (cell[0]->neighbor(face)->has_children())
                {
                  //first: neighbour is finer
                  const unsigned int neighbor2=
                  cell[0]->neighbor_of_neighbor(face);

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    const auto
                    neighbor_child
                    = cell[0]->neighbor_child_on_subface (face, subface_no);

                    Assert (neighbor_child->face(neighbor2) ==
                        cell[0]->face(face)->child(subface_no),
                        ExcInternalError());
                    Assert (neighbor_child->has_children() == false,
                        ExcInternalError());

                    //TODO Now here we have to initialise the subface_values on the
                    // actual cell and then the facevalues of the neighbours
                    pde.InterfaceEquation(fdc, local_cell_vector);

                  }

                }
                else if (cell[0]->has_children())
                {// So this cell is finer than the previous one
                  Assert(cell[0]->neighbor(face)->level() == cell[0]->level()-1,
                      ExcInternalError());

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    //TODO initialise the subface_values ont he neighbor_cell and the
                    // facevalues on the actual cell
                    pde.InterfaceEquation(fdc, local_cell_vector);
                  }

                }
                else
                {
                  //the 'normal' case: both cells are on the same level

                  Assert(cell[0]->neighbor(face)->level() == cell[0]->level(),
                      ExcInternalError());
                  fdc.ReInit(face);
                  fdc.ReInitNbr();
                  pde.InterfaceEquation(fdc, local_cell_vector);
                }

              }//endif atinterface
            }
          }//endif need_interfaces
#endif
          //LocalToGlobal
          cell[0]->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            residual(local_dof_indices[i]) += local_cell_vector(i);
          }

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            cell[dh]++;
          }
        }

        if (apply_boundary_values)
        {
          ApplyNewtonBoundaryValues(pde, residual);
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearLhs(
          PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {
        {

          residual = 0.;
          // Begin integration
          unsigned int dofs_per_cell;

          dealii::Vector<SCALAR> local_cell_vector;

          std::vector<unsigned int> local_dof_indices;

          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto
          cell =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          // Generate the data containers.
          _idc.InitializeCDC(pde.GetUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
              this->GetParamData(), this->GetDomainData());
          auto& cdc = _idc.GetCellDataContainer();
          //            CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim> cdc(
          //                *(this->GetQuadratureFormula()), pde.GetUpdateFlags(),
          //                *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, this->GetParamData(),
          //                this->GetDomainData());

#if deal_II_dimension == 2 || deal_II_dimension == 3
          bool need_faces = pde.HasFaces();
          bool need_interfaces = pde.HasInterfaces();
          std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
          bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

          _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData(),
              need_interfaces);
          auto & fdc = _idc.GetFaceDataContainer();
#endif

          for (; cell[0] != endc[0]; cell[0]++)
          {
            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (cell[dh] == endc[dh])
              {
                throw DOpEException(
                    "Cellnumbers in DoFHandlers are not matching!",
                    "mIntegrator::ComputeNonlinearLhs");
              }
            }

            cdc.ReInit();
            dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

            local_cell_vector.reinit(dofs_per_cell);
            local_cell_vector = 0;

            local_dof_indices.resize(0);
            local_dof_indices.resize(dofs_per_cell, 0);

            //the second '1' plays only a role in the stationary case. In the non-stationary
            //case, scale_ico is set by the time-stepping-scheme
            pde.CellEquation(cdc, local_cell_vector, 1., 1.);

#if deal_II_dimension == 2 || deal_II_dimension == 3
            if(need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
                {
                  fdc.ReInit(face);
                  pde.BoundaryEquation(fdc,local_cell_vector);
                }
              }
            }
            if(need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->neighbor_index(face) != -1)
                {
                  fdc.ReInit(face);
                  pde.FaceEquation(fdc, local_cell_vector);
                }
              }
            }

            if( need_interfaces)
            {

              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                auto face_it = cell[0]->face(face);
                // first, check if we are at an interface, i.e. not the neighbour exists and
                // it has a different material_id than the actual cell
                if(AtInterface(cell, face))
                {
                  //now we distinguish between 3 cases:

                  //note: neighbours->level is at most the actual level, i.e. they may or may not be refined!
                  if (cell[0]->neighbor(face)->has_children())
                  {
                    //first: neighbour is finer
                    const unsigned int neighbor2=
                    cell[0]->neighbor_of_neighbor(face);

                    for (unsigned int subface_no=0;
                        subface_no < cell[0]->face(face)->n_children();
                        ++subface_no)
                    {
                      const auto
                      neighbor_child
                      = cell[0]->neighbor_child_on_subface (face, subface_no);

                      Assert (neighbor_child->face(neighbor2) ==
                          cell[0]->face(face)->child(subface_no),
                          ExcInternalError());
                      Assert (neighbor_child->has_children() == false,
                          ExcInternalError());

                      //TODO Now here we have to initialise the subface_values on the
                      // actual cell and then the facevalues of the neighbours
                      pde.InterfaceEquation(fdc, local_cell_vector);

                    }

                  }
                  else if (cell[0]->has_children())
                  {// So this cell is finer than the previous one
                    Assert(cell[0]->neighbor(face)->level() == cell[0]->level()-1,
                        ExcInternalError());

                    for (unsigned int subface_no=0;
                        subface_no < cell[0]->face(face)->n_children();
                        ++subface_no)
                    {
                      //TODO initialise the subface_values ont he neighbor_cell and the
                      // facevalues on the actual cell
                      pde.InterfaceEquation(fdc, local_cell_vector);
                    }

                  }
                  else
                  {
                    //the 'normal' case: both cells are on the same level

                    Assert(cell[0]->neighbor(face)->level() == cell[0]->level(),
                        ExcInternalError());
                    fdc.ReInit(face);
                    fdc.ReInitNbr();
                    pde.InterfaceEquation(fdc, local_cell_vector);
                  }

                }//endif atinterface
              }
            }//endif need_interfaces
#endif
            //LocalToGlobal
            cell[0]->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              residual(local_dof_indices[i]) += local_cell_vector(i);
            }

            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              cell[dh]++;
            }
          }

          if (apply_boundary_values)
          {
            ApplyNewtonBoundaryValues(pde, residual);
          }
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs(
          PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {

        {
          residual = 0.;
          // Begin integration
          unsigned int dofs_per_cell;// = pde.GetFESystem()->dofs_per_cell;

          dealii::Vector<SCALAR> local_cell_vector;//(dofs_per_cell);

          std::vector<unsigned int> local_dof_indices;//(dofs_per_cell);

          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto
          cell =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          // Initialize the data containers.
          _idc.InitializeCDC(pde.GetUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
              this->GetParamData(), this->GetDomainData());
          auto& cdc = _idc.GetCellDataContainer();
#if deal_II_dimension == 2 || deal_II_dimension == 3
          bool need_faces = pde.HasFaces();
          std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
          bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

          _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData());
          auto & fdc = _idc.GetFaceDataContainer();
#endif

          for (; cell[0] != endc[0]; cell[0]++)
          {
            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (cell[dh] == endc[dh])
              {
                throw DOpEException(
                    "Cellnumbers in DoFHandlers are not matching!",
                    "Integrator::ComputeNonlinearRhs");
              }
            }

            cdc.ReInit();
            dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

            local_cell_vector.reinit(dofs_per_cell);
            local_cell_vector = 0;

            local_dof_indices.resize(0);
            local_dof_indices.resize(dofs_per_cell, 0);
            pde.CellRhs(cdc, local_cell_vector, 1.);

#if deal_II_dimension == 2 || deal_II_dimension == 3
            if(need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
                {
                  fdc.ReInit(face);
                  pde.BoundaryRhs(fdc,local_cell_vector,1.);
                }
              }
            }
            if(need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->neighbor_index(face) != -1)
                {
                  fdc.ReInit(face);
                  pde.FaceRhs(fdc, local_cell_vector);
                }
              }
            }
#endif
            //LocalToGlobal
            cell[0]->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              residual(local_dof_indices[i]) += local_cell_vector(i);
            }

            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              cell[dh]++;
            }
          }

          if (apply_boundary_values)
          {
            ApplyNewtonBoundaryValues(pde, residual);
          }
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM, typename MATRIX>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix(
          PROBLEM& pde, MATRIX &matrix)
      {
        matrix = 0.;
        // Begin integration
        unsigned int dofs_per_cell;
        std::vector<unsigned int> local_dof_indices;

        const auto& dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto
        cell =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        _idc.InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = _idc.GetCellDataContainer();

#if deal_II_dimension == 2 || deal_II_dimension == 3
        //for the interface-case
        unsigned int nbr_dofs_per_cell;
        std::vector<unsigned int> nbr_local_dof_indices;

        bool need_faces = pde.HasFaces();
        bool need_interfaces = pde.HasInterfaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            need_interfaces);
        auto & fdc = _idc.GetFaceDataContainer();
#endif

        for (; cell[0] != endc[0]; cell[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            if (cell[dh] == endc[dh])
            {
              throw DOpEException(
                  "Cellnumbers in DoFHandlers are not matching!",
                  "Integrator::ComputeMatrix");
            }
          }
          cdc.ReInit();
          dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

          dealii::FullMatrix<SCALAR>
          local_cell_matrix(dofs_per_cell, dofs_per_cell);
          local_cell_matrix = 0;

          local_dof_indices.resize(0);
          local_dof_indices.resize(dofs_per_cell, 0);
          pde.CellMatrix(cdc, local_cell_matrix);

#if deal_II_dimension == 2 || deal_II_dimension == 3
          if(need_boundary_integrals)
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell[0]->face(face)->at_boundary()
                  &&
                  (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
              {
                fdc.ReInit(face);
                pde.BoundaryMatrix(fdc, local_cell_matrix);
              }
            }
          }
          if(need_faces)
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell[0]->neighbor_index(face) != -1)
              {
                fdc.ReInit(face);
                pde.FaceMatrix(fdc, local_cell_matrix);
              }
            }
          }

          if( need_interfaces)
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              auto face_it = cell[0]->face(face);
              // first, check if we are at an interface, i.e. not the neighbour exists and
              // it has a different material_id than the actual cell
              if(AtInterface(cell, face))
              {
                //now we distinguish between 3 cases:

                //note: neighbours->level is at most the actual level, i.e. they may or may not be refined!
                if (cell[0]->neighbor(face)->has_children())
                {
                  //first: neighbour is finer
                  const unsigned int neighbor2=
                  cell[0]->neighbor_of_neighbor(face);

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    const auto
                    neighbor_child
                    = cell[0]->neighbor_child_on_subface (face, subface_no);

                    Assert (neighbor_child->face(neighbor2) ==
                        cell[0]->face(face)->child(subface_no),
                        ExcInternalError());
                    Assert (neighbor_child->has_children() == false,
                        ExcInternalError());

                    //TODO Now here we have to initialise the subface_values on the
                    // actual cell and then the facevalues of the neighbours
                    //matrix

                  }

                }
                else if (cell[0]->has_children())
                {// So this cell is finer than the previous one
                  Assert(cell[0]->neighbor(face)->level() == cell[0]->level()-1,
                      ExcInternalError());

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    //TODO initialise the subface_values ont he neighbor_cell and the
                    // facevalues on the actual cell
                    //matrix
                  }

                }
                else
                {
                  //the 'normal' case: both cells are on the same level

                  Assert(cell[0]->neighbor(face)->level() == cell[0]->level(),
                      ExcInternalError());
                  fdc.ReInit(face);
                  fdc.ReInitNbr();
                  nbr_dofs_per_cell = fdc.GetNbrNDoFsPerCell();
                  nbr_local_dof_indices.resize(0);
                  nbr_local_dof_indices.resize(nbr_dofs_per_cell, 0);
                  dealii::FullMatrix<SCALAR> local_interface_matrix(dofs_per_cell,nbr_dofs_per_cell );
                  local_interface_matrix = 0;

                  pde.InterfaceMatrix(fdc, local_interface_matrix);

                  cell[0]->get_dof_indices(local_dof_indices);
                  cell[0]->neighbor(face)->get_dof_indices(nbr_local_dof_indices);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < nbr_dofs_per_cell; ++j)
                    {
                      matrix.add(local_dof_indices[i], nbr_local_dof_indices[j],
                          local_interface_matrix(i, j));
                    } //endfor j
                  } //endfor i
                }

              }//endif atinterface
            }
          }//endif need_interfaces
#endif

          //LocalToGlobal
          cell[0]->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              matrix.add(local_dof_indices[i], local_dof_indices[j],
                  local_cell_matrix(i, j));
            }
          }

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            cell[dh]++;
          }
        }
      }

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar(
          PROBLEM& pde)
      {
        {
          SCALAR ret = 0.;

          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto
          cell =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
          _idc.InitializeCDC(pde.GetUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
              this->GetParamData(), this->GetDomainData());
          auto& cdc = _idc.GetCellDataContainer();

          bool need_faces = pde.HasFaces();

          for (; cell[0] != endc[0]; cell[0]++)
          {
            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (cell[dh] == endc[dh])
              {
                throw DOpEException(
                    "Cellnumbers in DoFHandlers are not matching!",
                    "Integrator::ComputeDomainScalar");
              }
            }

            cdc.ReInit();
            ret += pde.CellFunctional(cdc);

            if (need_faces)
            {
              throw DOpEException("Face Integrals not Implemented!",
                  "Integrator::ComputeDomainScalar");
            }

            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              cell[dh]++;
            }
          }
          return ret;
        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePointScalar(
          PROBLEM& pde)
      {

        {
          SCALAR ret = 0.;
          ret += pde.PointFunctional(this->GetParamData(),
              this->GetDomainData());

          return ret;
        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar(
#if deal_II_dimension == 2 || deal_II_dimension == 3
          PROBLEM& pde
#else
          PROBLEM& /*pde*/
#endif
      )
      {
        {
          SCALAR ret = 0.;
#if deal_II_dimension == 2 || deal_II_dimension == 3
          // Begin integration
          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData());
          auto & fdc = _idc.GetFaceDataContainer();

          std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
          bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
          if(!need_boundary_integrals)
          {
            throw DOpEException("No boundary colors given!","Integrator::ComputeBoundaryScalar");
          }

          for (;cell[0]!=endc[0]; cell[0]++)
          {
            for(unsigned int dh=1; dh<dof_handler.size(); dh++)
            {
              if( cell[dh] == endc[dh])
              {
                throw DOpEException("Cellnumbers in DoFHandlers are not matching!","Integrator::ComputeBoundaryScalar");
              }
            }

            if(need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->face(face)->at_boundary()
                    &&
                    (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                            cell[0]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
                {
                  fdc.ReInit(face);
                  ret += pde.BoundaryFunctional(fdc);
                }
              }
            }
            for(unsigned int dh=1; dh<dof_handler.size(); dh++)
            {
              cell[dh]++;
            }
          }
#else
          throw DOpEException("Not implemented in this dimension!",
              "Integrator::ComputeBoundaryScalar");
#endif

          return ret;

        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeFaceScalar(
#if deal_II_dimension == 2 || deal_II_dimension == 3
          PROBLEM& pde
#else
          PROBLEM& /*pde*/
#endif
      )
      {

        {
          SCALAR ret = 0.;
#if deal_II_dimension == 2 || deal_II_dimension == 3
          // Begin integration
          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData());
          auto & fdc = _idc.GetFaceDataContainer();

          bool need_faces = pde.HasFaces();
          if(!need_faces)
          {
            throw DOpEException("No faces required!","Integrator::ComputeFaceScalar");
          }

          for (;cell[0]!=endc[0]; cell[0]++)
          {
            for(unsigned int dh=1; dh<dof_handler.size(); dh++)
            {
              if( cell[dh] == endc[dh])
              {
                throw DOpEException("Cellnumbers in DoFHandlers are not matching!","Integrator::ComputeFaceScalar");
              }
            }

            if(need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell[0]->neighbor_index(face) != -1)
                {
                  fdc.ReInit(face);
                  ret +=pde.FaceFunctional(fdc);
                }
              }
            }
            for(unsigned int dh=1; dh<dof_handler.size(); dh++)
            {
              cell[dh]++;
            }
          }
#else
          throw DOpEException("Not implemented in this dimension!",
              "Integrator::ComputeFaceScalar");
#endif
          return ret;
        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeAlgebraicScalar(
          PROBLEM& pde)
      {

        {
          SCALAR ret = 0.;
          ret = pde.AlgebraicFunctional(this->GetParamData(),
              this->GetDomainData());
          return ret;
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearAlgebraicResidual(
          PROBLEM& pde, VECTOR &residual)
      {
        residual = 0.;
        pde.AlgebraicResidual(residual, this->GetParamData(),
            this->GetDomainData());
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeLocalControlConstraints(
          PROBLEM& pde, VECTOR &constraints)
      {
        constraints = 0.;
        pde.ComputeLocalControlConstraints(constraints, this->GetParamData(),
            this->GetDomainData());
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyTransposedInitialBoundaryValues(
          PROBLEM& /*pde*/, VECTOR &u __attribute__((unused)))
      {
        //Wird  hier nicht gebraucht...
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyInitialBoundaryValues(
          PROBLEM& pde, VECTOR &u)
      {
        //TODO Apply constraints locally, see, e.g., dealii step-27 ? But howto do this in the newton iter
        // e.g. sometimes we need zero sometimes we need other values.

        //Never Condense Nodes Here ! Or All will fail if the state is not initialized with zero!
        //pde.GetHangingNodeConstraints().condense(u);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues(
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0],
              color, pde.GetDirichletValues(color, this->GetParamData(),
                  this->GetDomainData()), boundary_values, comp_mask);

          for (typename std::map<unsigned int, SCALAR>::const_iterator p =
              boundary_values.begin(); p != boundary_values.end(); p++)
          {
            u(p->first) = p->second;
          }
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyNewtonBoundaryValues(
          PROBLEM& pde, VECTOR &u)
      {
        //TODO Apply constraints locally, see, e.g., dealii step-27 ? But howto do this in the newton iter
        // e.g. sometimes we need zero sometimes we need other values.

        pde.GetHangingNodeConstraints().condense(u);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues(
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0],
              color, dealii::ZeroFunction<dim>(comp_mask.size()),
              boundary_values, comp_mask);

          for (typename std::map<unsigned int, SCALAR>::const_iterator p =
              boundary_values.begin(); p != boundary_values.end(); p++)
          {
            u(p->first) = p->second;
          }
        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM, typename MATRIX>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyNewtonBoundaryValues(
          PROBLEM& pde, MATRIX& matrix, VECTOR &rhs, VECTOR &sol)
      {
        //TODO Apply constraints locally, see, e.g., dealii step-27 ? But howto do this in the newton iter
        // e.g. sometimes we need zero sometimes we need other values.
        pde.GetHangingNodeConstraints().condense(rhs);
        pde.GetHangingNodeConstraints().condense(matrix);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues(
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0],
              color, dealii::ZeroFunction<dim>(comp_mask.size()),
              boundary_values, comp_mask);

          dealii::MatrixTools::apply_boundary_values(boundary_values, matrix,
              sol, rhs);
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddDomainData(
        std::string name, const VECTOR* new_data)
    {
      if (_domain_data.find(name) != _domain_data.end())
      {
        throw DOpEException("Adding multiple Data with name " + name
            + " is prohibited!", "Integrator::AddDomainData");
      }
      _domain_data.insert(std::pair<std::string, const VECTOR*>(name, new_data));
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteDomainData(
        std::string name)
    {
      typename std::map<std::string, const VECTOR *>::iterator it =
          _domain_data.find(name);
      if (it == _domain_data.end())
      {
        throw DOpEException("Deleting Data " + name
            + " is impossible! Data not found", "Integrator::DeleteDomainData");
      }
      _domain_data.erase(it);
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    const std::map<std::string, const VECTOR*>&
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetDomainData() const
    {
      return _domain_data;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddParamData(
        std::string name, const dealii::Vector<SCALAR>* new_data)
    {
      if (_param_data.find(name) != _param_data.end())
      {
        throw DOpEException("Adding multiple Data with name " + name
            + " is prohibited!", "Integrator::AddParamData");
      }
      _param_data.insert(std::pair<std::string, const dealii::Vector<SCALAR>*>(
          name, new_data));
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteParamData(
        std::string name)
    {
      typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator
          it = _param_data.find(name);
      if (it == _param_data.end())
      {
        throw DOpEException("Deleting Data " + name
            + " is impossible! Data not found", "Integrator::DeleteParamData");
      }
      _param_data.erase(it);
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    const std::map<std::string, const dealii::Vector<SCALAR>*>&
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetParamData() const
    {
      return _param_data;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    INTEGRATORDATACONT&
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainer() const
    {
      return _idc;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename DOFHANDLER>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::InterpolateBoundaryValues(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER>* dof_handler,
          const unsigned int color, const dealii::Function<dim>& function,
          std::map<unsigned int, SCALAR>& boundary_values, const std::vector<
              bool>& comp_mask) const
      {
        dealii::VectorTools::interpolate_boundary_values(
            *(static_cast<const DOFHANDLER*> (dof_handler)), color, function,
            boundary_values, comp_mask);
      }

}
#endif

