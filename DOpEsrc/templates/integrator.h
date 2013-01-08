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
#include "dwrdatacontainer.h"
#include "residualestimator.h"
#include "dopetypes.h"

namespace DOpE
{
  /**
   * This class is used to integrate the righthand side, matrix and so on.
   * It assumes that one uses the same triangulation for the control and state variable.
   *
   * @template INTEGRATORDATACONT       The type of the integratordatacontainer, which has
   *                                    manages the basic data for integration (quadrature,
   *                                    celldatacontainer, facedatacontainer etc.)
   * @template VECTOR                   Class of the vectors which we use in the integrator.
   * @template SCALAR                   Type of the scalars we use in the integrator.
   * @template dim
   */
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    class Integrator
    {
      public:
        Integrator(INTEGRATORDATACONT& idc1);
        /**
         * This constructor gets two INTEGRATORDATACONT. One for the evaluation
         * of the integrals during the assembling and residual computation, the
         * second one for the evaluation of functionals.
         */
        Integrator(INTEGRATORDATACONT& idc1, INTEGRATORDATACONT& idc2);

        ~Integrator();

        /**
         This Function should be called once after grid refinement, or changes in boundary values
         to  recompute sparsity patterns, and constraint matrices.
         */
        void
        ReInit();

        /**
         * Mainly self-explanatory functions.
         */
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

        template<typename PROBLEM, class STH, class CDC, class FDC>
          void
          ComputeRefinementIndicators(PROBLEM& pde,
              DWRDataContainer<STH, INTEGRATORDATACONT, CDC, FDC, VECTOR>& dwrc);
        template<typename PROBLEM>
          void
          ComputeRefinementIndicators(PROBLEM& pde,
              ResidualErrorContainer<VECTOR>& dwrc);

        inline INTEGRATORDATACONT&
        GetIntegratorDataContainer() const;

        inline INTEGRATORDATACONT&
        GetIntegratorDataContainerFunc() const;

      private:
        template<typename DOFHANDLER>
          void
          InterpolateBoundaryValues(
              const DOpEWrapper::Mapping<dim, DOFHANDLER>& mapping,
              const DOpEWrapper::DoFHandler<dim, DOFHANDLER>* dof_handler,
              const unsigned int color, const dealii::Function<dim>& function,
              std::map<unsigned int, SCALAR>& boundary_values,
              const std::vector<bool>& comp_mask) const;

        /**
         * Given a vector of active cell iterators and a facenumber, checks if the face
         * belongs to an 'interface' (i.e. the adjoining cells have different material ids).
         *
         * @template CELLITERATOR   Class of the celliterator.
         *
         * @param   cell            The cell in question.
         * @param   face            Local number of the face for which we ask if it is
         *                          at the interface.
         */
        template<typename CELLITERATOR>
          bool
          AtInterface(CELLITERATOR& cell, unsigned int face)
          {
            if (cell[0]->neighbor_index(face) != -1)
              if (cell[0]->material_id()
                  != cell[0]->neighbor(face)->material_id())
                return true;
            return false;
          }

        INTEGRATORDATACONT & _idc1;
        INTEGRATORDATACONT & _idc2;

        std::map<std::string, const VECTOR*> _domain_data;
        std::map<std::string, const dealii::Vector<SCALAR>*> _param_data;
    };

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::Integrator(
        INTEGRATORDATACONT& idc)
        : _idc1(idc), _idc2(idc)
    {
    }

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::Integrator(
        INTEGRATORDATACONT& idc1, INTEGRATORDATACONT& idc2)
        : _idc1(idc1), _idc2(idc2)
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

        const bool need_point_rhs = pde.HasPoints();

        const auto& dof_handler =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto cell =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        // Generate the data containers.
        GetIntegratorDataContainer().InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

        bool need_faces = pde.HasFaces();
        bool need_interfaces = pde.HasInterfaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
        GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            need_interfaces);
        auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

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
                pde.BoundaryEquation(fdc,local_cell_vector, 1., 1.);
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
                pde.FaceEquation(fdc, local_cell_vector, 1., 1.);
                pde.FaceRhs(fdc, local_cell_vector,-1.);
              }
            }
          }
          if( need_interfaces)
          {

            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            {
              // auto face_it = cell[0]->face(face);
              // first, check if we are at an interface, i.e. not the neighbour exists and
              // it has a different material_id than the actual cell
              if(AtInterface(cell, face))
              {
                //There exist now 3 different scenarios, given the actual cell and face:
                // The neighbour behind this face is [ more | as much | less] refined
                // than/as the actual cell. We have to distinguish here only between the case 1
                // and the other two, because these will be distinguished in in the FaceDataContainer.

                if (cell[0]->neighbor(face)->has_children())
                {
                  //first: neighbour is finer

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    //TODO Now here we have to initialise the subface_values on the
                    // actual cell and then the facevalues of the neighbours
                    fdc.ReInit(face, subface_no);
                    fdc.ReInitNbr();

                    pde.InterfaceEquation(fdc, local_cell_vector, 1., 1.);

                  }
                }
                else
                {
                  // either neighbor is as fine as this cell or
                  // it is coarser

                  fdc.ReInit(face);
                  fdc.ReInitNbr();
                  pde.InterfaceEquation(fdc, local_cell_vector, 1., 1.);
                }

              }                  //endif atinterface
            }                  //endfor faces
          }                  //endif need_interfaces
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

        //check if we need the evaluation of PointRhs
        if (need_point_rhs)
        {
          VECTOR point_rhs;
          pde.PointRhs(this->GetParamData(), this->GetDomainData(), point_rhs,
              -1.);
          residual += point_rhs;
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
          auto cell =
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc =
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          // Generate the data containers.
          GetIntegratorDataContainer().InitializeCDC(pde.GetUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
              this->GetParamData(), this->GetDomainData());
          auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

          bool need_faces = pde.HasFaces();
          bool need_interfaces = pde.HasInterfaces();
          std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
          bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

          GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData(),
              need_interfaces);
          auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

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
                  pde.BoundaryEquation(fdc,local_cell_vector, 1., 1.);
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
                  pde.FaceEquation(fdc, local_cell_vector, 1., 1.);
                }
              }
            }

            if( need_interfaces)
            {

              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                //auto face_it = cell[0]->face(face);
                // first, check if we are at an interface, i.e. not the neighbour exists and
                // it has a different material_id than the actual cell
                if(AtInterface(cell, face))
                {
                  //There exist now 3 different scenarios, given the actual cell and face:
                  // The neighbour behind this face is [ more | as much | less] refined
                  // than/as the actual cell. We have to distinguish here only between the case 1
                  // and the other two, because these will be distinguished in in the FaceDataContainer.

                  if (cell[0]->neighbor(face)->has_children())
                  {
                    //first: neighbour is finer

                    for (unsigned int subface_no=0;
                        subface_no < cell[0]->face(face)->n_children();
                        ++subface_no)
                    {
                      //TODO Now here we have to initialise the subface_values on the
                      // actual cell and then the facevalues of the neighbours
                      fdc.ReInit(face, subface_no);
                      fdc.ReInitNbr();

                      pde.InterfaceEquation(fdc, local_cell_vector, 1., 1.);

                    }
                  }
                  else
                  {
                    // either neighbor is as fine as this cell or
                    // it is coarser

                    fdc.ReInit(face);
                    fdc.ReInitNbr();
                    pde.InterfaceEquation(fdc, local_cell_vector, 1., 1.);
                  }
                }                    //endif atinterface
              }                    //endfor face
            }                    //endif need_interfaces
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
        residual = 0.;
        // Begin integration
        unsigned int dofs_per_cell;
        dealii::Vector<SCALAR> local_cell_vector;
        std::vector<unsigned int> local_dof_indices;

        const bool need_point_rhs = pde.HasPoints();

        const auto& dof_handler =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto cell =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        // Initialize the data containers.
        GetIntegratorDataContainer().InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

        bool need_interfaces = pde.HasInterfaces();
        if(need_interfaces )
        {
          throw DOpEException(" Faces on multiple meshes not implemented yet!",
              "IntegratorMultiMesh::ComputeNonlinearRhs_Recursive");
        }
        bool need_faces = pde.HasFaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData());
        auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

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

        //check if we need the evaluation of PointRhs
        if (need_point_rhs)
        {
          VECTOR point_rhs;
          pde.PointRhs(this->GetParamData(), this->GetDomainData(), point_rhs,
              1.);
          residual += point_rhs;
        }

        if (apply_boundary_values)
        {
          ApplyNewtonBoundaryValues(pde, residual);
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
        auto cell =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        GetIntegratorDataContainer().InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

        //for the interface-case
        unsigned int nbr_dofs_per_cell;
        std::vector<unsigned int> nbr_local_dof_indices;

        bool need_faces = pde.HasFaces();
        bool need_interfaces = pde.HasInterfaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            need_interfaces);
        auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

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

          dealii::FullMatrix<SCALAR> local_cell_matrix(dofs_per_cell,
              dofs_per_cell);
          local_cell_matrix = 0;

          local_dof_indices.resize(0);
          local_dof_indices.resize(dofs_per_cell, 0);
          pde.CellMatrix(cdc, local_cell_matrix);

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
              //auto face_it = cell[0]->face(face);
              // first, check if we are at an interface, i.e. not the neighbour exists and
              // it has a different material_id than the actual cell
              if(AtInterface(cell, face))
              {
                //There exist now 3 different scenarios, given the actual cell and face:
                // The neighbour behind this face is [ more | as much | less] refined
                // than/as the actual cell. We have to distinguish here only between the case 1
                // and the other two, because these will be distinguished in in the FaceDataContainer.

                if (cell[0]->neighbor(face)->has_children())
                {
                  //first: neighbour is finer

                  for (unsigned int subface_no=0;
                      subface_no < cell[0]->face(face)->n_children();
                      ++subface_no)
                  {
                    //TODO Now here we have to initialise the subface_values on the
                    // actual cell and then the facevalues of the neighbours
                    fdc.ReInit(face, subface_no);
                    fdc.ReInitNbr();

                    //TODO auslagern?
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
                }
                else
                {
                  // either neighbor is as fine as this cell or it is coarser
                  fdc.ReInit(face);
                  fdc.ReInitNbr();

                  //TODO auslagern?
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
              } //endif atinterface

            }
          } //endif need_interfaces

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
          auto cell =
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc =
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
          GetIntegratorDataContainerFunc().InitializeCDC(pde.GetUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
              this->GetParamData(), this->GetDomainData());
          auto& cdc = GetIntegratorDataContainerFunc().GetCellDataContainer();

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
          PROBLEM& pde
          )
      {
        {
          SCALAR ret = 0.;
          // Begin integration
          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          GetIntegratorDataContainerFunc().InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData());
          auto & fdc = GetIntegratorDataContainerFunc().GetFaceDataContainer();

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

          return ret;

        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeFaceScalar(
          PROBLEM& pde
          )
      {

        {
          SCALAR ret = 0.;
          // Begin integration
          const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
          auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
          auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

          GetIntegratorDataContainerFunc().InitializeFDC(pde.GetFaceUpdateFlags(),
              *(pde.GetBaseProblem().GetSpaceTimeHandler()),
              cell,
              this->GetParamData(),
              this->GetDomainData());
          auto & fdc = GetIntegratorDataContainerFunc().GetFaceDataContainer();

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
        //pde.GetDoFConstraints().condense(u);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues( pde.GetBaseProblem().GetSpaceTimeHandler()->GetMapping(),
              pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0],
              color,
              pde.GetDirichletValues(color, this->GetParamData(),
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

        pde.GetDoFConstraints().condense(u);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues( pde.GetBaseProblem().GetSpaceTimeHandler()->GetMapping(),
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
        pde.GetDoFConstraints().condense(rhs);
        pde.GetDoFConstraints().condense(matrix);
        std::vector<unsigned int> dirichlet_colors = pde.GetDirichletColors();
        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
        {
          unsigned int color = dirichlet_colors[i];
          std::vector<bool> comp_mask = pde.GetDirichletCompMask(color);
          std::map<unsigned int, SCALAR> boundary_values;

          InterpolateBoundaryValues( pde.GetBaseProblem().GetSpaceTimeHandler()->GetMapping(),
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
        throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "Integrator::AddDomainData");
      }
      _domain_data.insert(
          std::pair<std::string, const VECTOR*>(name, new_data));
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
        throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "Integrator::DeleteDomainData");
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
        throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "Integrator::AddParamData");
      }
      _param_data.insert(
          std::pair<std::string, const dealii::Vector<SCALAR>*>(name,
              new_data));
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteParamData(
        std::string name)
    {
      typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator it =
          _param_data.find(name);
      if (it == _param_data.end())
      {
        throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "Integrator::DeleteParamData");
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
    template<typename PROBLEM, class STH, class CDC, class FDC>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeRefinementIndicators(
          PROBLEM& pde,
          DWRDataContainer<STH, INTEGRATORDATACONT, CDC, FDC, VECTOR>& dwrc)
      {
        unsigned int dofs_per_cell;

        //for primal and dual part of the error
        std::vector<double> cell_sum(2, 0);
        cell_sum.resize(2, 0);

        // Begin integration
        const auto& dof_handler =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto cell =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        const auto& dof_handler_weight = dwrc.GetWeightSTH().GetDoFHandler();
        auto cell_weight = dwrc.GetWeightSTH().GetDoFHandlerBeginActive();
        auto endc_high = dwrc.GetWeightSTH().GetDoFHandlerEnd();

        // Generate the data containers. Notice that we use the quadrature
        //formula from the higher order idc!.
        GetIntegratorDataContainer().InitializeCDC(
            dwrc.GetWeightIDC().GetQuad(), pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

        dwrc.GetWeightIDC().InitializeCDC(pde.GetUpdateFlags(),
            dwrc.GetWeightSTH(), cell_weight, this->GetParamData(),
            dwrc.GetWeightData());
        auto& cdc_weight = dwrc.GetCellWeight();

        // we want to integrate the face-terms only once, so
        // we store the values on each face in this map
        // and distribute it at the end to the adjacent cells.
        typename std::map<typename dealii::Triangulation<dim>::face_iterator,std::vector<double> >
        face_integrals;
        // initialize the map with a big value to make sure
        // that we take notice if we forget to add a face
        // during the error estimation process
        auto cell_it = cell[0];
        std::vector<double> face_init(2,-1e20);
        for (; cell_it != endc[0]; cell_it++)
        {
          for (unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
          {
            face_integrals[cell_it->face(face_no)] = face_init;
          }
        }

//        bool need_faces = pde.HasFaces();
//        bool need_interfaces = pde.HasInterfaces();
//        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
//        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        GetIntegratorDataContainer().InitializeFDC(dwrc.GetWeightIDC().GetFaceQuad(),
            pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            true);
        auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

        dwrc.GetWeightIDC().InitializeFDC(pde.GetFaceUpdateFlags(),
            dwrc.GetWeightSTH(),
            cell_weight,
            this->GetParamData(),
            dwrc.GetWeightData(),
            true);

        for (unsigned int cell_index = 0; cell[0] != endc[0];
            cell[0]++, cell_index++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            if (cell[dh] == endc[dh])
            {
              throw DOpEException(
                  "Cellnumbers in DoFHandlers are not matching!",
                  "Integrator::ComputeRefinementIndicators");
            }
          }
          for (unsigned int dh = 0; dh < dof_handler_weight.size(); dh++)
          {
            if (cell_weight[dh] == endc_high[dh])
            {
              throw DOpEException(
                  "Cellnumbers in DoFHandlers are not matching!",
                  "Integrator::ComputeRefinementIndicators");
            }
          }
          cell_sum.clear();
          cell_sum.resize(2, 0);

          cdc.ReInit();
          cdc_weight.ReInit();

          //first the cell-residual
          pde.CellErrorContribution(cdc, dwrc, cell_sum, 1., 1.);
          dwrc.GetPrimalErrorIndicators()(cell_index) = cell_sum[0];
          dwrc.GetDualErrorIndicators()(cell_index) = cell_sum[1];
          cell_sum.clear();
          cell_sum.resize(2, 0);

          //Now to the face terms. We compute them only once for each face and distribute the
          //afterwards. We choose always to work from the coarser cell, if both neigbors of the
          //face are on the same level, we pick the one with the lower index
          for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
            auto face_it = cell[0]->face(face);

            //check if the face lies at a boundary
            if(face_it->at_boundary())
            {
              fdc.ReInit(face);
              dwrc.GetFaceWeight().ReInit(face);
              pde.BoundaryErrorContribution(fdc, dwrc, cell_sum, 1.);

              Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                  ExcInternalError());
              Assert (face_integrals[cell[0]->face(face)] == face_init,
                  ExcInternalError());

              face_integrals[cell[0]->face(face)] = cell_sum;
              cell_sum.clear();
              cell_sum.resize(2,0.);
            }
            else
            {
              //There exist now 3 different scenarios, given the actual cell and face:
              // The neighbour behind this face is [ more | as much | less] refined
              // than/as the actual cell. We have to distinguish here only between the case 1
              // and the other two, because these will be distinguished in in the FaceDataContainer.
              if (cell[0]->neighbor(face)->has_children())
              {
                //first: neighbour is finer
                std::vector<double> sum(2,0.);
                for (unsigned int subface_no=0;
                    subface_no < cell[0]->face(face)->n_children();
                    ++subface_no)
                {
                  //TODO Now here we have to initialise the subface_values on the
                  // actual cell and then the facevalues of the neighbours
                  fdc.ReInit(face, subface_no);
                  fdc.ReInitNbr();
                  dwrc.GetFaceWeight().ReInit(face, subface_no);

                  pde.FaceErrorContribution(fdc, dwrc, cell_sum, 1.);
                  sum[0] += cell_sum[0];
                  sum[1] += cell_sum[1];
                  face_integrals[cell[0]->neighbor_child_on_subface(face, subface_no)
                  ->face(cell[0]->neighbor_of_neighbor(face))] = cell_sum;
                  cell_sum.clear();
                  cell_sum.resize(2,0);
                }

                Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                    ExcInternalError());
                Assert (face_integrals[cell[0]->face(face)] == face_init,
                    ExcInternalError());

                face_integrals[cell[0]->face(face)] = sum;
                cell_sum.clear();
                cell_sum.resize(2,0);
              }
              else
              {
                // either neighbor is as fine as this cell or
                // it is coarser
                Assert(cell[0]->neighbor(face)->level() <= cell[0]->level(),ExcInternalError());
                //now we work always from the coarser cell. if both cells
                //are on the same level, we pick the one with the lower index
                if(cell[0]->level() == cell[0]->neighbor(face)->level()
                    && cell[0]->index() < cell[0]->neighbor(face)->index())
                {
                  fdc.ReInit(face);
                  fdc.ReInitNbr();
                  dwrc.GetFaceWeight().ReInit(face);

                  pde.FaceErrorContribution(fdc, dwrc, cell_sum, 1.);
                  Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                      ExcInternalError());
                  Assert (face_integrals[cell[0]->face(face)] == face_init,
                      ExcInternalError());

                  face_integrals[cell[0]->face(face)] = cell_sum;
                  cell_sum.clear();
                  cell_sum.resize(2,0);
                }
              }
            }
          }                  //endfor faces
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            cell[dh]++;
          }
          for (unsigned int dh = 0; dh < dof_handler_weight.size(); dh++)
          {
            cell_weight[dh]++;
          }
        }                  //endfor cell
        //now we have to incorporate the face and boundary_values
        //into
        unsigned int present_cell = 0;
        cell =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        for (;
            cell[0] !=endc[0]; cell[0]++, ++present_cell)
        {
          for (unsigned int face_no = 0;
              face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
          {
            Assert(
                face_integrals.find(cell[0]->face(face_no)) != face_integrals.end(),
                ExcInternalError());
            if(cell[0]->face(face_no)->at_boundary())
            {
              dwrc.GetPrimalErrorIndicators()(present_cell) +=
              face_integrals[cell[0]->face(face_no)][0];
              dwrc.GetDualErrorIndicators()(present_cell) +=
              face_integrals[cell[0]->face(face_no)][1];
            }
            else
            {
              dwrc.GetPrimalErrorIndicators()(present_cell) +=
              0.5 * face_integrals[cell[0]->face(face_no)][0];
              dwrc.GetDualErrorIndicators()(present_cell) +=
              0.5 * face_integrals[cell[0]->face(face_no)][1];
            }

          }
        }
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeRefinementIndicators(
          PROBLEM& pde,
          ResidualErrorContainer<VECTOR>& dwrc)
      {
        //for primal and dual part of the error
        std::vector<double> cell_sum(2, 0);
        cell_sum.resize(2, 0);
        // Begin integration
        const auto& dof_handler =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto cell =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

        {
          //Add Weights
          auto wd = dwrc.GetWeightData().begin();
          auto wend = dwrc.GetWeightData().end();
          for (; wd != wend; wd++)
          {
            AddDomainData(wd->first, wd->second);
          }
        }

        // Generate the data containers. Notice that we use the quadrature
        // formula from the higher order idc!.
        GetIntegratorDataContainer().InitializeCDC(pde.GetUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
            this->GetParamData(), this->GetDomainData());
        auto& cdc = GetIntegratorDataContainer().GetCellDataContainer();

        //we want to integrate the face-terms only once
        typename std::map<typename dealii::Triangulation<dim>::face_iterator,std::vector<double> >
        face_integrals;
        //initialize the map
        auto cell_it = cell[0];
        std::vector<double> face_init(2,-1e20);
        for (; cell_it != endc[0]; cell_it++)
        {
          for (unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
          {
            face_integrals[cell_it->face(face_no)] = face_init;
          }
        }

//        bool need_faces = pde.HasFaces();
//        bool need_interfaces = pde.HasInterfaces();
//        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
//        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        GetIntegratorDataContainer().InitializeFDC(
            pde.GetFaceUpdateFlags(),
            *(pde.GetBaseProblem().GetSpaceTimeHandler()),
            cell,
            this->GetParamData(),
            this->GetDomainData(),
            true);
        auto & fdc = GetIntegratorDataContainer().GetFaceDataContainer();

        for (unsigned int cell_index = 0; cell[0] != endc[0];
            cell[0]++, cell_index++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            if (cell[dh] == endc[dh])
            {
              throw DOpEException(
                  "Cellnumbers in DoFHandlers are not matching!",
                  "Integrator::ComputeRefinementIndicators");
            }
          }

          cell_sum.clear();
          cell_sum.resize(2, 0);

          cdc.ReInit();
          dwrc.InitCell(cell[0]->diameter());
          //first the cell-residual
          pde.CellErrorContribution(cdc, dwrc, cell_sum, 1., 1.);
          dwrc.GetPrimalErrorIndicators()(cell_index) = cell_sum[0];
          dwrc.GetDualErrorIndicators()(cell_index) = cell_sum[1];
          cell_sum.clear();
          cell_sum.resize(2, 0);
          //Now to the face terms. We compute them only once for each face and distribute the
          //afterwards. We choose always to work from the coarser cell, if both neigbors of the
          //face are on the same level, we pick the one with the lower index

          for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
            auto face_it = cell[0]->face(face);

            //check if the face lies at a boundary
            if(face_it->at_boundary())
            {
              fdc.ReInit(face);
              dwrc.InitFace(cell[0]->face(face)->diameter());

              pde.BoundaryErrorContribution(fdc, dwrc, cell_sum, 1.);

              Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                  ExcInternalError());
              Assert (face_integrals[cell[0]->face(face)] == face_init,
                  ExcInternalError());
              face_integrals[cell[0]->face(face)] = cell_sum;
              cell_sum.clear();
              cell_sum.resize(2,0.);
            }
            else
            {
              //There exist now 3 different scenarios, given the actual cell and face:
              // The neighbour behind this face is [ more | as much | less] refined
              // than/as the actual cell. We have to distinguish here only between the case 1
              // and the other two, because these will be distinguished in in the FaceDataContainer.
              if (cell[0]->neighbor(face)->has_children())
              {
                //first: neighbour is finer
                std::vector<double> sum(2,0.);
                for (unsigned int subface_no=0;
                    subface_no < cell[0]->face(face)->n_children();
                    ++subface_no)
                {
                  //TODO Now here we have to initialise the subface_values on the
                  // actual cell and then the facevalues of the neighbours
                  fdc.ReInit(face, subface_no);
                  fdc.ReInitNbr();
                  dwrc.InitFace(cell[0]->face(face)->diameter());

                  pde.FaceErrorContribution(fdc, dwrc, cell_sum, 1.);
                  sum[0]= cell_sum[0];
                  sum[1]= cell_sum[1];
                  cell_sum.clear();
                  cell_sum.resize(2,0);
                  face_integrals[cell[0]->neighbor_child_on_subface(face, subface_no)
                  ->face(cell[0]->neighbor_of_neighbor(face))] = cell_sum;
                  cell_sum.clear();
                  cell_sum.resize(2,0.);
                }

                Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                    ExcInternalError());
                Assert (face_integrals[cell[0]->face(face)] == face_init,
                    ExcInternalError());

                face_integrals[cell[0]->face(face)] = sum;

              }
              else
              {
                // either neighbor is as fine as this cell or
                // it is coarser
                Assert(cell[0]->neighbor(face)->level() <= cell[0]->level(),ExcInternalError());
                //now we work always from the coarser cell. if both cells
                //are on the same level, we pick the one with the lower index
                if(cell[0]->level() == cell[0]->neighbor(face)->level()
                    && cell[0]->index() < cell[0]->neighbor(face)->index())
                {
                  fdc.ReInit(face);
                  fdc.ReInitNbr();
                  dwrc.InitFace(cell[0]->face(face)->diameter());

                  pde.FaceErrorContribution(fdc, dwrc, cell_sum, 1.);
                  Assert (face_integrals.find (cell[0]->face(face)) != face_integrals.end(),
                      ExcInternalError());
                  Assert (face_integrals[cell[0]->face(face)] == face_init,
                      ExcInternalError());

                  face_integrals[cell[0]->face(face)] = cell_sum;
                  cell_sum.clear();
                  cell_sum.resize(2,0);
                }
              }
            }
          }                  //endfor faces
//          }//end else

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
          {
            cell[dh]++;
          }
        }                  //endfor cell
        //now we have to incorporate the face and boundary_values
        //into
        unsigned int present_cell = 0;
        cell =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        for (;
            cell[0] !=endc[0]; cell[0]++, ++present_cell)
        for (unsigned int face_no = 0;
            face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          Assert(
              face_integrals.find(cell[0]->face(face_no)) != face_integrals.end(),
              ExcInternalError());
          dwrc.GetPrimalErrorIndicators()(present_cell) +=
          0.5 * face_integrals[cell[0]->face(face_no)][0];
          dwrc.GetDualErrorIndicators()(present_cell) +=
          0.5 * face_integrals[cell[0]->face(face_no)][1];
        }
        {
          //Remove Weights
          auto wd = dwrc.GetWeightData().begin();
          auto wend = dwrc.GetWeightData().end();
          for (; wd != wend; wd++)
          {
            DeleteDomainData(wd->first);
          }
        }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    INTEGRATORDATACONT&
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainer() const
    {
      return _idc1;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    INTEGRATORDATACONT&
    Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainerFunc() const
    {
      return _idc2;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename DOFHANDLER>
      void
      Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::InterpolateBoundaryValues(
          const DOpEWrapper::Mapping<dim, DOFHANDLER>& mapping,
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER>* dof_handler,
          const unsigned int color, const dealii::Function<dim>& function,
          std::map<unsigned int, SCALAR>& boundary_values,
          const std::vector<bool>& comp_mask) const
      {
    //TODO: mapping[0] is a workaround, as deal does not support interpolate
    // boundary_values with a mapping collection at this point.
        dealii::VectorTools::interpolate_boundary_values(mapping[0],
            dof_handler->GetDEALDoFHandler(), color, function,
            boundary_values, comp_mask);
      }


}
#endif

