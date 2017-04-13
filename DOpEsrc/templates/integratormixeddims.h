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

#ifndef IntegratorMixed_H_
#define IntegratorMixed_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <vector>

#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/optproblemcontainer.h>
#if DEAL_II_VERSION_GTE(7,3,0)
#include <deal.II/base/types.h>
#endif

namespace DOpE
{
  /**
   * This class is used to integrate the righthand side, matrix and so on.
   * This class is used when the control is 0 dimensional and the state is
   * in dimension 1, 2, or 3. This is then used to ,,integrate'' residuals
   * for the 0 dim variable that may depend upon integrals over the
   * highdimensional domain for the other variable.
   *
   * For details on the functions see Integrator.
   *
   * @template INTEGRATORDATACONT       The type of the integratordatacontainer, which has
   *                                    manages the basic data for integration (quadrature,
   *                                    elementdatacontainer, facedatacontainer etc.)
   * @template VECTOR                   Class of the vectors which we use in the integrator.
   * @template SCALAR                   Type of the scalars we use in the integrator.
   * @template dimlow                   The dimension of the lowdimensional object (should be 0!)
   * @template dimhigh                  The dimension of the highdimensional object (should be 1,2, or 3)
   */
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  class IntegratorMixedDimensions
  {
  public:
    IntegratorMixedDimensions(INTEGRATORDATACONT &idc);

    ~IntegratorMixedDimensions();

    void ReInit();

    template<typename PROBLEM>
    void ComputeNonlinearResidual(PROBLEM &pde, VECTOR &residual);
    template<typename PROBLEM, typename MATRIX>
    void ComputeMatrix(PROBLEM &pde, MATRIX &matrix);
    template<typename PROBLEM>
    void
    ComputeNonlinearRhs(PROBLEM &pde, VECTOR &residual);

    template<typename PROBLEM>
    void ComputeLocalControlConstraints (PROBLEM &pde, VECTOR &constraints);
    template<typename PROBLEM>
    SCALAR ComputeDomainScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR ComputePointScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR ComputeBoundaryScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR ComputeFaceScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR ComputeAlgebraicScalar(PROBLEM &pde);

    template<typename PROBLEM>
    void ApplyInitialBoundaryValues(PROBLEM &pde, VECTOR &u);
    template<typename PROBLEM>
    void ApplyTransposedInitialBoundaryValues(PROBLEM &pde, VECTOR &u, SCALAR scale);


    inline void AddDomainData(std::string name, const VECTOR *new_data);
    inline void DeleteDomainData(std::string name);

    inline void AddParamData(std::string name, const dealii::Vector<SCALAR> *new_data);
    inline void DeleteParamData(std::string name);

    inline  const INTEGRATORDATACONT &GetIntegratorDataContainer() const;

  protected:
    inline const std::map<std::string, const VECTOR *> &GetDomainData() const;
    inline const std::map<std::string, const dealii::Vector<SCALAR>*> &GetParamData() const;

    inline void AddPresetRightHandSide(double s, dealii::Vector<SCALAR> &residual) const;

  private:
    INTEGRATORDATACONT &idc_;

    std::map<std::string, const VECTOR *> domain_data_;
    std::map<std::string, const dealii::Vector<SCALAR>*> param_data_;
  };

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dimlow, int dimhigh>
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow,
                            dimhigh>::IntegratorMixedDimensions(INTEGRATORDATACONT &idc) :
                              idc_(idc)
  {
  }

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::~IntegratorMixedDimensions()
  {

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ReInit()
  {

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeNonlinearResidual(PROBLEM &pde,
      VECTOR &residual)
  {
    {
      residual = 0.;
      // Begin integration
      const unsigned int dofs =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetControlNDoFs();

      dealii::Vector<SCALAR> local_vector(dofs);
      const auto &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

      // Initialize the data containers.
      idc_.InitializeEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
                         this->GetParamData(), this->GetDomainData());
      auto &edc = idc_.GetElementDataContainer();

      bool need_faces = pde.HasFaces();
      bool need_interfaces = pde.HasInterfaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
      idc_.InitializeFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element,
                         this->GetParamData(),
                         this->GetDomainData());
      auto &fdc = idc_.GetFaceDataContainer();

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!",
                                      "IntegratorMixedDimensions::ComputeNonlinearResidual");
                }
            }

          local_vector = 0;
          edc.ReInit();

          pde.ElementRhs(edc,local_vector, -1.);

          if (need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
#if DEAL_II_VERSION_GTE(8,3,0)
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                    {
                      fdc.ReInit(face);
                      pde.BoundaryRhs(fdc,local_vector,-1.);
                    }
                }
            }
          if (need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
                  if (element[0]->neighbor_index(face) != -1)
                    {
                      fdc.ReInit(face);
                      pde.FaceRhs(fdc,local_vector,-1.);
                    }
                }
            }
          if ( need_interfaces)
            {
              throw DOpEException("Interfaces not implemented!",
                                  "IntegratorMixedDimensions::ComputeNonlinearResidual");
            }
          //LocalToGlobal
          for (unsigned int i = 0; i < dofs; ++i)
            {
              residual(i) += local_vector(i);
            }

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }
      //The Equation should not be space dependend.
      local_vector = 0;
      pde.ElementEquation(edc, local_vector, 1., 1.);

      for (unsigned int i = 0; i < dofs; ++i)
        {
          residual(i) += local_vector(i);
        }

      if (pde.HasControlInDirichletData())
        {
          ApplyTransposedInitialBoundaryValues(pde,residual, -1.);
        }
      //Check if some preset righthandside exists.
      local_vector = 0;
      AddPresetRightHandSide(-1.,local_vector);
      for (unsigned int i = 0; i < dofs; ++i)
        {
          residual(i) += local_vector(i);
        }

//      if (apply_boundary_values)
//      {
//        ApplyNewtonBoundaryValues(pde,residual);
//      }
    }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeNonlinearRhs(PROBLEM &pde,
      VECTOR &residual)
  {
    {
      residual = 0.;
      // Begin integration
      const unsigned int dofs =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetControlNDoFs();

      dealii::Vector<SCALAR> local_vector(dofs);
      const auto &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

      // Initialize the data containers.
      idc_.InitializeEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
                         this->GetParamData(), this->GetDomainData());
      auto &edc = idc_.GetElementDataContainer();

      bool need_faces = pde.HasFaces();
      bool need_interfaces = pde.HasInterfaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
      idc_.InitializeFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element,
                         this->GetParamData(),
                         this->GetDomainData());
      auto &fdc = idc_.GetFaceDataContainer();

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!",
                                      "IntegratorMixedDimensions::ComputeNonlinearResidual");
                }
            }

          local_vector = 0;
          edc.ReInit();

          pde.ElementRhs(edc,local_vector, 1.);

          if (need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
#if DEAL_II_VERSION_GTE(8,3,0)
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                    {
                      fdc.ReInit(face);
                      pde.BoundaryRhs(fdc,local_vector,1.);
                    }
                }
            }
          if (need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
                  if (element[0]->neighbor_index(face) != -1)
                    {
                      fdc.ReInit(face);
                      pde.FaceRhs(fdc,local_vector,1.);
                    }
                }
            }
          if ( need_interfaces)
            {
              throw DOpEException("Interfaces not implemented!",
                                  "IntegratorMixedDimensions::ComputeNonlinearRhs");
            }
          //LocalToGlobal
          for (unsigned int i = 0; i < dofs; ++i)
            {
              residual(i) += local_vector(i);
            }

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }

      if (pde.HasControlInDirichletData())
        {
          ApplyTransposedInitialBoundaryValues(pde,residual, -1.);
        }
      //Check if some preset righthandside exists.
      local_vector = 0;
      AddPresetRightHandSide(-1.,local_vector);
      for (unsigned int i = 0; i < dofs; ++i)
        {
          residual(i) += local_vector(i);
        }

//      if (apply_boundary_values)
//      {
//        ApplyNewtonBoundaryValues(pde,residual);
//      }
    }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM, typename MATRIX>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeMatrix(PROBLEM &pde, MATRIX &matrix)
  {
    throw DOpEException("You should not use this function, try VoidLinearSolver instead.",
                        "IntegratorMixedDimensions::ComputeMatrix");
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeLocalControlConstraints (PROBLEM &pde, VECTOR &constraints)
  {
    constraints = 0.;
    pde.ComputeLocalControlConstraints(constraints,this->GetParamData(),this->GetDomainData());
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dimlow, int dimhigh>
  template<typename PROBLEM>
  SCALAR
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow,
                            dimhigh>::ComputeDomainScalar(PROBLEM &pde)
  {
    {
      SCALAR ret = 0.;
      const unsigned int n_q_points = this->GetQuadratureFormula()->size();

      const auto &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

      idc_.InitializeEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
                         this->GetParamData(), this->GetDomainData());
      auto &edc = idc_.GetElementDataContainer();

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!",
                                      "IntegratorMixedDimensions::ComputeDomainScalar");
                }
            }

          ret += pde.ElementFunctional(edc);

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }
      return ret;
    }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
  int dimhigh>
  template<typename PROBLEM>
  SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputePointScalar(PROBLEM &pde)
  {
    if (pde.GetFEValuesNeededToBeInitialized())
      {
        this->InitializeFEValues();
      }

    {
      SCALAR ret = 0.;

      ret += pde.PointFunctional(this->GetParamData(), this->GetDomainData());

      return ret;
    }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeBoundaryScalar(PROBLEM &pde)
  {

    {
      SCALAR ret = 0.;
      // Begin integration

      const std::vector<const DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler >*> &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler >::active_element_iterator>
      element(dof_handler.size());
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler>::active_element_iterator>
      endc(dof_handler.size());

      for (unsigned int dh = 0; dh < dof_handler.size(); dh++)
        {
          element[dh] = dof_handler[dh]->begin_active();
          endc[dh] = dof_handler[dh]->end();
        }

      // Generate the data containers.
      FaceDataContainer<dealii::DoFHandler, VECTOR, dimhigh> fdc(*(this->GetFaceQuadratureFormula()),
                                                                 pde.GetFaceUpdateFlags(),
                                                                 *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
                                                                 this->GetParamData(), this->GetDomainData());

      std::vector<unsigned int> boundary_functional_colors =
        pde.GetBoundaryFunctionalColors();
      bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
      if (!need_boundary_integrals)
        {
          throw DOpEException("No boundary colors given!",
                              "IntegratorMixedDimensions::ComputeBoundaryScalar");
        }

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!",
                                      "IntegratorMixedDimensions::ComputeBoundaryScalar");
                }
            }

          if (need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
#if DEAL_II_VERSION_GTE(8,3,0)
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                            element[0]->face(face)->boundary_id()) != boundary_functional_colors.end()))
#else
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                            element[0]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
#endif
                    {
//              pde.GetBaseProblem().GetSpaceTimeHandler()->ComputeFaceFEValues(element, face, pde.GetType());
                      fdc.ReInit(face);
                      ret += pde.BoundaryFunctional(fdc);
                    }
                }
            }
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }
      return ret;
    }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeFaceScalar(PROBLEM &pde)
  {
    {
      SCALAR ret = 0.;

      // Begin integration
      const std::vector<const DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler >*> &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler>::active_element_iterator>
      element(dof_handler.size());
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler>::active_element_iterator>
      endc(dof_handler.size());

      for (unsigned int dh = 0; dh < dof_handler.size(); dh++)
        {
          element[dh] = dof_handler[dh]->begin_active();
          endc[dh] = dof_handler[dh]->end();
        }
      // Generate the data containers.
      FaceDataContainer<dealii::DoFHandler, VECTOR, dimhigh> fdc(*(this->GetFaceQuadratureFormula()),
                                                                 pde.GetFaceUpdateFlags(),
                                                                 *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
                                                                 this->GetParamData(), this->GetDomainData());

      bool need_faces = pde.HasFaces();
      if (!need_faces)
        {
          throw DOpEException("No faces required!", "IntegratorMixedDimensions::ComputeFaceScalar");
        }

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!",
                                      "IntegratorMixedDimensions::ComputeFaceScalar");
                }
            }

          if (need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
                {
                  fdc.ReInit(face);
                  ret +=pde.FaceFunctional(fdc);
                }
            }
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }
      return ret;
    }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeAlgebraicScalar(PROBLEM &pde)
  {
    SCALAR ret = 0.;
    ret = pde.AlgebraicFunctional(this->GetParamData(), this->GetDomainData());
    return ret;
  }
  /*******************************************************************************************/
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ApplyInitialBoundaryValues(PROBLEM & /*pde*/,
      VECTOR &/*u*/)
  {
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>
  ::ApplyTransposedInitialBoundaryValues(PROBLEM &pde,
                                         VECTOR &u,
                                         SCALAR scale)
  {
    //Das macht nur sinn, wenn es um "Transponierte Dirichletdaten geht.
    unsigned int dofs =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetControlNDoFs();
    dealii::Vector<SCALAR> local_vector(dofs);

    std::vector<unsigned int> dirichlet_colors = pde.GetTransposedDirichletColors();
    std::vector<bool> selected_components;
    if (dirichlet_colors.size() > 0)
      {
        selected_components.resize(
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0]->n_dofs());
        const std::vector<Point<dimhigh> > &support_points =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetMapDoFToSupportPoints();

        for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
          {
            unsigned int color = dirichlet_colors[i];
            std::vector<bool> comp_mask = pde.GetTransposedDirichletCompMask(color);
            std::vector<bool> current_comp(comp_mask.size(), false);
#if DEAL_II_VERSION_GTE(7,3,0)
            std::set<types::boundary_id> boundary_indicators;
#else
            std::set<unsigned char> boundary_indicators;
#endif
            boundary_indicators.insert(color);
            for (unsigned int j = 0; j < comp_mask.size(); j++)
              {
                if (j > 0)
                  current_comp[j - 1] = false;
                if (comp_mask[j])
                  {
                    current_comp[j] = true;
                    //Hole eine Liste der DoFs auf dem Rand und die zugehoerigen Knoten
                    DoFTools::extract_boundary_dofs(pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0]->GetDEALDoFHandler(),
                                                    current_comp, selected_components, boundary_indicators);
                  }
                const TransposedDirichletDataInterface<dimhigh> &DD =
                  pde.GetTransposedDirichletValues(color,
                                                   this->GetParamData(),
                                                   this->GetDomainData());
                for (unsigned int k = 0; k < selected_components.size(); k++)
                  {
                    if (selected_components[k])
                      {
                        local_vector = 0.;
                        DD.value(support_points[k], j, k, local_vector);
                        for (unsigned int l = 0; l < dofs; ++l)
                          {
                            u(l) += scale * local_vector(l);
                          }
                      }
                  }
              }
            //end loop over components
          }
      }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::AddDomainData(
    std::string name,
    const VECTOR *new_data)
  {
    if (domain_data_.find(name) != domain_data_.end())
      {
        throw DOpEException("Adding multiple Data with name " + name + " is prohibited!",
                            "IntegratorMixedDimensions::AddDomainData");
      }
    domain_data_.insert(std::pair<std::string, const VECTOR *>(name, new_data));
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::DeleteDomainData(
    std::string name)
  {
    typename std::map<std::string, const VECTOR *>::iterator it = domain_data_.find(name);
    if (it == domain_data_.end())
      {
        throw DOpEException("Deleting Data " + name + " is impossible! Data not found",
                            "IntegratorMixedDimensions::DeleteDomainData");
      }
    domain_data_.erase(it);
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  const std::map<std::string, const VECTOR *> &IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR,
        SCALAR, dimlow, dimhigh>::GetDomainData() const
  {
    return domain_data_;
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
  int dimhigh>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::AddParamData(
    std::string name,
    const dealii::Vector<
    SCALAR>* new_data)
  {
    if (param_data_.find(name) != param_data_.end())
      {
        throw DOpEException("Adding multiple Data with name " + name + " is prohibited!",
                            "IntegratorMixedDimensions::AddParamData");
      }
    param_data_.insert(std::pair<std::string, const dealii::Vector<SCALAR>*>(name, new_data));
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::DeleteParamData(
    std::string name)
  {
    typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator it =
      param_data_.find(name);
    if (it == param_data_.end())
      {
        throw DOpEException("Deleting Data " + name + " is impossible! Data not found",
                            "IntegratorMixedDimensions::DeleteParamData");
      }
    param_data_.erase(it);
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
           int dimhigh>
  const std::map<std::string, const dealii::Vector<SCALAR>*> &IntegratorMixedDimensions<
  INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::GetParamData() const
  {
    return param_data_;
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dimlow, int dimhigh>
  const INTEGRATORDATACONT &
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::GetIntegratorDataContainer() const
  {
    return idc_;
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dimlow, int dimhigh>
  void
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::AddPresetRightHandSide(double s,
      dealii::Vector<SCALAR> &residual) const
  {
    typename std::map<std::string, const dealii::Vector<SCALAR>*>::const_iterator it =
      param_data_.find("fixed_rhs");
    if (it != param_data_.end())
      {
        assert(residual.size() == it->second->size());
        residual.add(s,*(it->second));
      }
  }

  /*******************************************************************************************/
}
#endif

