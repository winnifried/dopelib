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

#ifndef _IntegratorMixed_H_
#define _IntegratorMixed_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include <numerics/vectors.h>
#include <numerics/matrices.h>

#include <base/function.h>

#include <dofs/dof_tools.h>

#include <fe/mapping_q1.h>

#include <vector>

#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "optproblemcontainer.h"

namespace DOpE
{

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
class IntegratorMixedDimensions 
{
  public:
    IntegratorMixedDimensions(INTEGRATORDATACONT& idc);

    ~IntegratorMixedDimensions();

    /**
     This Function should be called once after grid refinement, or changes in boundary values
     to  recompute sparsity patterns, and constraint matrices.
     */
    void ReInit();

    template<typename PROBLEM>
    void ComputeNonlinearResidual(PROBLEM& pde, VECTOR &residual, bool apply_boundary_values = true);
    template<typename PROBLEM, typename MATRIX>
    void ComputeMatrix(PROBLEM& pde, MATRIX &matrix);
    template<typename PROBLEM>
      void
      ComputeNonlinearRhs(PROBLEM& pde, VECTOR &residual, bool apply_boundary_values = true);

    template<typename PROBLEM>
      void ComputeLocalControlConstraints (PROBLEM& pde, VECTOR &constraints);
    template<typename PROBLEM>
    SCALAR ComputeDomainScalar(PROBLEM& pde);
    template<typename PROBLEM>
    SCALAR ComputePointScalar(PROBLEM& pde);
    template<typename PROBLEM>
    SCALAR ComputeBoundaryScalar(PROBLEM& pde);
    template<typename PROBLEM>
    SCALAR ComputeFaceScalar(PROBLEM& pde);
    template<typename PROBLEM>
    SCALAR ComputeAlgebraicScalar(PROBLEM& pde);

    template<typename PROBLEM>
    void ApplyInitialBoundaryValues(PROBLEM& pde, VECTOR &u);
    template<typename PROBLEM>
    void ApplyTransposedInitialBoundaryValues(PROBLEM& pde, VECTOR &u, SCALAR scale);
    template<typename PROBLEM>
    void ApplyNewtonBoundaryValues(PROBLEM& pde, VECTOR &u);
    template<typename PROBLEM, typename MATRIX>
    void ApplyNewtonBoundaryValues(PROBLEM& pde, MATRIX &matrix, VECTOR &rhs, VECTOR &sol);

    inline void AddDomainData(std::string name, const VECTOR* new_data);
    inline void DeleteDomainData(std::string name);
    inline const std::map<std::string, const VECTOR*>& GetDomainData() const;

    inline void AddParamData(std::string name, const dealii::Vector<SCALAR>* new_data);
    inline void DeleteParamData(std::string name);
    inline const std::map<std::string, const dealii::Vector<SCALAR>*>& GetParamData() const;

    inline  const INTEGRATORDATACONT& GetIntegratorDataContainer() const;
  private:
    INTEGRATORDATACONT & _idc;

    std::map<std::string, const VECTOR*> _domain_data;
    std::map<std::string, const dealii::Vector<SCALAR>*> _param_data;
};

/**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dimlow, int dimhigh>
    IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow,
        dimhigh>::IntegratorMixedDimensions(INTEGRATORDATACONT& idc) :
      _idc(idc)
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
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeNonlinearResidual(PROBLEM& pde,
												    VECTOR &residual,
												    bool apply_boundary_values)
{
  {
      residual = 0.;
      // Begin integration
      const unsigned int dofs =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetControlNDoFs();

      dealii::Vector<SCALAR> local_cell_vector(dofs);
      const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

     // Initialize the data containers.
      _idc.InitializeCDC(pde.GetUpdateFlags(),
                *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
                this->GetParamData(), this->GetDomainData());
      auto& cdc = _idc.GetCellDataContainer();

      bool need_faces = pde.HasFaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
      _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
                *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                cell,
                this->GetParamData(),
                this->GetDomainData());
      auto & fdc = _idc.GetFaceDataContainer();

      for (; cell[0] != endc[0]; cell[0]++)
      {
        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          if (cell[dh] == endc[dh])
          {
            throw DOpEException("Cellnumbers in DoFHandlers are not matching!",
                                "IntegratorMixedDimensions::ComputeNonlinearResidual");
          }
        }

        local_cell_vector = 0;
	cdc.ReInit();

        pde.CellRhs(cdc,local_cell_vector, -1.);

        if(need_boundary_integrals)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            if (cell[0]->face(face)->at_boundary()
                &&
                (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
            {
	      fdc.ReInit(face);
	      pde.BoundaryRhs(fdc,local_cell_vector,-1.);
            }
          }
        }
        if(need_faces)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            if (cell[0]->neighbor_index(face) != -1)
            {
              fdc.ReInit(face);
	      pde.FaceRhs(fdc,local_cell_vector,-1.);
            }
          }
        }
        //LocalToGlobal
        for (unsigned int i = 0; i < dofs; ++i)
        {
          residual(i) += local_cell_vector(i);
        }

        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          cell[dh]++;
        }
      }
      //The Equation should not be space dependend.
      local_cell_vector = 0;
      pde.CellEquation(cdc, local_cell_vector, 1., 1.);

      for (unsigned int i = 0; i < dofs; ++i)
      {
        residual(i) += local_cell_vector(i);
      }

      if (pde.HasControlInDirichletData())
      {
        ApplyTransposedInitialBoundaryValues(pde,residual, -1.);
      }

      if (apply_boundary_values)
      {
        ApplyNewtonBoundaryValues(pde,residual);
      }
  }
}
/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeNonlinearRhs(PROBLEM& pde,
												    VECTOR &residual,
												    bool apply_boundary_values)
{
  {
      residual = 0.;
      // Begin integration
      const unsigned int dofs =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetControlNDoFs();

      dealii::Vector<SCALAR> local_cell_vector(dofs);
      const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

     // Initialize the data containers.
      _idc.InitializeCDC(pde.GetUpdateFlags(),
                *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
                this->GetParamData(), this->GetDomainData());
      auto& cdc = _idc.GetCellDataContainer();

      bool need_faces = pde.HasFaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
      _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
                *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                cell,
                this->GetParamData(),
                this->GetDomainData());
      auto & fdc = _idc.GetFaceDataContainer();

      for (; cell[0] != endc[0]; cell[0]++)
      {
        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          if (cell[dh] == endc[dh])
          {
            throw DOpEException("Cellnumbers in DoFHandlers are not matching!",
                                "IntegratorMixedDimensions::ComputeNonlinearResidual");
          }
        }

        local_cell_vector = 0;
	cdc.ReInit();

        pde.CellRhs(cdc,local_cell_vector, 1.);

        if(need_boundary_integrals)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            if (cell[0]->face(face)->at_boundary()
                &&
                (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
            {
	      fdc.ReInit(face);
	      pde.BoundaryRhs(fdc,local_cell_vector,1.);
            }
          }
        }
        if(need_faces)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            if (cell[0]->neighbor_index(face) != -1)
            {
              fdc.ReInit(face);
	      pde.FaceRhs(fdc,local_cell_vector,1.);
            }
          }
        }
        //LocalToGlobal
        for (unsigned int i = 0; i < dofs; ++i)
        {
          residual(i) += local_cell_vector(i);
        }

        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          cell[dh]++;
        }
      }

      if (pde.HasControlInDirichletData())
      {
        ApplyTransposedInitialBoundaryValues(pde,residual, -1.);
      }

      if (apply_boundary_values)
      {
        ApplyNewtonBoundaryValues(pde,residual);
      }
  }
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM, typename MATRIX>
    void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeMatrix(PROBLEM& pde, MATRIX &matrix)
{
  throw DOpEException("You should not use this function, try VoidLinearSolver instead.",
                          "IntegratorMixedDimensions::ComputeMatrix");
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
    void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeLocalControlConstraints (PROBLEM& pde, VECTOR &constraints)
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
          dimhigh>::ComputeDomainScalar(PROBLEM& pde)
      {
  {
      SCALAR ret = 0.;
      const unsigned int n_q_points = this->GetQuadratureFormula()->size();

      const auto& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

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
            throw DOpEException("Cellnumbers in DoFHandlers are not matching!",
                                "IntegratorMixedDimensions::ComputeDomainScalar");
          }
        }

        ret += pde.CellFunctional(cdc);

        if (need_faces)
        {
          throw DOpEException("Face Integrals not Implemented!",
                              "IntegratorMixedDimensions::ComputeDomainScalar");
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

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputePointScalar(PROBLEM& pde)
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
    SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeBoundaryScalar(PROBLEM& pde)
{

  {
      SCALAR ret = 0.;
      // Begin integration

      const std::vector<const DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >*>& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >::active_cell_iterator>
          cell(dof_handler.size());
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >::active_cell_iterator>
          endc(dof_handler.size());

      for (unsigned int dh = 0; dh < dof_handler.size(); dh++)
      {
        cell[dh] = dof_handler[dh]->begin_active();
        endc[dh] = dof_handler[dh]->end();
      }

      // Generate the data containers.
      FaceDataContainer<dealii::DoFHandler<dimhigh>, VECTOR, dimhigh> fdc(*(this->GetFaceQuadratureFormula()),
					     pde.GetFaceUpdateFlags(),
					     *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
					     this->GetParamData(), this->GetDomainData());

      std::vector<unsigned int> boundary_functional_colors =
          pde.GetBoundaryFunctionalColors();
      bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
      if (!need_boundary_integrals)
      {
        throw DOpEException("No boundary colors given!",
                            "IntegratorMixedDimensions::ComputeBoundaryScalar");
      }

      for (; cell[0] != endc[0]; cell[0]++)
      {
        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          if (cell[dh] == endc[dh])
          {
            throw DOpEException("Cellnumbers in DoFHandlers are not matching!",
                                "IntegratorMixedDimensions::ComputeBoundaryScalar");
          }
        }

        if(need_boundary_integrals)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            if (cell[0]->face(face)->at_boundary()
                &&
                (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                        cell[0]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
            {
//              pde.GetBaseProblem().GetSpaceTimeHandler()->ComputeFaceFEValues(cell, face, pde.GetType());
              fdc.ReInit(face);
	      ret += pde.BoundaryFunctional(fdc);
            }
          }
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

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
    SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeFaceScalar(PROBLEM& pde)
{
  {
      SCALAR ret = 0.;

      // Begin integration
      const std::vector<const DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >*>& dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >::active_cell_iterator>
          cell(dof_handler.size());
      std::vector<typename DOpEWrapper::DoFHandler<dimhigh, dealii::DoFHandler<dimhigh> >::active_cell_iterator>
          endc(dof_handler.size());

      for (unsigned int dh = 0; dh < dof_handler.size(); dh++)
      {
        cell[dh] = dof_handler[dh]->begin_active();
        endc[dh] = dof_handler[dh]->end();
      }
      // Generate the data containers.
      FaceDataContainer<dealii::DoFHandler<dimhigh>, VECTOR, dimhigh> fdc(*(this->GetFaceQuadratureFormula()),
					     pde.GetFaceUpdateFlags(),
					     *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
					     this->GetParamData(), this->GetDomainData());

      bool need_faces = pde.HasFaces();
      if (!need_faces)
      {
        throw DOpEException("No faces required!", "IntegratorMixedDimensions::ComputeFaceScalar");
      }

      for (; cell[0] != endc[0]; cell[0]++)
      {
        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
        {
          if (cell[dh] == endc[dh])
          {
            throw DOpEException("Cellnumbers in DoFHandlers are not matching!",
                                "IntegratorMixedDimensions::ComputeFaceScalar");
          }
        }

        if(need_faces)
        {
          for (unsigned int face=0; face < dealii::GeometryInfo<dimhigh>::faces_per_cell; ++face)
          {
            fdc.ReInit(face);
            ret +=pde.FaceFunctional(fdc);
          }
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

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
    SCALAR IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ComputeAlgebraicScalar(PROBLEM& pde)
{
    SCALAR ret = 0.;
    ret = pde.AlgebraicFunctional(this->GetParamData(), this->GetDomainData());
    return ret;
}
/*******************************************************************************************/
template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ApplyInitialBoundaryValues(PROBLEM& /*pde*/,
												      VECTOR &u __attribute__((unused)))
{
}
/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>
  ::ApplyTransposedInitialBoundaryValues(PROBLEM& pde,
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
    const std::vector<Point<dimhigh> >& support_points =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetMapDoFToSupportPoints();

    for (unsigned int i = 0; i < dirichlet_colors.size(); i++)
    {
      unsigned int color = dirichlet_colors[i];
      std::vector<bool> comp_mask = pde.GetTransposedDirichletCompMask(color);
      std::vector<bool> current_comp(comp_mask.size(), false);
      std::set<unsigned char> boundary_indicators;
      boundary_indicators.insert(color);
      for (unsigned int j = 0; j < comp_mask.size(); j++)
      {
        if (j > 0)
          current_comp[j - 1] = false;
        if (comp_mask[j])
        {
          current_comp[j] = true;
          //Hole eine Liste der DoFs auf dem Rand und die zugehoerigen Knoten
          DoFTools::extract_boundary_dofs(*static_cast<const dealii::DoFHandler<dimhigh>*> ((pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0])),
                                          current_comp, selected_components, boundary_indicators);
        }
        const TransposedDirichletDataInterface<dimlow, dimhigh> & DD =
            pde.GetTransposedDirichletValues(
                                                               color,
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
template<typename PROBLEM>
  void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ApplyNewtonBoundaryValues(PROBLEM& /*pde*/,
												     VECTOR &u __attribute__((unused)))
{
  //We don't need those in the mixed case...
}
/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
template<typename PROBLEM, typename MATRIX>
    void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::ApplyNewtonBoundaryValues(PROBLEM& pde,
												       MATRIX& matrix __attribute__((unused)),
												       VECTOR &rhs __attribute__((unused)),
												       VECTOR &sol __attribute__((unused)))
{
  //We don't need those in the mixed case...
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::AddDomainData(
                                                                                                   std::string name,
                                                                                                   const VECTOR* new_data)
{
  if (_domain_data.find(name) != _domain_data.end())
  {
    throw DOpEException("Adding multiple Data with name " + name + " is prohibited!",
                        "IntegratorMixedDimensions::AddDomainData");
  }
  _domain_data.insert(std::pair<std::string, const VECTOR*>(name, new_data));
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::DeleteDomainData(
                                                                                                      std::string name)
{
  typename std::map<std::string, const VECTOR *>::iterator it = _domain_data.find(name);
  if (it == _domain_data.end())
  {
    throw DOpEException("Deleting Data " + name + " is impossible! Data not found",
                        "IntegratorMixedDimensions::DeleteDomainData");
  }
  _domain_data.erase(it);
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
const std::map<std::string, const VECTOR*>& IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR,
    SCALAR, dimlow, dimhigh>::GetDomainData() const
{
  return _domain_data;
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::AddParamData(
                                                                                                  std::string name,
                                                                                                  const dealii::Vector<
                                                                                                      SCALAR>* new_data)
{
  if (_param_data.find(name) != _param_data.end())
  {
    throw DOpEException("Adding multiple Data with name " + name + " is prohibited!",
                        "IntegratorMixedDimensions::AddParamData");
  }
  _param_data.insert(std::pair<std::string, const dealii::Vector<SCALAR>*>(name, new_data));
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
void IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::DeleteParamData(
                                                                                                     std::string name)
{
  typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator it =
      _param_data.find(name);
  if (it == _param_data.end())
  {
    throw DOpEException("Deleting Data " + name + " is impossible! Data not found",
                        "IntegratorMixedDimensions::DeleteParamData");
  }
  _param_data.erase(it);
}

/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR, int dimlow,
    int dimhigh>
const std::map<std::string, const dealii::Vector<SCALAR>*>& IntegratorMixedDimensions<
    INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::GetParamData() const
{
  return _param_data;
}
/*******************************************************************************************/

template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
    int dimlow, int dimhigh>
  const INTEGRATORDATACONT&
  IntegratorMixedDimensions<INTEGRATORDATACONT, VECTOR, SCALAR, dimlow, dimhigh>::GetIntegratorDataContainer() const
  {
    return _idc;
  }

/*******************************************************************************************/
}
#endif

