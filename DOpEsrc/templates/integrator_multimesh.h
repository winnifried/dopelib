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

#ifndef IntegratorMultiMesh_H_
#define IntegratorMultiMesh_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/function.h>

#include <vector>

#include <container/multimesh_elementdatacontainer.h>
#include <container/multimesh_facedatacontainer.h>

namespace DOpE
{
  /**
   * This class is used to integrate the righthand side, matrix and so on.
   * This class is used when the control and the state are given on two different meshes
   * based upon the same initial mesh.
   *
   * For details on the functions see Integrator.
   *
   * Note that integration on faces is not yet supported.
   *
   * @template INTEGRATORDATACONT       The type of the integratordatacontainer, which has
   *                                    manages the basic data for integration (quadrature,
   *                                    elementdatacontainer, facedatacontainer etc.)
   * @template VECTOR                   Class of the vectors which we use in the integrator.
   * @template SCALAR                   Type of the scalars we use in the integrator.
   * @template dim                      The dimension of the domain
   */

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  class IntegratorMultiMesh
  {
  public:
    IntegratorMultiMesh(INTEGRATORDATACONT &idc);

    ~IntegratorMultiMesh();

    void
    ReInit();

    template<typename PROBLEM>
    void
    ComputeNonlinearResidual(PROBLEM &pde, VECTOR &residual);
    template<typename PROBLEM>
    void
    ComputeNonlinearLhs(PROBLEM &pde, VECTOR &residual);
    template<typename PROBLEM>
    void
    ComputeNonlinearRhs(PROBLEM &pde, VECTOR &residual);
    template<typename PROBLEM, typename MATRIX>
    void
    ComputeMatrix(PROBLEM &pde, MATRIX &matrix);

    template<typename PROBLEM>
    void ComputeNonlinearAlgebraicResidual (PROBLEM &pde, VECTOR &residual);
    template<typename PROBLEM>
    void ComputeLocalControlConstraints (PROBLEM &pde, VECTOR &constraints);

    template<typename PROBLEM>
    SCALAR
    ComputeDomainScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR
    ComputePointScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR
    ComputeBoundaryScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR
    ComputeFaceScalar(PROBLEM &pde);
    template<typename PROBLEM>
    SCALAR
    ComputeAlgebraicScalar(PROBLEM &pde);

    template<typename PROBLEM>
    void
    ApplyInitialBoundaryValues(PROBLEM &pde, VECTOR &u);


    inline void
    AddDomainData(std::string name, const VECTOR *new_data);
    inline void
    DeleteDomainData(std::string name);

    inline void
    AddParamData(std::string name, const dealii::Vector<SCALAR> *new_data);
    inline void
    DeleteParamData(std::string name);

  protected:
    inline const std::map<std::string, const dealii::Vector<SCALAR>*> &
    GetParamData() const;

    inline  INTEGRATORDATACONT &
    GetIntegratorDataContainer() const;
    inline const std::map<std::string, const VECTOR *> &
    GetDomainData() const;
    inline void AddPresetRightHandSide(double s, VECTOR &residual) const;

  private:
    template<template<int, int> class DH>
    void
    InterpolateBoundaryValues(
      const DOpEWrapper::DoFHandler<dim, DH>  *dof_handler,
      const unsigned int color, const dealii::Function<dim> &function,
      std::map<unsigned int, SCALAR> &boundary_values,
      const std::vector<bool> &comp_mask) const;

    /**
     * Used by to ComputeNonlinearResidual to loop until both variables are on
     * the same local element. See also deal.ii step-28
     */
    template<typename PROBLEM, template<int, int> class DH>
    inline void ComputeNonlinearResidual_Recursive(
      PROBLEM &pde,
      VECTOR &residual,
      typename std::vector<typename DH<dim, dim>::cell_iterator> &element_iter,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_iter,
      const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,
      Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc);

    /**
     * Used by to ComputeNonlinearRhs to loop until both variables are on
     * the same local element. See also deal.ii step-28
     */
    template<typename PROBLEM, template<int, int> class DH>
    inline void ComputeNonlinearRhs_Recursive(
      PROBLEM &pde,
      VECTOR &residual,
      typename std::vector<typename DH<dim, dim>::cell_iterator> &element_iter,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_iter,
      const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,
      Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc);

    /**
     * Used by to ComputeMatrix to loop until both variables are on
     * the same local element. See also deal.ii step-28
     */
    template<typename PROBLEM, typename MATRIX, template<int, int> class DH>
    inline void ComputeMatrix_Recursive(
      PROBLEM &pde,
      MATRIX &matrix,
      typename std::vector<typename DH<dim, dim>::cell_iterator> &element_iter,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_iter,
      const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,
      Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc);

    /**
     * Used by to ComputeDomainScalar to loop until both variables are on
     * the same local element. See also deal.ii step-28
     */
    template<typename PROBLEM, template<int, int> class DH>
    inline SCALAR ComputeDomainScalar_Recursive(
      PROBLEM &pde,
      typename std::vector<typename DH<dim, dim>::cell_iterator> &element_iter,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_iter,
      const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc);

    /**
     * Used by to ComputeBoundaryScalar to loop until both variables are on
     * the same local element. See also deal.ii step-28
     */
    template<typename PROBLEM, template<int, int> class DH>
    inline SCALAR ComputeBoundaryScalar_Recursive(
      PROBLEM &pde,
      typename std::vector<typename DH<dim, dim>::cell_iterator> &element_iter,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_iter,
      const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_FaceDataContainer<DH, VECTOR, dim> &edc);

    INTEGRATORDATACONT &idc_;

    std::map<std::string, const VECTOR *> domain_data_;
    std::map<std::string, const dealii::Vector<SCALAR>*> param_data_;
  };

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::IntegratorMultiMesh(
    INTEGRATORDATACONT &idc) :
    idc_(idc)
  {
  }

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::~IntegratorMultiMesh()
  {

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ReInit()
  {

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearResidual(
    PROBLEM &pde, VECTOR &residual)
  {
    residual = 0.;

    const auto &dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

    assert(dof_handler.size() == 2);

#if DEAL_II_VERSION_GTE(8,4,0)
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_triangulation(),
                                   dof_handler[1]->GetDEALDoFHandler().get_triangulation());
#else
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
                                   dof_handler[1]->GetDEALDoFHandler().get_tria());
#endif

    const auto element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
                              dof_handler[1]->GetDEALDoFHandler());
    auto element_iter = element_list.begin();
    auto tria_element_iter = tria_element_list.begin();

    std::vector<decltype(tria_element_iter->first)> tria_element(2);
    tria_element[0] = tria_element_iter->first;
    tria_element[1] = tria_element_iter->second;

    std::vector<decltype(element_iter->first)> element(2);
    element[0] = element_iter->first;
    element[1] = element_iter->second;
    int coarse_index = 0; //element[coarse_index] is the coarser of the two.
    int fine_index = 0; //element[fine_index] is the finer of the two (both indices are 2 = element.size() if they are equally refined.

    // Generate the data containers.
    idc_.InitializeMMEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element, tria_element,
                         this->GetParamData(), this->GetDomainData());
    auto &edc = idc_.GetMultimeshElementDataContainer();

    bool need_interfaces = pde.HasInterfaces();
    idc_.InitializeMMFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element,
                         tria_element,
                         this->GetParamData(),
                         this->GetDomainData(),
                         need_interfaces);
    auto &fdc = idc_.GetMultimeshFaceDataContainer();

    for (; element_iter != element_list.end(); element_iter++)
      {
        element[0] = element_iter->first;
        element[1] = element_iter->second;
        tria_element[0] = tria_element_iter->first;
        tria_element[1] = tria_element_iter->second;
        FullMatrix<SCALAR> prolong_matrix;

        if (element[0]->has_children())
          {
            prolong_matrix = IdentityMatrix(element[1]->get_fe().dofs_per_cell);
            coarse_index =1;
            fine_index = 0;
          }
        else
          {
            if (element[1]->has_children())
              {
                prolong_matrix = IdentityMatrix(element[0]->get_fe().dofs_per_cell);
                coarse_index =0;
                fine_index = 1;
              }
            else
              {
                assert(element.size() ==2);
                coarse_index = fine_index = 2;
              }
          }
        ComputeNonlinearResidual_Recursive(pde,residual,element,tria_element,prolong_matrix,coarse_index,fine_index,edc,fdc);
        tria_element_iter++;
      }
    //Check if some preset righthandside exists.
    AddPresetRightHandSide(-1.,residual);

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearLhs(
    PROBLEM &pde, VECTOR &residual)
  {
    {
      throw DOpEException("This function needs to be implemented!", "IntegratorMultiMesh::ComputeNonlinearLhs");
    }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs(
    PROBLEM &pde, VECTOR &residual)
  {
    residual = 0.;

    const auto &dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

    assert(dof_handler.size() == 2);
#if DEAL_II_VERSION_GTE(8,4,0)
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_triangulation(),
                                   dof_handler[1]->GetDEALDoFHandler().get_triangulation());
#else
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
                                   dof_handler[1]->GetDEALDoFHandler().get_tria());
#endif

    const auto element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
                              dof_handler[1]->GetDEALDoFHandler());
    auto element_iter = element_list.begin();
    auto tria_element_iter = tria_element_list.begin();

    std::vector<decltype(tria_element_iter->first)> tria_element(2);
    tria_element[0] = tria_element_iter->first;
    tria_element[1] = tria_element_iter->second;

    std::vector<decltype(element_iter->first)> element(2);
    element[0] = element_iter->first;
    element[1] = element_iter->second;
    int coarse_index = 0; //element[coarse_index] is the coarser of the two.
    int fine_index = 0; //element[fine_index] is the finer of the two (both indices are 2 = element.size() if they are equally refined.

    // Generate the data containers.
    idc_.InitializeMMEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element, tria_element,
                         this->GetParamData(), this->GetDomainData());
    auto &edc = idc_.GetMultimeshElementDataContainer();

    bool need_interfaces = pde.HasInterfaces();
    idc_.InitializeMMFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element,
                         tria_element,
                         this->GetParamData(),
                         this->GetDomainData(),
                         need_interfaces);
    auto &fdc = idc_.GetMultimeshFaceDataContainer();

    for (; element_iter != element_list.end(); element_iter++)
      {
        element[0] = element_iter->first;
        element[1] = element_iter->second;
        tria_element[0] = tria_element_iter->first;
        tria_element[1] = tria_element_iter->second;
        FullMatrix<SCALAR> prolong_matrix;

        if (element[0]->has_children())
          {
            prolong_matrix = IdentityMatrix(element[1]->get_fe().dofs_per_cell);
            coarse_index =1;
            fine_index = 0;
          }
        else
          {
            if (element[1]->has_children())
              {
                prolong_matrix = IdentityMatrix(element[0]->get_fe().dofs_per_cell);
                coarse_index =0;
                fine_index = 1;
              }
            else
              {
                assert(element.size() ==2);
                coarse_index = fine_index = 2;
              }
          }
        ComputeNonlinearRhs_Recursive(pde,residual,element,tria_element,prolong_matrix,coarse_index,fine_index,edc,fdc);
        tria_element_iter++;
      }
    //Check if some preset righthandside exists.
    AddPresetRightHandSide(1.,residual);

  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM, typename MATRIX>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix(
    PROBLEM &pde, MATRIX &matrix)
  {
    matrix = 0.;

    const auto &dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

    assert(dof_handler.size() == 2);

#if DEAL_II_VERSION_GTE(8,4,0)
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_triangulation(),
                                   dof_handler[1]->GetDEALDoFHandler().get_triangulation());
#else
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
                                   dof_handler[1]->GetDEALDoFHandler().get_tria());
#endif
    const auto element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
                              dof_handler[1]->GetDEALDoFHandler());
    auto element_iter = element_list.begin();
    auto tria_element_iter = tria_element_list.begin();

    std::vector<decltype(tria_element_iter->first)> tria_element(2);
    tria_element[0] = tria_element_iter->first;
    tria_element[1] = tria_element_iter->second;

    std::vector<decltype(element_iter->first)> element(2);
    element[0] = element_iter->first;
    element[1] = element_iter->second;
    int coarse_index = 0; //element[coarse_index] is the coarser of the two.
    int fine_index = 0; //element[fine_index] is the finer of the two (both indices are 2 = element.size() if they are equally refined.

    // Generate the data containers.
    idc_.InitializeMMEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element, tria_element,
                         this->GetParamData(), this->GetDomainData());
    auto &edc = idc_.GetMultimeshElementDataContainer();

    bool need_interfaces = pde.HasInterfaces();
    idc_.InitializeMMFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element,
                         tria_element,
                         this->GetParamData(),
                         this->GetDomainData(),
                         need_interfaces);
    auto &fdc = idc_.GetMultimeshFaceDataContainer();

    for (; element_iter != element_list.end(); element_iter++)
      {
        element[0] = element_iter->first;
        element[1] = element_iter->second;
        tria_element[0] = tria_element_iter->first;
        tria_element[1] = tria_element_iter->second;
        FullMatrix<SCALAR> prolong_matrix;

        if (element[0]->has_children())
          {
            prolong_matrix = IdentityMatrix(element[1]->get_fe().dofs_per_cell);
            coarse_index =1;
            fine_index = 0;
          }
        else
          {
            if (element[1]->has_children())
              {
                prolong_matrix = IdentityMatrix(element[0]->get_fe().dofs_per_cell);
                coarse_index =0;
                fine_index = 1;
              }
            else
              {
                assert(element.size() ==2);
                coarse_index = fine_index = 2;
              }
          }
        ComputeMatrix_Recursive(pde,matrix,element,tria_element,prolong_matrix,coarse_index,fine_index,edc,fdc);
        tria_element_iter++;
      }
  }

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  SCALAR
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar(
    PROBLEM &pde)
  {
    SCALAR ret = 0.;

    const auto &dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

    assert(dof_handler.size() == 2);

#if DEAL_II_VERSION_GTE(8,4,0)
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_triangulation(),
                                   dof_handler[1]->GetDEALDoFHandler().get_triangulation());
#else
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
                                   dof_handler[1]->GetDEALDoFHandler().get_tria());
#endif
    const auto element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
                              dof_handler[1]->GetDEALDoFHandler());
    auto element_iter = element_list.begin();
    auto tria_element_iter = tria_element_list.begin();

    std::vector<decltype(tria_element_iter->first)> tria_element(2);
    tria_element[0] = tria_element_iter->first;
    tria_element[1] = tria_element_iter->second;

    std::vector<decltype(element_iter->first)> element(2);
    element[0] = element_iter->first;
    element[1] = element_iter->second;
    int coarse_index = 0; //element[coarse_index] is the coarser of the two.
    int fine_index = 0; //element[fine_index] is the finer of the two (both indices are 2 = element.size() if they are equally refined.

    if (pde.HasFaces())
      {
        throw DOpEException("This function should not be called when faces are needed!",
                            "IntegratorMultiMesh::ComputeDomainScalar");
      }

    // Generate the data containers.
    idc_.InitializeMMEDC(pde.GetUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()), element, tria_element,
                         this->GetParamData(), this->GetDomainData());
    auto &edc = idc_.GetMultimeshElementDataContainer();

    for (; element_iter != element_list.end(); element_iter++)
      {
        element[0] = element_iter->first;
        element[1] = element_iter->second;
        tria_element[0] = tria_element_iter->first;
        tria_element[1] = tria_element_iter->second;
        FullMatrix<SCALAR> prolong_matrix;

        if (element[0]->has_children())
          {
            prolong_matrix = IdentityMatrix(element[1]->get_fe().dofs_per_cell);
            coarse_index =1;
            fine_index = 0;
          }
        else
          {
            if (element[1]->has_children())
              {
                prolong_matrix = IdentityMatrix(element[0]->get_fe().dofs_per_cell);
                coarse_index =0;
                fine_index = 1;
              }
            else
              {
                assert(element.size() ==2);
                coarse_index = fine_index = 2;
              }
          }
        ret += ComputeDomainScalar_Recursive(pde,element,tria_element,prolong_matrix,coarse_index,fine_index,edc);
        tria_element_iter++;
      }
    return ret;
  }


  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  SCALAR
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePointScalar(
    PROBLEM &pde)
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
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar(
    PROBLEM &pde
  )
  {
    SCALAR ret = 0.;
    // Begin integration
    const auto &dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

#if DEAL_II_VERSION_GTE(8,4,0)
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_triangulation(),
                                   dof_handler[1]->GetDEALDoFHandler().get_triangulation());
#else
    const auto tria_element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
                                   dof_handler[1]->GetDEALDoFHandler().get_tria());
#endif
    const auto element_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
                              dof_handler[1]->GetDEALDoFHandler());
    auto element_iter = element_list.begin();
    auto tria_element_iter = tria_element_list.begin();

    std::vector<decltype(tria_element_iter->first)> tria_element(2);
    tria_element[0] = tria_element_iter->first;
    tria_element[1] = tria_element_iter->second;

    std::vector<decltype(element_iter->first)> element(2);
    element[0] = element_iter->first;
    element[1] = element_iter->second;
    int coarse_index = 0; //element[coarse_index] is the coarser of the two.
    int fine_index = 0; //element[fine_index] is the finer of the two (both indices are 2 = element.size() if they are equally refined.


    idc_.InitializeMMFDC(pde.GetFaceUpdateFlags(),
                         *(pde.GetBaseProblem().GetSpaceTimeHandler()),
                         element, tria_element,
                         this->GetParamData(),
                         this->GetDomainData());
    auto &fdc = idc_.GetMultimeshFaceDataContainer();

    std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
    bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
    if (!need_boundary_integrals)
      {
        throw DOpEException("No boundary colors given!","IntegratorMultiMesh::ComputeBoundaryScalar");
      }

    for (; element_iter != element_list.end(); element_iter++)
      {
        element[0] = element_iter->first;
        element[1] = element_iter->second;
        tria_element[0] = tria_element_iter->first;
        tria_element[1] = tria_element_iter->second;
        FullMatrix<SCALAR> prolong_matrix;

        if (element[0]->has_children())
          {
            prolong_matrix = IdentityMatrix(element[1]->get_fe().dofs_per_cell);
            coarse_index =1;
            fine_index = 0;
          }
        else
          {
            if (element[1]->has_children())
              {
                prolong_matrix = IdentityMatrix(element[0]->get_fe().dofs_per_cell);
                coarse_index =0;
                fine_index = 1;
              }
            else
              {
                assert(element.size() ==2);
                coarse_index = fine_index = 2;
              }
          }
        ret += ComputeBoundaryScalar_Recursive(pde,element,tria_element,prolong_matrix,coarse_index,fine_index,fdc);
        tria_element_iter++;
      }

    return ret;
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  SCALAR
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeFaceScalar(
    PROBLEM & /*pde*/
  )
  {
    throw DOpEException("This function needs to be implemented!", "IntegratorMultiMesh::ComputeFaceScalar");
//          {
//            SCALAR ret = 0.;
//#if deal_II_dimension == 2 || deal_II_dimension == 3
//            // Begin integration
//            const auto& dof_handler =
//            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//            auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//            auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//            idc_.InitializeFDC(pde.GetFaceUpdateFlags(),
//                *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                element,
//                this->GetParamData(),
//                this->GetDomainData());
//            auto & fdc = idc_.GetFaceDataContainer();
//
//            bool need_faces = pde.HasFaces();
//            if(!need_faces)
//              {
//                throw DOpEException("No faces required!","IntegratorMultiMesh::ComputeFaceScalar");
//              }
//
//            for (;element[0]!=endc[0]; element[0]++)
//              {
//                for(unsigned int dh=1; dh<dof_handler.size(); dh++)
//                  {
//                    if( element[dh] == endc[dh])
//                      {
//                        throw DOpEException("Elementnumbers in DoFHandlers are not matching!","IntegratorMultiMesh::ComputeFaceScalar");
//                      }
//                  }
//
//                if(need_faces)
//                  {
//                    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                      {
//                        if (element[0]->neighbor_index(face) != -1)
//                          {
//                            fdc.ReInit(face);
//                            ret +=pde.FaceFunctional(fdc);
//                          }
//                      }
//                  }
//                for(unsigned int dh=1; dh<dof_handler.size(); dh++)
//                  {
//                    element[dh]++;
//                  }
//              }
//#else
//            throw DOpEException("Not implemented in this dimension!",
//                "IntegratorMultiMesh::ComputeFaceScalar");
//#endif
//            return ret;
//          }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  SCALAR
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeAlgebraicScalar(
    PROBLEM &pde)
  {

    {
      SCALAR ret = 0.;
      ret = pde.AlgebraicFunctional(this->GetParamData(),
                                    this->GetDomainData());
      return ret;
    }
  }

  /*******************************************************************************************/

  template <typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM>
  void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>
  ::ComputeNonlinearAlgebraicResidual (PROBLEM &pde, VECTOR &residual)
  {
    residual = 0.;
    pde.AlgebraicResidual(residual,this->GetParamData(),this->GetDomainData());
    //Check if some preset righthandside exists.
    AddPresetRightHandSide(-1.,residual);
  }

  /*******************************************************************************************/

  template <typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM>
  void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>
  ::ComputeLocalControlConstraints (PROBLEM &pde, VECTOR &constraints)
  {
    constraints = 0.;
    pde.ComputeLocalControlConstraints(constraints,this->GetParamData(),this->GetDomainData());
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyInitialBoundaryValues(
    PROBLEM &pde, VECTOR &u)
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

        InterpolateBoundaryValues(pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler()[0], color, pde.GetDirichletValues(color, this->GetParamData(),
                                  this->GetDomainData()), boundary_values, comp_mask );

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
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddDomainData(
    std::string name, const VECTOR *new_data)
  {
    if (domain_data_.find(name) != domain_data_.end())
      {
        throw DOpEException(
          "Adding multiple Data with name " + name + " is prohibited!",
          "IntegratorMultiMesh::AddDomainData");
      }
    domain_data_.insert(std::pair<std::string, const VECTOR *>(name, new_data));
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteDomainData(
    std::string name)
  {
    typename std::map<std::string, const VECTOR *>::iterator it =
      domain_data_.find(name);
    if (it == domain_data_.end())
      {
        throw DOpEException(
          "Deleting Data " + name + " is impossible! Data not found",
          "IntegratorMultiMesh::DeleteDomainData");
      }
    domain_data_.erase(it);
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddPresetRightHandSide(double s,
      VECTOR &residual) const
  {
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      domain_data_.find("fixed_rhs");
    if (it != domain_data_.end())
      {
        assert(residual.size() == it->second->size());
        residual.add(s,*(it->second));
      }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  const std::map<std::string, const VECTOR *> &
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetDomainData() const
  {
    return domain_data_;
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddParamData(
    std::string name, const dealii::Vector<SCALAR> *new_data)
  {
    if (param_data_.find(name) != param_data_.end())
      {
        throw DOpEException(
          "Adding multiple Data with name " + name + " is prohibited!",
          "IntegratorMultiMesh::AddParamData");
      }
    param_data_.insert(
      std::pair<std::string, const dealii::Vector<SCALAR>*>(name, new_data));
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteParamData(
    std::string name)
  {
    typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator
    it = param_data_.find(name);
    if (it == param_data_.end())
      {
        throw DOpEException(
          "Deleting Data " + name + " is impossible! Data not found",
          "IntegratorMultiMesh::DeleteParamData");
      }
    param_data_.erase(it);
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  const std::map<std::string, const dealii::Vector<SCALAR>*> &
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetParamData() const
  {
    return param_data_;
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  INTEGRATORDATACONT &
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainer() const
  {
    return idc_;
  }


  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<template<int, int> class DH>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::InterpolateBoundaryValues(
    const DOpEWrapper::DoFHandler<dim, DH> *dof_handler,
    const unsigned int color, const dealii::Function<dim> &function,
    std::map<unsigned int, SCALAR> &boundary_values,
    const std::vector<bool> &comp_mask) const
  {
    dealii::VectorTools::interpolate_boundary_values(
      dof_handler->GetDEALDoFHandler(), color, function,
      boundary_values, comp_mask);
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM, template<int, int> class DH>
  void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearResidual_Recursive(
    PROBLEM &pde,
    VECTOR &residual,
    typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
    const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
    Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc)
  {
    if (!element[0]->has_children() && ! element[1]->has_children())
      {
        unsigned int dofs_per_element;
        dealii::Vector<SCALAR> local_vector;
        std::vector<unsigned int> local_dof_indices;

        bool need_faces = pde.HasFaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
        bool need_interfaces = pde.HasInterfaces();

        edc.ReInit(coarse_index,fine_index,prolong_matrix);

        dofs_per_element = element[0]->get_fe().dofs_per_cell;

        local_vector.reinit(dofs_per_element);
        local_vector = 0;

        local_dof_indices.resize(0);
        local_dof_indices.resize(dofs_per_element, 0);

        //the second '1' plays only a role in the stationary case. In the non-stationary
        //case, scale_ico is set by the time-stepping-scheme
        pde.ElementEquation(edc, local_vector, 1., 1.);
        pde.ElementRhs(edc, local_vector, -1.);

        //FIXME Integrate on Faces of Element[0] that contain a fine-element face.
        if (need_faces || need_interfaces)
          {
            throw DOpEException(" Faces on multiple meshes not implemented yet!",
                                "IntegratorMultiMesh::ComputeNonlinearResidual_Recursive");
          }
        unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the
        //zeros entry in the element vector

        if (need_boundary_integrals && element[b_index]->at_boundary())
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
#if DEAL_II_VERSION_GTE(8,3,0)
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                  {
                    fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
                    pde.BoundaryEquation(fdc,local_vector, 1., 1.);
                    pde.BoundaryRhs(fdc,local_vector,-1.);
                  }
              }
          }
        //     if(need_faces)
        //       {
        //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        //           {
        //             if (element[0]->neighbor_index(face) != -1)
        //               {
        //                 fdc.ReInit(face);
        //                 pde.FaceEquation(fdc, local_vector);
        //                 pde.FaceRhs(fdc, local_vector,-1.);
        //               }
        //           }
        //       }
        //     if( need_interfaces)
        //       {
        //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        //           {
        //             fdc.ReInit(face);
        //             if (element[0]->neighbor_index(face) != -1
        //                 &&
        //                 fdc.GetMaterialId()!= fdc.GetNbrMaterialId())
        //               {
        //                 fdc.ReInitNbr();
        //                 pde.InterfaceEquation(fdc, local_vector);
        //               }
        //           }
        //       }
        if (coarse_index == 0) //Need to transfer the computed residual to the one on the coarse element
          {
            dealii::Vector<SCALAR> tmp(dofs_per_element);
            prolong_matrix.Tvmult(tmp,local_vector);

            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(tmp, local_dof_indices, residual);

          }
        else //Testfunctions are already the right ones...
          {
            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(local_vector, local_dof_indices, residual);
          }

      }//Endof the case on the finest level
    else
      {
        assert(fine_index != coarse_index);
        assert(element[fine_index]->has_children());
        assert(!element[coarse_index]->has_children());
        assert(tria_element[fine_index]->has_children());
        assert(!tria_element[coarse_index]->has_children());

        unsigned int local_n_dofs = element[coarse_index]->get_fe().dofs_per_cell;

        typename DH<dim, dim>::cell_iterator dofh_fine = element[fine_index];
        typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_element[fine_index];

        for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
          {
            FullMatrix<SCALAR>   new_matrix(local_n_dofs);
            element[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
                prolong_matrix);
            element[fine_index] = dofh_fine->child(child);
            tria_element[fine_index] = tria_fine->child(child);

            ComputeNonlinearResidual_Recursive(pde,residual,element,tria_element,new_matrix, coarse_index, fine_index, edc, fdc);
          }
      }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM, template<int, int> class DH>
  void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs_Recursive(
    PROBLEM &pde,
    VECTOR &residual,
    typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
    const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
    Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc)
  {
    if (!element[0]->has_children() && ! element[1]->has_children())
      {
        unsigned int dofs_per_element;
        dealii::Vector<SCALAR> local_vector;
        std::vector<unsigned int> local_dof_indices;

        bool need_faces = pde.HasFaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
        bool need_interfaces = pde.HasInterfaces();

        edc.ReInit(coarse_index,fine_index,prolong_matrix);

        dofs_per_element = element[0]->get_fe().dofs_per_cell;

        local_vector.reinit(dofs_per_element);
        local_vector = 0;

        local_dof_indices.resize(0);
        local_dof_indices.resize(dofs_per_element, 0);

        //the second '1' plays only a role in the stationary case. In the non-stationary
        //case, scale_ico is set by the time-stepping-scheme
        pde.ElementRhs(edc, local_vector, 1.);

        //FIXME Integrate on Faces of Element[0] that contain a fine-element face.
        if (need_faces )
          {
            throw DOpEException(" Faces on multiple meshes not implemented yet!",
                                "IntegratorMultiMesh::ComputeNonlinearRhs_Recursive");
          }
        if (need_interfaces )
          {
            throw DOpEException(" Faces on multiple meshes not implemented yet!",
                                "IntegratorMultiMesh::ComputeNonlinearRhs_Recursive");
          }
        unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the
        //zeros entry in the element vector

        if (need_boundary_integrals && element[b_index]->at_boundary())
          {
            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
#if DEAL_II_VERSION_GTE(8,3,0)
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                  {
                    fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
                    pde.BoundaryRhs(fdc,local_vector,1.);
                  }
              }
          }
        //     if(need_faces)
        //       {
        //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        //           {
        //             if (element[0]->neighbor_index(face) != -1)
        //               {
        //                 fdc.ReInit(face);
        //                 pde.FaceRhs(fdc, local_vector,1.);
        //               }
        //           }
        //       }
        if (coarse_index == 0) //Need to transfer the computed residual to the one on the coarse element
          {
            dealii::Vector<SCALAR> tmp(dofs_per_element);
            prolong_matrix.Tvmult(tmp,local_vector);

            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(tmp, local_dof_indices, residual);
          }
        else //Testfunctions are already the right ones...
          {
            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(local_vector, local_dof_indices, residual);
          }

      }//Endof the case on the finest level
    else
      {
        assert(fine_index != coarse_index);
        assert(element[fine_index]->has_children());
        assert(!element[coarse_index]->has_children());
        assert(tria_element[fine_index]->has_children());
        assert(!tria_element[coarse_index]->has_children());

        unsigned int local_n_dofs = element[coarse_index]->get_fe().dofs_per_cell;

        typename DH<dim, dim>::cell_iterator dofh_fine = element[fine_index];
        typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_element[fine_index];

        for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
          {
            FullMatrix<SCALAR>   new_matrix(local_n_dofs);
            element[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
                prolong_matrix);
            element[fine_index] = dofh_fine->child(child);
            tria_element[fine_index] = tria_fine->child(child);

            ComputeNonlinearRhs_Recursive(pde,residual,element,tria_element,new_matrix, coarse_index, fine_index, edc, fdc);
          }
      }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
           int dim>
  template<typename PROBLEM, typename MATRIX, template<int, int> class DH>
  void
  IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix_Recursive(
    PROBLEM &pde, MATRIX &matrix, typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
    const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
    Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc,
    Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc)
  {

    if (!element[0]->has_children() && ! element[1]->has_children())
      {
        unsigned int dofs_per_element;
        std::vector<unsigned int> local_dof_indices;

        bool need_faces = pde.HasFaces();
        bool need_interfaces = pde.HasInterfaces();
        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

        edc.ReInit(coarse_index,fine_index,prolong_matrix);
        dofs_per_element = element[0]->get_fe().dofs_per_cell;

        dealii::FullMatrix<SCALAR> local_matrix(dofs_per_element,
                                                dofs_per_element);
        local_matrix = 0;

        local_dof_indices.resize(0);
        local_dof_indices.resize(dofs_per_element, 0);
        pde.ElementMatrix(edc, local_matrix);

        //FIXME Integrate on Faces of Element[0] that contain a fine-element face.
        if (need_faces || need_interfaces)
          {
            throw DOpEException(" Faces on multiple meshes not implemented yet!",
                                "IntegratorMultiMesh::ComputeMatrix_Recursive");
          }
        unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the
        //zeros entry in the element vector
        if (need_boundary_integrals && element[b_index]->at_boundary())
          {

            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
#if DEAL_II_VERSION_GTE(8,3,0)
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                if (element[b_index]->face(face)->at_boundary()
                    &&
                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                          element[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                  {
                    fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
                    pde.BoundaryMatrix(fdc, local_matrix);
                  }
              }
          }
        //  if(need_faces)
        //    {
        //      for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        //        {
        //          if (element[0]->neighbor_index(face) != -1)
        //            {
        //              fdc.ReInit(face);
        //              pde.FaceMatrix(fdc, local_matrix);
        //            }
        //        }
        //    }
        //  if( need_interfaces)
        //    {
        //      for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        //        {
        //          fdc.ReInit(face);
        //          if (element[0]->neighbor_index(face) != -1
        //              &&
        //              fdc.GetMaterialId()!= fdc.GetNbrMaterialId())
        //            {
        //              fdc.ReInitNbr();
        //
        //              nbr_dofs_per_element = fdc.GetNbrNDoFsPerElement();
        //              nbr_local_dof_indices.resize(0);
        //              nbr_local_dof_indices.resize(nbr_dofs_per_element, 0);
        //              dealii::FullMatrix<SCALAR>  local_interface_matrix(dofs_per_element,nbr_dofs_per_element );
        //              local_interface_matrix = 0;
        //
        //              pde.InterfaceMatrix(fdc, local_interface_matrix);
        //
        //              element[0]->get_dof_indices(local_dof_indices);
        //              element[0]->neighbor(face)->get_dof_indices(nbr_local_dof_indices);
        //
        //              for (unsigned int i = 0; i < dofs_per_element; ++i)
        //                {
        //                  for (unsigned int j = 0; j < nbr_dofs_per_element; ++j)
        //                    {
        //                      matrix.add(local_dof_indices[i], nbr_local_dof_indices[j],
        //                          local_interface_matrix(i, j));
        //                    } //endfor j
        //                } //endfor i
        //            }
        //        }
        //    }
        if (coarse_index == 0) //Need to transfer the computed residual to the one on the coarse element
          {
            dealii::FullMatrix<SCALAR> tmp(dofs_per_element);
            tmp = 0.;
            prolong_matrix.Tmmult(tmp,local_matrix);
            local_matrix = 0.;
            tmp.mmult(local_matrix,prolong_matrix);

            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(local_matrix, local_dof_indices, matrix);
          }
        else
          {
            //LocalToGlobal
            const auto &C = pde.GetDoFConstraints();
            element[0]->get_dof_indices(local_dof_indices);
            C.distribute_local_to_global(local_matrix, local_dof_indices, matrix);
          }

      }//Endof the case on the finest level
    else
      {
        assert(fine_index != coarse_index);
        assert(element[fine_index]->has_children());
        assert(!element[coarse_index]->has_children());
        assert(tria_element[fine_index]->has_children());
        assert(!tria_element[coarse_index]->has_children());

        unsigned int local_n_dofs = element[coarse_index]->get_fe().dofs_per_cell;

        typename DH<dim, dim>::cell_iterator dofh_fine = element[fine_index];
        typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_element[fine_index];

        for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
          {
            FullMatrix<SCALAR>   new_matrix(local_n_dofs);
            element[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
                prolong_matrix);
            element[fine_index] = dofh_fine->child(child);
            tria_element[fine_index] = tria_fine->child(child);

            ComputeMatrix_Recursive(pde,matrix,element,tria_element,new_matrix, coarse_index, fine_index, edc, fdc);
          }
      }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM, template<int, int> class DH>
  SCALAR IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar_Recursive(
    PROBLEM &pde,
    typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
    const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
    Multimesh_ElementDataContainer<DH, VECTOR, dim> &edc)
  {
    if (!element[0]->has_children() && ! element[1]->has_children())
      {
        SCALAR ret = 0.;
        edc.ReInit(coarse_index,fine_index,prolong_matrix);
        ret += pde.ElementFunctional(edc);
        return ret;
      }    //Endof the case on the finest level
    else
      {
        assert(fine_index != coarse_index);
        assert(element[fine_index]->has_children());
        assert(!element[coarse_index]->has_children());
        assert(tria_element[fine_index]->has_children());
        assert(!tria_element[coarse_index]->has_children());

        unsigned int local_n_dofs = element[coarse_index]->get_fe().dofs_per_cell;

        typename DH<dim, dim>::cell_iterator dofh_fine = element[fine_index];
        typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_element[fine_index];
        SCALAR ret = 0.;
        for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
          {
            FullMatrix<SCALAR>   new_matrix(local_n_dofs);
            element[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
                prolong_matrix);
            element[fine_index] = dofh_fine->child(child);
            tria_element[fine_index] = tria_fine->child(child);

            ret += ComputeDomainScalar_Recursive(pde,element,tria_element,new_matrix, coarse_index, fine_index, edc);
          }
        return ret;
      }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
  template<typename PROBLEM, template<int, int> class DH>
  SCALAR IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar_Recursive(
    PROBLEM &pde,
    typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
    const FullMatrix<SCALAR> &prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
    Multimesh_FaceDataContainer<DH, VECTOR, dim> &fdc)
  {
    if (!element[0]->has_children() && ! element[1]->has_children())
      {
        SCALAR ret = 0.;
        std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
        unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the
        //zeros entry in the element vector
        for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            if (element[b_index]->face(face)->at_boundary()
                &&
                (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                      element[b_index]->face(face)->boundary_id()) != boundary_functional_colors.end()))
#else
            if (element[b_index]->face(face)->at_boundary()
                &&
                (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
                      element[b_index]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
#endif
              {
                fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
                ret += pde.BoundaryFunctional(fdc);
              }
          }
        return ret;
      }    //Endof the case on the finest level
    else
      {
        assert(fine_index != coarse_index);
        assert(element[fine_index]->has_children());
        assert(!element[coarse_index]->has_children());
        assert(tria_element[fine_index]->has_children());
        assert(!tria_element[coarse_index]->has_children());

        unsigned int local_n_dofs = element[coarse_index]->get_fe().dofs_per_cell;

        typename DH<dim, dim>::cell_iterator dofh_fine = element[fine_index];
        typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_element[fine_index];
        SCALAR ret = 0.;
        for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
          {
            FullMatrix<SCALAR>   new_matrix(local_n_dofs);
            element[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
                prolong_matrix);
            element[fine_index] = dofh_fine->child(child);
            tria_element[fine_index] = tria_fine->child(child);

            ret += ComputeBoundaryScalar_Recursive(pde,element,tria_element,new_matrix, coarse_index, fine_index, fdc);
          }
        return ret;
      }
  }
//ENDOF NAMESPACE DOpE
}
#endif

