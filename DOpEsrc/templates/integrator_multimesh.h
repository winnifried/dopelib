#ifndef _IntegratorMultiMesh_H_
#define _IntegratorMultiMesh_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include <numerics/vectors.h>
#include <numerics/matrices.h>
#include <deal.II/grid/grid_tools.h>
#include <base/function.h>

#include <vector>

#include "multimesh_celldatacontainer.h"
#include "multimesh_facedatacontainer.h"

namespace DOpE
{
  /**
   * This class is used to integrate the righthand side, matrix and so on
   * In contrast to the Integrator this class assumes that we may have 
   * different meshes for the control and state variable.
   */
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    class IntegratorMultiMesh
    {
      public:
        IntegratorMultiMesh(INTEGRATORDATACONT& idc);

        ~IntegratorMultiMesh();

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
	  void ComputeNonlinearAlgebraicResidual (PROBLEM& pde, VECTOR &residual);
	template<typename PROBLEM>
	  void ComputeLocalControlConstraints (PROBLEM& pde, VECTOR &constraints);

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

        inline  INTEGRATORDATACONT&
        GetIntegratorDataContainer() const;

      private:
        template<typename DOFHANDLER>
          void
          InterpolateBoundaryValues(
              const DOpEWrapper::DoFHandler<dim, DOFHANDLER>*  dof_handler,
              const unsigned int color, const dealii::Function<dim>& function,
              std::map<unsigned int, SCALAR>& boundary_values,
              const std::vector<bool>& comp_mask) const;

	template<typename PROBLEM, typename DOFHANDLER>
	  inline void ComputeNonlinearResidual_Recursive(
	    PROBLEM& pde, 
	    VECTOR& residual, 
	    typename std::vector<typename DOFHANDLER::cell_iterator>& cell_iter,
	    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell_iter,
	    const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	    Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,
	    Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc);
	
	template<typename PROBLEM, typename DOFHANDLER>
	  inline void ComputeNonlinearRhs_Recursive(
	    PROBLEM& pde, 
	    VECTOR& residual, 
	    typename std::vector<typename DOFHANDLER::cell_iterator>& cell_iter,
	    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell_iter,
	    const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	    Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,
	    Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc);
	
	template<typename PROBLEM, typename MATRIX, typename DOFHANDLER>
	  inline void ComputeMatrix_Recursive(
	    PROBLEM& pde, 
	    MATRIX& matrix, 
	    typename std::vector<typename DOFHANDLER::cell_iterator>& cell_iter,
	    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell_iter,
	    const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	    Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,
	    Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc);

	template<typename PROBLEM, typename DOFHANDLER>
	  inline SCALAR ComputeDomainScalar_Recursive(
	    PROBLEM& pde, 
	    typename std::vector<typename DOFHANDLER::cell_iterator>& cell_iter,
	    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell_iter,
	    const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	    Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc);

	template<typename PROBLEM, typename DOFHANDLER>
	  inline SCALAR ComputeBoundaryScalar_Recursive(
	    PROBLEM& pde, 
	    typename std::vector<typename DOFHANDLER::cell_iterator>& cell_iter,
	    typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell_iter,
	    const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	    Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& cdc);

        INTEGRATORDATACONT & _idc;

        std::map<std::string, const VECTOR*> _domain_data;
        std::map<std::string, const dealii::Vector<SCALAR>*> _param_data;
    };

  /**********************************Implementation*******************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::IntegratorMultiMesh(
        INTEGRATORDATACONT& idc) :
      _idc(idc)
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
          PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {
            residual = 0.;
   
	    const auto& dof_handler =
                pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

	    assert(dof_handler.size() == 2);
	    
	    const auto tria_cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
								       dof_handler[1]->GetDEALDoFHandler().get_tria());
	    const auto cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
								       dof_handler[1]->GetDEALDoFHandler());

	    auto cell_iter = cell_list.begin();
	    auto tria_cell_iter = tria_cell_list.begin();
	    
	    std::vector<decltype(tria_cell_iter->first)> tria_cell(2);
	    tria_cell[0] = tria_cell_iter->first;
	    tria_cell[1] = tria_cell_iter->second;

	    std::vector<decltype(cell_iter->first)> cell(2);
	    cell[0] = cell_iter->first;
	    cell[1] = cell_iter->second;
	    int coarse_index = 0; //cell[coarse_index] is the coarser of the two.
	    int fine_index = 0; //cell[fine_index] is the finer of the two (both indices are 2 = cell.size() if they are equally refined.

            // Generate the data containers.
            _idc.InitializeMMCDC(pde.GetUpdateFlags(),
				 *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, tria_cell,
				 this->GetParamData(), this->GetDomainData());
            auto& cdc = _idc.GetMultimeshCellDataContainer();


#if deal_II_dimension == 2 || deal_II_dimension == 3
            
            bool need_interfaces = pde.HasInterfaces();
            _idc.InitializeMMFDC(pde.GetFaceUpdateFlags(),
				 *(pde.GetBaseProblem().GetSpaceTimeHandler()),
				 cell,
				 tria_cell,
				 this->GetParamData(),
				 this->GetDomainData(),
				 need_interfaces);
            auto& fdc = _idc.GetMultimeshFaceDataContainer();
#endif

	    for(; cell_iter != cell_list.end(); cell_iter++)
	    {
	      cell[0] = cell_iter->first;
	      cell[1] = cell_iter->second;
	      tria_cell[0] = tria_cell_iter->first;
	      tria_cell[1] = tria_cell_iter->second;
	      FullMatrix<SCALAR> prolong_matrix;

	      if(cell[0]->has_children())
	      {
		prolong_matrix = IdentityMatrix(cell[1]->get_fe().dofs_per_cell);
		coarse_index =1;  
		fine_index = 0;
	      }
	      else
	      {
		if(cell[1]->has_children())
		{
		  prolong_matrix = IdentityMatrix(cell[0]->get_fe().dofs_per_cell);
		  coarse_index =0;  
		  fine_index = 1;
		}
		else
		{
		  assert(cell.size() ==2);
		  coarse_index = fine_index = 2;
		}
	      }
	      ComputeNonlinearResidual_Recursive(pde,residual,cell,tria_cell,prolong_matrix,coarse_index,fine_index,cdc,fdc);
	      tria_cell_iter++;
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
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearLhs(
          PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {
          {
	    throw DOpEException("This function needs to be implemented!", "IntegratorMultiMesh::ComputeNonlinearLhs");
//            residual = 0.;
//            // Begin integration
//            unsigned int dofs_per_cell ;
//
//            dealii::Vector<SCALAR> local_cell_vector;
//
//            std::vector<unsigned int> local_dof_indices;
//
//            const auto& dof_handler =
//                pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//            auto
//                cell =
//                    pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//            auto endc =
//                pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//            // Generate the data containers.
//            _idc.InitializeCDC(pde.GetUpdateFlags(),
//                *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell,
//                this->GetParamData(), this->GetDomainData());
//            auto& cdc = _idc.GetCellDataContainer();
//            //            CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim> cdc(
//            //                *(this->GetQuadratureFormula()), pde.GetUpdateFlags(),
//            //                *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, this->GetParamData(),
//            //                this->GetDomainData());
//
//#if deal_II_dimension == 2 || deal_II_dimension == 3
//            bool need_faces = pde.HasFaces();
//            bool need_interfaces = pde.HasInterfaces();
//            std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
//            bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
//
//            _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
//                *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                cell,
//                this->GetParamData(),
//                this->GetDomainData(),
//                need_interfaces);
//            auto & fdc = _idc.GetFaceDataContainer();
//#endif
//
//            for (; cell[0] != endc[0]; cell[0]++)
//              {
//                for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//                  {
//                    if (cell[dh] == endc[dh])
//                      {
//                        throw DOpEException(
//                            "Cellnumbers in DoFHandlers are not matching!",
//                            "IntegratorMultiMesh::ComputeNonlinearLhs");
//                      }
//                  }
//
//                cdc.ReInit();
//                dofs_per_cell = cell[0]->get_fe().dofs_per_cell;
//
//                local_cell_vector.reinit(dofs_per_cell);
//                local_cell_vector = 0;
//
//                local_dof_indices.resize(0);
//                local_dof_indices.resize(dofs_per_cell, 0);
//
//                //the second '1' plays only a role in the stationary case. In the non-stationary
//                //case, scale_ico is set by the time-stepping-scheme
//                pde.CellEquation(cdc, local_cell_vector, 1., 1.);
//
//#if deal_II_dimension == 2 || deal_II_dimension == 3
//                if(need_boundary_integrals)
//                  {
//                    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                      {
//                        if (cell[0]->face(face)->at_boundary()
//                            &&
//                            (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
//                                    cell[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
//                          {
//                            fdc.ReInit(face);
//                            pde.BoundaryEquation(fdc,local_cell_vector);
//                          }
//                      }
//                  }
//                if(need_faces)
//                  {
//                    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                      {
//                        if (cell[0]->neighbor_index(face) != -1)
//                          {
//                            fdc.ReInit(face);
//                            pde.FaceEquation(fdc, local_cell_vector);
//                          }
//                      }
//                  }
//                if( need_interfaces)
//                  {
//                    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                      {
//                        fdc.ReInit(face);
//                        if (cell[0]->neighbor_index(face) != -1
//                            &&
//                            fdc.GetMaterialId()!= fdc.GetNbrMaterialId())
//                          {
//                            fdc.ReInitNbr();
//                            pde.InterfaceEquation(fdc, local_cell_vector);
//                          }
//                      }
//                  }
//#endif
//                //LocalToGlobal
//                cell[0]->get_dof_indices(local_dof_indices);
//                for (unsigned int i = 0; i < dofs_per_cell; ++i)
//                  {
//                    residual(local_dof_indices[i]) += local_cell_vector(i);
//                  }
//
//                for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//                  {
//                    cell[dh]++;
//                  }
//              }
//
//            if (apply_boundary_values)
//              {
//                ApplyNewtonBoundaryValues(pde, residual);
//              }
          }
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      void
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs(
	PROBLEM& pde, VECTOR &residual, bool apply_boundary_values)
      {
	residual = 0.;
   
	const auto& dof_handler =
	  pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
	
	assert(dof_handler.size() == 2);
	
	const auto tria_cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
									dof_handler[1]->GetDEALDoFHandler().get_tria());
	const auto cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
								   dof_handler[1]->GetDEALDoFHandler());
	
	auto cell_iter = cell_list.begin();
	auto tria_cell_iter = tria_cell_list.begin();
	
	std::vector<decltype(tria_cell_iter->first)> tria_cell(2);
	tria_cell[0] = tria_cell_iter->first;
	tria_cell[1] = tria_cell_iter->second;
	
	std::vector<decltype(cell_iter->first)> cell(2);
	cell[0] = cell_iter->first;
	cell[1] = cell_iter->second;
	int coarse_index = 0; //cell[coarse_index] is the coarser of the two.
	int fine_index = 0; //cell[fine_index] is the finer of the two (both indices are 2 = cell.size() if they are equally refined.
	
	// Generate the data containers.
	_idc.InitializeMMCDC(pde.GetUpdateFlags(),
			     *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, tria_cell,
			     this->GetParamData(), this->GetDomainData());
	auto& cdc = _idc.GetMultimeshCellDataContainer();
	
#if deal_II_dimension == 2 || deal_II_dimension == 3
           
            _idc.InitializeMMFDC(pde.GetFaceUpdateFlags(),
				 *(pde.GetBaseProblem().GetSpaceTimeHandler()),
				 cell,
				 tria_cell,
				 this->GetParamData(),
				 this->GetDomainData());
            auto& fdc = _idc.GetMultimeshFaceDataContainer();
#endif

	    for(; cell_iter != cell_list.end(); cell_iter++)
	    {
	      cell[0] = cell_iter->first;
	      cell[1] = cell_iter->second;
	      tria_cell[0] = tria_cell_iter->first;
	      tria_cell[1] = tria_cell_iter->second;
	      FullMatrix<SCALAR> prolong_matrix;

	      if(cell[0]->has_children())
	      {
		prolong_matrix = IdentityMatrix(cell[1]->get_fe().dofs_per_cell);
		coarse_index =1;  
		fine_index = 0;
	      }
	      else
	      {
		if(cell[1]->has_children())
		{
		  prolong_matrix = IdentityMatrix(cell[0]->get_fe().dofs_per_cell);
		  coarse_index =0;  
		  fine_index = 1;
		}
		else
		{
		  assert(cell.size() ==2);
		  coarse_index = fine_index = 2;
		}
	      }
	      ComputeNonlinearRhs_Recursive(pde,residual,cell,tria_cell,prolong_matrix,coarse_index,fine_index,cdc,fdc);
	      tria_cell_iter++;
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
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix(
          PROBLEM& pde, MATRIX &matrix)
      {
        matrix = 0.;

        const auto& dof_handler =
            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();

	assert(dof_handler.size() == 2);
	    
	const auto tria_cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
									dof_handler[1]->GetDEALDoFHandler().get_tria());
	const auto cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
								   dof_handler[1]->GetDEALDoFHandler());

	auto cell_iter = cell_list.begin();
	auto tria_cell_iter = tria_cell_list.begin();
	
	std::vector<decltype(tria_cell_iter->first)> tria_cell(2);
	tria_cell[0] = tria_cell_iter->first;
	tria_cell[1] = tria_cell_iter->second;
	
	std::vector<decltype(cell_iter->first)> cell(2);
	cell[0] = cell_iter->first;
	cell[1] = cell_iter->second;
	int coarse_index = 0; //cell[coarse_index] is the coarser of the two.
	int fine_index = 0; //cell[fine_index] is the finer of the two (both indices are 2 = cell.size() if they are equally refined.

	// Generate the data containers.
        _idc.InitializeMMCDC(pde.GetUpdateFlags(),
			     *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, tria_cell,
			     this->GetParamData(), this->GetDomainData());
        auto& cdc = _idc.GetMultimeshCellDataContainer();

#if deal_II_dimension == 2 || deal_II_dimension == 3
        bool need_interfaces = pde.HasInterfaces();
        _idc.InitializeMMFDC(pde.GetFaceUpdateFlags(),
			     *(pde.GetBaseProblem().GetSpaceTimeHandler()),
			     cell,
			     tria_cell,
			     this->GetParamData(),
			     this->GetDomainData(),
			     need_interfaces);
        auto & fdc = _idc.GetMultimeshFaceDataContainer();
#endif

	for(; cell_iter != cell_list.end(); cell_iter++)
	{
	  cell[0] = cell_iter->first;
	  cell[1] = cell_iter->second;
	  tria_cell[0] = tria_cell_iter->first;
	  tria_cell[1] = tria_cell_iter->second;
	  FullMatrix<SCALAR> prolong_matrix;

	  if(cell[0]->has_children())
	  {
	    prolong_matrix = IdentityMatrix(cell[1]->get_fe().dofs_per_cell);
	    coarse_index =1;  
	    fine_index = 0;
	  }
	  else
	  {
	    if(cell[1]->has_children())
	    {
	      prolong_matrix = IdentityMatrix(cell[0]->get_fe().dofs_per_cell);
	      coarse_index =0;  
	      fine_index = 1;
	    }
	    else
	    {
	      assert(cell.size() ==2);
	      coarse_index = fine_index = 2;
	    }
	  }
	  ComputeMatrix_Recursive(pde,matrix,cell,tria_cell,prolong_matrix,coarse_index,fine_index,cdc,fdc);
	  tria_cell_iter++;
	}
      }

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
    int dim>
    template<typename PROBLEM>
      SCALAR
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar(
      PROBLEM& pde)
  {
    SCALAR ret = 0.;

    const auto& dof_handler =
      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      
    assert(dof_handler.size() == 2);
	    
    const auto tria_cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
								    dof_handler[1]->GetDEALDoFHandler().get_tria());
    const auto cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
							       dof_handler[1]->GetDEALDoFHandler());

    auto cell_iter = cell_list.begin();
    auto tria_cell_iter = tria_cell_list.begin();
    
    std::vector<decltype(tria_cell_iter->first)> tria_cell(2);
    tria_cell[0] = tria_cell_iter->first;
    tria_cell[1] = tria_cell_iter->second;
    
    std::vector<decltype(cell_iter->first)> cell(2);
    cell[0] = cell_iter->first;
    cell[1] = cell_iter->second;
    int coarse_index = 0; //cell[coarse_index] is the coarser of the two.
    int fine_index = 0; //cell[fine_index] is the finer of the two (both indices are 2 = cell.size() if they are equally refined.

    if (pde.HasFaces())
    {
      throw DOpEException("This function should not be called when faces are needed!",
			  "IntegratorMultiMesh::ComputeDomainScalar");
    }
    
    // Generate the data containers.
    _idc.InitializeMMCDC(pde.GetUpdateFlags(),
			 *(pde.GetBaseProblem().GetSpaceTimeHandler()), cell, tria_cell,
			 this->GetParamData(), this->GetDomainData());
    auto& cdc = _idc.GetMultimeshCellDataContainer();
    
    for(; cell_iter != cell_list.end(); cell_iter++)
    {
      cell[0] = cell_iter->first;
      cell[1] = cell_iter->second;
      tria_cell[0] = tria_cell_iter->first;
      tria_cell[1] = tria_cell_iter->second;
      FullMatrix<SCALAR> prolong_matrix;

      if(cell[0]->has_children())
      {
	prolong_matrix = IdentityMatrix(cell[1]->get_fe().dofs_per_cell);
	coarse_index =1;  
	fine_index = 0;
      }
      else
      {
	if(cell[1]->has_children())
	{
	  prolong_matrix = IdentityMatrix(cell[0]->get_fe().dofs_per_cell);
	  coarse_index =0;  
	  fine_index = 1;
	}
	else
	{
	  assert(cell.size() ==2);
	  coarse_index = fine_index = 2;
	}
      }
      ret += ComputeDomainScalar_Recursive(pde,cell,tria_cell,prolong_matrix,coarse_index,fine_index,cdc);
      tria_cell_iter++;
    }
    return ret; 
  }


  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePointScalar(
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
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar(
#if deal_II_dimension == 2 || deal_II_dimension == 3
	PROBLEM& pde
#else
          PROBLEM& /*pde*/
#endif	
      )
      {
	SCALAR ret = 0.;
#if deal_II_dimension == 2 || deal_II_dimension == 3
	// Begin integration
	const auto& dof_handler =
	  pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
	
	const auto tria_cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler().get_tria(),
									dof_handler[1]->GetDEALDoFHandler().get_tria());
	const auto cell_list = GridTools::get_finest_common_cells (dof_handler[0]->GetDEALDoFHandler(),
								   dof_handler[1]->GetDEALDoFHandler());
	    
	auto cell_iter = cell_list.begin();
	auto tria_cell_iter = tria_cell_list.begin();
	
	std::vector<decltype(tria_cell_iter->first)> tria_cell(2);
	tria_cell[0] = tria_cell_iter->first;
	tria_cell[1] = tria_cell_iter->second;
	
	std::vector<decltype(cell_iter->first)> cell(2);
	cell[0] = cell_iter->first;
	cell[1] = cell_iter->second;
	int coarse_index = 0; //cell[coarse_index] is the coarser of the two.
	int fine_index = 0; //cell[fine_index] is the finer of the two (both indices are 2 = cell.size() if they are equally refined.
	

	_idc.InitializeMMFDC(pde.GetFaceUpdateFlags(),
			   *(pde.GetBaseProblem().GetSpaceTimeHandler()),
			   cell, tria_cell,
			   this->GetParamData(),
			   this->GetDomainData());
	auto & fdc = _idc.GetMultimeshFaceDataContainer();

	std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
	bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
	if(!need_boundary_integrals)
	{
	  throw DOpEException("No boundary colors given!","IntegratorMultiMesh::ComputeBoundaryScalar");
	}

        for(; cell_iter != cell_list.end(); cell_iter++)
	{
	  cell[0] = cell_iter->first;
	  cell[1] = cell_iter->second;
	  tria_cell[0] = tria_cell_iter->first;
	  tria_cell[1] = tria_cell_iter->second;
	  FullMatrix<SCALAR> prolong_matrix;
	  
	  if(cell[0]->has_children())
	  {
	    prolong_matrix = IdentityMatrix(cell[1]->get_fe().dofs_per_cell);
	    coarse_index =1;  
	    fine_index = 0;
	  }
	  else
	  {
	    if(cell[1]->has_children())
	    {
	      prolong_matrix = IdentityMatrix(cell[0]->get_fe().dofs_per_cell);
	      coarse_index =0;  
	      fine_index = 1;
	    }
	    else
	    {
	      assert(cell.size() ==2);
	      coarse_index = fine_index = 2;
	    }
	  }
	  ret += ComputeBoundaryScalar_Recursive(pde,cell,tria_cell,prolong_matrix,coarse_index,fine_index,fdc);
	  tria_cell_iter++;
	}
#else
            throw DOpEException("Not implemented in this dimension!",
                "IntegratorMultiMesh::ComputeBoundaryScalar");
#endif

            return ret;
      }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM>
      SCALAR
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeFaceScalar(
#if deal_II_dimension == 2 || deal_II_dimension == 3
	PROBLEM& /*pde*/
#else
          PROBLEM& /*pde*/
#endif
      )
      {
	throw DOpEException("This function needs to be implemented!", "IntegratorMultiMesh::ComputeFaceScalar");
//          {
//            SCALAR ret = 0.;
//#if deal_II_dimension == 2 || deal_II_dimension == 3
//            // Begin integration
//            const auto& dof_handler =
//            pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//            auto cell = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//            auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//            _idc.InitializeFDC(pde.GetFaceUpdateFlags(),
//                *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                cell,
//                this->GetParamData(),
//                this->GetDomainData());
//            auto & fdc = _idc.GetFaceDataContainer();
//
//            bool need_faces = pde.HasFaces();
//            if(!need_faces)
//              {
//                throw DOpEException("No faces required!","IntegratorMultiMesh::ComputeFaceScalar");
//              }
//
//            for (;cell[0]!=endc[0]; cell[0]++)
//              {
//                for(unsigned int dh=1; dh<dof_handler.size(); dh++)
//                  {
//                    if( cell[dh] == endc[dh])
//                      {
//                        throw DOpEException("Cellnumbers in DoFHandlers are not matching!","IntegratorMultiMesh::ComputeFaceScalar");
//                      }
//                  }
//
//                if(need_faces)
//                  {
//                    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                      {
//                        if (cell[0]->neighbor_index(face) != -1)
//                          {
//                            fdc.ReInit(face);
//                            ret +=pde.FaceFunctional(fdc);
//                          }
//                      }
//                  }
//                for(unsigned int dh=1; dh<dof_handler.size(); dh++)
//                  {
//                    cell[dh]++;
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

  template <typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM>
    void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>
    ::ComputeNonlinearAlgebraicResidual (PROBLEM& pde, VECTOR &residual)
  {
    residual = 0.;
    pde.AlgebraicResidual(residual,this->GetParamData(),this->GetDomainData());
  }

  /*******************************************************************************************/
  
  template <typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM>
    void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>
    ::ComputeLocalControlConstraints (PROBLEM& pde, VECTOR &constraints)
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
    template<typename PROBLEM>
      void
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyNewtonBoundaryValues(
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
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyNewtonBoundaryValues(
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
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddDomainData(
        std::string name, const VECTOR* new_data)
    {
      if (_domain_data.find(name) != _domain_data.end())
        {
          throw DOpEException(
              "Adding multiple Data with name " + name + " is prohibited!",
              "IntegratorMultiMesh::AddDomainData");
        }
      _domain_data.insert(std::pair<std::string, const VECTOR*>(name, new_data));
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteDomainData(
        std::string name)
    {
      typename std::map<std::string, const VECTOR *>::iterator it =
          _domain_data.find(name);
      if (it == _domain_data.end())
        {
          throw DOpEException(
              "Deleting Data " + name + " is impossible! Data not found",
              "IntegratorMultiMesh::DeleteDomainData");
        }
      _domain_data.erase(it);
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    const std::map<std::string, const VECTOR*>&
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetDomainData() const
    {
      return _domain_data;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    void
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddParamData(
        std::string name, const dealii::Vector<SCALAR>* new_data)
    {
      if (_param_data.find(name) != _param_data.end())
        {
          throw DOpEException(
              "Adding multiple Data with name " + name + " is prohibited!",
              "IntegratorMultiMesh::AddParamData");
        }
      _param_data.insert(
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
          it = _param_data.find(name);
      if (it == _param_data.end())
        {
          throw DOpEException(
              "Deleting Data " + name + " is impossible! Data not found",
              "IntegratorMultiMesh::DeleteParamData");
        }
      _param_data.erase(it);
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    const std::map<std::string, const dealii::Vector<SCALAR>*>&
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetParamData() const
    {
      return _param_data;
    }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
     INTEGRATORDATACONT&
    IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainer() const
    {
      return _idc;
    }


  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename DOFHANDLER>
      void
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::InterpolateBoundaryValues(
          const DOpEWrapper::DoFHandler<dim, DOFHANDLER>* dof_handler,
          const unsigned int color, const dealii::Function<dim>& function,
          std::map<unsigned int, SCALAR>& boundary_values,
          const std::vector<bool>& comp_mask) const
      {
        dealii::VectorTools::interpolate_boundary_values(
            *(static_cast<const DOFHANDLER*> (dof_handler)), color, function,
            boundary_values, comp_mask);
      }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM, typename DOFHANDLER>
    void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearResidual_Recursive(
      PROBLEM& pde, 
      VECTOR& residual, 
      typename std::vector<typename DOFHANDLER::cell_iterator> &cell,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_cell,
      const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc)
  {
    if(!cell[0]->has_children() && ! cell[1]->has_children())
    {
          unsigned int dofs_per_cell;
	  dealii::Vector<SCALAR> local_cell_vector;
	  std::vector<unsigned int> local_dof_indices;

#if deal_II_dimension == 2 || deal_II_dimension == 3
	  bool need_faces = pde.HasFaces();
	  std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
	  bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
	  bool need_interfaces = pde.HasInterfaces();  
#endif
	  cdc.ReInit(coarse_index,fine_index,prolong_matrix);
	  
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
	  //FIXME Integrate on Faces of Cell[0] that contain a fine-cell face.
	  if(need_faces || need_interfaces)
	  {
	    throw DOpEException(" Faces on multiple meshes not implemented yet!", 
				"IntegratorMultiMesh::ComputeNonlinearResidual_Recursive");
	  }
	  unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the 
	                                       //zeros entry in the cell vector
	  
	  if(need_boundary_integrals && cell[b_index]->at_boundary())
	  {
	    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell[b_index]->face(face)->at_boundary()
		  &&
		  (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
			cell[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
	      {
		fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
		pde.BoundaryEquation(fdc,local_cell_vector);
		pde.BoundaryRhs(fdc,local_cell_vector,-1.);
	      }
	    }
	  }
          //     if(need_faces)
          //       {
          //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          //           {
          //             if (cell[0]->neighbor_index(face) != -1)
          //               {
          //                 fdc.ReInit(face);
          //                 pde.FaceEquation(fdc, local_cell_vector);
          //                 pde.FaceRhs(fdc, local_cell_vector,-1.);
          //               }
          //           }
          //       }
          //     if( need_interfaces)
          //       {
          //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          //           {
          //             fdc.ReInit(face);
          //             if (cell[0]->neighbor_index(face) != -1
          //                 &&
          //                 fdc.GetMaterialId()!= fdc.GetNbrMaterialId())
          //               {
          //                 fdc.ReInitNbr();
          //                 pde.InterfaceEquation(fdc, local_cell_vector);
          //               }
          //           }
          //       }
#endif
	  if(coarse_index == 0) //Need to transfer the computed residual to the one on the coarse cell
	  {
	    dealii::Vector<SCALAR> tmp(dofs_per_cell);
	    prolong_matrix.Tvmult(tmp,local_cell_vector);
	    
	    //LocalToGlobal
	    cell[0]->get_dof_indices(local_dof_indices);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      residual(local_dof_indices[i]) += tmp(i);
	    }
	  }
	  else //Testfunctions are already the right ones...
	  {
	    //LocalToGlobal
	    cell[0]->get_dof_indices(local_dof_indices);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      residual(local_dof_indices[i]) += local_cell_vector(i);
	    }
	  }
	  
    }//Endof the case on the finest level
    else
    {
      assert(fine_index != coarse_index);
      assert(cell[fine_index]->has_children());
      assert(!cell[coarse_index]->has_children());
      assert(tria_cell[fine_index]->has_children());
      assert(!tria_cell[coarse_index]->has_children());
      
      unsigned int local_n_dofs = cell[coarse_index]->get_fe().dofs_per_cell;
      
      typename DOFHANDLER::cell_iterator dofh_fine = cell[fine_index];
      typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_cell[fine_index];
      
      for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell;++child)
      {
	FullMatrix<SCALAR>   new_matrix(local_n_dofs);
	cell[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
								  prolong_matrix);
	cell[fine_index] = dofh_fine->child(child);
	tria_cell[fine_index] = tria_fine->child(child);
	
	ComputeNonlinearResidual_Recursive(pde,residual,cell,tria_cell,new_matrix, coarse_index, fine_index, cdc, fdc);
      }
    }
  }
  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM, typename DOFHANDLER>
    void IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs_Recursive(
      PROBLEM& pde, 
      VECTOR& residual, 
      typename std::vector<typename DOFHANDLER::cell_iterator> &cell,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_cell,
      const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc)
  {
    if(!cell[0]->has_children() && ! cell[1]->has_children())
    {
          unsigned int dofs_per_cell;
	  dealii::Vector<SCALAR> local_cell_vector;
	  std::vector<unsigned int> local_dof_indices;

#if deal_II_dimension == 2 || deal_II_dimension == 3
	  bool need_faces = pde.HasFaces();
	  std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
	  bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
	  bool need_interfaces = pde.HasInterfaces();  
#endif
	  cdc.ReInit(coarse_index,fine_index,prolong_matrix);
	  
	  dofs_per_cell = cell[0]->get_fe().dofs_per_cell;
	  
	  local_cell_vector.reinit(dofs_per_cell);
	  local_cell_vector = 0;
	  
	  local_dof_indices.resize(0);
	  local_dof_indices.resize(dofs_per_cell, 0);
	  
	  //the second '1' plays only a role in the stationary case. In the non-stationary
	  //case, scale_ico is set by the time-stepping-scheme
	  pde.CellRhs(cdc, local_cell_vector, 1.);
	  
#if deal_II_dimension == 2 || deal_II_dimension == 3
	  //FIXME Integrate on Faces of Cell[0] that contain a fine-cell face.
	  if(need_faces )
	  {
	    throw DOpEException(" Faces on multiple meshes not implemented yet!", 
				"IntegratorMultiMesh::ComputeNonlinearRhs_Recursive");
	  }
	  if(need_interfaces )
	  {
	    throw DOpEException(" Faces on multiple meshes not implemented yet!", 
				"IntegratorMultiMesh::ComputeNonlinearRhs_Recursive");
	  }
	  unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the 
	                                       //zeros entry in the cell vector
	  
	  if(need_boundary_integrals && cell[b_index]->at_boundary())
	  {
	    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell[b_index]->face(face)->at_boundary()
		  &&
		  (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
			cell[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
	      {
		fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
		pde.BoundaryRhs(fdc,local_cell_vector,1.);
	      }
	    }
	  }
          //     if(need_faces)
          //       {
          //         for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          //           {
          //             if (cell[0]->neighbor_index(face) != -1)
          //               {
          //                 fdc.ReInit(face);
          //                 pde.FaceRhs(fdc, local_cell_vector,1.);
          //               }
          //           }
          //       }
#endif
	  if(coarse_index == 0) //Need to transfer the computed residual to the one on the coarse cell
	  {
	    dealii::Vector<SCALAR> tmp(dofs_per_cell);
	    prolong_matrix.Tvmult(tmp,local_cell_vector);
	    
	    //LocalToGlobal
	    cell[0]->get_dof_indices(local_dof_indices);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      residual(local_dof_indices[i]) += tmp(i);
	    }
	  }
	  else //Testfunctions are already the right ones...
	  {
	    //LocalToGlobal
	    cell[0]->get_dof_indices(local_dof_indices);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      residual(local_dof_indices[i]) += local_cell_vector(i);
	    }
	  }
	  
    }//Endof the case on the finest level
    else
    {
      assert(fine_index != coarse_index);
      assert(cell[fine_index]->has_children());
      assert(!cell[coarse_index]->has_children());
      assert(tria_cell[fine_index]->has_children());
      assert(!tria_cell[coarse_index]->has_children());
      
      unsigned int local_n_dofs = cell[coarse_index]->get_fe().dofs_per_cell;
      
      typename DOFHANDLER::cell_iterator dofh_fine = cell[fine_index];
      typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_cell[fine_index];
      
      for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell;++child)
      {
	FullMatrix<SCALAR>   new_matrix(local_n_dofs);
	cell[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
								  prolong_matrix);
	cell[fine_index] = dofh_fine->child(child);
	tria_cell[fine_index] = tria_fine->child(child);
	
	ComputeNonlinearRhs_Recursive(pde,residual,cell,tria_cell,new_matrix, coarse_index, fine_index, cdc, fdc);
      }
    }
  }

  /*******************************************************************************************/

  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
      int dim>
    template<typename PROBLEM, typename MATRIX, typename DOFHANDLER>
      void
      IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix_Recursive(
	PROBLEM& pde, MATRIX &matrix, typename std::vector<typename DOFHANDLER::cell_iterator>& cell,
	typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell,
	const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
	Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc,
	Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc)
      {

	if(!cell[0]->has_children() && ! cell[1]->has_children())
	{
	  unsigned int dofs_per_cell;
	  std::vector<unsigned int> local_dof_indices;

#if deal_II_dimension == 2 || deal_II_dimension == 3
	  bool need_faces = pde.HasFaces();
	  bool need_interfaces = pde.HasInterfaces();
	  std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
	  bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
#endif

	  cdc.ReInit(coarse_index,fine_index,prolong_matrix);
	  dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

	  dealii::FullMatrix<SCALAR> local_cell_matrix(dofs_per_cell,
						       dofs_per_cell);
	  local_cell_matrix = 0;

	  local_dof_indices.resize(0);
	  local_dof_indices.resize(dofs_per_cell, 0);
	  pde.CellMatrix(cdc, local_cell_matrix);

#if deal_II_dimension == 2 || deal_II_dimension == 3
	  //FIXME Integrate on Faces of Cell[0] that contain a fine-cell face.
	  if(need_faces || need_interfaces)
	  {
	    throw DOpEException(" Faces on multiple meshes not implemented yet!", 
				"IntegratorMultiMesh::ComputeMatrix_Recursive");
	  }
	  unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the 
	                                       //zeros entry in the cell vector
	  if(need_boundary_integrals && cell[b_index]->at_boundary())
	  {
	    
	    for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell[b_index]->face(face)->at_boundary()
		  &&
		  (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
			cell[b_index]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
	      {
		fdc.ReInit(coarse_index,fine_index,prolong_matrix,face);
		pde.BoundaryMatrix(fdc, local_cell_matrix);
	      }
	    }
	  }
         //  if(need_faces)
         //    {
         //      for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
         //        {
         //          if (cell[0]->neighbor_index(face) != -1)
         //            {
         //              fdc.ReInit(face);
         //              pde.FaceMatrix(fdc, local_cell_matrix);
         //            }
         //        }
         //    }
         //  if( need_interfaces)
         //    {
         //      for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
         //        {
         //          fdc.ReInit(face);
         //          if (cell[0]->neighbor_index(face) != -1
         //              &&
         //              fdc.GetMaterialId()!= fdc.GetNbrMaterialId())
         //            {
         //              fdc.ReInitNbr();
	 //
         //              nbr_dofs_per_cell = fdc.GetNbrNDoFsPerCell();
         //              nbr_local_dof_indices.resize(0);
         //              nbr_local_dof_indices.resize(nbr_dofs_per_cell, 0);
         //              dealii::FullMatrix<SCALAR>  local_interface_matrix(dofs_per_cell,nbr_dofs_per_cell );
         //              local_interface_matrix = 0;
	 //
         //              pde.InterfaceMatrix(fdc, local_interface_matrix);
	 //
         //              cell[0]->get_dof_indices(local_dof_indices);
         //              cell[0]->neighbor(face)->get_dof_indices(nbr_local_dof_indices);
	 //
         //              for (unsigned int i = 0; i < dofs_per_cell; ++i)
         //                {
         //                  for (unsigned int j = 0; j < nbr_dofs_per_cell; ++j)
         //                    {
         //                      matrix.add(local_dof_indices[i], nbr_local_dof_indices[j],
         //                          local_interface_matrix(i, j));
         //                    } //endfor j
         //                } //endfor i
         //            }
         //        }
         //    }
#endif
	  if(coarse_index == 0) //Need to transfer the computed residual to the one on the coarse cell
	  {
	    dealii::FullMatrix<SCALAR> tmp(dofs_per_cell);
	    tmp = 0.;
	    prolong_matrix.Tmmult(tmp,local_cell_matrix);
	    local_cell_matrix = 0.;
	    tmp.mmult(local_cell_matrix,prolong_matrix);

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
          }
	  else
	  {
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
	  }

	}//Endof the case on the finest level
	else
	{
	  assert(fine_index != coarse_index);
	  assert(cell[fine_index]->has_children());
	  assert(!cell[coarse_index]->has_children());
	  assert(tria_cell[fine_index]->has_children());
	  assert(!tria_cell[coarse_index]->has_children());
	  
	  unsigned int local_n_dofs = cell[coarse_index]->get_fe().dofs_per_cell;
	  
	  typename DOFHANDLER::cell_iterator dofh_fine = cell[fine_index];
	  typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_cell[fine_index];
	  
	  for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell;++child)
	  {
	    FullMatrix<SCALAR>   new_matrix(local_n_dofs);
	    cell[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
									       prolong_matrix);
	    cell[fine_index] = dofh_fine->child(child);
	    tria_cell[fine_index] = tria_fine->child(child);
	    
	    ComputeMatrix_Recursive(pde,matrix,cell,tria_cell,new_matrix, coarse_index, fine_index, cdc, fdc);
	  }
	}
      } 

  /*******************************************************************************************/
  
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM, typename DOFHANDLER>
    SCALAR IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar_Recursive(
      PROBLEM& pde, 
      typename std::vector<typename DOFHANDLER::cell_iterator> &cell,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_cell,
      const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_CellDataContainer<DOFHANDLER, VECTOR, dim>& cdc)
  {
    if(!cell[0]->has_children() && ! cell[1]->has_children())
    {
      SCALAR ret = 0.;
      cdc.ReInit(coarse_index,fine_index,prolong_matrix);
      ret += pde.CellFunctional(cdc);
      return ret;
    }    //Endof the case on the finest level
    else
    {
      assert(fine_index != coarse_index);
      assert(cell[fine_index]->has_children());
      assert(!cell[coarse_index]->has_children());
      assert(tria_cell[fine_index]->has_children());
      assert(!tria_cell[coarse_index]->has_children());
      
      unsigned int local_n_dofs = cell[coarse_index]->get_fe().dofs_per_cell;
      
      typename DOFHANDLER::cell_iterator dofh_fine = cell[fine_index];
      typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_cell[fine_index];
      SCALAR ret = 0.;
      for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell;++child)
      {
	FullMatrix<SCALAR>   new_matrix(local_n_dofs);
	cell[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
									   prolong_matrix);
	cell[fine_index] = dofh_fine->child(child);
	tria_cell[fine_index] = tria_fine->child(child);
	
	ret += ComputeDomainScalar_Recursive(pde,cell,tria_cell,new_matrix, coarse_index, fine_index, cdc);
      }
      return ret;
    }
  }  
  /*******************************************************************************************/
  
  template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,int dim>
    template<typename PROBLEM, typename DOFHANDLER>
    SCALAR IntegratorMultiMesh<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar_Recursive(
      PROBLEM& pde, 
      typename std::vector<typename DOFHANDLER::cell_iterator> &cell,
      typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_cell,
      const FullMatrix<SCALAR>& prolong_matrix,unsigned int coarse_index,unsigned int fine_index,
      Multimesh_FaceDataContainer<DOFHANDLER, VECTOR, dim>& fdc)
  {
    if(!cell[0]->has_children() && ! cell[1]->has_children())
    {
      SCALAR ret = 0.;
      std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
      unsigned int b_index = fine_index%2; //This takes care that if fine_index ==2 then we select the 
                                           //zeros entry in the cell vector
      for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
      {
	if (cell[b_index]->face(face)->at_boundary()
	    &&
	    (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
		  cell[b_index]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
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
      assert(cell[fine_index]->has_children());
      assert(!cell[coarse_index]->has_children());
      assert(tria_cell[fine_index]->has_children());
      assert(!tria_cell[coarse_index]->has_children());
      
      unsigned int local_n_dofs = cell[coarse_index]->get_fe().dofs_per_cell;
      
      typename DOFHANDLER::cell_iterator dofh_fine = cell[fine_index];
      typename dealii::Triangulation<dim>::cell_iterator tria_fine = tria_cell[fine_index];
      SCALAR ret = 0.;
      for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell;++child)
      {
	FullMatrix<SCALAR>   new_matrix(local_n_dofs);
	cell[coarse_index]->get_fe().get_prolongation_matrix(child).mmult (new_matrix,
									   prolong_matrix);
	cell[fine_index] = dofh_fine->child(child);
	tria_cell[fine_index] = tria_fine->child(child);
	
	ret += ComputeBoundaryScalar_Recursive(pde,cell,tria_cell,new_matrix, coarse_index, fine_index, fdc);
      }
      return ret;
    }
  }
//ENDOF NAMESPACE DOpE
}
#endif

