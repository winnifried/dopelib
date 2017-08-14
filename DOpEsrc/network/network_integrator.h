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

#ifndef Network_Integrator_H_
#define Network_Integrator_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

#include <vector>

#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/dwrdatacontainer.h>
#include <container/residualestimator.h>
#include <basic/dopetypes.h>


namespace DOpE
{
  namespace Networks
  {
    /**
     * This class is used to integrate the righthand side, matrix and so on.
     * It assumes that one uses the same triangulation for the control and state variable.
     *
     * @template INTEGRATORDATACONT       The type of the integratordatacontainer, which has
     *                                    manages the basic data for integration (quadrature,
     *                                    elementdatacontainer, facedatacontainer etc.)
     * @template VECTOR                   Class of the vectors which we use in the integrator.
     * @template SCALAR                   Type of the scalars we use in the integrator.
     * @template dim                      dimesion of the domain
     */
    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    class Network_Integrator
    {
    public:
      Network_Integrator(INTEGRATORDATACONT &idc1);
      /**
       * This constructor gets two INTEGRATORDATACONT. One for the evaluation
       * of the integrals during the assembling and residual computation, the
       * second one for the evaluation of functionals.
       */
      Network_Integrator(INTEGRATORDATACONT &idc1, INTEGRATORDATACONT &idc2);

      ~Network_Integrator();

      /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
       */
      void
      ReInit();

      /**
       * This method is used to evaluate the residual of the nonlinear equation.
      *
      * It assumes that the PROBLEM provides methods named
      * ElementEquation, ElementRhs, BoundaryEquation, BoundaryRhs, FaceEquation, FaceRhs,
      * InterfaceEquation, and PointRhs
       * For the calculation of the respective quantities, see, e.g., OptProblemContainer for
      * details on these methods.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param resiual                   A vector which contains the residual after completing the method.
      */
      template<typename PROBLEM>
      void
      ComputeNonlinearResidual(PROBLEM &pde, VECTOR &residual);
      /**
       * This method is used to evaluate the left hand side of the residual of the nonlinear equation
      * This are all terms depending on the nonlinear variable.
      * The use of this function can be advantageous to avoid recalculation of terms that do
      * not change with update of the evaluation point.
      *
      * It assumes that the PROBLEM provides methods named
      * ElementEquation,BoundaryEquation, FaceEquation, and InterfaceEquation
       * For the calculation of the respective quantities, see, e.g., OptProblemContainer for
      * details on these methods.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param resiual                   A vector which contains the residual after completing the method.
      */
      template<typename PROBLEM>
      void
      ComputeNonlinearLhs(PROBLEM &pde, VECTOR &residual);
      /**
       * This method is used to evaluate the righ hand side of the residual of the nonlinear equation
      * This are all terms independent of the nonlinear variable.
      * The use of this function can be advantageous to avoid recalculation of terms that do
      * not change with update of the evaluation point.
      *
      * It assumes that the PROBLEM provides methods named
      * ElementRhs, BoundaryRhs, FaceRhs, and PointRhs
       * For the calculation of the respective quantities, see, e.g., OptProblemContainer for
      * details on these methods.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param resiual                   A vector which contains the residual after completing the method.
      */
      template<typename PROBLEM>
      void
      ComputeNonlinearRhs(PROBLEM &pde, VECTOR &residual);
      /**
        * This method is used to calculate the matrix corresponding to the linearized equation.
      *
      * It assumes that the PROBLEM provides methods named
      * ElementMatrix, BoundaryMatrix, FaceMatrix, and InterfaceRhs
       * For the calculation of the respective quantities, see, e.g., OptProblemContainer for
      * details on these methods.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param matrix                    A matrix which contains the matrix after completing the method.
      */
      template<typename PROBLEM>
      void
      ComputeMatrix(PROBLEM &pde, dealii::BlockSparseMatrix<double> &matrix);

      /**
       * This routine is used as a dummy to allow for the solutions of problems that do not need any
      * integration, i.e., that don't involve integration.
      *
      * It is assumed that a method AlgebraicResidual in
      * PROBLEM provides a method for the calculation of the residual.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param resiual                   A vector which contains the residual after completing the method.
      */
      template<typename PROBLEM>
      void
      ComputeNonlinearAlgebraicResidual(PROBLEM &pde, VECTOR &residual);
      /**
       * This method is used to calculate local constraints, i.e., constraints that do not need
      * any integration but can be calculated directly from the values in the unknowns.
      * Typical examples are box constraints in Lagrange elements.
      *
      *
      * It is assumed that a method ComputeLocalControlConstraints in
      * PROBLEM provides a method for the calculation of the residual.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the nonlinear pde.
      * @param constraints               A vector which contains the constraints.
      */
      template<typename PROBLEM>
      void
      ComputeLocalControlConstraints(PROBLEM &pde, VECTOR &constraints);

      /**
      * This methods evaluates functionals that are given by an integration over the spatial domain.
      *
      * It is assumed that PROBLEM provides a method ElementFunctional.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the functional.
      *
      * @return                          The value of the functional
      */
      template<typename PROBLEM>
      SCALAR
      ComputeDomainScalar(PROBLEM &pde);
      /**
      * This methods evaluates functionals that are given by evaluation in certain
      * fixed points in the domain.
      *
      * It is assumed that PROBLEM provides a method PointFunctional.
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the functional.
      *
      * @return                          The value of the functional
      */
      template<typename PROBLEM>
      SCALAR
      ComputePointScalar(PROBLEM &pde);
      /**
      * This methods evaluates functionals that are given by evaluation of integrals
      * over parts of the boundary.
      *
      * It is assumed that PROBLEM provides a method BoundaryFunctional.
      *
      * This routine assumes that a corresponding calculation method is provided by PROBLEM
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the functional.
      *
      * @return                          The value of the functional
      */
      template<typename PROBLEM>
      SCALAR
      ComputeBoundaryScalar(PROBLEM &pde);
      /**
      * This methods evaluates functionals that are given by evaluation of integrals over
      * certain faces in the domain.
      *
      * It is assumed that PROBLEM provides a method FaceFunctional.
      *
      * This routine assumes that a corresponding calculation method is provided by PROBLEM
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the functional.
      *
      * @return                          The value of the functional
      */
      template<typename PROBLEM>
      SCALAR
      ComputeFaceScalar(PROBLEM &pde);
      /**
      * This methods evaluates functionals that are given algebraic manipulation
      * of the unknowns.
      *
      * It is assumed that PROBLEM provides a method ComputeAlgebraicScalar.
      *
      * This routine assumes that a corresponding calculation method is provided by PROBLEM
      *
      * @tparam <PROBLEM>                The problem description
      *
      * @param pde                       The object containing the description of the functional.
      *
      * @return                          The value of the functional
      */
      template<typename PROBLEM>
      SCALAR
      ComputeAlgebraicScalar(PROBLEM &pde);

      /**
       * This method applies inhomogeneous dirichlet boundary values.
       *
       * It is assumed that PROBLEM provides functions GetDirichletCompMask,
       * GetDirichletColors, and GetDirichletValues to find the appropriate
       * boundary values.
       *
       * @tparam <PROBLEM>                The problem description
       *
       * @param pde                       The object containing the description of the functional.
       * @param u                         The vector to which the boundary values are applied.
       */
      template<typename PROBLEM>
      void
      ApplyInitialBoundaryValues(PROBLEM &pde, VECTOR &u);

      /**
        * This method is used in optimization problems with
        * control in the dirichlet values, it then needs
        * to calculate the transposed to the control-to-dirichletvalues
        * mapping. Note that this is currently only usable
        * of the control is 0-dimensional, i.e., a fixed number of
        * parameters. Thus it is only implemented in
        * IntegratorMixedDimensions and provided here for compatibility reasons
        * only.
        *
        * @tparam <PROBLEM>                The problem description
        *
        * @param pde                       The object containing the description of the functional.
        * @param u                         The vector to which the boundary values are applied.
        */
      template<typename PROBLEM>
      void
      ApplyTransposedInitialBoundaryValues(PROBLEM &pde, VECTOR &u);

      /**
       * This function can be used to pass domain data, i.e., finite element
       * functions, to the problem. The added data will automatically be
       * initialized on the elements where the integration takes place.
       *
       * Adding multiple data with the same name is prohibited.
       *
       * @param name         An identifier by which the PROBLEM in the
       *                     corresponding Compute* method can access the data.
       * @param new_data     A pointer to the data to be added.
       */
      inline void
      AddDomainData(std::string name, const VECTOR *new_data);
      /**
       * This function is used to remove previously added domain data from the
       * integrator.
       *
       * Deleting data that is not present in the integrator will
       * cause an exception.
       *
       * @param name         The identifier for the data to be removed from
       *                     the integrator.
       */
      inline void
      DeleteDomainData(std::string name);
      /**
       * This function can be used to pass parameter data, i.e., data independent
       * of the spatial position, to the problem.
       *
       * Adding multiple data with the same name is prohibited.
       *
       * @param name         An identifier by which the PROBLEM in the
       *                     corresponding Compute* method can access the data.
       * @param new_data     A pointer to the data to be added.
       */
      inline void
      AddParamData(std::string name, const dealii::Vector<SCALAR> *new_data);
      /**
       * This function is used to remove previously added parameter data from the
       * integrator.
       *
       * Deleting data that is not present in the integrator will
       * cause an exception.
       *
       * @param name         The identifier for the data to be removed from
       *                     the integrator.
       */inline void
      DeleteParamData(std::string name);

      /**
       * This function is used to calculate indicators based
       * on DWR-type data containes. This means it is
       * assumed that an additional SpaceTimeHandler is used
       * for the calculation of the weights.
       * See DWRDataContainter for details.
       *
       * @tparam <PROBLEM>    The problem description
       * @tparam <STH>        An additional SpaceTimeHandler
       * @tparam <EDC>        An additional ElementDataContainer
       * @tparam <FDC>        An additional FaceDataContainer
       *
       * @param pde           The problem
       * @param dwrc          The data container for the error estimation.
       *                      This object also stores the refinement indicators.
       */
      template<typename PROBLEM, class STH, class EDC, class FDC>
      void
      ComputeRefinementIndicators(PROBLEM &pde,
                                  DWRDataContainer<STH, INTEGRATORDATACONT, EDC, FDC, VECTOR> &dwrc);
      /**
       * This function is used to calculate indicators based
       * on Residual-type data containes. See ResidualErrorContainer
       * for details.
       *
       * @tparam <PROBLEM>    The problem description
       *
       * @param pde           The problem
       * @param dwrc          The data container for the error estimation.
       *                      This object also stores the refinement indicators.
       */
      template<typename PROBLEM>
      void
      ComputeRefinementIndicators(PROBLEM &pde,
                                  ResidualErrorContainer<VECTOR> &dwrc);

      inline INTEGRATORDATACONT &
      GetIntegratorDataContainer() const;

      inline INTEGRATORDATACONT &
      GetIntegratorDataContainerFunc() const;

    protected:
      /**
       * This grants access to the domain data stored in the integrator.
       *
       * @return The map with the stored data.
       */
      inline const std::map<std::string, const VECTOR *> &
      GetDomainData() const;
      /**
       * This grants access to the parameter data stored in the integrator.
       *
       * @return The map with the stored data.
       */
      inline const std::map<std::string, const dealii::Vector<SCALAR>*> &
      GetParamData() const;

      /**
       * This function is used to add usergiven righthandsides to the residual.
       * This is usefull if the rhs is expensive to compute.
       *
       * The given rhs needs to be given to the integrator as domain data
       * with the name "fixed_rhs"
       *
       *
       * @param s          A scaling parameter
       * @param residual   The given residual. Upon exit from this method
       *                   This vector is residual = residual + s* "fixed_rhs"
       *
       */
      inline void AddPresetRightHandSide(double s, VECTOR &residual) const;

    private:
      typedef MethodOfLines_Network_SpaceTimeHandler<FESystem,DoFHandler,BlockVector<double>,0,1> STH_;
      typedef MethodOfLines_StateSpaceTimeHandler<FESystem,DoFHandler,SparsityPattern,Vector<double>,1> PIPE_STH_;

      template<template<int, int> class DH>
      void
      InterpolateBoundaryValues(
        const DOpEWrapper::Mapping<dim, DH> &mapping,
        const DOpEWrapper::DoFHandler<dim, DH> *dof_handler,
        const unsigned int color, const dealii::Function<dim> &function,
        std::map<unsigned int, SCALAR> &boundary_values,
        const std::vector<bool> &comp_mask) const;

      /**
       * Integrate the nonlinear residual on a given pipe
       * Assumes that the pipe is selected prior to calling this function
       *
       *
       */
      template<typename PROBLEM>
      void
      ComputeNonlinearPipeResidual(unsigned int pipe,
                                   unsigned int n_pipes,
                                   unsigned int n_comp,
                                   PIPE_STH_* sth, PROBLEM &pde,
                                   dealii::Vector<double> &residual,
                                   dealii::Vector<double> &flow,
                                   std::vector<bool> &fluxes_in_outflow);
      /**
       * Integrate the matrix on a given pipe
       * Assumes that the pipe is selected prior to calling this function
       *
       *
       */
      template<typename PROBLEM>
      void
      ComputePipeMatrix(unsigned int pipe,
                        unsigned int n_pipes,
                        unsigned int n_comp,
                        PIPE_STH_* sth, PROBLEM &pde,
                        dealii::SparseMatrix<double> &matrix,
                        dealii::FullMatrix<double> &inflow_coupling,
                        dealii::FullMatrix<double> &outflow_coupling,
                        std::vector<bool> &fluxes_in_outflow);

      /**
       * Integrate a functional on a pipe
       * Assumes that the pipe is selected prior to calling this function
       *
       *
       */
      template<typename PROBLEM>
      SCALAR
      ComputePipeDomainScalar(unsigned int pipe,
                              PIPE_STH_* sth, PROBLEM &pde);

//        /**
//         * Given a vector of active element iterators and a facenumber, checks if the face
//         * belongs to an 'interface' (i.e. the adjoining elements have different material ids).
//         *
//         * @template ELEMENTITERATOR   Class of the elementiterator.
//         *
//         * @param   element            The element in question.
//         * @param   face            Local number of the face for which we ask if it is
//         *                          at the interface.
//         */
//        template<typename ELEMENTITERATOR>
//          bool
//          AtInterface(ELEMENTITERATOR& element, unsigned int face)
//          {
//            if (element[0]->neighbor_index(face) != -1)
//              if (element[0]->material_id()
//                  != element[0]->neighbor(face)->material_id())
//                return true;
//            return false;
//          }

      INTEGRATORDATACONT &idc1_;
      INTEGRATORDATACONT &idc2_;

      std::map<std::string, const VECTOR *> domain_data_;
      std::map<std::string, const dealii::Vector<SCALAR>*> param_data_;
    };

    /**********************************Implementation*******************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::Network_Integrator(
      INTEGRATORDATACONT &idc)
      : idc1_(idc), idc2_(idc)
    {
      if (dim != 1)
        {
          throw DOpEException("Not implemented in given dimensions!","Network_Integrator::Network_Integrator");
        }
    }

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::Network_Integrator(
      INTEGRATORDATACONT &idc1, INTEGRATORDATACONT &idc2)
      : idc1_(idc1), idc2_(idc2)
    {
      if (dim != 1)
        {
          throw DOpEException("Not implemented in given dimensions!","Network_Integrator::Network_Integrator");
        }
    }

    /**********************************Implementation*******************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::~Network_Integrator()
    {

    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ReInit()
    {

    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearResidual(
      PROBLEM &pde, VECTOR &residual)
    {
      residual = 0.;

      STH_* sth = dynamic_cast<STH_ *>(pde.GetBaseProblem().GetSpaceTimeHandler());

      if (sth == NULL)
        {
          throw DOpEException("Using Networks::Network_Integrator with wrong SpaceTimeHandler","Networks::Network_Integrator::ComputeNonlinearResidual");
        }
      unsigned int n_pipes = sth->GetNPipes();
      unsigned int n_comp = sth->GetFESystem("state").n_components();
      std::vector<bool> present_in_outflow(2*n_comp*n_pipes,false);
      dealii::Vector<double> flow(4*n_comp);

      for (unsigned int p = 0; p < n_pipes; p++)
        {
          flow = 0.;
          std::vector<bool> tmp(2*n_comp,false);

          sth->SelectPipe(p);
          ComputeNonlinearPipeResidual(p,n_pipes,n_comp,sth->GetPipeSTH(),pde,residual.block(p),flow,tmp);

          //Sort the inconsistency in the outflow variable to the right position
          for (unsigned int c = 0; c < n_comp; c++)
            {
              //Residual for the first flow stored at n_comp*p+c in block n_pipes
              residual.block(n_pipes)[n_comp*p+c] = flow[c];
              residual.block(n_pipes)[n_comp*p+c] += flow[n_comp+c];
              residual.block(n_pipes)[n_pipes*n_comp+n_comp*p+c] = flow[2*n_comp+c];
              residual.block(n_pipes)[n_pipes*n_comp+n_comp*p+c] += flow[3*n_comp+c];
              //Sort the bool-flags for the fluxes in the outflow to the right place

              assert(tmp[c]||tmp[n_comp+c]);//At least on must be outflow
              //(Both are allowed to deal with ) algebraic pipes where the boundary is induced by the
              //pipe
              present_in_outflow[p*n_comp+c]=tmp[c];
              present_in_outflow[n_pipes*n_comp+p*n_comp+c]=tmp[n_comp+c];
            }
        }

      sth->SelectPipe(n_pipes);

      //Find the current linearization point
      typename std::map<std::string,const VECTOR *>::iterator it = domain_data_.find("last_newton_solution");
      if (it == domain_data_.end())
        {
          throw DOpEException(
            "Could not find data named last_newton_solution!",
            "Network_Integrator::ComputeNonlinearResidual");
        }
      const VECTOR &lin_pt = *(it->second);

      //Cross Coupling of the nodal variables, e.g. continuity of the flux...
      dealii::Vector<double> coupling(2*n_pipes*n_comp);
      pde.PipeCouplingResidual(coupling,lin_pt.block(n_pipes),present_in_outflow);
      assert(coupling.size() == 2*n_pipes*n_comp);
      for (unsigned int i = 0 ; i < coupling.size(); i++)
        {
          residual.block(n_pipes)[i]+=coupling[i];
        }

      //Check if some preset righthandside exists.
      AddPresetRightHandSide(-1.,residual);
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearLhs(
      PROBLEM &/*pde*/, VECTOR &/*residual*/)
    {
//    {
//
//      residual = 0.;
//      // Begin integration
//      unsigned int dofs_per_element;
//
//      dealii::Vector<SCALAR> local_vector;
//
//      std::vector<unsigned int> local_dof_indices;
//
//      const auto &dof_handler =
//        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//      auto element =
//        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//      auto endc =
//        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//      // Generate the data containers.
//      GetIntegratorDataContainer().InitializeEDC(pde.GetUpdateFlags(),
//                                                 *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
//                                                 this->GetParamData(), this->GetDomainData());
//      auto &edc = GetIntegratorDataContainer().GetElementDataContainer();
//
//      bool need_faces = pde.HasFaces();
//      bool need_interfaces = pde.HasInterfaces();
//      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
//      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
//
//      GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
//                                                 *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                                                 element,
//                                                 this->GetParamData(),
//                                                 this->GetDomainData(),
//                                                 need_interfaces);
//      auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();
//
//      for (; element[0] != endc[0]; element[0]++)
//        {
//          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//            {
//              if (element[dh] == endc[dh])
//                {
//                  throw DOpEException(
//                    "Elementnumbers in DoFHandlers are not matching!",
//                    "mNetwork_Integrator::ComputeNonlinearLhs");
//                }
//            }
//
//          edc.ReInit();
//          dofs_per_element = element[0]->get_fe().dofs_per_cell;
//
//          local_vector.reinit(dofs_per_element);
//          local_vector = 0;
//
//          local_dof_indices.resize(0);
//          local_dof_indices.resize(dofs_per_element, 0);
//
//          //the second '1' plays only a role in the stationary case. In the non-stationary
//          //case, scale_ico is set by the time-stepping-scheme
//          pde.ElementEquation(edc, local_vector, 1., 1.);
//
//          if (need_boundary_integrals)
//            {
//              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//#if DEAL_II_VERSION_GTE(8,3,0)
//                  if (element[0]->face(face)->at_boundary()
//                      &&
//                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
//                            element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
//#else
//                  if (element[0]->face(face)->at_boundary()
//                      &&
//                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
//                            element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
//#endif
//                    {
//                      fdc.ReInit(face);
//                      pde.BoundaryEquation(fdc,local_vector, 1., 1.);
//                    }
//                }
//            }
//          if (need_faces)
//            {
//              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//                  if (element[0]->neighbor_index(face) != -1)
//                    {
//                      fdc.ReInit(face);
//                      pde.FaceEquation(fdc, local_vector, 1., 1.);
//                    }
//                }
//            }
//
//          if ( need_interfaces)
//            {
//
//              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//                  //auto face_it = element[0]->face(face);
//                  // first, check if we are at an interface, i.e. not the neighbour exists and
//                  // it has a different material_id than the actual element
//                  if (pde.AtInterface(element, face))
//                    {
//                      //There exist now 3 different scenarios, given the actual element and face:
//                      // The neighbour behind this face is [ more | as much | less] refined
//                      // than/as the actual element. We have to distinguish here only between the case 1
//                      // and the other two, because these will be distinguished in in the FaceDataContainer.
//
//                      if (element[0]->neighbor(face)->has_children())
//                        {
//                          //first: neighbour is finer
//
//                          for (unsigned int subface_no=0;
//                               subface_no < element[0]->face(face)->n_children();
//                               ++subface_no)
//                            {
//                              //TODO Now here we have to initialise the subface_values on the
//                              // actual element and then the facevalues of the neighbours
//                              fdc.ReInit(face, subface_no);
//                              fdc.ReInitNbr();
//
//                              pde.InterfaceEquation(fdc, local_vector, 1., 1.);
//
//                            }
//                        }
//                      else
//                        {
//                          // either neighbor is as fine as this element or
//                          // it is coarser
//
//                          fdc.ReInit(face);
//                          fdc.ReInitNbr();
//                          pde.InterfaceEquation(fdc, local_vector, 1., 1.);
//                        }
//                    }                    //endif atinterface
//                }                    //endfor face
//            }                    //endif need_interfaces
//          //LocalToGlobal
//          const auto &C = pde.GetDoFConstraints();
//          element[0]->get_dof_indices(local_dof_indices);
//          C.distribute_local_to_global(local_vector, local_dof_indices, residual);
//
//          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//            {
//              element[dh]++;
//            }
//        }
//
//    }
      abort();
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearRhs(
      PROBLEM &/*pde*/, VECTOR &/*residual*/)
    {
//    residual = 0.;
//    // Begin integration
//    unsigned int dofs_per_element;
//    dealii::Vector<SCALAR> local_vector;
//    std::vector<unsigned int> local_dof_indices;
//
//    const bool need_point_rhs = pde.HasPoints();
//
//    const auto &dof_handler =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//    auto element =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//    auto endc =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//    // Initialize the data containers.
//    GetIntegratorDataContainer().InitializeEDC(pde.GetUpdateFlags(),
//                                               *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
//                                               this->GetParamData(), this->GetDomainData());
//    auto &edc = GetIntegratorDataContainer().GetElementDataContainer();
//
////       We don't have interface terms in the Rhs! They are all to be included in the Equation!
////        bool need_interfaces = pde.HasInterfaces();
////        if(need_interfaces )
////        {
////          throw DOpEException(" Interfaces not implemented yet!",
////              "Network_Integrator::ComputeNonlinearRhs");
////        }
//    bool need_faces = pde.HasFaces();
//    std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
//    bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
//
//    GetIntegratorDataContainer().InitializeFDC(pde.GetFaceUpdateFlags(),
//                                               *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                                               element,
//                                               this->GetParamData(),
//                                               this->GetDomainData());
//    auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();
//
//    for (; element[0] != endc[0]; element[0]++)
//      {
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            if (element[dh] == endc[dh])
//              {
//                throw DOpEException(
//                  "Elementnumbers in DoFHandlers are not matching!",
//                  "Network_Integrator::ComputeNonlinearRhs");
//              }
//          }
//
//        edc.ReInit();
//        dofs_per_element = element[0]->get_fe().dofs_per_cell;
//
//        local_vector.reinit(dofs_per_element);
//        local_vector = 0;
//
//        local_dof_indices.resize(0);
//        local_dof_indices.resize(dofs_per_element, 0);
//        pde.ElementRhs(edc, local_vector, 1.);
//
//        if (need_boundary_integrals)
//          {
//            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//              {
//#if DEAL_II_VERSION_GTE(8,3,0)
//                if (element[0]->face(face)->at_boundary()
//                    &&
//                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
//                          element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
//#else
//                if (element[0]->face(face)->at_boundary()
//                    &&
//                    (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
//                          element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
//#endif
//                  {
//                    fdc.ReInit(face);
//                    pde.BoundaryRhs(fdc,local_vector,1.);
//                  }
//              }
//          }
//        if (need_faces)
//          {
//            for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//              {
//                if (element[0]->neighbor_index(face) != -1)
//                  {
//                    fdc.ReInit(face);
//                    pde.FaceRhs(fdc, local_vector);
//                  }
//              }
//          }
//        //LocalToGlobal
//        const auto &C = pde.GetDoFConstraints();
//        element[0]->get_dof_indices(local_dof_indices);
//        C.distribute_local_to_global(local_vector, local_dof_indices, residual);
//
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            element[dh]++;
//          }
//      }
//
//    //check if we need the evaluation of PointRhs
//    if (need_point_rhs)
//      {
//        VECTOR point_rhs;
//        pde.PointRhs(this->GetParamData(), this->GetDomainData(), point_rhs,
//                     1.);
//        residual += point_rhs;
//      }
//    //Check if some preset righthandside exists.
//    AddPresetRightHandSide(1.,residual);
      abort();
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeMatrix(
      PROBLEM &pde, dealii::BlockSparseMatrix<double> &matrix)
    {
      matrix = 0.;

      STH_* sth = dynamic_cast<STH_ *>(pde.GetBaseProblem().GetSpaceTimeHandler());

      if (sth == NULL)
        {
          throw DOpEException("Using Networks::Network_Integrator with wrong SpaceTimeHandler","Networks::Network_Integrator::ComputeMatrix");
        }

      const std::vector<std::vector<unsigned int> > &left_vals = sth->GetPipeToLeftDoF();
      const std::vector<std::vector<unsigned int> > &right_vals = sth->GetPipeToRightDoF();

      unsigned int n_pipes = sth->GetNPipes();
      unsigned int n_comp = sth->GetFESystem("state").n_components();
      std::vector<bool> present_in_outflow(2*n_comp*n_pipes,false);
      dealii::FullMatrix<double> inflow_matrix(2*n_comp);
      dealii::FullMatrix<double> outflow_matrix(2*n_comp);


      for (unsigned int p = 0; p < n_pipes; p++)
        {
          inflow_matrix = 0.;
          outflow_matrix = 0.;
          sth->SelectPipe(p);
          std::vector<bool> tmp(2*n_comp,false);
          ComputePipeMatrix(p,n_pipes,n_comp,sth->GetPipeSTH(),pde,matrix.block(p,p),inflow_matrix,outflow_matrix,tmp);
          assert(inflow_matrix.n()==inflow_matrix.m());
          assert(outflow_matrix.n()==outflow_matrix.m());
          assert(inflow_matrix.n()==outflow_matrix.n());
          assert(inflow_matrix.n()==2*n_comp);
          assert(left_vals[p].size()==n_comp);
          //Sort in and outflow matrices to the right places.
          for (unsigned int c = 0; c < n_comp; c++)
            {
              for (unsigned int d = 0; d < n_comp; d++)
                {
                  //Inflow coupling
                  if (fabs(inflow_matrix(c,d)) > std::numeric_limits<double>::min())
                    matrix.block(p,n_pipes).set(left_vals[p][c],p*n_comp+d,inflow_matrix(c,d));
                  if (fabs(inflow_matrix(n_comp+c,d)) > std::numeric_limits<double>::min())
                    matrix.block(p,n_pipes).set(right_vals[p][c],p*n_comp+d,inflow_matrix(n_comp+c,d));
                  if (fabs(inflow_matrix(c,n_comp+d)) > std::numeric_limits<double>::min())
                    matrix.block(p,n_pipes).set(left_vals[p][c],n_pipes*n_comp+p*n_comp+d,inflow_matrix(c,n_comp+d));
                  if (fabs(inflow_matrix(n_comp+c,n_comp+d)) > std::numeric_limits<double>::min())
                    matrix.block(p,n_pipes).set(right_vals[p][c],n_pipes*n_comp+p*n_comp+d,inflow_matrix(n_comp+c,n_comp+d));
                  //Outflow coupling
                  if (fabs(outflow_matrix(c,d)) > std::numeric_limits<double>::min()) //left value is outflow
                    matrix.block(n_pipes,p).set(p*n_comp+c,left_vals[p][d],outflow_matrix(c,d));
                  if (fabs(outflow_matrix(n_comp+c,d)) > std::numeric_limits<double>::min())
                    matrix.block(n_pipes,p).set(n_pipes*n_comp+p*n_comp+c,left_vals[p][d],outflow_matrix(n_comp+c,d));
                  if (fabs(outflow_matrix(c,n_comp+d)) > std::numeric_limits<double>::min())
                    matrix.block(n_pipes,p).set(p*n_comp+c,right_vals[p][d],outflow_matrix(c,n_comp+d));
                  if (fabs(outflow_matrix(n_comp+c,n_comp+d)) > std::numeric_limits<double>::min())
                    matrix.block(n_pipes,p).set(n_pipes*n_comp+p*n_comp+c,right_vals[p][d],outflow_matrix(n_comp+c,n_comp+d));
                }
              //Sort the bool-flags for the fluxes in the outflow to the right place
              assert(tmp[c]||tmp[n_comp+c]);
              present_in_outflow[p*n_comp+c]=tmp[c];
              present_in_outflow[n_pipes*n_comp+p*n_comp+c]=tmp[n_comp+c];
            }
        }
      sth->SelectPipe(n_pipes);
      //Now the coupling block(n_pipes,n_pipes)
      //No linearization point needed since coupling conditions are linear!
      pde.CouplingMatrix(matrix.block(n_pipes,n_pipes),present_in_outflow);

//    //Print Matrix
//    for(unsigned int p = 0; p <= n_pipes; p++)
//    {
//      for(unsigned int i = 0; i < matrix.block(p,0).m(); i++)
//      {
//  for(unsigned int r = 0; r <= n_pipes; r++)
//  {
//    for(unsigned int j = 0; j < matrix.block(p,r).n(); j++)
//    {
//      if( p != n_pipes && r != n_pipes)
//      {
//        std::cout<<"     *";
//      }
//      else
//      {
//        std::cout<<std::setw(6)<<std::setprecision(1)<<matrix.block(p,r).el(i,j);
//      }
//    }
//    std::cout<<" | ";
//  }
//  std::cout<<std::endl;
//      }
//      for(unsigned int r = 0; r <= n_pipes; r++)
//      {
//  for(unsigned int j = 0; j < matrix.block(p,r).n(); j++)
//  {
//    std::cout<<"------";
//  }
//  std::cout<<" | ";
//      }
//      std::cout<<std::endl;
//    }

    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeDomainScalar(
      PROBLEM &pde)
    {
      SCALAR ret = 0;

      STH_* sth = dynamic_cast<STH_ *>(pde.GetBaseProblem().GetSpaceTimeHandler());

      if (sth == NULL)
        {
          throw DOpEException("Using Networks::Network_Integrator with wrong SpaceTimeHandler","Networks::Network_Integrator::ComputeDomainScalar");
        }

      unsigned int n_pipes = sth->GetNPipes();

      for (unsigned int p = 0; p < n_pipes; p++)
        {
          sth->SelectPipe(p);
          ret += ComputePipeDomainScalar(p, sth->GetPipeSTH(), pde);
        }
      sth->SelectPipe(n_pipes);

      return ret;
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePointScalar(
      PROBLEM &/*pde*/)
    {
//
//    {
//      SCALAR ret = 0.;
//      ret += pde.PointFunctional(this->GetParamData(),
//                                 this->GetDomainData());
//
//      return ret;
//    }
      abort();
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeBoundaryScalar(
      PROBLEM &/*pde*/
    )
    {
//    {
//      SCALAR ret = 0.;
//      // Begin integration
//      const auto &dof_handler =
//        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//      auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//      GetIntegratorDataContainerFunc().InitializeFDC(pde.GetFaceUpdateFlags(),
//                                                     *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                                                     element,
//                                                     this->GetParamData(),
//                                                     this->GetDomainData());
//      auto &fdc = GetIntegratorDataContainerFunc().GetFaceDataContainer();
//
//      std::vector<unsigned int> boundary_functional_colors = pde.GetBoundaryFunctionalColors();
//      bool need_boundary_integrals = (boundary_functional_colors.size() > 0);
//      if (!need_boundary_integrals)
//        {
//          throw DOpEException("No boundary colors given!","Network_Integrator::ComputeBoundaryScalar");
//        }
//
//      for (; element[0]!=endc[0]; element[0]++)
//        {
//          for (unsigned int dh=1; dh<dof_handler.size(); dh++)
//            {
//              if ( element[dh] == endc[dh])
//                {
//                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!","Network_Integrator::ComputeBoundaryScalar");
//                }
//            }
//
//          if (need_boundary_integrals)
//            {
//              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//#if DEAL_II_VERSION_GTE(8,3,0)
//                  if (element[0]->face(face)->at_boundary()
//                      &&
//                      (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
//                            element[0]->face(face)->boundary_id()) != boundary_functional_colors.end()))
//#else
//                  if (element[0]->face(face)->at_boundary()
//                      &&
//                      (find(boundary_functional_colors.begin(),boundary_functional_colors.end(),
//                            element[0]->face(face)->boundary_indicator()) != boundary_functional_colors.end()))
//#endif
//                    {
//                      fdc.ReInit(face);
//                      ret += pde.BoundaryFunctional(fdc);
//                    }
//                }
//            }
//          for (unsigned int dh=1; dh<dof_handler.size(); dh++)
//            {
//              element[dh]++;
//            }
//        }
//
//      return ret;
//
//    }
      abort();
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeFaceScalar(
      PROBLEM &/*pde*/
    )
    {

//    {
//      SCALAR ret = 0.;
//      // Begin integration
//      const auto &dof_handler =
//        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//      auto element = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//      auto endc = pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//      GetIntegratorDataContainerFunc().InitializeFDC(pde.GetFaceUpdateFlags(),
//                                                     *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                                                     element,
//                                                     this->GetParamData(),
//                                                     this->GetDomainData());
//      auto &fdc = GetIntegratorDataContainerFunc().GetFaceDataContainer();
//
//      bool need_faces = pde.HasFaces();
//      if (!need_faces)
//        {
//          throw DOpEException("No faces required!","Network_Integrator::ComputeFaceScalar");
//        }
//
//      for (; element[0]!=endc[0]; element[0]++)
//        {
//          for (unsigned int dh=1; dh<dof_handler.size(); dh++)
//            {
//              if ( element[dh] == endc[dh])
//                {
//                  throw DOpEException("Elementnumbers in DoFHandlers are not matching!","Network_Integrator::ComputeFaceScalar");
//                }
//            }
//
//          if (need_faces)
//            {
//              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//                  if (element[0]->neighbor_index(face) != -1)
//                    {
//                      fdc.ReInit(face);
//                      ret +=pde.FaceFunctional(fdc);
//                    }
//                }
//            }
//          for (unsigned int dh=1; dh<dof_handler.size(); dh++)
//            {
//              element[dh]++;
//            }
//        }
//      return ret;
//    }
      abort();
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeAlgebraicScalar(
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

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearAlgebraicResidual(
      PROBLEM &pde, VECTOR &residual)
    {
      residual = 0.;
      pde.AlgebraicResidual(residual, this->GetParamData(),
                            this->GetDomainData());
      //Check if some preset righthandside exists.
      AddPresetRightHandSide(-1.,residual);
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeLocalControlConstraints(
      PROBLEM &pde, VECTOR &constraints)
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
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyTransposedInitialBoundaryValues(
      PROBLEM & /*pde*/, VECTOR &/*u*/)
    {
      // Is not required here ...
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ApplyInitialBoundaryValues(
      PROBLEM &pde, VECTOR &u)
    {

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
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddDomainData(
      std::string name, const VECTOR *new_data)
    {
      if (domain_data_.find(name) != domain_data_.end())
        {
          throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "Network_Integrator::AddDomainData");
        }
      domain_data_.insert(
        std::pair<std::string, const VECTOR *>(name, new_data));
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteDomainData(
      std::string name)
    {
      typename std::map<std::string, const VECTOR *>::iterator it =
        domain_data_.find(name);
      if (it == domain_data_.end())
        {
          throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "Network_Integrator::DeleteDomainData");
        }
      domain_data_.erase(it);
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    const std::map<std::string, const VECTOR *> &
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetDomainData() const
    {
      return domain_data_;
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddParamData(
      std::string name, const dealii::Vector<SCALAR> *new_data)
    {
      if (param_data_.find(name) != param_data_.end())
        {
          throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "Network_Integrator::AddParamData");
        }
      param_data_.insert(
        std::pair<std::string, const dealii::Vector<SCALAR>*>(name,
                                                              new_data));
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::DeleteParamData(
      std::string name)
    {
      typename std::map<std::string, const dealii::Vector<SCALAR>*>::iterator it =
        param_data_.find(name);
      if (it == param_data_.end())
        {
          throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "Network_Integrator::DeleteParamData");
        }
      param_data_.erase(it);
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    const std::map<std::string, const dealii::Vector<SCALAR>*> &
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetParamData() const
    {
      return param_data_;
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM, class STH, class EDC, class FDC>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeRefinementIndicators(
      PROBLEM &pde,
      DWRDataContainer<STH, INTEGRATORDATACONT, EDC, FDC, VECTOR> &dwrc)
    {
//    unsigned int n_error_comps = dwrc.GetNErrorComps();
//    //for primal and dual part of the error
//    std::vector<double> element_sum(n_error_comps, 0);
//    element_sum.resize(n_error_comps, 0);
//
//    // Begin integration
//    const auto &dof_handler =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//    auto element =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//    auto endc =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//    const auto &dof_handler_weight = dwrc.GetWeightSTH().GetDoFHandler();
//    auto element_weight = dwrc.GetWeightSTH().GetDoFHandlerBeginActive();
//    auto endc_high = dwrc.GetWeightSTH().GetDoFHandlerEnd();
//
//    // Generate the data containers. Notice that we use the quadrature
//    //formula from the higher order idc!.
//    GetIntegratorDataContainer().InitializeEDC(
//      dwrc.GetWeightIDC().GetQuad(), pde.GetUpdateFlags(),
//      *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
//      this->GetParamData(), this->GetDomainData());
//    auto &edc = GetIntegratorDataContainer().GetElementDataContainer();
//
//    dwrc.GetWeightIDC().InitializeEDC(pde.GetUpdateFlags(),
//                                      dwrc.GetWeightSTH(), element_weight, this->GetParamData(),
//                                      dwrc.GetWeightData());
//    auto &edc_weight = dwrc.GetElementWeight();
//
//    // we want to integrate the face-terms only once, so
//    // we store the values on each face in this map
//    // and distribute it at the end to the adjacent elements.
//#if deal_II_dimension > 1
//    typename std::map<typename dealii::Triangulation<dim>::face_iterator,std::vector<double> >
//    face_integrals;
//#else
//    //Points (Faces in 1d) have no working iterator, use vertex_number instead
//    typename std::map<unsigned int,std::vector<double> >
//    face_integrals;
//#endif
//    // initialize the map with a big value to make sure
//    // that we take notice if we forget to add a face
//    // during the error estimation process
//    auto element_it = element[0];
//    std::vector<double> face_init(n_error_comps,-1e20);
//    for (; element_it != endc[0]; element_it++)
//      {
//        for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
//          {
//#if deal_II_dimension > 1
//            face_integrals[element_it->face(face_no)] = face_init;
//#else
//      face_integrals[element_it->face(face_no)->vertex_index()] = face_init;
//#endif
//          }
//      }
//
////        bool need_faces = pde.HasFaces();
////        bool need_interfaces = pde.HasInterfaces();
////        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
////        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
//
//    GetIntegratorDataContainer().InitializeFDC(dwrc.GetWeightIDC().GetFaceQuad(),
//                                               pde.GetFaceUpdateFlags(),
//                                               *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//                                               element,
//                                               this->GetParamData(),
//                                               this->GetDomainData(),
//                                               true);
//    auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();
//
//    dwrc.GetWeightIDC().InitializeFDC(pde.GetFaceUpdateFlags(),
//                                      dwrc.GetWeightSTH(),
//                                      element_weight,
//                                      this->GetParamData(),
//                                      dwrc.GetWeightData(),
//                                      true);
//
//    for (unsigned int element_index = 0; element[0] != endc[0];
//         element[0]++, element_index++)
//      {
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            if (element[dh] == endc[dh])
//              {
//                throw DOpEException(
//                  "Elementnumbers in DoFHandlers are not matching!",
//                  "Network_Integrator::ComputeRefinementIndicators");
//              }
//          }
//        for (unsigned int dh = 0; dh < dof_handler_weight.size(); dh++)
//          {
//            if (element_weight[dh] == endc_high[dh])
//              {
//                throw DOpEException(
//                  "Elementnumbers in DoFHandlers are not matching!",
//                  "Network_Integrator::ComputeRefinementIndicators");
//              }
//          }
//        element_sum.clear();
//        element_sum.resize(n_error_comps, 0);
//
//        edc.ReInit();
//        edc_weight.ReInit();
//
//        //first the element-residual
//        pde.ElementErrorContribution(edc, dwrc, element_sum, 1.);
//        for (unsigned int l =0; l < n_error_comps; l ++)
//          {
//            dwrc.GetErrorIndicators(l)(element_index) = element_sum[l];
//          }
//        //dwrc.GetPrimalErrorIndicators()(element_index) = element_sum[0];
//        //dwrc.GetDualErrorIndicators()(element_index) = element_sum[1];
//        element_sum.clear();
//        element_sum.resize(n_error_comps, 0);
//
//
//        //Now to the face terms. We compute them only once for each face and distribute the
//        //afterwards. We choose always to work from the coarser element, if both neigbors of the
//        //face are on the same level, we pick the one with the lower index
//        for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//          {
//            auto face_it = element[0]->face(face);
//
//            //check if the face lies at a boundary
//            if (face_it->at_boundary())
//              {
//                fdc.ReInit(face);
//                dwrc.GetFaceWeight().ReInit(face);
//                pde.BoundaryErrorContribution(fdc, dwrc, element_sum, 1.);
//
//#if deal_II_dimension > 1
//                Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                        ExcInternalError());
//                Assert (face_integrals[element[0]->face(face)] == face_init,
//                        ExcInternalError());
//
//                face_integrals[element[0]->face(face)] = element_sum;
//#else
//                Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                        ExcInternalError());
//                Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                        ExcInternalError());
//
//                face_integrals[element[0]->face(face)->vertex_index()] = element_sum;
//#endif
//                element_sum.clear();
//                element_sum.resize(n_error_comps,0.);
//              }
//            else
//              {
//                //There exist now 3 different scenarios, given the actual element and face:
//                // The neighbour behind this face is [ more | as much | less] refined
//                // than/as the actual element. We have to distinguish here only between the case 1
//                // and the other two, because these will be distinguished in in the FaceDataContainer.
//                if (element[0]->neighbor(face)->has_children())
//                  {
//                    //first: neighbour is finer
//                    std::vector<double> sum(n_error_comps,0.);
//                    for (unsigned int subface_no=0;
//                         subface_no < element[0]->face(face)->n_children();
//                         ++subface_no)
//                      {
//                        //TODO Now here we have to initialise the subface_values on the
//                        // actual element and then the facevalues of the neighbours
//                        fdc.ReInit(face, subface_no);
//                        fdc.ReInitNbr();
//                        dwrc.GetFaceWeight().ReInit(face, subface_no);
//
//                        pde.FaceErrorContribution(fdc, dwrc, element_sum, 1.);
//                        for (unsigned int l =0; l < n_error_comps; l ++)
//                          {
//                            sum[l] += element_sum[l];
//                          }
//#if deal_II_dimension > 1
//                        face_integrals[element[0]->neighbor_child_on_subface(face, subface_no)
//                                       ->face(element[0]->neighbor_of_neighbor(face))] = element_sum;
//#else
//                        face_integrals[element[0]->neighbor_child_on_subface(face, subface_no)
//                                       ->face(element[0]->neighbor_of_neighbor(face))->vertex_index()] = element_sum;
//#endif
//                        element_sum.clear();
//                        element_sum.resize(n_error_comps,0);
//                      }
//
//#if deal_II_dimension > 1
//                    Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                            ExcInternalError());
//                    Assert (face_integrals[element[0]->face(face)] == face_init,
//                            ExcInternalError());
//
//                    face_integrals[element[0]->face(face)] = sum;
//#else
//                    Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                            ExcInternalError());
//                    Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                            ExcInternalError());
//
//                    face_integrals[element[0]->face(face)->vertex_index()] = sum;
//#endif
//                    element_sum.clear();
//                    element_sum.resize(n_error_comps,0);
//                  }
//                else
//                  {
//                    // either neighbor is as fine as this element or
//                    // it is coarser
//                    Assert(element[0]->neighbor(face)->level() <= element[0]->level(),ExcInternalError());
//                    //now we work always from the coarser element. if both elements
//                    //are on the same level, we pick the one with the lower index
//                    if (element[0]->level() == element[0]->neighbor(face)->level()
//                        && element[0]->index() < element[0]->neighbor(face)->index())
//                      {
//                        fdc.ReInit(face);
//                        fdc.ReInitNbr();
//                        dwrc.GetFaceWeight().ReInit(face);
//
//                        pde.FaceErrorContribution(fdc, dwrc, element_sum, 1.);
//#if deal_II_dimension > 1
//                        Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                                ExcInternalError());
//                        Assert (face_integrals[element[0]->face(face)] == face_init,
//                                ExcInternalError());
//
//                        face_integrals[element[0]->face(face)] = element_sum;
//#else
//                        Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                                ExcInternalError());
//                        Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                                ExcInternalError());
//
//                        face_integrals[element[0]->face(face)->vertex_index()] = element_sum;
//#endif
//                        element_sum.clear();
//                        element_sum.resize(n_error_comps,0);
//                      }
//                  }
//              }
//          }                  //endfor faces
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            element[dh]++;
//          }
//        for (unsigned int dh = 0; dh < dof_handler_weight.size(); dh++)
//          {
//            element_weight[dh]++;
//          }
//      }                  //endfor element
//    //now we have to incorporate the face and boundary_values
//    //into
//    unsigned int present_element = 0;
//    element =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//    for (;
//         element[0] !=endc[0]; element[0]++, ++present_element)
//      {
//        for (unsigned int face_no = 0;
//             face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
//          {
//#if deal_II_dimension > 1
//            Assert(
//              face_integrals.find(element[0]->face(face_no)) != face_integrals.end(),
//              ExcInternalError());
//#else
//            Assert(
//              face_integrals.find(element[0]->face(face_no)->vertex_index()) != face_integrals.end(),
//              ExcInternalError());
//#endif
//            if (element[0]->face(face_no)->at_boundary())
//              {
//                for (unsigned int l =0; l < n_error_comps; l ++)
//                  {
//#if deal_II_dimension > 1
//                    dwrc.GetErrorIndicators(l)(present_element) +=
//                      face_integrals[element[0]->face(face_no)][l];
//#else
//                    dwrc.GetErrorIndicators(l)(present_element) +=
//                      face_integrals[element[0]->face(face_no)->vertex_index()][l];
//#endif
//                  }
////              dwrc.GetPrimalErrorIndicators()(present_element) +=
////              face_integrals[element[0]->face(face_no)][0];
////              dwrc.GetDualErrorIndicators()(present_element) +=
////              face_integrals[element[0]->face(face_no)][1];
//              }
//            else
//              {
//                for (unsigned int l =0; l < n_error_comps; l ++)
//                  {
//#if deal_II_dimension > 1
//                    dwrc.GetErrorIndicators(l)(present_element) +=
//                      0.5*face_integrals[element[0]->face(face_no)][l];
//#else
//                    dwrc.GetErrorIndicators(l)(present_element) +=
//                      0.5*face_integrals[element[0]->face(face_no)->vertex_index()][l];
//#endif
//                  }
////              dwrc.GetPrimalErrorIndicators()(present_element) +=
////              0.5 * face_integrals[element[0]->face(face_no)][0];
////              dwrc.GetDualErrorIndicators()(present_element) +=
////              0.5 * face_integrals[element[0]->face(face_no)][1];
//              }
//
//          }
//      }
      abort();
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeRefinementIndicators(
      PROBLEM &pde,
      ResidualErrorContainer<VECTOR> &dwrc)
    {
//    //for primal and dual part of the error
//    std::vector<double> element_sum(2, 0);
//    element_sum.resize(2, 0);
//    // Begin integration
//    const auto &dof_handler =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
//    auto element =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//    auto endc =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
//
//    {
//      //Add Weights
//      auto wd = dwrc.GetWeightData().begin();
//      auto wend = dwrc.GetWeightData().end();
//      for (; wd != wend; wd++)
//        {
//          AddDomainData(wd->first, wd->second);
//        }
//    }
//
//    // Generate the data containers. Notice that we use the quadrature
//    // formula from the higher order idc!.
//    GetIntegratorDataContainer().InitializeEDC(pde.GetUpdateFlags(),
//                                               *(pde.GetBaseProblem().GetSpaceTimeHandler()), element,
//                                               this->GetParamData(), this->GetDomainData());
//    auto &edc = GetIntegratorDataContainer().GetElementDataContainer();
//
//    //we want to integrate the face-terms only once
//#if deal_II_dimension > 1
//    typename std::map<typename dealii::Triangulation<dim>::face_iterator,std::vector<double> >
//    face_integrals;
//#else
//    //Points (Faces in 1d) have no working iterator, use vertex_number instead
//    typename std::map<unsigned int,std::vector<double> >
//    face_integrals;
//#endif
////initialize the map
//    auto element_it = element[0];
//    std::vector<double> face_init(2,-1e20);
//    for (; element_it != endc[0]; element_it++)
//      {
//        for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
//          {
//#if deal_II_dimension > 1
//            face_integrals[element_it->face(face_no)] = face_init;
//#else
//            face_integrals[element_it->face(face_no)->vertex_index()] = face_init;
//#endif
//    }
//      }
//
////        bool need_faces = pde.HasFaces();
////        bool need_interfaces = pde.HasInterfaces();
////        std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
////        bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
//
//    GetIntegratorDataContainer().InitializeFDC(
//      pde.GetFaceUpdateFlags(),
//      *(pde.GetBaseProblem().GetSpaceTimeHandler()),
//      element,
//      this->GetParamData(),
//      this->GetDomainData(),
//      true);
//    auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();
//
//    for (unsigned int element_index = 0; element[0] != endc[0];
//         element[0]++, element_index++)
//      {
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            if (element[dh] == endc[dh])
//              {
//                throw DOpEException(
//                  "Elementnumbers in DoFHandlers are not matching!",
//                  "Network_Integrator::ComputeRefinementIndicators");
//              }
//          }
//
//        element_sum.clear();
//        element_sum.resize(2, 0);
//
//        edc.ReInit();
//        dwrc.InitElement(element[0]->diameter());
//        //first the element-residual
//        pde.ElementErrorContribution(edc, dwrc, element_sum, 1.);
//        dwrc.GetPrimalErrorIndicators()(element_index) = element_sum[0];
//        dwrc.GetDualErrorIndicators()(element_index) = element_sum[1];
//        element_sum.clear();
//        element_sum.resize(2, 0);
//        //Now to the face terms. We compute them only once for each face and distribute the
//        //afterwards. We choose always to work from the coarser element, if both neigbors of the
//        //face are on the same level, we pick the one with the lower index
//
//        for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
//          {
//            auto face_it = element[0]->face(face);
//
//            //check if the face lies at a boundary
//            if (face_it->at_boundary())
//              {
//                fdc.ReInit(face);
//#if deal_II_dimension > 1
//                dwrc.InitFace(element[0]->face(face)->diameter());
//#else
//                dwrc.InitFace(element[0]->diameter());
//#endif
//                pde.BoundaryErrorContribution(fdc, dwrc, element_sum, 1.);
//
//#if deal_II_dimension > 1
//                Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                        ExcInternalError());
//                Assert (face_integrals[element[0]->face(face)] == face_init,
//                        ExcInternalError());
//                face_integrals[element[0]->face(face)] = element_sum;
//#else
//                Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                        ExcInternalError());
//                Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                        ExcInternalError());
//                face_integrals[element[0]->face(face)->vertex_index()] = element_sum;
//#endif
//                element_sum.clear();
//                element_sum.resize(2,0.);
//              }
//            else
//              {
//                //There exist now 3 different scenarios, given the actual element and face:
//                // The neighbour behind this face is [ more | as much | less] refined
//                // than/as the actual element. We have to distinguish here only between the case 1
//                // and the other two, because these will be distinguished in in the FaceDataContainer.
//                if (element[0]->neighbor(face)->has_children())
//                  {
//                    //first: neighbour is finer
//                    std::vector<double> sum(2,0.);
//                    for (unsigned int subface_no=0;
//                         subface_no < element[0]->face(face)->n_children();
//                         ++subface_no)
//                      {
//                        //TODO Now here we have to initialise the subface_values on the
//                        // actual element and then the facevalues of the neighbours
//                        fdc.ReInit(face, subface_no);
//                        fdc.ReInitNbr();
//#if deal_II_dimension > 1
//                        dwrc.InitFace(element[0]->face(face)->diameter());
//#else
//                        dwrc.InitFace(element[0]->diameter());
//#endif
//
//                        pde.FaceErrorContribution(fdc, dwrc, element_sum, 1.);
//                        sum[0]= element_sum[0];
//                        sum[1]= element_sum[1];
//                        element_sum.clear();
//                        element_sum.resize(2,0);
//#if deal_II_dimension > 1
//                        face_integrals[element[0]->neighbor_child_on_subface(face, subface_no)
//                                       ->face(element[0]->neighbor_of_neighbor(face))] = element_sum;
//#else
//                        face_integrals[element[0]->neighbor_child_on_subface(face, subface_no)
//                                       ->face(element[0]->neighbor_of_neighbor(face))->vertex_index()] = element_sum;
//#endif
//                        element_sum.clear();
//                        element_sum.resize(2,0.);
//                      }
//
//#if deal_II_dimension > 1
//                    Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                            ExcInternalError());
//                    Assert (face_integrals[element[0]->face(face)] == face_init,
//                            ExcInternalError());
//
//                    face_integrals[element[0]->face(face)] = sum;
//#else
//                    Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                            ExcInternalError());
//                    Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                            ExcInternalError());
//
//                    face_integrals[element[0]->face(face)->vertex_index()] = sum;
//#endif
//                  }
//                else
//                  {
//                    // either neighbor is as fine as this element or
//                    // it is coarser
//                    Assert(element[0]->neighbor(face)->level() <= element[0]->level(),ExcInternalError());
//                    //now we work always from the coarser element. if both elements
//                    //are on the same level, we pick the one with the lower index
//                    if (element[0]->level() == element[0]->neighbor(face)->level()
//                        && element[0]->index() < element[0]->neighbor(face)->index())
//                      {
//                        fdc.ReInit(face);
//                        fdc.ReInitNbr();
//#if deal_II_dimension > 1
//                        dwrc.InitFace(element[0]->face(face)->diameter());
//#else
//                        dwrc.InitFace(element[0]->diameter());
//#endif
//
//                        pde.FaceErrorContribution(fdc, dwrc, element_sum, 1.);
//#if deal_II_dimension > 1
//                        Assert (face_integrals.find (element[0]->face(face)) != face_integrals.end(),
//                                ExcInternalError());
//                        Assert (face_integrals[element[0]->face(face)] == face_init,
//                                ExcInternalError());
//
//                        face_integrals[element[0]->face(face)] = element_sum;
//#else
//                        Assert (face_integrals.find (element[0]->face(face)->vertex_index()) != face_integrals.end(),
//                                ExcInternalError());
//                        Assert (face_integrals[element[0]->face(face)->vertex_index()] == face_init,
//                                ExcInternalError());
//
//                        face_integrals[element[0]->face(face)->vertex_index()] = element_sum;
//#endif
//                        element_sum.clear();
//                        element_sum.resize(2,0);
//                      }
//                  }
//              }
//          }                  //endfor faces
////          }//end else
//
//        for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
//          {
//            element[dh]++;
//          }
//      }                  //endfor element
//    //now we have to incorporate the face and boundary_values
//    //into
//    unsigned int present_element = 0;
//    element =
//      pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
//    for (;
//         element[0] !=endc[0]; element[0]++, ++present_element)
//      for (unsigned int face_no = 0;
//           face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
//      {
//#if deal_II_dimension > 1
//  Assert(
//    face_integrals.find(element[0]->face(face_no)) != face_integrals.end(),
//    ExcInternalError());
//  dwrc.GetPrimalErrorIndicators()(present_element) +=
//    0.5 * face_integrals[element[0]->face(face_no)][0];
//  dwrc.GetDualErrorIndicators()(present_element) +=
//    0.5 * face_integrals[element[0]->face(face_no)][1];
//#else
//  Assert(
//    face_integrals.find(element[0]->face(face_no)->vertex_index()) != face_integrals.end(),
//    ExcInternalError());
//  dwrc.GetPrimalErrorIndicators()(present_element) +=
//    0.5 * face_integrals[element[0]->face(face_no)->vertex_index()][0];
//  dwrc.GetDualErrorIndicators()(present_element) +=
//    0.5 * face_integrals[element[0]->face(face_no)->vertex_index()][1];
//
//#endif
//      }
//    {
//      //Remove Weights
//      auto wd = dwrc.GetWeightData().begin();
//      auto wend = dwrc.GetWeightData().end();
//      for (; wd != wend; wd++)
//        {
//          DeleteDomainData(wd->first);
//        }
//    }
      abort();
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    INTEGRATORDATACONT &
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainer() const
    {
      return idc1_;
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    INTEGRATORDATACONT &
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::GetIntegratorDataContainerFunc() const
    {
      return idc2_;
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<template<int, int> class DH>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::InterpolateBoundaryValues(
      const DOpEWrapper::Mapping<dim, DH> &mapping,
      const DOpEWrapper::DoFHandler<dim, DH> *dof_handler,
      const unsigned int color, const dealii::Function<dim> &function,
      std::map<unsigned int, SCALAR> &boundary_values,
      const std::vector<bool> &comp_mask) const
    {
      //TODO: mapping[0] is a workaround, as deal does not support interpolate
      // boundary_values with a mapping collection at this point.
      dealii::VectorTools::interpolate_boundary_values(mapping[0],
                                                       dof_handler->GetDEALDoFHandler(), color, function,
                                                       boundary_values, comp_mask);
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::AddPresetRightHandSide(double s,
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

//*************************Methods per Pipe*********************************/
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputeNonlinearPipeResidual(
      unsigned int pipe,
      unsigned int n_pipes,
      unsigned int n_comp,
      PIPE_STH_* sth, PROBLEM &pde,
      dealii::Vector<double> &residual,
      dealii::Vector<double> &flow,
      std::vector<bool> &fluxes_in_outflow)
    {
      residual = 0.;

      // Begin integration
      unsigned int dofs_per_element;
      dealii::Vector<SCALAR> local_vector;
      std::vector<unsigned int> local_dof_indices;

      const bool need_point_rhs = pde.HasPoints();

      const auto &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto element =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

      // Generate the data containers.
      GetIntegratorDataContainer().InitializeEDC(pipe,pde.GetUpdateFlags(),
                                                 *(sth), element,
                                                 this->GetParamData(), this->GetDomainData());
      auto &edc = GetIntegratorDataContainer().GetElementDataContainer();

      bool need_faces = pde.HasFaces();
      bool need_interfaces = pde.HasInterfaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);
      GetIntegratorDataContainer().InitializeFDC(pipe,
                                                 n_pipes,
                                                 n_comp,
                                                 pde.GetFaceUpdateFlags(),
                                                 *(sth),
                                                 element,
                                                 this->GetParamData(),
                                                 this->GetDomainData(),
                                                 need_interfaces);
      auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException(
                    "Elementnumbers in DoFHandlers are not matching!",
                    "Network_Integrator::ComputeNonlinearResidual");
                }
            }

          edc.ReInit();

          dofs_per_element = element[0]->get_fe().dofs_per_cell;

          local_vector.reinit(dofs_per_element);
          local_vector = 0;

          local_dof_indices.resize(0);
          local_dof_indices.resize(dofs_per_element, 0);

          //the second '1' plays only a role in the stationary case. In the non-stationary
          //case, scale_ico is set by the time-stepping-scheme
          pde.ElementEquation(edc, local_vector, 1., 1.);
          pde.ElementRhs(edc, local_vector, -1.);

          if (need_boundary_integrals && element[0]->at_boundary())
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
#if DEAL_II_VERSION_GTE(8,3,0)
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                    {
                      fdc.ReInit(face);
                      pde.BoundaryEquation(fdc,local_vector, 1., 1.);
                      pde.BoundaryRhs(fdc,local_vector,-1.);

                      //Compute the partial entries in the last column of the residual
                      pde.OutflowValues(fdc,fluxes_in_outflow,flow,1.,1.);
                    }
                }
            }
          if (need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  if (element[0]->neighbor_index(face) != -1)
                    {
                      fdc.ReInit(face);
                      pde.FaceEquation(fdc, local_vector, 1., 1.);
                      pde.FaceRhs(fdc, local_vector,-1.);
                    }
                }
            }
          if ( need_interfaces)
            {

              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  // auto face_it = element[0]->face(face);
                  // first, check if we are at an interface, i.e. not the neighbour exists and
                  // it has a different material_id than the actual element
                  if (pde.AtInterface(element, face))
                    {
                      //There exist now 3 different scenarios, given the actual element and face:
                      // The neighbour behind this face is [ more | as much | less] refined
                      // than/as the actual element. We have to distinguish here only between the case 1
                      // and the other two, because these will be distinguished in in the FaceDataContainer.

                      if (element[0]->neighbor(face)->has_children())
                        {
                          //first: neighbour is finer

                          for (unsigned int subface_no=0;
                               subface_no < element[0]->face(face)->n_children();
                               ++subface_no)
                            {
                              //TODO Now here we have to initialise the subface_values on the
                              // actual element and then the facevalues of the neighbours
                              fdc.ReInit(face, subface_no);
                              fdc.ReInitNbr();

                              pde.InterfaceEquation(fdc, local_vector, 1., 1.);

                            }
                        }
                      else
                        {
                          // either neighbor is as fine as this element or
                          // it is coarser

                          fdc.ReInit(face);
                          fdc.ReInitNbr();
                          pde.InterfaceEquation(fdc, local_vector, 1., 1.);
                        }

                    }                  //endif atinterface
                }                  //endfor faces
            }                  //endif need_interfaces
          //LocalToGlobal
          const auto &C = pde.GetDoFConstraints();
          element[0]->get_dof_indices(local_dof_indices);
          C.distribute_local_to_global(local_vector, local_dof_indices, residual);

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }

      //check if we need the evaluation of PointRhs
      if (need_point_rhs)
        {
          throw DOpEException("No Point Rhs implemented!","Networks::Network_Integrator::ComputeNonlinearPipeResidual");
        }
    }

    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    void
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePipeMatrix(
      unsigned int pipe,
      unsigned int n_pipes,
      unsigned int n_comp,
      PIPE_STH_* sth, PROBLEM &pde,
      dealii::SparseMatrix<double> &matrix,
      dealii::FullMatrix<double> &inflow_coupling,
      dealii::FullMatrix<double> &outflow_coupling,
      std::vector<bool> &fluxes_in_outflow)
    {
      matrix = 0.;
      // Begin integration
      unsigned int dofs_per_element;
      std::vector<unsigned int> local_dof_indices;

      const auto &dof_handler =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
      auto element =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
      auto endc =
        pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();

      GetIntegratorDataContainer().InitializeEDC(pipe,pde.GetUpdateFlags(),
                                                 *(sth), element,
                                                 this->GetParamData(), this->GetDomainData());
      auto &edc = GetIntegratorDataContainer().GetElementDataContainer();

      //for the interface-case
      unsigned int nbr_dofs_per_element;
      std::vector<unsigned int> nbr_local_dof_indices;

      bool need_faces = pde.HasFaces();
      bool need_interfaces = pde.HasInterfaces();
      std::vector<unsigned int> boundary_equation_colors = pde.GetBoundaryEquationColors();
      bool need_boundary_integrals = (boundary_equation_colors.size() > 0);

      GetIntegratorDataContainer().InitializeFDC(pipe, n_pipes, n_comp,
                                                 pde.GetFaceUpdateFlags(),
                                                 *(sth),
                                                 element,
                                                 this->GetParamData(),
                                                 this->GetDomainData(),
                                                 need_interfaces);
      auto &fdc = GetIntegratorDataContainer().GetFaceDataContainer();

      for (; element[0] != endc[0]; element[0]++)
        {
          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              if (element[dh] == endc[dh])
                {
                  throw DOpEException(
                    "Elementnumbers in DoFHandlers are not matching!",
                    "Network_Integrator::ComputePipeMatrix");
                }
            }
          edc.ReInit();
          dofs_per_element = element[0]->get_fe().dofs_per_cell;

          dealii::FullMatrix<SCALAR> local_matrix(dofs_per_element,
                                                  dofs_per_element);
          local_matrix = 0;

          local_dof_indices.resize(0);
          local_dof_indices.resize(dofs_per_element, 0);
          pde.ElementMatrix(edc, local_matrix);

          if (need_boundary_integrals)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
#if DEAL_II_VERSION_GTE(8,3,0)
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            element[0]->face(face)->boundary_id()) != boundary_equation_colors.end()))
#else
                  if (element[0]->face(face)->at_boundary()
                      &&
                      (find(boundary_equation_colors.begin(),boundary_equation_colors.end(),
                            element[0]->face(face)->boundary_indicator()) != boundary_equation_colors.end()))
#endif
                    {
                      fdc.ReInit(face);
                      pde.BoundaryMatrix(fdc, local_matrix);

                      //Calculate Outflow Matrix
                      pde.OutflowMatrix(fdc, fluxes_in_outflow,
                                        outflow_coupling, 1.,1.);
                      //Calculate Matrix for Coupling between flux and pde (inflow)
                      pde.BoundaryMatrix_BV(fdc,fluxes_in_outflow,
                                            inflow_coupling,1.,1.);
                    }
                }
            }
          if (need_faces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  if (element[0]->neighbor_index(face) != -1)
                    {
                      fdc.ReInit(face);
                      pde.FaceMatrix(fdc, local_matrix);
                    }
                }
            }

          if ( need_interfaces)
            {
              for (unsigned int face=0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  //auto face_it = element[0]->face(face);
                  // first, check if we are at an interface, i.e. not the neighbour exists and
                  // it has a different material_id than the actual element
                  if (pde.AtInterface(element, face))
                    {
                      //There exist now 3 different scenarios, given the actual element and face:
                      // The neighbour behind this face is [ more | as much | less] refined
                      // than/as the actual element. We have to distinguish here only between the case 1
                      // and the other two, because these will be distinguished in in the FaceDataContainer.

                      if (element[0]->neighbor(face)->has_children())
                        {
                          //first: neighbour is finer

                          for (unsigned int subface_no=0;
                               subface_no < element[0]->face(face)->n_children();
                               ++subface_no)
                            {
                              //TODO Now here we have to initialise the subface_values on the
                              // actual element and then the facevalues of the neighbours
                              fdc.ReInit(face, subface_no);
                              fdc.ReInitNbr();

                              //TODO to be swapped out?
                              nbr_dofs_per_element = fdc.GetNbrNDoFsPerElement();
                              nbr_local_dof_indices.resize(0);
                              nbr_local_dof_indices.resize(nbr_dofs_per_element, 0);
                              dealii::FullMatrix<SCALAR> local_interface_matrix(dofs_per_element,nbr_dofs_per_element );
                              local_interface_matrix = 0;

                              pde.InterfaceMatrix(fdc, local_interface_matrix);

                              element[0]->get_dof_indices(local_dof_indices);
                              element[0]->neighbor(face)->get_dof_indices(nbr_local_dof_indices);

                              const auto &C = pde.GetDoFConstraints();
                              C.distribute_local_to_global(local_interface_matrix, local_dof_indices, nbr_local_dof_indices, matrix);

                            }
                        }
                      else
                        {
                          // either neighbor is as fine as this element or it is coarser
                          fdc.ReInit(face);
                          fdc.ReInitNbr();

                          //TODO to be swapped out?
                          nbr_dofs_per_element = fdc.GetNbrNDoFsPerElement();
                          nbr_local_dof_indices.resize(0);
                          nbr_local_dof_indices.resize(nbr_dofs_per_element, 0);
                          dealii::FullMatrix<SCALAR> local_interface_matrix(dofs_per_element,nbr_dofs_per_element );
                          local_interface_matrix = 0;

                          pde.InterfaceMatrix(fdc, local_interface_matrix);

                          element[0]->get_dof_indices(local_dof_indices);
                          element[0]->neighbor(face)->get_dof_indices(nbr_local_dof_indices);

                          const auto &C = pde.GetDoFConstraints();
                          C.distribute_local_to_global(local_interface_matrix, local_dof_indices, nbr_local_dof_indices, matrix);

                        }
                    } //endif atinterface

                }
            } //endif need_interfaces

          //LocalToGlobal
          const auto &C = pde.GetDoFConstraints();
          element[0]->get_dof_indices(local_dof_indices);
          C.distribute_local_to_global(local_matrix, local_dof_indices, matrix);

          for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
            {
              element[dh]++;
            }
        }
    }
    /*******************************************************************************************/

    template<typename INTEGRATORDATACONT, typename VECTOR, typename SCALAR,
             int dim>
    template<typename PROBLEM>
    SCALAR
    Network_Integrator<INTEGRATORDATACONT, VECTOR, SCALAR, dim>::ComputePipeDomainScalar(
      unsigned int pipe,
      PIPE_STH_* sth, PROBLEM &pde)
    {
      {
        SCALAR ret = 0.;

        const auto &dof_handler =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandler();
        auto element =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerBeginActive();
        auto endc =
          pde.GetBaseProblem().GetSpaceTimeHandler()->GetDoFHandlerEnd();
        GetIntegratorDataContainerFunc().InitializeEDC(pipe,pde.GetUpdateFlags(),
                                                       *(sth), element,
                                                       this->GetParamData(), this->GetDomainData());
        auto &edc = GetIntegratorDataContainerFunc().GetElementDataContainer();

        for (; element[0] != endc[0]; element[0]++)
          {
            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
              {
                if (element[dh] == endc[dh])
                  {
                    throw DOpEException(
                      "Elementnumbers in DoFHandlers are not matching!",
                      "Network_Integrator::ComputePipeDomainScalar");
                  }
              }

            edc.ReInit();
            ret += pde.ElementFunctional(edc);

            for (unsigned int dh = 1; dh < dof_handler.size(); dh++)
              {
                element[dh]++;
              }
          }
        return ret;
      }
    }




//////////////////////////End of Namespaces//////////////////////////////////
  }
}
#endif

