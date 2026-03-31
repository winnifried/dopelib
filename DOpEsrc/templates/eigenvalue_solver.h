/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef EIGENVALUE_SOLVER_H_
#define EIGENVALUE_SOLVER_H_

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <include/parameterreader.h>

#if DOPELIB_WITH_SLEPC
#include <deal.II/lac/slepc_solver.h>
#endif

#if DOPELIB_WITH_PETSC
#if DEAL_II_VERSION_GTE(9,4,0)
#include <deal.II/lac/petsc_vector.h>
#else
#include <deal.II/lac/petsc_parallel_vector.h>
#endif
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#if DOPELIB_WITH_PETSC
#include <petscconf.h>
#include <petscksp.h>
#endif
#if DOPELIB_WITH_SLEPC
#include <slepceps.h>
#endif



namespace DOpE
{

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  class EigenvalueSolver
  {
  public:
    EigenvalueSolver( INTEGRATOR &integrator, ParameterReader &param_reader);
    ~EigenvalueSolver();

    static void declare_params(ParameterReader &param_reader);

    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    template<typename PROBLEM>
    void GetNormalizedVectorState(PROBLEM &pde,StateVector<VECTOR> &stateeigenfunction);
    template<typename PROBLEM>
    void GetNormalizedVectorAdjoint(PROBLEM &pde, StateVector<VECTOR> &adjeigenfunction,StateVector<VECTOR> &stateeigenfunction, double value);

    template<typename PROBLEM>
    bool EigenvalueSolve(PROBLEM &pde, dealii::Vector<double> &eigenvalues, std::vector<StateVector<VECTOR>> &eigenfunctions,
                         bool apply_boundary_values=true,
                         bool force_matrix_build=false,
                         int priority = 5, std::string algo_level = "\t\t ");

    inline INTEGRATOR &GetIntegrator();

  private:
    INTEGRATOR &integrator_;

    //TODO testen, ob extra precon_matrix benötigt wird. (Cast über Wrapper scheint problematisch, also erstmal nur kopieren)
    MATRIX matrixK_, matrixM_,/*;
    Mat */ matrixMprecon_;

    IndexSet eigenfunction_index_set;
#ifdef DOPELIB_WITH_PETSC
    std::vector<PETScWrappers::MPI::Vector> eigenvectors_;

    PETScWrappers::MPI::Vector state_for_normalization;
    PETScWrappers::MPI::Vector adjoint_for_normalization;
#endif
    std::vector<double> eigenvalues_;
    bool build_matrix_ = true;

    int maxiter_;
    double target_eigenvalue_, tol_, matshift_;
    bool first_iteration = true;
#ifdef DOPELIB_WITH_PETSC
   PetscScalar target_ ;
#endif

    std::string epstype_, sttype_, ksptype_, pctype_;
    const char *char_epstype_, *char_sttype_, *char_ksptype_, *char_pctype_ ;

  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  void EigenvalueSolver<INTEGRATOR,VECTOR, MATRIX>
  ::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("eigenvalue_solver parameters");
    param_reader.declare_entry("tol", "0.00001", Patterns::Double(0),"tolerance");
    param_reader.declare_entry("maxiter", "1000", Patterns::Integer(0),"maxiter");
    param_reader.declare_entry("target_eigenvalue", "0.001", Patterns::Double(0),"target eigenvalue");
    param_reader.declare_entry("EPSType", "lapack", Patterns::Anything(),"EPS Type");
    param_reader.declare_entry("STType", "sinvert", Patterns::Anything(),"ST Type");
    param_reader.declare_entry("KSPType", "richardson", Patterns::Anything(),"KSP Type");
    param_reader.declare_entry("PCType", "cholesky", Patterns::Anything(),"PC Type");
    param_reader.declare_entry("MatShift", "0", Patterns::Double(0),"Matrix Shift");
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  EigenvalueSolver<INTEGRATOR,VECTOR, MATRIX>
  ::EigenvalueSolver(INTEGRATOR &integrator, ParameterReader &param_reader)
    : integrator_(integrator)
  {
#ifndef DOPELIB_WITH_PETSC
    throw DOpEException("To use this algorithm you need to deal.II compiled with petsc!","EigenvalueSolver");
#endif
#ifndef DOPELIB_WITH_SELPC
    throw DOpEException("To use this algorithm you need to deal.II compiled with slepc!","EigenvalueSolver");
#endif

    param_reader.SetSubsection("eigenvalue_solver parameters");
    tol_ = param_reader.get_double ("tol");
    maxiter_ = param_reader.get_integer ("maxiter");
    target_eigenvalue_ = param_reader.get_double ("target_eigenvalue");
    epstype_ = param_reader.get_string ("EPSType");
    char_epstype_= epstype_.c_str();
    sttype_ = param_reader.get_string ("STType");
    char_sttype_= sttype_.c_str();
    ksptype_ = param_reader.get_string ("KSPType");
    char_ksptype_= ksptype_.c_str();
    pctype_ = param_reader.get_string ("PCType");
    char_pctype_= pctype_.c_str();
    matshift_ = param_reader.get_double ("MatShift");

  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR,  typename MATRIX>
  EigenvalueSolver<INTEGRATOR,VECTOR, MATRIX>
  ::~EigenvalueSolver()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  void EigenvalueSolver<INTEGRATOR, VECTOR, MATRIX>
  ::ReInit(PROBLEM &pde)
  {
#ifdef DOPELIB_WITH_PETSC
#ifdef DOPELIB_WITH_SLEPC
    matrixK_.clear();
    matrixK_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
                    pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
    matrixM_.clear();
    matrixM_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
                    pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());


    matrixMprecon_.clear();
    matrixMprecon_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
                          pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());


    eigenvectors_.clear();
    eigenvalues_.clear();
#endif
#endif
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  bool EigenvalueSolver<INTEGRATOR, VECTOR, MATRIX>
  ::EigenvalueSolve(PROBLEM &pde,
                    dealii::Vector<double> &eigenvalues,
                    std::vector<StateVector<VECTOR>> &eigenfunctions,
                    bool /*apply_boundary_values*/,
                    bool force_matrix_build,
                    int /*priority*/,
                    std::string /*algo_level*/
                   )
  {
    bool build_matrix = force_matrix_build;
#ifdef DOPELIB_WITH_PETSC
#ifdef DOPELIB_WITH_SLEPC

    if (force_matrix_build)
      {
        this->ReInit(pde);
        integrator_.ComputeMatrix(pde,matrixK_);
        integrator_.ComputeMassMatrix(pde,matrixM_);

        integrator_.ComputeMassMatrix(pde,matrixMprecon_);
      }
    target_=target_eigenvalue_;

    eigenvectors_.clear();

    eigenvalues_.clear();

    eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();
    eigenvectors_.resize(eigenvalues.size());
    for (unsigned int i = 0; i < eigenvectors_.size(); ++i)
      {
        eigenvectors_[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);
      }
    eigenvalues_.resize(eigenvalues.size());
//      int max_it = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs()*pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs();

    EPS eps;
    EPSCreate(MPI_COMM_WORLD, &eps);

    MatShift(matrixK_,matshift_);
    MatShift(matrixMprecon_, matshift_);

    EPSSetOperators(eps, matrixK_, matrixMprecon_);

    EPSSetType(eps,char_epstype_);
    EPSSetTarget(eps, target_);
    EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE); //this is from type EPSWhich, not included as parameter in the parameterfile
    EPSSetTolerances(eps, tol_, maxiter_);
    EPSSetFromOptions(eps);

    ST st;
    EPSGetST(eps, &st);
    STSetType(st, char_sttype_);
    STSetShift(st,target_);

    KSP ksp;
    STGetKSP(st, &ksp);
    KSPSetType(ksp, char_ksptype_);

    KSPSetOperators(ksp,matrixK_, matrixK_);
    KSPSetOperators(ksp,matrixM_, matrixMprecon_);

    //Diverges without preconditioner
    PC pc;

    KSPGetPC(ksp, &pc);
    PCSetType(pc, char_pctype_);

    EPSSetProblemType(eps, EPS_GHEP); // EPS_type is from type EPSProblemType
    EPSSetDimensions(eps,eigenvalues_.size(),PETSC_DECIDE,PETSC_DECIDE);

    EPSSolve(eps);
    PetscScalar eig;
    for (unsigned int i = 0; i<eigenvalues_.size(); i++)
      {
        EPSGetEigenpair(eps, i, &eig, nullptr, eigenvectors_[i], nullptr);  //realteil und imaginärteil (imaginärteil nullptr)
        eigenvalues[i] = eig;
        eigenfunctions[i].GetSpacialVector() = eigenvectors_[i];
      }

    MatShift(matrixK_, -matshift_);
    MatShift(matrixMprecon_, -matshift_);
    EPSDestroy(&eps);
#endif
#endif  
    
    return build_matrix;
  }

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  void EigenvalueSolver<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorState(PROBLEM &/*pde*/,  StateVector<VECTOR> &stateeigenfunction)
  {
#ifdef DOPELIB_WITH_PETSC

    state_for_normalization.clear();
    state_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    state_for_normalization = stateeigenfunction.GetSpacialVector();


    PETScWrappers::MPI::Vector vec;
    PetscScalar scalar_factor[1];
    vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    MatMult(matrixM_,state_for_normalization,vec);
    VecDot(state_for_normalization,vec,scalar_factor);
    scalar_factor[0] = sqrt(scalar_factor[0]);
    state_for_normalization/= scalar_factor[0];
    stateeigenfunction.GetSpacialVector() = state_for_normalization;
#endif
  }


  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  void EigenvalueSolver<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorAdjoint(PROBLEM &pde,  StateVector<VECTOR> &adjointeigenfunction, StateVector<VECTOR> &stateeigenfunction, double value)
  {
#ifdef DOPELIB_WITH_PETSC

    state_for_normalization.clear();
    adjoint_for_normalization.clear();

    eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();

    state_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    state_for_normalization = stateeigenfunction.GetSpacialVector();

    adjoint_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    adjoint_for_normalization =  adjointeigenfunction.GetSpacialVector();

    PETScWrappers::MPI::Vector vec;
    PetscScalar scalar_factor[1];
    vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    MatMult(matrixM_,adjoint_for_normalization,vec); // = matrixM*adjoint_eigenfunctions[i];
    VecDot(state_for_normalization,vec,scalar_factor);//= scalar_product(vec,eigenfunctions[i]);
    adjoint_for_normalization/= (scalar_factor[0]);
    adjoint_for_normalization *= value;
    adjointeigenfunction.GetSpacialVector() = adjoint_for_normalization;
#endif
  }


  /*******************************************************************************************/
  template <typename INTEGRATOR,  typename VECTOR, typename MATRIX>
  INTEGRATOR &EigenvalueSolver<INTEGRATOR, VECTOR, MATRIX>
  ::GetIntegrator()
  {
    return integrator_;
  }

}
#endif





