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
#include <iostream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>

#include <opt_algorithms/reducedgradientdescentalgorithm.h>
#include <opt_algorithms/reducedbfgsalgorithm.h>

#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <include/parameterreader.h>
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>

#include <templates/integrator.h>

#include <templates/voidlinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/cglinearsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/trilinosdirectlinearsolver.h>
#include <templates/richardsonlinearsolver.h>

#include <templates/newtonsolver.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>


#include "localpde.h"
#include "localfunctional.h"
#include "functions.h"
#include "eigenvector_solver.h"
#include "eigenvalueproblem.h"
#include "integrator_eigenval.h"
#include "mol_spacetimehandler.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;
const static int CDIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef PETScWrappers::SparseMatrix MATRIX;
typedef SparseMatrix<double> MATRIXFORLINSOLVE;
typedef /*Block*/SparsityPattern SPARSITYPATTERN;
typedef /*Block*/Vector<double> VECTOR;

typedef std::vector<double> EIGENVALUES;
typedef std::vector<PETScWrappers::MPI::Vector> EIGENVECTORS;

#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

typedef Integrator_eigenval<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef Integrator_eigenval<IDC, VECTOR, double, DIM> INTEGRATOR_CONTROL;

//typedef VoidLinearSolver<VECTOR> LINEARSOLVER;

//typedef RichardsonLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIXFORLINSOLVE>,SPARSITYPATTERN, MATRIXFORLINSOLVE, VECTOR> LINEARSOLVER;
typedef GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIXFORLINSOLVE>,SPARSITYPATTERN, MATRIXFORLINSOLVE, VECTOR> LINEARSOLVER;

//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef EigenvectorSolver<INTEGRATOR,VECTOR,EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER> EVS;

typedef EigenvalueProblem<EVS, EVS,INTEGRATOR_CONTROL, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;
typedef ReducedGradientDescentAlgorithm<OP, VECTOR> RNA;
//typedef ReducedBFGSAlgorithm<OP, VECTOR> RNA;



typedef MethodOfLines_SpaceTimeHandler/*_Curl*/<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
        CDIM, DIM> STH;

int
main(int argc, char **argv){
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv,1);

  string paramfile = "dope.prm";

  if (argc == 2)
    {
      paramfile = argv[1];
    }
  else if (argc > 2)
    {
      std::cout << "Usage: " << argv[0] << " [ paramfile ] " << std::endl;
      return -1;
    }
  ParameterReader pr;
  EVS::declare_params(pr);
  RNA::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, M_PI);
  triangulation.refine_global(3);

  //------------- FE-System ---------------------------------------------
   FE<DIM> control_fe(FE_Q<DIM>(1), 2);
   FESystem<DIM> state_fe(FE_Q<DIM>(1), 1 , FE_Nedelec<DIM>(0),1);


  QUADRATURE quadrature_formula(4);
  FACEQUADRATURE face_quadrature_formula(4);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  COSTFUNCTIONAL LFunc(0.00001);
  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  std::vector<bool> comp_mask(3);

   comp_mask[0] = true;
   DOpEWrapper::ZeroFunction<DIM> zf(3);
   SimpleDirichletData<VECTOR, DIM> DD1(zf);

//TODO in  mol space time handler angepasst für NedelecRB

   P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(1, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(3, comp_mask, &DD1);


  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc, 2);
  RNA Alg(&P, &solver, pr);

  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);

  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);

  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  solver.ReInit();

//  out.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem,pr);
  local::Q_Control q_initial;
  VectorTools::interpolate(DOFH.GetControlDoFHandler().GetDEALDoFHandler(), q_initial,  q.GetSpacialVector());
  ControlVector<VECTOR> dq(q);


      try
        {
    	  //TODO es muss noch überall der Lambda Zielwert richtig übergeben werden
//   	  solver.ComputeReducedCostFunctional(q);

    	  Alg.ReInit();
    	  Alg.Solve(q);
//	  const double eps_diff = 1;
//  	  Alg.CheckGrads(eps_diff, q, dq, 5);
        }
      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
        }


  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
