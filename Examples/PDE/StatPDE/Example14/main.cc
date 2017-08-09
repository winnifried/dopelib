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

#include <iostream>
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <templates/newtonsolver.h>
#include <templates/voidlinearsolver.h>
#include <wrapper/preconditioner_wrapper.h>
#include <include/sparsitymaker.h>
#include <interfaces/functionalinterface.h>
#include <problemdata/noconstraints.h>

#include <include/parameterreader.h>

#include <problemdata/simpledirichletdata.h>
#include <interfaces/active_fe_index_setter_interface.h>

#include <opt_algorithms/reducednewtonalgorithm.h>
#include <network/network_elementdatacontainer.h>
#include <network/network_facedatacontainer.h>
#include <network/network_integratordatacontainer.h>

#include <network/network_statreducedproblem.h>
#include <network/mol_network_spacetimehandler.h>
#include <network/network_integrator.h>
#include <network/network_integratormixeddims.h>
#include <network/network_directlinearsolver.h>
#include "functionals.h"
#include "localpde.h"
#include "localnetwork.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int CDIM = 0;
const static int DIM = 1;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC Networks::Network_ElementDataContainer
#define FDC Networks::Network_FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;


typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNC;

// Typedefs for timestep problem
//#define TSP ShiftedCrankNicolsonProblem
#define TSP BackwardEulerProblem
//#define TSP CrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

typedef Networks::Network_IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTOR, DIM> IDC;
typedef Networks::Network_Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef Networks::Network_IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> CINTEGRATOR;

typedef Networks::DirectLinearSolverWithMatrix LINEARSOLVER;


//dummy solver for the 0d control
typedef VoidLinearSolver<VECTOR> VOIDLS;


//special newtonsolver for the mixed dims
typedef NewtonSolverMixedDimensions<CINTEGRATOR, VOIDLS, VECTOR> CNLS;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef Networks::MethodOfLines_Network_SpaceTimeHandler<FE, DOFHANDLER,
        VECTOR, CDIM, DIM> STH;

typedef OptProblemContainer<
FUNC,
LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
SimpleDirichletData<VECTOR, DIM>,
NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
VECTOR, CDIM, DIM> OP;

typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef Networks::Network_StatReducedProblem<CNLS, NLS, CINTEGRATOR, INTEGRATOR, OP, CDIM,
        DIM> RP;




void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
                             "How many iterations?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
                             "How often should we refine the coarse grid?");
}

int
main(int argc, char **argv)
{
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

  RP::declare_params(pr);
  RNA::declare_params(pr);
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  int max_iter = pr.get_integer("max_iter");
  int prerefine = pr.get_integer("prerefine");

  //Make triangulation *************************************************
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 50);
  Triangulation<DIM> triangulation2;
  GridGenerator::hyper_cube(triangulation2, 50, 100);
  triangulation.refine_global(prerefine-1);
  triangulation2.refine_global(prerefine-1);
  std::vector<dealii::Triangulation<DIM> *> tria_s(2,NULL);
  tria_s[0] = &triangulation;
  tria_s[1] = &triangulation2;
  //*************************************************************

  //FiniteElemente*************************************************
  FESystem<DIM> control_fe(FE_Nothing<DIM>(1),1);
  FE<DIM> state_fe(FE_DGQ<DIM>(0), 2);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula(1);
  QGauss<DIM-1> face_quadrature_formula(1);
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> MVF(pr);
  LocalFunctional2<CDC, FDC, DOFHANDLER, VECTOR, DIM> MVF2(pr);
  //*************************************************

  //Network-description
  //*************************************************
  LocalNetwork mynet(pr);

  //pde*************************************************
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr,mynet);
  //*************************************************

  //space time handler***********************************/
  STH DOFH(tria_s, control_fe, state_fe, DOpEtypes::stationary, mynet, true);
  /***********************************/
  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;

  OP P(MVF, LPDE, Constraints, DOFH);
  //Boundary conditions************************************************
  P.SetBoundaryEquationColors(0);
  P.SetBoundaryEquationColors(1);
  /************************************************/
  P.AddFunctional(&MVF2);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  RNA Alg(&P, &solver, pr);


  //**************************************************************************************************
  Alg.GetOutputHandler()->Write("Solving ...",1);

  for (int i = 0; i < max_iter; i++)
    {
      try
        {
          Alg.ReInit();
          ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);

          Alg.SolveForward(q);
        }
      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
        }
      if (i != max_iter - 1)
        {
          //For global mesh refinement, uncomment the next line
          DOFH.RefineSpace(DOpEtypes::RefinementType::global); //or just DOFH.RefineSpace()
        }
    }

//*************************************************

  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
