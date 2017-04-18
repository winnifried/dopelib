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

//c++ includes
#include <iostream>
#include <fstream>

//deal.ii includes
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

//DOpE includes
//This steers the whole solution process
#include <reducedproblems/statpdeproblem.h>
//The newtonsolver
#include <templates/newtonsolver.h>
//The linear solver
#include <templates/directlinearsolver.h>
//The integrator
#include <templates/integrator.h>
//This one handles the param files
#include <include/parameterreader.h>
//The SpaceTimeHandler manages the DoFs in space and time
#include <basic/mol_statespacetimehandler.h>
//This represents the dirichletvalues
#include <problemdata/simpledirichletdata.h>
//Container class for the integrator
#include <container/integratordatacontainer.h>

//local includes
//Defines the functionals we want to evaluate
#include "functionals.h"
//Defines the PDE we want to solve.
#include "localpde.h"
//Defines some functions like the inflow-condition.
#include "my_functions.h"
//Determins where we use which Finite Element.
#include "indexsetter.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

//Some abbreviations for better readability
const static int DIM = 2;

//Abbreviations of deal.II types
//The dofhandler and finite element we want to use.
#define DOFHANDLER hp::DoFHandler
#define FE hp::FECollection

//The qudrature rules we want to use.
typedef hp::QCollection<DIM> QUADRATURE;
typedef hp::QCollection<DIM - 1> FACEQUADRATURE;

//Which Matrix, Sparsitypattern and Vector-type do we want to use?
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

//Abbreviations of various DOpE types, see in the respective headers for more
//information.

//Elementdatacontainer and Facedatacontainer handle all the informations we need
//to integrate or evaluate an FE function on a element resp. face.
#define CDC ElementDataContainer
#define FDC FaceDataContainer

//The PDEProblemContainer holds all the information regarding the PDE.
typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM, FE,
        DOFHANDLER> OP;

//The IntegratorDataContainer holds quadrature formulas as
//well as element- and facedatacontainer.
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

//The integrator which handels the integration.
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

//The linear solver we want to use.
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

//The newtonsolver we want to use.
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;

//This class represents the PDEProblem and steers the solution process.
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;

//The spacetimehandler manages all the things related to the degrees of
//freedom in space and time.
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   *  In this example we solve stationary (linear) Stokes' equations
   *  with symmetric stress tensor and do-nothing condition on
   *  the outflow boundary. In this case we employ an additional
   *  term on the outflow boundary due the symmetry of the stress tensor.
   *
   *  This example uses hp finite elements as extension to PDE/StatPDE/Example 1
   */

  //Handling of the param file.
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
  ParameterReader pr; //The parameter reader is responsible for the param-files handling

  //Declaration of the parameters
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile); //we read the parameters

  //We build the triangulation. We read it in from the file channel.inp
  Triangulation<DIM> triangulation;

  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("channel.inp");
  grid_in.read_ucd(input_file);

  triangulation.refine_global(3);

  FESystem<DIM> state_fe(FE_Q<DIM>(2), 2, FE_Q<DIM>(1), 1); //Q2Q1

  /******hp******************/
  //We use two different finite elements. If the x - component of
  //the middle point of the element is smaller than zero, we use the Q2Q1
  //element, else a Q3Q2 element.
  FESystem<DIM> state_fe_2(FE_Q<DIM>(3), 2, FE_Q<DIM>(2), 1); //Q3Q2
  FE<DIM> state_fe_collection(state_fe);
  state_fe_collection.push_back(state_fe_2);
  /******hp******************/

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);

  /******hp******************/
  //Adapt the quadrature rule to the Finite Element in use.
  QGauss<DIM> quadrature_formula_2(4);
  QGauss<DIM - 1> face_quadrature_formula_2(4);
  QUADRATURE q_coll(quadrature_formula);
  q_coll.push_back(quadrature_formula_2);
  FACEQUADRATURE face_q_coll(face_quadrature_formula);
  face_q_coll.push_back(face_quadrature_formula_2);
  IDC idc(q_coll, face_q_coll);
  /******hp******************/

  //Definition of the pde we want to solve.
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;

  //Definition of the functionals we want to evaluate.
  LocalPointFunctionalX<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFX;
  LocalBoundaryFluxFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFF;

  /***************hp********************/
  //The indexsetter determines on which part of the domain
  //which FE, quadrature rule etc. is used.
  ActiveFEIndexSetter<DIM> indexsetter(pr);
  STH DOFH(triangulation, state_fe_collection, false, indexsetter);
  /***************hp********************/

  OP P(LPDE, DOFH);

  //We add the functionals to the problemcontainer
  P.AddFunctional(&LPFX);
  P.AddFunctional(&LBFF);

  //Here we specify the boundary colors for the boundaries
  //on which we want to evaluate some functionals (here LBFF).
  P.SetBoundaryFunctionalColors(1);

  //Specification of the dirichlet values

  //We need zero dirichletvalues as well as an inflow condition.
  //So first define two functions representing these values...
  DOpEWrapper::ZeroFunction<DIM> zf(3);
  BoundaryParabel boundary_parabel;
  //...then we put them into an object of type SimpleDirichletData,
  //as this is the format in which DOpE expects the Dirichletvalues.
  SimpleDirichletData<VECTOR, DIM> DD1(zf);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);

  //Nex, we define on which boundaries (identified via
  //boundary-colors) and which components (specified via an component mask)
  //we want to impose the dirichlet conditions and give all
  //these informations to the problemcontainer P. Note that we
  //do not impose any boundary condition on the outflow boundary (number 1).
  std::vector<bool> comp_mask(3, true);
  comp_mask[DIM] = false;

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  //As our weak formulation has some boundary-integrals, we tell the problemcontainer
  //on which part of the boundary these live.
  P.SetBoundaryEquationColors(1);

  //We define the stateproblem, which steers the solution process.
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems: We define and register
  //the output- and exception handler. The first handels the
  //output on the screen as well as the output of files. The
  //amount of the output can be steered by the paramfile.
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  try
    {
      //Before solving we have to reinitialize the stateproblem and outputhandler.
      solver.ReInit();
      out.ReInit();

      stringstream outp;
      outp << "**************************************************\n";
      outp << "*             Starting Forward Solve             *\n";
      outp << "*   Solving : " << P.GetName() << "\t*\n";
      outp << "*   SDoFs   : ";
      solver.StateSizeInfo(outp);
      outp << "**************************************************";
      //We print this header with priority 1 and 1 empty line in front and after.
      out.Write(outp, 1, 1, 1);

      //We compute the value of the functionals. To this end, we have to solve
      //the PDE at hand. The output of numbers on the screen as well
      solver.ComputeReducedFunctionals();
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
