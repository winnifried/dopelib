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
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <include/userdefineddofconstraints.h>
#include <include/sparsitymaker.h>
#include <container/integratordatacontainer.h>

#include <templates/integrator.h>
#include <include/parameterreader.h>

#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <interfaces/active_fe_index_setter_interface.h>

#include "localpde.h"
#include "functionals.h"
#include <container/higher_order_dwrc.h>
#include <container/residualestimator.h>
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

typedef PDEProblemContainer<LocalPDELaplace<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;
typedef HigherOrderDWRContainer<STH, IDC, CDC<DOFHANDLER, VECTOR, DIM>,
        FDC<DOFHANDLER, VECTOR, DIM>, VECTOR> HO_DWRC;
typedef L2ResidualErrorContainer<STH, VECTOR, DIM> L2_RESC;
typedef H1ResidualErrorContainer<STH, VECTOR, DIM> H1_RESC;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
                             "How many iterations?");
  param_reader.declare_entry("quad order", "2", Patterns::Integer(1),
                             "Order of the quad formula?");
  param_reader.declare_entry("facequad order", "2", Patterns::Integer(1),
                             "Order of the face quad formula?");
  param_reader.declare_entry("order fe", "2", Patterns::Integer(1),
                             "Order of the finite element?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
                             "How often should we refine the coarse grid?");
}

int
main(int argc, char **argv)
{
  /**
   * We solve the standard laplace equation in 2d. The
   * main feature is the use of the DWR method for error
   * estimation and grid refinement.
   */
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
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  int max_iter = pr.get_integer("max_iter");
  int prerefine = pr.get_integer("prerefine");

  //*************************************************

  //Make triangulation *************************************************
  const Point<DIM> center(0, 0);
  const HyperShellBoundary<DIM> boundary_description(center);
  Triangulation<DIM> triangulation(
    Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.5, 2., 1, 1);
  triangulation.set_boundary(1, boundary_description);
  triangulation.refine_global(1); //because we need the face located at x==0;
  for (auto it = triangulation.begin_active(); it != triangulation.end(); it++)
    if (it->center()[1] <= 0)
      {
        if (it->center()[0] < 0)
          {
            it->set_material_id(1);
          }
        else
          {
            it->set_material_id(2);
          }
      }
  if (prerefine > 0)
    triangulation.refine_global(prerefine);
  //*************************************************************

  //FiniteElemente*************************************************
  pr.SetSubsection("main parameters");
  FE<DIM> state_fe(FE_Q<DIM>(pr.get_integer("order fe")), 1);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula(pr.get_integer("quad order"));
  QGauss<1> face_quadrature_formula(pr.get_integer("facequad order"));
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  LocalFaceFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> LFF;
  LocalPDELaplace<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  //*************************************************

  //space time handler***********************************/
  STH DOFH(triangulation, state_fe);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&LFF);
  //Boundary conditions************************************************
  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  ExactSolution ex_sol;

  SimpleDirichletData<VECTOR, DIM> DD1(ex_sol);
  //Set dirichlet boundary values all around
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  /************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  /**********************************************************************/
  //DWR**********************************************************************/
  //Set dual functional for ee
  P.SetFunctionalForErrorEstimation(LFF.GetName());
  //FiniteElemente for DWR*************************************************
  pr.SetSubsection("main parameters");
  FE<DIM> state_fe_high(FE_Q<DIM>(2 * pr.get_integer("order fe")), 1);
  //Quadrature formulas for DWR*************************************************
  pr.SetSubsection("main parameters");
  QUADRATURE quadrature_formula_high(pr.get_integer("quad order") + 1);
  FACEQUADRATURE face_quadrature_formula_high(
    pr.get_integer("facequad order") + 1);
  IDC idc_high(quadrature_formula_high, face_quadrature_formula_high);
  STH DOFH_higher_order(triangulation, state_fe_high);
  HO_DWRC dwrc(DOFH_higher_order, idc_high, DOpEtypes::VectorStorageType::fullmem, pr,
               DOpEtypes::primal_only);

  L2_RESC l2resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);
  H1_RESC h1resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);

  P.InitializeDWRC(dwrc);
  //**************************************************************************************************

  for (int i = 0; i < max_iter; i++)
    {
      try
        {
          solver.ReInit();
          out.ReInit();
          stringstream outp;

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************";
          out.Write(outp, 1, 1, 1);

          solver.ComputeReducedFunctionals();
          solver.ComputeRefinementIndicators(dwrc, LPDE);
          solver.ComputeRefinementIndicators(l2resc, LPDE);
          solver.ComputeRefinementIndicators(h1resc, LPDE);

          const double exact_value = 0.441956231972232;

          double error = exact_value - solver.GetFunctionalValue(LFF.GetName());
          outp << "Mean value error: " << error << "  Ieff (eh/e)= "
               << dwrc.GetError() / error << std::endl;
          outp << "L2-Error estimator: " << sqrt(l2resc.GetError()) << std::endl;
          outp << "H1-Error estimator: " << sqrt(h1resc.GetError()) << std::endl;
          out.Write(outp, 1, 1, 1);
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
          // DOFH.RefineSpace(DOpEtypes::RefinementType::global); //or just DOFH.RefineSpace()

          Vector<float> error_ind(dwrc.GetErrorIndicators());
          for (unsigned int i = 0; i < error_ind.size(); i++)
            error_ind(i) = std::fabs(error_ind(i));

          DOFH.RefineSpace(RefineOptimized(error_ind));
          //There are other mesh refinement strategies implemented, for example
          //DOFH.RefineSpace(RefineFixedNumber(error_ind, 0.4));
          //DOFH.RefineSpace(RefineFixedFraction(error_ind, 0.8));
        }
    }
  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
