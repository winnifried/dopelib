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
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/manifold_lib.h>
#if DEAL_II_VERSION_GTE(9, 1, 1)
#else
#include <deal.II/grid/tria_boundary_lib.h>
#endif

#include <basic/mol_spacetimehandler.h>
#include <container/optproblemcontainer.h>
#include <container/integratordatacontainer.h>
#include <container/higher_order_dwrc_control.h>
#include <interfaces/functionalinterface.h>
#include <include/parameterreader.h>
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/directlinearsolver.h>
#include <templates/newtonsolver.h>
#include <templates/integrator.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functions.h"

#include <opt_algorithms/reducednewtonalgorithm.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;
const static int CDIM = 2;

#if DEAL_II_VERSION_GTE(9, 3, 0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif
#define FE FESystem
#define EDC ElementDataContainer
#define FDC FaceDataContainer

//GaussLobatto Quadrature is required to compute the integrals, because we need to make sure
//that the vertices are used as quadrature points, so that we are able to compute lambda
typedef QGaussLobatto<DIM> QUADRATURE;
typedef QGaussLobatto<DIM - 1> FACEQUADRATURE;

//Fix for a bug in deal.ii 8.5.0
#if DEAL_II_VERSION_GTE(8, 5, 0)
#if DEAL_II_VERSION_GTE(9, 0, 0)//post deal 8.5.0
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;
#else //dealii 8.5.0
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;
#endif
#else //pre deal 8.5.0
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;
#endif

typedef LocalFunctional<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
					        LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
					        SimpleDirichletData<VECTOR, DIM>,
					        NoConstraints<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
					        SPARSITYPATTERN, VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;

typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM, DIM> RP;

typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM, DIM> STH;

void
declare_params(ParameterReader &param_reader) {
    param_reader.SetSubsection("main parameters");
    param_reader.declare_entry("mode", "solve", Patterns::Selection("solve|generateOutput|checkGrad|errorTesting"),
                               "The mode in which the algorithm is running");
    param_reader.declare_entry("quad_order", "2", Patterns::Integer(1), "Order of the quad formula?");
    param_reader.declare_entry("facequad_order", "2", Patterns::Integer(1), "Order of the face quad formula?");
    param_reader.declare_entry("order_state_fe", "1", Patterns::Integer(1), "Order of the finite element?");
    param_reader.declare_entry("order_control_fe", "1", Patterns::Integer(1), "Order of the finite element?");
    param_reader.declare_entry("prerefine", "4", Patterns::Integer(0), "How often should we refine the coarse grid?");

    param_reader.SetSubsection("problem-specific parameters");
    param_reader.declare_entry("weight_alpha", "0.001", Patterns::Double(0),"Weight of the Tychonoff Term");
    param_reader.declare_entry("weight_beta", "0.00001", Patterns::Double(0), "Weight of the Barrier Term");
    param_reader.declare_entry("obst_scale", "1", Patterns::Double(0.01), "Scaling of obstacle in max-term");
    param_reader.declare_entry("obst_value", "1000", Patterns::Double(0), "Value of the obstacle");
    param_reader.declare_entry("lower_bound_control", "0.1", Patterns::Double(0.01),
                               "Lower bound for semidefinite ordering");
    param_reader.declare_entry("upper_bound_control", "10", Patterns::Double(0.01),
                               "Upper bound for semidefinite ordering");
}

int
main(int argc, char **argv) {

    dealii::Utilities::MPI::MPI_InitFinalize const mpi(argc, argv);

	//Set Parameters *****************************************************
    string paramfile = "dope.prm";

    if (argc == 2) {
        paramfile = argv[1];
    }
    else if (argc > 2) {
        std::cout << "Usage: " << argv[0] << " [ paramfile ] " << std::endl;
        return -1;
    }

    ParameterReader pr;
    RP::declare_params(pr);
    RNA::declare_params(pr);
    DOpEOutputHandler<VECTOR>::declare_params(pr);
    declare_params(pr);

    pr.read_parameters(paramfile);

    //Define constants****************************************************
    pr.SetSubsection("main parameters");
    const std::string cases = pr.get_string("mode");
    const int prerefine = pr.get_integer("prerefine");
    //********************************************************************

    //Make triangulation *************************************************
    Triangulation<DIM> triangulation(Triangulation<DIM>::MeshSmoothing::patch_level_1);
    GridGenerator::hyper_cube(triangulation, -1, 1);
    //Refinement**********************************************************
    triangulation.refine_global(prerefine);
    //********************************************************************

    //Finite elements*****************************************************
    pr.SetSubsection("main parameters");
    //Components 0-2 describe the 4 entries of the control matrix (due to symmetry, the off diagonal entry is only saved once).
    FE<CDIM> const control_fe(FE_Q<CDIM>(pr.get_integer("order_control_fe")), 3);
    //Component 0 describes the PDE, Component 1 describes the multiplier
    FE<DIM> const state_fe(FE_Q<DIM>(pr.get_integer("order_state_fe")), 2);
	//********************************************************************

    //Quadrature formulas*************************************************
    pr.SetSubsection("main parameters");
    QUADRATURE quadrature_formula(pr.get_integer("quad_order"));
    FACEQUADRATURE face_quadrature_formula(pr.get_integer("facequad_order"));
    IDC idc(quadrature_formula, face_quadrature_formula);
    //********************************************************************

    //Weights*************************************************************
    pr.SetSubsection("problem-specific parameters");
    const double alpha = pr.get_double("weight_alpha");
    const double beta = pr.get_double("weight_beta");
    const double obst_scale = pr.get_double("obst_scale");
    const double obst_value = pr.get_double("obst_value");
    const double lower_bound_control = pr.get_double("lower_bound_control");
    const double upper_bound_control = pr.get_double("upper_bound_control");
    //********************************************************************

    //Functionals*********************************************************
    LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(obst_scale);
    COSTFUNCTIONAL LFunc(alpha, beta, lower_bound_control, upper_bound_control);
    //********************************************************************

    //SpaceTimeHandler****************************************************
    STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);
    //********************************************************************

    //Problem description*************************************************
    NoConstraints<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;
    OP P(LFunc, LPDE, Constraints, DOFH);
    //********************************************************************

    //Boundary conditions*************************************************
    //CompMask is set so that the multiplier is not subject to the boundary constraints
#if DEAL_II_VERSION_GTE(9,7,0)
    dealii::ComponentMask comp_mask(2, true);
    comp_mask.set(1,false);
#else
    std::vector<bool> comp_mask(2, true);
    comp_mask[1] = false;
#endif
    DOpEWrapper::ZeroFunction<DIM> const zf(2);
    SimpleDirichletData<VECTOR, DIM> const DD(zf);
    //Set dirichlet boundary values all around
    P.SetDirichletBoundaryColors(0, comp_mask, &DD);
    //********************************************************************

    //Solver**************************************************************
    RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc, 2);
    DOpEOutputHandler<VECTOR> out(&solver, pr);
    DOpEExceptionHandler<VECTOR> ex(&out);
    RNA Alg1(&P, &solver, pr, &ex, &out);
    //********************************************************************

    Alg1.ReInit();
    out.ReInit();

    //******************************************************************************************************************
    //Set the initial control values
    ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
    local::InitControl const q_exact;
    VectorTools::interpolate(DOFH.GetControlDoFHandler().GetDEALDoFHandler(), q_exact, q.GetSpacialVector());
    //Set the desired state
    StateVector<VECTOR> desired_state(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
    local::DesState const desired_state_exact;
    VectorTools::interpolate(DOFH.GetStateDoFHandler().GetDEALDoFHandler(), desired_state_exact,
                             desired_state.GetSpacialVector());
    //Set the desired control,  this is simply set to 0
    ControlVector<VECTOR> desired_control(q);
    local::DesControl const desired_control_exact;
    VectorTools::interpolate(DOFH.GetControlDoFHandler().GetDEALDoFHandler(), desired_control_exact,
                             desired_control.GetSpacialVector());
    //******************************************************************************************************************

    try {
        if (cases == "checkGrad") {
            //Initialize Obstacle*****************************************************
            StateVector<VECTOR> obstacle(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
            local::Obstacle const exact_obstacle(obst_value);
            VectorTools::interpolate(DOFH.GetStateDoFHandler().GetDEALDoFHandler(), exact_obstacle,
                                     obstacle.GetSpacialVector());
            P.AddAuxiliaryState(&obstacle, "obstacle");
            //Done with Obstacle*******************************************************
            P.AddAuxiliaryState(&desired_state, "udvalues");
            P.AddAuxiliaryControl(&desired_control, "qdvalues");
            ControlVector<VECTOR> dq(q);
            Alg1.CheckGrads(1, q, dq, 5, 0.1);
            Alg1.CheckHessian(1, q, dq, 5, 0.1);
            P.DeleteAuxiliaryState("obstacle");
            P.DeleteAuxiliaryState("udvalues");
            P.DeleteAuxiliaryControl("qdvalues");
        } else {

            stringstream outputString;
            //Initialize Obstacle******************************************************
            StateVector<VECTOR> obstacle(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
            local::Obstacle const exact_obstacle(obst_value);
            VectorTools::interpolate(DOFH.GetStateDoFHandler().GetDEALDoFHandler(), exact_obstacle,
                                     obstacle.GetSpacialVector());
            P.AddAuxiliaryState(&obstacle, "obstacle");
            //Done with Obstacle********************************************************

            P.AddAuxiliaryState(&desired_state, "udvalues");
            P.AddAuxiliaryControl(&desired_control, "qdvalues");

            Alg1.Solve(q);

            const double value = solver.GetFunctionalValue(LFunc.GetName());
            outputString << "\t Computed: " << value;
            out.Write(outputString, 1, 1, 1);

            //Write Output**********************************************************************************************
            pr.SetSubsection("main parameters");
            outputString << "  Refinementlevel: " << pr.get_integer("prerefine") << std::endl;
            outputString << "  Tychonoff Weight: " << alpha << std::endl;
            outputString << "  Barrier Weight: " << beta << std::endl;
            out.Write(outputString, 1, 1, 1);

        	P.DeleteAuxiliaryState("obstacle");
        	P.DeleteAuxiliaryState("udvalues");
        	P.DeleteAuxiliaryControl("qdvalues");

            return 0;
        }
    }
    catch (DOpEException &e) {
        std::cout
                << "Warning: During execution of `" + e.GetThrowingInstance()
                   + "` the following Problem occurred!" << std::endl;
        std::cout << e.GetErrorMessage() << std::endl;
        P.DeleteAuxiliaryState("obstacle");
        return 1;
    }

}

#undef FDC
#undef EDC
#undef FE
#undef DOFHANDLER
