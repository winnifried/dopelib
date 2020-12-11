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

#ifndef EIGENVAL_PROBLEM_H_
#define EIGENVAL_PROBLEM_H_

#include <interfaces/reducedprobleminterface.h>
#include "integrator_eigenval.h"
#include <include/parameterreader.h>
#include <include/statevector.h>
#include <problemdata/eigenvaluestateproblem.h>
#include <problemdata/eigenvaluederivativeproblem.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/lac/petsc_parallel_vector.h>

#include <container/optproblemcontainer.h>
#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/dirichletdatainterface.h>
#include <include/dopeexception.h>
//#include <templates/newtonsolver.h>
//#include <templates/newtonsolvermixeddims.h>
//#include <templates/cglinearsolver.h>
//#include <templates/gmreslinearsolver.h>
//#include <templates/directlinearsolver.h>
//#include <templates/voidlinearsolver.h>
#include <interfaces/constraintinterface.h>
#include <include/solutionextractor.h>

#include <deal.II/base/data_out_base.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/sparse_matrix.h>
#if DEAL_II_VERSION_GTE(8,5,0)
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#else
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#endif
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/petsc_parallel_vector.h>

#include <fstream>
namespace DOpE {
/**
 * Basic class to solve stationary PDE- and optimization problems.
 *
 * @tparam <CONTROLNONLINEARSOLVER>    Newton solver for the control variables.
 * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
 * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions.
 * @tparam <INTEGRATOR>                An integrator for the state variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions.
 * @tparam <PROBLEM>                   PDE- or optimization problem under consideration.
 * @tparam <VECTOR>                    Class in which we want to store the spatial vector
 *                                     (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
 * @tparam <dopedim>                   The dimension for the control variable.
 * @tparam <dealdim>                   The dimension for the state variable.
 */
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> class EigenvalueProblem : public ReducedProblemInterface<PROBLEM,
		VECTOR> { public:
		/**
		 * Constructor for the StatReducedProblem.
		 *
		 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
		 *
		 * @param OP                Problem is given to the stationary solver.
		 * @param state_behavior    Indicates the behavior of the StateVector.
		 * @param param_reader      An object which has run time data.
		 * @param idc               The InegratorDataContainer for state and control integration
		 * @param base_priority     An offset for the priority of the output written to
		 *                          the OutputHandler
		 */
		template<typename INTEGRATORDATACONT> EigenvalueProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader, INTEGRATORDATACONT &idc, int base_priority = 0);

		/**
		 * Constructor for the StatReducedProblem.
		 *
		 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
		 *
		 * @param OP                Problem is given to the stationary solver.
		 * @param state_behavior    Indicates the behavior of the StateVector.
		 * @param param_reader      An object which has run time data.
		 * @param c_idc             The InegratorDataContainer for control integration
		 * @param s_idc             The InegratorDataContainer for state integration
		 * @param base_priority     An offset for the priority of the output written to
		 *                          the OutputHandler
		 */
		template<typename STATEINTEGRATORDATACONT,
		typename CONTROLINTEGRATORCONT> EigenvalueProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc, STATEINTEGRATORDATACONT &s_idc, int base_priority = 0);

		virtual ~EigenvalueProblem();

		/******************************************************/

		/**
		 * Static member function for run time parameters.
		 *
		 * @param param_reader      An object which has run time data.
		 */
		static void declare_params(ParameterReader &param_reader);

		/******************************************************/

		/**
		 * This function sets state- and dual vectors to their correct sizes.
		 * Further, the flags to build the system matrices are set to true.
		 *
		 */
		void ReInit();

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		bool ComputeReducedConstraints(const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface,
		 *
		 */
		void GetControlBoxConstraints(ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		void ComputeReducedGradient(const ControlVector<VECTOR> &q, ControlVector<VECTOR> &gradient, ControlVector<VECTOR> &gradient_transposed);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		double ComputeReducedCostFunctional(const ControlVector<VECTOR> &q);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		void ComputeReducedFunctionals(const ControlVector<VECTOR> &q);

		/******************************************************/

		/**
		 * Computes the error indicators for the error of a previosly
		 * specified functional. Assumes that the primal state solution
		 * is already computed and the functional is specified (see
		 * problem::SetFunctionalForErrorEstimation).
		 *
		 * Everything else is determined by the DWRDataContainer
		 * you use (represented by the template parameter DWRC).
		 *
		 * @tparam <DWRC>           A container for the refinement indicators
		 *                          See, e.g., DWRDataContainer
		 * @tparam <PDE>            The problem contrainer
		 *
		 * @param q                 The ControlVector at which the indicators
		 *                          are to be evaluated.
		 * @param dwrc              The data container
		 * @param pde               The problem
		 *
		 */
		template<class DWRC,
		class PDE> void ComputeRefinementIndicators(const ControlVector<VECTOR> &q, DWRC &dwrc, PDE &pde);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		void ComputeReducedHessianVector(const ControlVector<VECTOR> &q, const ControlVector<VECTOR> &direction, ControlVector<VECTOR> &hessian_direction, ControlVector<VECTOR> &hessian_direction_transposed);

		/******************************************************/

		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 *
		 */
		void ComputeReducedGradientOfGlobalConstraints(unsigned int num, const ControlVector<VECTOR> &q, const ConstraintVector<VECTOR> &g, ControlVector<VECTOR> &gradient, ControlVector<VECTOR> &gradient_transposed);

		/******************************************************/
		void ComputeEigenvalueAdjoint(const ControlVector<VECTOR> &v, PETScWrappers::MPI::Vector &eigenfunction, double eigenvalue, std::vector<PETScWrappers::MPI::Vector> &adjoint_eigenfunctions,
		std::vector<double> &adjoint_eigenvalues);
		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 */
		void StateSizeInfo(std::stringstream &out) { GetU().PrintInfos(out); }

		/******************************************************/

		/**
		 *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
		 *  However, in later implementations other file formats will be available.
		 *
		 *  @param v           The ControlVector<VECTOR> to write to a file.
		 *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
		 *  @param dof_type    Has the DoF type: state or control.
		 */
		virtual void WriteToFile(const ControlVector<VECTOR> &v, std::string name, std::string dof_type) { this->GetOutputHandler()->Write(v.GetSpacialVector(), name, dof_type); }

		/**
		 * Basic function to write a std::vector to a file.
		 *
		 *  @param v           A std::vector to write to a file.
		 *  @param outfile     The basic name for the output file to print.
		 *  Doesn't make sense here so aborts if called!
		 */
		virtual void WriteToFile(const std::vector<double> &/*v*/, std::string /*outfile*/) { abort(); }

		/**
		 * Import overloads from base class.
		 */
		using ReducedProblemInterface<PROBLEM,
		VECTOR>::WriteToFile;

		protected:
		/**
		 * This function computes the solution for the state variable.
		 * The nonlinear solver is called, even for
		 * linear problems where the solution is computed within one iteration step.
		 *
		 * @param q            The ControlVector<VECTOR> is given to this function.
		 */
		void ComputeReducedState(const ControlVector<VECTOR> &q);

		/******************************************************/
		/**
		 * This function computes the adjoint, i.e., the Lagrange
		 * multiplier to constraint given by the state equation.
		 * It is assumed that the state u(q) corresponding to
		 * the argument q is already calculated.
		 *
		 * @param q            The ControlVector<VECTOR> is given to this function.
		 */
		void ComputeReducedAdjoint(const ControlVector<VECTOR> &q);

		/******************************************************/

		/**
		 * This function computes the solution for the dual variable
		 * for error estimation.
		 *
		 * I is assumed that the state u(q) corresponding to
		 * the argument q is already calculated.
		 *
		 * @param q            The ControlVector<VECTOR> is given to this function.
		 * @param weight_comp  A flag deciding how the weights should be calculated
		 */
		void ComputeDualForErrorEstimation(const ControlVector<VECTOR> &q, DOpEtypes::WeightComputation weight_comp);

		const StateVector<VECTOR> & GetU() const { return u_; } StateVector<VECTOR> & GetU() { return u_; } StateVector<VECTOR> & GetZ() { return z_; } StateVector<VECTOR> & GetDU() { return du_; } StateVector<VECTOR> & GetDZ() { return dz_; }
		/**
		 * Returns the solution of the dual equation for error estimation.
		 */
		const StateVector<VECTOR> & GetZForEE() const { return z_for_ee_; } StateVector<VECTOR> & GetZForEE() { return z_for_ee_; }

		NONLINEARSOLVER & GetNonlinearSolver(std::string type); CONTROLNONLINEARSOLVER & GetControlNonlinearSolver(); INTEGRATOR & GetIntegrator() { return integrator_; } CONTROLINTEGRATOR & GetControlIntegrator() { return control_integrator_; }

		private:
		/**
		 * Helper function to prevent code duplicity. Adds the user deeigenvalue_[q_point][0] *fined
		 * user Data to the Integrator.
		 */
		void AddUDD() { for (auto it = this->GetUserDomainData().begin(); it != this->GetUserDomainData().end(); it++) { this->GetIntegrator().AddDomainData(it->first, it->second); } }

		/**
		 * Helper function to prevent code duplicity. Deletes the user defined
		 * user Data from the Integrator.
		 */
		void DeleteUDD() { for (auto it = this->GetUserDomainData().begin(); it != this->GetUserDomainData().end(); it++) { this->GetIntegrator().DeleteDomainData(it->first); } }

		/**
		 * This function is used to allocate space for auxiliary parameters.
		 *
		 * @param name         The name under wich the params are stored.
		 * @param n_components The number of components needed in the paramerter vector
		 *                     at each time-point.
		 **/
		void AllocateAuxiliaryParams(std::string name, unsigned int n_components);

		std::map<std::string,
		dealii::Vector<double> >::iterator GetAuxiliaryParams(std::string name);

		/**
		 *
		 * This function calulates the functional pre-values and stores them
		 * in an auxilliary param-vector of the same name that needs
		 * to be allocated prior to calling this function.
		 *
		 * @param name        The name of the precomputation
		 *                    either `cost_functional` or
		 *                    `aux_functional`
		 * @param postfix     A postfix to be attached to the name for the problem type of the
		 *                    precalculation
		 * @param n_pre       Number of pre-iteration cycles
		 * @param prob_num    The number of the functional (only relevant for aux_functionals)
		 *
		 * After finishing the problem type is reset to the value of the `name` param
		 **/
		void CalculatePreFunctional(std::string name, std::string postfix, unsigned int n_pre, unsigned int prob_num);

		StateVector<VECTOR> u_; StateVector<VECTOR> z_; StateVector<VECTOR> du_; StateVector<VECTOR> dz_; StateVector<VECTOR> z_for_ee_;

		std::map<std::string,
		dealii::Vector<double> > auxiliary_params_;

		INTEGRATOR integrator_; CONTROLINTEGRATOR control_integrator_; NONLINEARSOLVER nonlinear_state_solver_; NONLINEARSOLVER nonlinear_adjoint_solver_; CONTROLNONLINEARSOLVER nonlinear_gradient_solver_;

		bool build_state_matrix_ = false, build_adjoint_matrix_ = false,
		build_control_matrix_ = false; bool state_reinit_, adjoint_reinit_, eigenvaluestate_reinit_, eigenvalueadjoint_reinit_,eigenvaluederivative_reinit_,
		gradient_reinit_; unsigned int cost_needs_precomputations_;
		std::vector<PETScWrappers::MPI::Vector> eigenfunctions, adjoint_eigenfunctions, tangent_eigenfunctions, adjoint_hessian_eigenfunctions;
		std::vector<double> eigenvalues, adjoint_eigenvalues, tangent_eigenvalues, adjoint_hessian_eigenvalues;
		double numOfEigenval = 2;
		double eigenvalueFIXED;

		friend class SolutionExtractor< EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim, dealdim>, VECTOR> ; };

		/*************************************************************************/
		/*****************************IMPLEMENTATION******************************/
		/*************************************************************************/
		using namespace dealii;

		/******************************************************/
		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::declare_params( ParameterReader &param_reader) { NONLINEARSOLVER::declare_params(param_reader); }
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> template<typename INTEGRATORDATACONT> EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::EigenvalueProblem( PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader, INTEGRATORDATACONT &idc, int base_priority) : ReducedProblemInterface<PROBLEM,
		VECTOR>(OP, base_priority),
		u_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		z_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		du_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		dz_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		z_for_ee_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		integrator_(idc), control_integrator_(idc),
		nonlinear_state_solver_(integrator_, param_reader),
		nonlinear_adjoint_solver_(integrator_, param_reader),
		nonlinear_gradient_solver_( control_integrator_, param_reader)

		{
		//ReducedProblems should be ReInited
		{ state_reinit_ = true; adjoint_reinit_ = true; gradient_reinit_ = true; eigenvaluestate_reinit_ = true;   eigenvalueadjoint_reinit_ = true;
		eigenvaluederivative_reinit_ = true;} cost_needs_precomputations_=0; eigenvalueFIXED=1; }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> template<typename STATEINTEGRATORDATACONT,
		typename CONTROLINTEGRATORCONT> EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::EigenvalueProblem( PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc, STATEINTEGRATORDATACONT &s_idc, int base_priority) : ReducedProblemInterface<PROBLEM,
		VECTOR>(OP, base_priority),
		u_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		z_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		du_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		dz_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		z_for_ee_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		integrator_(s_idc), control_integrator_(c_idc),
		nonlinear_state_solver_(integrator_, param_reader),
		nonlinear_adjoint_solver_(integrator_, param_reader),
		nonlinear_gradient_solver_( control_integrator_, param_reader)

		{
		//EigenvalueProblem should be ReInited
		{ state_reinit_ = true; eigenvaluestate_reinit_ = true; eigenvalueadjoint_reinit_ = true;
		eigenvaluederivative_reinit_ = true; adjoint_reinit_ = true; gradient_reinit_ = true; } cost_needs_precomputations_ = 0;
				eigenvalueFIXED = 1;


		eigenfunctions.resize((int) (numOfEigenval));
		eigenvalues.resize(numOfEigenval);
		adjoint_eigenfunctions.resize((int) (numOfEigenval));
		adjoint_eigenvalues.resize(numOfEigenval);
		adjoint_hessian_eigenfunctions.resize((int) (numOfEigenval));
		adjoint_hessian_eigenvalues.resize(numOfEigenval);
		tangent_eigenfunctions.resize((int) (numOfEigenval));
		tangent_eigenvalues.resize(numOfEigenval);
		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> EigenvalueProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
		CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim,
		dealdim>::~EigenvalueProblem() { }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> NONLINEARSOLVER & EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::GetNonlinearSolver( std::string type) { if ((type == "state") || (type == "tangent") ||(type == "eigenvaluetangent") ||(type == "eigenvaluestate")) {
			return nonlinear_state_solver_;
		} else if ((type == "adjoint") || (type == "adjoint_hessian")|| (type == "eigenvalueadjoint_hessian") || (type == "adjoint_for_ee") || (type == "eigenvalueadjoint")) {
			return nonlinear_adjoint_solver_;
		} else { throw DOpEException("No Solver for Problem type:`" + type + "' found", "EigenvalueProblem::GetNonlinearSolver");

		} }
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> CONTROLNONLINEARSOLVER & EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::GetControlNonlinearSolver() { if ((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() =="eigenvaluederivative")|| (this->GetProblem()->GetType() == "hessian")) {
			return nonlinear_gradient_solver_;
		} else {
			throw DOpEException( "No Solver for Problem type:`" + this->GetProblem()->GetType() + "' found", "EigenvalueProblem::GetControlNonlinearSolver");

		}
		}
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim, dealdim>::ReInit() { ReducedProblemInterface<PROBLEM,
		VECTOR>::ReInit();
		{

		state_reinit_ = true;
		adjoint_reinit_ = true;
		gradient_reinit_ = true;
		eigenvaluestate_reinit_ = true;
		eigenvaluederivative_reinit_ = true;
		eigenvalueadjoint_reinit_ = true;
		}

		build_state_matrix_ = true; build_adjoint_matrix_ = true;

		GetU().ReInit();
		GetZ().ReInit();
		GetDU().ReInit();
		GetDZ().ReInit();
		GetZForEE().ReInit();

		build_control_matrix_ = true;
		cost_needs_precomputations_ = 0;

		}

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
				typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
				typename VECTOR, int dopedim,
				int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
				NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
				dopedim,
				dealdim>::ComputeEigenvalueAdjoint(const ControlVector<VECTOR> &q, PETScWrappers::MPI::Vector &eigenfunction, double eigenvalue, std::vector<PETScWrappers::MPI::Vector> &adjoint_eigenfunctions,
		std::vector<double> &adjoint_eigenvalues) {
//			std::cout << "Ab hier in ComputeEigenvalueAdjoint" << std::endl;

			this->GetOutputHandler()->Write("Computing EigenvalueAdjoint:", 4 + this->GetBasePriority());
			this->SetProblemType("eigenvalueadjoint");
			auto &problem = this->GetProblem()->GetEigenvalueAdjointProblem();
			//double eigenvalue_derivative = 0;

//			if (eigenvalueadjoint_reinit_ == true) {
//				GetNonlinearSolver("eigenvalueadjoint").ReInit(problem);
//				eigenvalueadjoint_reinit_ = false;
//			}
			this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
//			    if (cost_needs_precomputations_ != 0)
//			      {
//			        auto func_vals = GetAuxiliaryParams("cost_functional_pre");
//			        this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
//			      }
			    VECTOR eigfun(eigenfunction.size());
			    eigfun = eigenfunction;
			    this->GetIntegrator().AddDomainData("state",&(eigfun));

			    if (dopedim == dealdim)
			      {
			        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
			      }
//			    else if (dopedim == 0)
//			      {
//			        this->GetIntegrator().AddParamData("control",
//			                                           &(q.GetSpacialVectorCopy()));
//			      }
			    else
			      {
			        throw DOpEException("dopedim not implemented",
			                            "EigenvalueProblem::ComputeEigenvalueAdjoint");
			      }

			    VECTOR eigval(eigenfunction.size());
			    for(unsigned int i=0; i < eigenfunction.size(); i++){
			    		eigval[i]=eigenvalue;
			     }

//			    VECTOR eigval(1);
//			    eigval[0]=eigenvalue;
			    this->GetIntegrator().AddParamData("eigenvalue",  &eigval);

			    adjoint_eigenfunctions.resize((int) (numOfEigenval)); //TODO hier reicht auch Übergabe von bestimmter EV und EW bei aktuellem Problem
			    adjoint_eigenvalues.resize(numOfEigenval);
			    build_adjoint_matrix_ = this->GetNonlinearSolver("eigenvalueadjoint").EigenvalueSolve(
			    	                             problem,adjoint_eigenvalues, adjoint_eigenfunctions, true, build_adjoint_matrix_/*, n*/);



			    			    std::cout << "################################################################" << std::endl;
			    for (unsigned int i = 0; i < 2/*numOfEigenval*/; ++i) {
			    		std::cout << "adjoint_k^2 " << " = " << adjoint_eigenvalues[i] << std::endl;
			    }
			    std::cout << "################################################################" << std::endl;
			    if (dopedim == dealdim)
			      {
			        this->GetIntegrator().DeleteDomainData("control");
			      }
//			    else if (dopedim == 0)
//			      {
//			        this->GetIntegrator().DeleteParamData("control");
//			        q.UnLockCopy();
//			      }
			    else
			      {
			        throw DOpEException("dopedim not implemented",
			                            "EigenvalueProblem::ComputeEigenvalueAdjoint");
			      }
//			    if (cost_needs_precomputations_ != 0)
//			      {
//			        this->GetIntegrator().DeleteParamData("cost_functional_pre");
//			      }
			    this->GetIntegrator().DeleteDomainData("state");
			    this->GetIntegrator().DeleteParamData("eigenvalue");

			    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

			    VECTOR adjeigfun(adjoint_eigenfunctions[0].size());
			    adjeigfun = adjoint_eigenfunctions[0];

			    //TODO Hier auch???
//			    adjeigfun *= (eigval[0]-3.9);
			    this->GetOutputHandler()->Write((adjeigfun),
			                                       "Adjoint" + this->GetPostIndex(), problem.GetDoFType());
		}





		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim>
		void EigenvalueProblem<CONTROLNONLINEARSOLVER,
				NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
				dopedim,
				dealdim>::ComputeReducedGradient(const ControlVector<VECTOR> &q, ControlVector<VECTOR> &gradient, ControlVector<VECTOR> &gradient_transposed){
			PETScWrappers::MPI::Vector eigenfunction = eigenfunctions[0];
			double eigenvalue = eigenvalues[0];
			this->ComputeEigenvalueAdjoint(q,eigenfunction, eigenvalue, adjoint_eigenfunctions, adjoint_eigenvalues);



//			std::cout << "Ab hier in ComputeDerivativesEigenvalue " << std::endl;
			this->GetOutputHandler()->Write("Computing Gradient for Eigenvalueproblem:",
			                                    4 + this->GetBasePriority());
			this->SetProblemType("eigenvaluederivative");
//			auto &problem = this->GetProblem()->GetEigenvalueDerivativeProblem();
			if (eigenvaluederivative_reinit_ == true){
			        GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
			        eigenvaluederivative_reinit_ = false;
			}

			 this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

			    if (dopedim == dealdim)
			      {
			        this->GetControlIntegrator().AddDomainData("control",
			                                                   &(q.GetSpacialVector()));
			      }
//			    else if (dopedim == 0)
//			      {
//			        this->GetControlIntegrator().AddParamData("control",
//			                                                  &(q.GetSpacialVectorCopy()));
//			      }
			    else
			      {
			        throw DOpEException("dopedim not implemented",
			                            "EigenvalueProblem::ComputeEigenvalueDerivative");
			      }
////			    if (ceigost_needs_precomputations_ != 0)
////			        {
////			          auto func_vals = GetAuxiliaryParams("cost_functional_pre");
////			          this->GetControlIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
////			        }

			    VECTOR eigval(eigenfunction.size());
			    for(unsigned int i=0; i < eigenfunction.size(); i++){
			    	eigval[i]=eigenvalue;
			    }

			    this->GetControlIntegrator().AddDomainData("eigenvalue",  &eigval);
			    this->GetIntegrator().AddDomainData("eigenvalue",  &eigval);

			    VECTOR eigfun(eigenfunction.size());
			    eigfun = eigenfunction;
			    this->GetControlIntegrator().AddDomainData("state",&(eigfun));

			    VECTOR adjeigfun(eigenfunction.size());
			    adjeigfun = adjoint_eigenfunctions[0];
			    adjeigfun *= (eigval[0]-3.8);

			    this->GetControlIntegrator().AddDomainData("adjoint",&(adjeigfun));

			      gradient_transposed = 0.;
			      if (dopedim == dealdim)
			        {
			          this->GetControlIntegrator().AddDomainData("last_newton_solution",
			                                                     &(gradient_transposed.GetSpacialVector()));
			          this->GetControlIntegrator().ComputeNonlinearResidual(
			            *(this->GetProblem()), gradient.GetSpacialVector(), eigenvalue);
			          this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
			        }
//			      else if (dopedim == 0)
//			        {
//			          this->GetControlIntegrator().AddParamData("last_newton_solution",
//			                                                    &(gradient_transposed.GetSpacialVectorCopy()));
//			  //        this->GetControlIntegrator().ComputeNonlinearResidual(
//			  //            *(this->GetProblem()), gradient.GetSpacialVector(), true);
//			          this->GetControlIntegrator().ComputeNonlinearResidual(
//			            *(this->GetProblem()), gradient.GetSpacialVector(),eigenvalue);
//
//			          this->GetControlIntegrator().DeleteParamData("last_newton_solution");
//			          gradient_transposed.UnLockCopy();
//			        }

			gradient *= -1.;
			gradient_transposed = gradient;

			//Compute l^2 representation of the Gradient
//			//TODO
				build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve(
						*(this->GetProblem()), adjoint_eigenvalues,gradient_transposed.GetSpacialVector(), true,
			                                    build_control_matrix_);




			          if(dopedim == dealdim)
			            {
			              this->GetControlIntegrator().DeleteDomainData("control");
			            }
//			          else if (dopedim == 0)
//			            {
//			              this->GetControlIntegrator().DeleteParamData("control");
//			              q.UnLockCopy();
//			            }
			          else
			            {
			              throw DOpEException("dopedim not implemented",
			                                  "EigenvalueProblem::ComputeEigenvalueDerivative");
			            }
			          if (cost_needs_precomputations_ != 0)
			            {
			              this->GetControlIntegrator().DeleteParamData("cost_functional_pre");
			            }
			          this->GetControlIntegrator().DeleteDomainData("state");
			          this->GetControlIntegrator().DeleteDomainData("adjoint");
			          this->GetControlIntegrator().DeleteDomainData("eigenvalue");
			          this->GetIntegrator().DeleteDomainData("eigenvalue");
//			          if (this->GetProblem()->HasControlInDirichletData())
//			            this->GetControlIntegrator().DeleteDomainData("adjoint_residual");

			          this->GetProblem()->DeleteAuxiliaryFromIntegrator(
			            this->GetControlIntegrator());

			          this->GetOutputHandler()->Write(gradient,
			                                          "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
			          this->GetOutputHandler()->Write(gradient_transposed,
			                                          "Gradient_Transposed" + this->GetPostIndex(),
			                                          this->GetProblem()->GetDoFType());

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedState( const ControlVector<VECTOR> &q) {
			this->InitializeFunctionalValues( this->GetProblem()->GetNFunctionals() + 1);



				this->SetProblemType("eigenvaluestate");
				auto &problem = this->GetProblem()->GetEigenvalueStateProblem();
				if(eigenvaluestate_reinit_ == true){
				 this->GetNonlinearSolver("eigenvaluestate").ReInit(problem);
					eigenvaluestate_reinit_ = false;
				}
				this->GetOutputHandler()->Write("Computing Eigenvalues and Eigenvectors (state solution):", 4 + this->GetBasePriority());
				this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

//				AddUDD();


	//			cost_needs_precomputations_ = this->GetProblem()->FunctionalNeedPrecomputations();
	//			if (cost_needs_precomputations_ != 0) {
	//				unsigned int n_pre = cost_needs_precomputations_;
	//				AllocateAuxiliaryParams("cost_functional_pre",n_pre);
	//			}
	//			this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector())); // TODO needed?

				if (dopedim == dealdim){
					 this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
	//		    } else if (dopedim == 0){
	//				 this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy()));
			    }else{
					throw DOpEException("dopedim not implemented","EigenvalueProblem::ComputeReducedState");
				}
				try{
					eigenfunctions.resize((int) (numOfEigenval));
					eigenvalues.resize(numOfEigenval);
					build_state_matrix_ = this->GetNonlinearSolver("eigenvaluestate").EigenvalueSolve(
		                 problem, eigenvalues, eigenfunctions, true, build_state_matrix_/*, n*/);

//					std::cout << "################################################################" << std::endl;
//					for (unsigned int i = 0; i < 2/*eigenvalues.size()*/; ++i) {
//						std::cout << " k^2 " << " = " << eigenvalues[i] << std::endl;
//					}
//					std::cout << "################################################################" << std::endl;
				}catch ( DOpEException &e){
					if (dopedim == dealdim){
						this->GetIntegrator().DeleteDomainData("control");
	//				}else if (dopedim == 0){
	//				    this->GetIntegrator().DeleteParamData("control");
	//				     q.UnLockCopy();
				}else{
					  throw DOpEException("dopedim not implemented",
					                                 "EigenvalueProblem::ComputeReducedCostFunctional");
				}
//				DeleteUDD();
				this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
				 //Reset Values
//				GetU().GetSpacialVector() = 0.;
				build_state_matrix_ = true;
				eigenvaluestate_reinit_ = true;
				throw e;
				}

				if (dopedim == dealdim) {
					this->GetIntegrator().DeleteDomainData("control");
	//			} else if (dopedim == 0) {
	//				this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy();
				} else {
					throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeReducedCostFunctional");
				}

				this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
				VECTOR eigfun(eigenfunctions[0].size());
				eigfun = eigenfunctions[0];

				this->GetOutputHandler()->Write(eigfun, "State" + this->GetPostIndex(), problem.GetDoFType());

		}
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> bool EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedConstraints( const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g) {
			this->GetOutputHandler()->Write("Evaluating Constraints:", 4 + this->GetBasePriority());

		this->SetProblemType("constraints");

		g = 0;
		//Local constraints
		//  this->GetProblem()->ComputeLocalConstraints(q.GetSpacialVector(), GetU().GetSpacialVector(),
		//                                              g.GetSpacialVector("local"));
		if (dopedim == dealdim) { this->GetControlIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetControlIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedConstraints"); } this->GetControlIntegrator().ComputeLocalControlConstraints( *(this->GetProblem()), g.GetSpacialVector("local")); if (dopedim == dealdim) { this->GetControlIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetControlIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedConstraints"); }
		//Global in Space-Time Constraints
		dealii::Vector<double> &gc = g.GetGlobalConstraints();
		//dealii::Vector<double> global_values(gc.size());

		unsigned int nglobal = gc.size();//global_values.size();

		if (nglobal > 0) { this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

		for (unsigned int i = 0; i < nglobal; i++) {
		//this->SetProblemType("local_global_constraints", i);
		this->SetProblemType("global_constraints", i); this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector())); if (dopedim == dealdim) { this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedConstraints"); }

		double ret = 0; bool found = false;

		if (this->GetProblem()->GetConstraintType().find("domain") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeDomainScalar( *(this->GetProblem())); } if (this->GetProblem()->GetConstraintType().find("point") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputePointScalar( *(this->GetProblem())); } if (this->GetProblem()->GetConstraintType().find("boundary") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeBoundaryScalar( *(this->GetProblem())); } if (this->GetProblem()->GetConstraintType().find("face") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeFaceScalar( *(this->GetProblem())); }

		if (!found) { throw DOpEException( "Unknown Constraint Type: " + this->GetProblem()->GetConstraintType(), "StatReducedProblem::ComputeReducedConstraints"); }
		//      global_values(i) = ret;
		gc(i) = ret;

		if (dopedim == dealdim) { this->GetIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedConstraints"); } this->GetIntegrator().DeleteDomainData("state"); }

		this->GetProblem()->DeleteAuxiliaryFromIntegrator( this->GetIntegrator());
		//gc = global_values;
		}

		//Check that no global in space, local in time constraints are given!
		if (g.HasType("local_global_control") || g.HasType("local_global_state")) { throw DOpEException( "There are global in space, local in time constraints given. In the stationary case they should be moved to global in space and time!", "StatReducedProblem::ComputeReducedConstraints"); }

		//this->GetProblem()->PostProcessConstraints(g, true);
		this->GetProblem()->PostProcessConstraints(g);
//
		return g.IsFeasible();

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::GetControlBoxConstraints( ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub) {
			this->GetProblem()->GetControlBoxConstraints(lb.GetSpacialVector(), ub.GetSpacialVector());
		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedAdjoint( const ControlVector<VECTOR> &q) {
//			this->GetOutputHandler()->Write("Computing Reduced Adjoint:", 4 + this->GetBasePriority());
//
//		this->SetProblemType("adjoint"); auto &problem = this->GetProblem()->GetAdjointProblem();
//
//		if (adjoint_reinit_ == true) { GetNonlinearSolver("adjoint").ReInit(problem); adjoint_reinit_ = false; }
//
//		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator()); if (cost_needs_precomputations_ != 0) { auto func_vals = GetAuxiliaryParams("cost_functional_pre"); this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second)); } this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector()));
//
//		if (dopedim == dealdim) { this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedAdjoint"); }
//
//		build_adjoint_matrix_ = this->GetNonlinearSolver("adjoint").NonlinearSolve( problem, (GetZ().GetSpacialVector()), true, build_adjoint_matrix_);
//
//		if (dopedim == dealdim) { this->GetIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedAdjoint"); } if (cost_needs_precomputations_ != 0) { this->GetIntegrator().DeleteParamData("cost_functional_pre"); } this->GetIntegrator().DeleteDomainData("state");
//
//		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
//
//		this->GetOutputHandler()->Write((GetZ().GetSpacialVector()), "Adjoint" + this->GetPostIndex(), problem.GetDoFType());
		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeDualForErrorEstimation( const ControlVector<VECTOR> &q, DOpEtypes::WeightComputation weight_comp) {
//			this->GetOutputHandler()->Write("Computing Dual for Error Estimation:", 4 + this->GetBasePriority());

//		if (weight_comp == DOpEtypes::higher_order_interpolation) { this->SetProblemType("adjoint_for_ee"); } else { throw DOpEException("Unknown WeightComputation", "StatPDEProblem::ComputeDualForErrorEstimation"); }
//
//		auto &problem = this->GetProblem()->GetAdjoint_For_EEProblem();
//
//		if (adjoint_reinit_ == true) { GetNonlinearSolver("adjoint_for_ee").ReInit(problem); adjoint_reinit_ = false; }
//
//		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
//
//		this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector())); //&(GetU().GetSpacialVector())
//
//		if (dopedim == dealdim) { this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedAdjoint"); }
//
//		build_adjoint_matrix_ = this->GetNonlinearSolver("adjoint_for_ee").NonlinearSolve(problem, (GetZForEE().GetSpacialVector()), true, build_adjoint_matrix_);
//
//		if (dopedim == dealdim) { this->GetIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedAdjoint"); }
//
//		this->GetIntegrator().DeleteDomainData("state");
//
//		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
//
//		this->GetOutputHandler()->Write((GetZForEE().GetSpacialVector()), "Adjoint_for_ee" + this->GetPostIndex(), problem.GetDoFType());

		}

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> double EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedCostFunctional( const ControlVector<VECTOR> &q) {
		this->ComputeReducedState(q);

		this->GetOutputHandler()->Write("Computing Cost Functional:", 4 + this->GetBasePriority());
		this->SetProblemType("cost_functional");
		cost_needs_precomputations_ = this->GetProblem()->FunctionalNeedPrecomputations();
		if (cost_needs_precomputations_ != 0) {
			unsigned int n_pre = cost_needs_precomputations_;
			AllocateAuxiliaryParams("cost_functional_pre",n_pre);

		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

		if (dopedim == dealdim) {
			this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
//		}
//		else if (dopedim == 0) {
//			this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy()));
		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}

		 VECTOR eigval(eigenfunctions[0].size());
		 for(unsigned int i = 0; i< eigval.size(); i++){
			 eigval[i]=eigenvalues[0];
		 }


		this->GetIntegrator().AddDomainData("eigenvalue",  &eigval);

		VECTOR eigfun(eigenfunctions[0].size());
		eigfun = eigenfunctions[0];
		this->GetIntegrator().AddDomainData("state",&(eigfun));


		CalculatePreFunctional("cost_functional","_pre",n_pre,0);

		if (dopedim == dealdim) {
			this->GetIntegrator().DeleteDomainData("control");
//		}
//		else if (dopedim == 0) {
//			this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy();
		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}
		this->GetIntegrator().DeleteDomainData("state");
		this->GetIntegrator().DeleteDomainData("eigenvalue");

		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
		this->SetProblemType("cost_functional");
		}
		//End of Precomputations

		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

		if (dopedim == dealdim) {
			this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
//		} else if (dopedim == 0) {
//			this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy()));
		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}

		if (cost_needs_precomputations_ != 0) {
			auto func_vals = GetAuxiliaryParams("cost_functional_pre");
			this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
		}
		VECTOR eigfun(eigenfunctions[0].size());
		eigfun = eigenfunctions[0];
//		for(unsigned int i = 0; i < eigfun.size(); i++){
//			std::cout << eigfun[i] << std::endl;
//		}
		this->GetIntegrator().AddDomainData("state",&(eigfun));

		VECTOR eigval(eigenfunctions[0].size());
//		for(unsigned int i = 0; i < eigval.size(); i++){
			eigval[0]=eigenvalues[0];
//		}

		this->GetIntegrator().AddDomainData("eigenvalue",  &eigval);

		double ret = 0;
		bool found = false;

		if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeBoundaryScalar( *(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("algebraic") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
		}

		if (!found) { throw DOpEException( "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(), "StatReducedProblem::ComputeReducedCostFunctional"); }

		if (dopedim == dealdim) {
			this->GetIntegrator().DeleteDomainData("control");
//		} else if (dopedim == 0) {
//			this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy();
		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}
		if (cost_needs_precomputations_ != 0) { this->GetIntegrator().DeleteParamData("cost_functional_pre");
		}
		this->GetIntegrator().DeleteDomainData("state");
		this->GetIntegrator().DeleteDomainData("eigenvalue");

		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

		this->GetFunctionalValues()[0].push_back(ret);
		return ret;
		}



		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedFunctionals( const ControlVector<VECTOR> &/*q*/) {
//			this->GetOutputHandler()->Write("Computing Functionals:", 4 + this->GetBasePriority());

//		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
//
//		if (dopedim == dealdim) { this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedFunctionals"); } this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector())); AddUDD();
//
//		for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++) { double ret = 0; bool found = false;
//
//		this->SetProblemType("aux_functional", i); if (this->GetProblem()->FunctionalNeedPrecomputations() != 0) { std::stringstream tmp; tmp << "aux_functional_"<<i<<"_pre"; AllocateAuxiliaryParams(tmp.str(),this->GetProblem()->FunctionalNeedPrecomputations()); CalculatePreFunctional("aux_functional","_pre", this->GetProblem()->FunctionalNeedPrecomputations(),i); auto func_vals = GetAuxiliaryParams(tmp.str()); this->GetIntegrator().AddParamData(tmp.str(),&(func_vals->second)); }
//
//		if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeDomainScalar( *(this->GetProblem())); } if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputePointScalar( *(this->GetProblem())); } if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeBoundaryScalar( *(this->GetProblem())); } if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem())); } if (this->GetProblem()->GetFunctionalType().find("algebraic") != std::string::npos) { found = true; ret += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem())); }
//
//		if (!found) { throw DOpEException( "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(), "StatReducedProblem::ComputeReducedFunctionals"); } if (this->GetProblem()->FunctionalNeedPrecomputations() != 0) { std::stringstream tmp; tmp << "aux_functional_"<<i<<"_pre"; this->GetIntegrator().DeleteParamData(tmp.str()); }
//
//		this->GetFunctionalValues()[i + 1].push_back(ret); std::stringstream out; this->GetOutputHandler()->InitOut(out); out << this->GetProblem()->GetFunctionalName() << ": " << ret; this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority()); }
//
//		if (dopedim == dealdim) { this->GetIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedFunctionals"); } this->GetIntegrator().DeleteDomainData("state"); DeleteUDD(); this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim, int dealdim> template<class DWRC,
		class PDE> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeRefinementIndicators( const ControlVector<VECTOR> &q, DWRC &dwrc, PDE &pde) {
//		//Attach the ResidualModifier to the PDE.
//		pde.ResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::ResidualModifier),boost::ref(dwrc),_1); pde.VectorResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::VectorResidualModifier),boost::ref(dwrc),_1);
//
//		//first we reinit the dwrdatacontainer (this
//		//sets the weight-vectors to their correct length)
//		dwrc.ReInit();
//
//		//Estimation for Costfunctional or if no dual is needed
//		if (this->GetProblem()->EEFunctionalIsCost() || !dwrc.NeedDual()) { this->GetOutputHandler()->Write("Computing Error Indicators:", 4 + this->GetBasePriority()); this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
//
//		//add the primal and dual solution to the integrator
//		this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector())); this->GetIntegrator().AddDomainData("adjoint_for_ee", &(GetZ().GetSpacialVector()));
//
//		if (dopedim == dealdim) { this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeRefinementIndicators"); }
//
//		this->SetProblemType("error_evaluation");
//
//		//prepare the weights...
//		dwrc.PrepareWeights(GetU(), GetZ());
//#if dope_dimension > 0
//        dwrc.PrepareWeights(q);
//#endif
//		//now we finally compute the refinement indicators
//		this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(), dwrc);
//		// release the lock on the refinement indicators (see dwrcontainer.h)
//		dwrc.ReleaseLock(); dwrc.ClearWeightData();
//
//		// clear the data
//		if (dopedim == dealdim) { this->GetIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeRefinementIndicators"); } this->GetIntegrator().DeleteDomainData("state"); this->GetIntegrator().DeleteDomainData("adjoint_for_ee"); this->GetProblem()->DeleteAuxiliaryFromIntegrator( this->GetIntegrator()); } else //Estimation for other (not the cost) functional
//		{ throw DOpEException("Estimating the error in other functionals than cost is not implemented", "StatReducedProblem::ComputeRefinementIndicators");
//
//		}
//
//		std::stringstream out; this->GetOutputHandler()->InitOut(out); out << "Error estimate using "<<dwrc.GetName(); if (dwrc.NeedDual()) out<<" For the computation of "<<this->GetProblem()->GetFunctionalName(); out<< ": "<< dwrc.GetError(); this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedHessianVector(
				const ControlVector<VECTOR> &/*q*/, const ControlVector<VECTOR> &/*direction*/, ControlVector<VECTOR> &/*hessian_direction*/, ControlVector<VECTOR> &/*hessian_direction_transposed*/) {
//		this->GetOutputHandler()->Write("Computing ReducedHessianVector:", 4 + this->GetBasePriority()); this->GetOutputHandler()->Write("\tSolving Tangent:", 5 + this->GetBasePriority());
//		std::cout << "In ComputeReducedHessianVector"<< std::endl;
//		this->SetProblemType("eigenvaluetangent");
//		{ //Start Tangent Calculatation
//		auto &problem = this->GetProblem()->GetEigenvalueTangentProblem();
//
//
//		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
//		VECTOR eigfun(eigenfunctions[0].size()); //TODO anpassen bei Problemänderung
//		eigfun = eigenfunctions[0];
//
//		this->GetIntegrator().AddDomainData("state", /*&(GetU().GetSpacialVector())*/&(eigfun));
//		this->GetControlIntegrator().AddDomainData("state",/*&(GetU().GetSpacialVector())*/&(eigfun));
//
//		if (dopedim == dealdim) {
//			this->GetIntegrator().AddDomainData("dq", &(direction.GetSpacialVector()));
//			this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
//		} else if (dopedim == 0) {
//			this->GetIntegrator().AddParamData("dq", &(direction.GetSpacialVectorCopy()));
//			this->GetIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy()));
//		} else {
//			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeReducedHessianVector");
//		}
//
//
////		tangent Matrix is the same as state matrix
//		tangent_eigenfunctions.resize((int) (numOfEigenval));
//		tangent_eigenvalues.resize(tangent_eigenfunctions.size());
//		build_state_matrix_ = this->GetNonlinearSolver("eigenvaluetangent").EigenvalueSolve(
//				                         problem, tangent_eigenvalues, tangent_eigenfunctions, true, build_state_matrix_/*, n*/);
//
//		std::cout << "################################################################" << std::endl;
//		for (unsigned int i = 0; i < tangent_eigenvalues.size(); ++i) {
//			std::cout << "tangent_k^2 " << i << " = " << tangent_eigenvalues[i] << std::endl;
//		}
//		std::cout << "################################################################" << std::endl;
//
//		VECTOR taneigfun(tangent_eigenfunctions[0].size());
//		taneigfun = tangent_eigenfunctions[0];
//		VECTOR adjeigfun(adjoint_eigenfunctions[0].size());
//		adjeigfun = adjoint_eigenfunctions[0];
//
//		this->GetOutputHandler()->Write(taneigfun, "Tangent" + this->GetPostIndex(), problem.GetDoFType());
//		//End Tangent Calculation
//
//		this->GetIntegrator().AddDomainData("adjoint", &(adjeigfun));
//		this->GetIntegrator().AddDomainData("tangent", &(taneigfun));
//		this->GetControlIntegrator().AddDomainData("adjoint", &(adjeigfun));
//		this->GetControlIntegrator().AddDomainData("tangent", &(taneigfun));
//
//		//After the Tangent has been computed, we can precompute
//		//cost functional derivative-values (if necessary)
////		if (cost_needs_precomputations_ != 0) {
////			unsigned int n_pre = cost_needs_precomputations_;
////			AllocateAuxiliaryParams("cost_functional_pre_tangent",n_pre);
////			CalculatePreFunctional("cost_functional","_pre_tangent",n_pre,0);
//		} // End precomputation of values
//
//		this->GetOutputHandler()->Write("\tSolving Eigenvalue Adjoint Hessian:", 5 + this->GetBasePriority());
//		this->SetProblemType("eigenvalueadjoint_hessian");
////		//Adjoint_Hessian
//		{
//		auto &problem = this->GetProblem()->GetEigenvalueAdjoint_HessianProblem();
////
////		if (cost_needs_precomputations_ != 0) { { auto func_vals = GetAuxiliaryParams("cost_functional_pre"); this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second)); this->GetControlIntegrator().AddParamData("cost_functional_pre",&(func_vals->second)); } { auto func_vals = GetAuxiliaryParams("cost_functional_pre_tangent"); this->GetIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second)); this->GetControlIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second)); } }
////
//		//adjoint_hessian Matrix is the same as adjoint matrix
//
//		adjoint_hessian_eigenfunctions.resize((int) (numOfEigenval));
//		adjoint_hessian_eigenvalues.resize(adjoint_hessian_eigenfunctions.size());
//		build_adjoint_matrix_ = this->GetNonlinearSolver("eigenvalueadjoint_hessian").EigenvalueSolve(
//						                         problem, adjoint_hessian_eigenvalues, adjoint_hessian_eigenfunctions, true, build_state_matrix_/*, n*/);
//
//				std::cout << "################################################################" << std::endl;
//				for (unsigned int i = 0; i < numOfEigenval; ++i) {
//					std::cout << "adjoint_hessian_k^2 " << i << " = " << adjoint_hessian_eigenvalues[i] << std::endl;
//				}
//				std::cout << "################################################################" << std::endl;
//
//		VECTOR adjhesseigfun(adjoint_hessian_eigenfunctions[0].size());
//		adjhesseigfun = adjoint_hessian_eigenfunctions[0];
//
//		this->GetOutputHandler()->Write(adjhesseigfun, "Hessian" + this->GetPostIndex(), problem.GetDoFType());
//
//		this->GetIntegrator().AddDomainData("adjoint_hessian", &(adjhesseigfun));
//		this->GetControlIntegrator().AddDomainData("adjoint_hessian", &(adjhesseigfun));
//
//		this->GetOutputHandler()->Write( "\tComputing Representation of the Hessian:", 5 + this->GetBasePriority());
//
//		} //End Adjoint Hessian
//
////		 //Preparations for Control In The Dirichlet Data
////		VECTOR tmp; VECTOR tmp_second;
////		if (this->GetProblem()->HasControlInDirichletData()) {
////			tmp.reinit(GetU().GetSpacialVector());
////			tmp_second.reinit(GetU().GetSpacialVector()); this->SetProblemType("adjoint"); {
////		}
//////		// Adjoint
//////		auto &problem = this->GetProblem()->GetAdjointProblem();
//////
//////		this->GetIntegrator().AddDomainData("last_newton_solution", &(GetZ().GetSpacialVector()));
//////
////////    this->GetIntegrator().ComputeNonlinearResidual(problem, tmp_second, false);
//////		this->GetIntegrator().ComputeNonlinearResidual(problem, tmp_second); tmp_second *= -1.;
//////
//////		this->GetIntegrator().DeleteDomainData("last_newton_solution"); } //End Adjoint
//////		this->SetProblemType("adjoint_hessian");
////		//{
//////		//Adjoint_Hessian
//////		auto &problem = this->GetProblem()->GetAdjoint_HessianProblem();
//////
//////		this->GetIntegrator().AddDomainData("last_newton_solution", &(GetDZ().GetSpacialVector()));
//////
////////    this->GetIntegrator().ComputeNonlinearResidual(problem, tmp, false);
//////		this->GetIntegrator().ComputeNonlinearResidual(problem, tmp); tmp *= -1.;
//////
//////		this->GetIntegrator().DeleteDomainData("last_newton_solution"); } }
//////		//Endof Dirichletdata Preparations
//
//
//
//		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
//
//		this->SetProblemType("eigenvaluehessian");
//		this->GetProblem()->AddAuxiliaryToIntegrator( this->GetControlIntegrator());
//		if (dopedim == dealdim) {
//			this->GetIntegrator().DeleteDomainData("dq");
//			this->GetIntegrator().DeleteDomainData("control");
//			this->GetControlIntegrator().AddDomainData("dq", &(direction.GetSpacialVector()));
//			this->GetControlIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
////		} else if (dopedim == 0) {
////			this->GetIntegrator().DeleteParamData("dq");
////			this->GetIntegrator().DeleteParamData("control");
////			direction.UnLockCopy(); q.UnLockCopy();
////			this->GetControlIntegrator().AddParamData("dq", &(direction.GetSpacialVectorCopy()));
////			this->GetControlIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy()));
//		} else {
//			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedHessianVector");
//		}
////		if (this->GetProblem()->HasControlInDirichletData()) {
////			this->GetControlIntegrator().AddDomainData("adjoint_residual", &tmp);
////			this->GetControlIntegrator().AddDomainData("hessian_residual", &tmp_second);
////		}
////
//		{
//			hessian_direction_transposed = 0.;
//			if (dopedim == dealdim) {
//				this->GetControlIntegrator().AddDomainData("last_newton_solution", &(hessian_direction_transposed.GetSpacialVector()));
//				this->GetControlIntegrator().ComputeNonlinearResidual( *(this->GetProblem()), hessian_direction.GetSpacialVector()/*,eigenvalues[0]*/);
//				this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
//			}
////		} else if (dopedim == 0) {
////			this->GetControlIntegrator().AddParamData("last_newton_solution", &(hessian_direction_transposed.GetSpacialVectorCopy()));
////////         this->GetControlIntegrator().ComputeNonlinearResidual(
////////             *(this->GetProblem()), hessian_direction.GetSpacialVector(),
////////              true);
//////		this->GetControlIntegrator().ComputeNonlinearResidual( *(this->GetProblem()), hessian_direction.GetSpacialVector()); this->GetControlIntegrator().DeleteParamData("last_newton_solution"); hessian_direction_transposed.UnLockCopy();
//		}
//		hessian_direction *= -1.; hessian_direction_transposed = hessian_direction;
////		//Compute l^2 representation of the HesssianVector
////		//hessian Matrix is the same as control matrix
////		build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve( *(this->GetProblem()), hessian_direction_transposed.GetSpacialVector(), true, build_control_matrix_);
//
//		this->GetOutputHandler()->Write(hessian_direction, "HessianDirection" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
//		this->GetOutputHandler()->Write(hessian_direction_transposed, "HessianDirection_Transposed" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
//
//
//		if (dopedim == dealdim) {
//			this->GetControlIntegrator().DeleteDomainData("dq");
//			this->GetControlIntegrator().DeleteDomainData("control");
//		}/* else if (dopedim == 0) {
//			this->GetControlIntegrator().DeleteParamData("dq");
//			this->GetControlIntegrator().DeleteParamData("control");
//			direction.UnLockCopy(); q.UnLockCopy();
//		} */else {
//			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedHessianVector");
//		}
//		this->GetIntegrator().DeleteDomainData("state");
//		this->GetIntegrator().DeleteDomainData("adjoint");
//		this->GetIntegrator().DeleteDomainData("tangent");
//		this->GetIntegrator().DeleteDomainData("adjoint_hessian");
//		this->GetControlIntegrator().DeleteDomainData("state");
//		this->GetControlIntegrator().DeleteDomainData("adjoint");
//		this->GetControlIntegrator().DeleteDomainData("tangent");
//		this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
//		if (this->GetProblem()->HasControlInDirichletData()) {
//			this->GetControlIntegrator().DeleteDomainData("adjoint_residual");
//			this->GetControlIntegrator().DeleteDomainData("hessian_residual");
//		}
//		this->GetProblem()->DeleteAuxiliaryFromIntegrator( this->GetControlIntegrator());
////
////		if (cost_needs_precomputations_ != 0) { this->GetIntegrator().DeleteParamData("cost_functional_pre"); this->GetIntegrator().DeleteParamData("cost_functional_pre_tangent"); this->GetControlIntegrator().DeleteParamData("cost_functional_pre"); this->GetControlIntegrator().DeleteParamData("cost_functional_pre_tangent"); }

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedGradientOfGlobalConstraints(unsigned int /*num*/, const ControlVector<VECTOR> &/*q*/, const ConstraintVector<VECTOR> &/*g*/, ControlVector<VECTOR> &/*gradient*/, ControlVector<VECTOR> &/*gradient_transposed*/) {
//		//FIXME: If the global constraints depend on u we need to calculate a corresponding
//		//       dual solution before we can calculate the gradient.
//		std::stringstream out; out << "Computing Reduced Gradient of global constraint " << num << " :"; this->GetOutputHandler()->Write(out, 4 + this->GetBasePriority());
//		//Compute derivatives of global constraints
//		this->SetProblemType("global_constraint_gradient", num);
//
//		if (dopedim == dealdim) { this->GetControlIntegrator().AddDomainData("control", &(q.GetSpacialVector())); } else if (dopedim == 0) { this->GetControlIntegrator().AddParamData("control", &(q.GetSpacialVectorCopy())); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedGradient"); } this->GetProblem()->AddAuxiliaryToIntegrator( this->GetControlIntegrator()); this->GetControlIntegrator().AddDomainData("constraints_local", &g.GetSpacialVector("local")); this->GetControlIntegrator().AddParamData("constraints_global", &g.GetGlobalConstraints());
//
//		//Compute
////      this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()), gradient.GetSpacialVector(), true);
//		this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()), gradient.GetSpacialVector()); gradient_transposed = gradient;
//
//		this->GetControlIntegrator().DeleteDomainData("constraints_local"); this->GetControlIntegrator().DeleteParamData("constraints_global"); this->GetProblem()->DeleteAuxiliaryFromIntegrator( this->GetControlIntegrator()); if (dopedim == dealdim) { this->GetControlIntegrator().DeleteDomainData("control"); } else if (dopedim == 0) { this->GetControlIntegrator().DeleteParamData("control"); q.UnLockCopy(); } else { throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedGradient"); }
		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>:: AllocateAuxiliaryParams(std::string name, unsigned int n_components) { std::map<std::string,
		dealii::Vector<double> >::iterator func_vals = auxiliary_params_.find(name); if (func_vals != auxiliary_params_.end()) { assert(func_vals->second.size() == n_components);
		//already created. Nothing to do
		} else { auto ret = auxiliary_params_.emplace(name,dealii::Vector<double>(n_components)); if (ret.second == false) { throw DOpEException("Creation of Storage for Auxiliary time params with name "+name+" failed!", "StatReducedProblem::AllocateAuxiliaryParams"); } } }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim, int dealdim> std::map<std::string,
		dealii::Vector<double> >::iterator EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim, dealdim>:: GetAuxiliaryParams(std::string name)

		{ return auxiliary_params_.find(name); }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
CalculatePreFunctional(std::string name,
		std::string postfix,
		unsigned int n_pre,
		unsigned int prob_num)
{
	//Checking input
	if (name != "aux_functional" && name != "cost_functional")
	{
		throw DOpEException("Only valid with name `aux_functional` or `cost_functional` but not: "+name ,
				"StatReducedProblem::CalculatePreFunctional");
	}
	if (postfix == "" || postfix == " ")
	{
		throw DOpEException("Postfix needs to be a non-empty string" ,
				"StatReducedProblem::CalculatePreFunctional");
	}
	//Create problem name
	std::string pname;
	{
		std::stringstream tmp;
		tmp << name;
		if (name == "aux_functional")
		{
			tmp<<"_"<<prob_num;
		}
		else
		{
			assert(prob_num == 0);
			assert(name == "cost_functional");
		}
		tmp<<postfix;
		pname = tmp.str();
	}
	//Begin Precomputation
	auto func_vals = GetAuxiliaryParams(pname);
	for (unsigned int i = 0; i < n_pre; i++)
	{
		this->SetProblemType(pname,i);
		this->GetOutputHandler()->Write("\tprecomputations for "+name,
				4 + this->GetBasePriority());
		//Begin Precomputations
		bool found = false;
		double pre = 0;

		if (this->GetProblem()->GetFunctionalType().find("domain")
				!= std::string::npos)
		{
			found = true;
			pre += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("point")
				!= std::string::npos)
		{
			found = true;
			pre += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("boundary")
				!= std::string::npos)
		{
			found = true;
			pre += this->GetIntegrator().ComputeBoundaryScalar(
					*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("face")
				!= std::string::npos)
		{
			found = true;
			pre += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("algebraic")
				!= std::string::npos)
		{
			found = true;
			pre += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
		}

		if (!found)
		{
			throw DOpEException(
					"Unknown Functional Type: "
					+ this->GetProblem()->GetFunctionalType(),
					"StatReducedProblem::CalculatePreFunctional");
		}
		//Store Precomputed Values
		func_vals->second[i] = pre;
	}
	if (name == "aux_functional")
	{
		this->SetProblemType(name,prob_num);
	}
	else
	{
		this->SetProblemType(name);
	}
}
//////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
