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
 * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,eigfun_
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
		void ComputeEigenvalueAdjoint(const ControlVector<VECTOR> &v, /*PETScWrappers::MPI::Vector*/ StateVector<VECTOR> &eigenfunction, double eigenvalue, /*std::vector<PETScWrappers::MPI::Vector>*/ StateVector<VECTOR> &adjoint_eigenfunction, /*std::vector<double> &*/double adjoint_eigenvalue);
		/**
		 * Implementation of Virtual Method in Base Class
		 * ReducedProblemInterface
		 */
		void StateSizeInfo(std::stringstream &out) { GetEigfun().PrintInfos(out); /*GetU()*/}

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

		StateVector<VECTOR> &GetEigfun(){
		      return eigfun_;
		}

		const StateVector<VECTOR> &GetEigfun() const{
			  return eigfun_;
		}

		StateVector<VECTOR> &GetAdjfun(){
		      return adjeigfun_;
		}

		const StateVector<VECTOR> &GetAdjfun() const{
			  return adjeigfun_;
		}


		NONLINEARSOLVER & GetNonlinearSolver(std::string type); CONTROLNONLINEARSOLVER & GetControlNonlinearSolver(); INTEGRATOR & GetIntegrator() { return integrator_; } CONTROLINTEGRATOR & GetControlIntegrator() { return control_integrator_; }

		private:
		/**
		 * Helper function to prevent code duplicity. Adds the user defined
		 * user Data to the Integrator.
		 */
		void AddUDD() {
			for (auto it = this->GetUserDomainData().begin(); it != this->GetUserDomainData().end(); it++) {
				this->GetIntegrator().AddDomainData(it->first, it->second);
			}
		}
		void AddUDDControl() {
					for (auto it = this->GetUserDomainData().begin(); it != this->GetUserDomainData().end(); it++) {
						this->GetControlIntegrator().AddDomainData(it->first, it->second);
			}
		}

		/**
		 * Helper function to prevent code duplicity. Deletes the user defined
		 * user Data from the Integrator.
		 */
		void DeleteUDD() {
			for (auto it = this->GetUserDomainData().begin();
					it != this->GetUserDomainData().end(); it++) {
				this->GetIntegrator().DeleteDomainData(it->first);
			}
		}

		void DeleteUDDControl() {
					for (auto it = this->GetUserDomainData().begin();
							it != this->GetUserDomainData().end(); it++) {
						this->GetControlIntegrator().DeleteDomainData(it->first);
					}
				}
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

		StateVector<VECTOR> eigfun_; StateVector<VECTOR>  adjeigfun_; double eigenval_;
		VECTOR  vecOfEigval_;  double adjeigenval_; VECTOR  vecOfAdjeigval_;
//		VECTOR  initial_control_;

		std::map<std::string,
		dealii::Vector<double> > auxiliary_params_;

//		double initial_counter;

		INTEGRATOR integrator_; CONTROLINTEGRATOR control_integrator_; NONLINEARSOLVER nonlinear_state_solver_; NONLINEARSOLVER nonlinear_adjoint_solver_; CONTROLNONLINEARSOLVER nonlinear_gradient_solver_;

		bool build_state_matrix_ = false, build_adjoint_matrix_ = false,
		build_control_matrix_ = false;
		bool eigenvaluestate_reinit_, eigenvalueadjoint_reinit_,
		eigenvaluederivative_reinit_;
		unsigned int cost_needs_precomputations_;
		std::vector<PETScWrappers::MPI::Vector> eigenfunctions,	adjoint_eigenfunctions;
		std::vector<double> eigenvalues,adjoint_eigenvalues;
		double numOfEigenval = 2;

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
		dealdim>::declare_params( ParameterReader &param_reader) {
			NONLINEARSOLVER::declare_params(param_reader);
		}
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> template<typename INTEGRATORDATACONT> EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::EigenvalueProblem( PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader, INTEGRATORDATACONT &idc, int base_priority) : ReducedProblemInterface<PROBLEM,
		VECTOR>(OP, base_priority),
		eigfun_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		adjeigfun_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		integrator_(idc), control_integrator_(idc),
		nonlinear_state_solver_(integrator_, param_reader),
		nonlinear_adjoint_solver_(integrator_, param_reader),
		nonlinear_gradient_solver_( control_integrator_, param_reader)

		{
		//ReducedProblems should be ReInited
		{
			eigenvaluestate_reinit_ = true;
			eigenvalueadjoint_reinit_ = true;
			eigenvaluederivative_reinit_ = true;
		}
			cost_needs_precomputations_=0;

			eigenfunctions.resize((int) (numOfEigenval));
			eigenvalues.resize(numOfEigenval);

//			initial_control_.resize(numOfEigenval);

			adjoint_eigenfunctions.resize((int) (numOfEigenval));
			adjoint_eigenvalues.resize(numOfEigenval);

			eigenval_=0.;
			adjeigenval_=0.;
//			initial_counter = 0;
		}

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
		eigfun_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		adjeigfun_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
		integrator_(s_idc), control_integrator_(c_idc),
		nonlinear_state_solver_(integrator_, param_reader),
		nonlinear_adjoint_solver_(integrator_, param_reader),
		nonlinear_gradient_solver_( control_integrator_, param_reader)

		{
		//EigenvalueProblem should be ReInited
		{
			eigenvaluestate_reinit_ = true;
			eigenvalueadjoint_reinit_ = true;
			eigenvaluederivative_reinit_ = true;
		}
		cost_needs_precomputations_ = 0;

		eigenfunctions.resize((int) (numOfEigenval));
		eigenvalues.resize(numOfEigenval);

		adjoint_eigenfunctions.resize((int) (numOfEigenval));
		adjoint_eigenvalues.resize(numOfEigenval);

//		initial_control_.resize(numOfEigenval);

//		initial_counter = 0;
		eigenval_=0.;
		adjeigenval_=0.;
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
		dealdim>::GetNonlinearSolver( std::string type) {
			if (type == "eigenvaluestate"){
				return nonlinear_state_solver_;
			} else if (type == "eigenvalueadjoint") {
				return nonlinear_adjoint_solver_;
			} else {
				throw DOpEException("No Solver for Problem type:`" + type + "' found", "EigenvalueProblem::GetNonlinearSolver");

		} }
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> CONTROLNONLINEARSOLVER & EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::GetControlNonlinearSolver() {
			if (this->GetProblem()->GetType() =="eigenvaluederivative") {
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
		dopedim, dealdim>::ReInit()
		{
			ReducedProblemInterface<PROBLEM,
			VECTOR>::ReInit();
			{
			eigenvaluestate_reinit_ = true;
			eigenvaluederivative_reinit_ = true;
			eigenvalueadjoint_reinit_ = true;
			}

		build_state_matrix_ = true;
		build_adjoint_matrix_ = true;
		GetAdjfun().ReInit();
		GetEigfun().ReInit();

//		initial_counter = 0;

		eigenval_=0.;
		adjeigenval_=0.;

		build_control_matrix_ = true;
		cost_needs_precomputations_ = 0;

		}

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeEigenvalueAdjoint(const ControlVector<VECTOR> &q, StateVector<VECTOR> &eigenfunction, double /*eigenvalue*/, StateVector<VECTOR> &adjoint_eigenfunction, double adjoint_eigenvalue) {
			this->GetOutputHandler()->Write("Computing EigenvalueAdjoint:", 4 + this->GetBasePriority());
			this->SetProblemType("eigenvalueadjoint");
			auto &problem = this->GetProblem()->GetEigenvalueAdjointProblem();

		if (eigenvalueadjoint_reinit_ == true) {
			GetNonlinearSolver("eigenvalueadjoint").ReInit(problem);
			eigenvalueadjoint_reinit_ = false;
		}
		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
		if (cost_needs_precomputations_ != 0) {
			auto func_vals = GetAuxiliaryParams("cost_functional_pre");
			this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
		}
		this->GetIntegrator().AddDomainData("state",&(GetEigfun().GetSpacialVector()));

		if (dopedim == dealdim) {
			this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
		} else {
			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeEigenvalueAdjoint");
		}

		this->GetIntegrator().AddParamData("eigenvalue", &(vecOfEigval_));
//		AddUDD();

		adjoint_eigenfunctions.resize((int) (numOfEigenval));
		adjoint_eigenvalues.resize(numOfEigenval);
		PETScWrappers::SparseMatrix matrixM;
		matrixM.clear();
//		SparseMatrix<double> testM = matrixM;

		build_adjoint_matrix_ = this->GetNonlinearSolver("eigenvalueadjoint").EigenvalueSolve(problem,matrixM, adjoint_eigenvalues, adjoint_eigenfunctions, true, build_adjoint_matrix_/*, n*/);



//			PETScWrappers::MPI::Vector vec;
		//TODO ACHTUNG adjoint_eigenfunctions[1] wird hier neu zugewiesen und hat nichts mehr mit dem eigenfunctions zutun -- ändern
			MatMult(matrixM,adjoint_eigenfunctions[0],adjoint_eigenfunctions[1]); // = matrixM*adjoint_eigenfunctions[i];
			PetscScalar scalar_factor[1];
			VecDot(adjoint_eigenfunctions[1],eigenfunctions[0],scalar_factor);//= scalar_product(vec,eigenfunctions[i]);
//			std::cout << scalar_factor[0] << std::endl;
			adjoint_eigenfunctions[0]/= scalar_factor[0];
			adjoint_eigenfunctions[0] *= -(eigenvalues[0]-1.5);


		GetAdjfun().GetSpacialVector() = 0.;
		/*adjoint_eigenfunction.GetSpacialVector() = adjoint_eigenfunctions[0];   //Needed?*/
		adjeigfun_ .GetSpacialVector() = adjoint_eigenfunctions[0];
		GetAdjfun().GetSpacialVector() = adjoint_eigenfunctions[0];

//		for (unsigned int i = 0; i < adjeigfun_ .GetSpacialVector().size();i++){
//			std::cout << adjeigfun_ .GetSpacialVector()[i] << std::endl;
//		}

		std::cout << "################################################################" << std::endl;
		for (unsigned int i = 0; i < numOfEigenval; ++i) {
			std::cout << "adjoint_k^2 " << " = " << adjoint_eigenvalues[i] << std::endl;
		}
		std::cout << "################################################################" << std::endl;

		if (dopedim == dealdim) {
			this->GetIntegrator().DeleteDomainData("control");
		} else {
			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeEigenvalueAdjoint");
		}

		if (cost_needs_precomputations_ != 0) {
			this->GetIntegrator().DeleteParamData("cost_functional_pre");
		}
//		DeleteUDD();
		this->GetIntegrator().DeleteDomainData("state");
		this->GetIntegrator().DeleteParamData("eigenvalue");

		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
		this->GetOutputHandler()->Write((GetAdjfun().GetSpacialVector()), "Adjoint" + this->GetPostIndex(), problem.GetDoFType());

		}

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedGradient(const ControlVector<VECTOR> &q, ControlVector<VECTOR> &gradient, ControlVector<VECTOR> &gradient_transposed){

			this->ComputeEigenvalueAdjoint(q, eigfun_, eigenval_, adjeigfun_, adjeigenval_);

			this->GetOutputHandler()->Write("Computing Gradient for Eigenvalueproblem:", 4 + this->GetBasePriority());

			if (this->GetProblem()->HasControlInDirichletData()) {
				std::cout << "TODO CONTROL IN DIRICHLETDATA " << std::endl;
			}

		this->SetProblemType("eigenvaluederivative");
		if (eigenvaluederivative_reinit_ == true){
			GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
			eigenvaluederivative_reinit_ = false;
		}

		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
		if (dopedim == dealdim) {
			this->GetControlIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
		} else{
			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeEigenvalueDerivative");
		}
		if (cost_needs_precomputations_ != 0) {
			auto func_vals = GetAuxiliaryParams("cost_functional_pre");
			this->GetControlIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
		}
		AddUDD();
		AddUDDControl();
		this->GetControlIntegrator().AddDomainData("eigenvalue", &(vecOfEigval_));
		this->GetControlIntegrator().AddDomainData("state",&(GetEigfun().GetSpacialVector()));
		this->GetControlIntegrator().AddDomainData("adjoint",&(GetAdjfun().GetSpacialVector()));

		gradient_transposed = 0.;
		if (dopedim == dealdim) {
			this->GetControlIntegrator().AddDomainData("last_newton_solution", &(gradient_transposed.GetSpacialVector()));
			this->GetControlIntegrator().ComputeNonlinearResidual( *(this->GetProblem()), gradient.GetSpacialVector(), eigenval_);
			this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
		} else if (dopedim == 0) {
			this->GetControlIntegrator().AddParamData("last_newton_solution", &(gradient_transposed.GetSpacialVectorCopy()));
			this->GetControlIntegrator().ComputeNonlinearResidual( *(this->GetProblem()), gradient.GetSpacialVector(),eigenval_);
			this->GetControlIntegrator().DeleteParamData("last_newton_solution"); gradient_transposed.UnLockCopy();
		}
		gradient *= -1.;
		gradient_transposed = gradient;

		//Compute l^2 representation of the Gradient
		build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve( *(this->GetProblem()), adjoint_eigenvalues,gradient_transposed.GetSpacialVector(), true, build_control_matrix_);

		if(dopedim == dealdim) {
		this->GetControlIntegrator().DeleteDomainData("control");
		} else {
			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeEigenvalueDerivative");
		}

		DeleteUDD();
		DeleteUDDControl();
		this->GetControlIntegrator().DeleteDomainData("state");
		this->GetControlIntegrator().DeleteDomainData("adjoint");
		this->GetControlIntegrator().DeleteDomainData("eigenvalue");

		this->GetProblem()->DeleteAuxiliaryFromIntegrator( this->GetControlIntegrator());

		this->GetOutputHandler()->Write(gradient, "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
		this->GetOutputHandler()->Write(gradient_transposed, "Gradient_Transposed" + this->GetPostIndex(), this->GetProblem()->GetDoFType());

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

			AddUDD();
		if (dopedim == dealdim){
			this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
		}else{
			throw DOpEException("dopedim not implemented","EigenvalueProblem::ComputeReducedState");
		}
		try{
			eigenfunctions.resize((int) (numOfEigenval));
			eigenvalues.resize(numOfEigenval);

			PETScWrappers::SparseMatrix matrixM;
			matrixM.clear();

			build_state_matrix_ = this->GetNonlinearSolver("eigenvaluestate").EigenvalueSolve(problem, matrixM, eigenvalues, eigenfunctions, true, build_state_matrix_ /*, n*/);



				PETScWrappers::MPI::Vector vec;
//
				MatMult(matrixM,eigenfunctions[0],eigenfunctions[1]); // VECTOR vec = matrixM.vmult(eigenfunctions[i]);

				//TODO ACHTUNG: eigenfunctions[1] wird hier neu definiert --- TODO eigenen PETSCVector initialisieren mit richtiger länge für mehr übersicht
				PetscScalar scalar_factor[1];
				VecDot(eigenfunctions[1],eigenfunctions[0],scalar_factor);
				scalar_factor[0] = sqrt(scalar_factor[0]);  //double scalar_factor = sqrt(scalar_product(vec,eigenfunctions[i]));
				eigenfunctions[0]/= scalar_factor[0];


//			std::cout << "################################################################" << std::endl;
//				for (unsigned int i = 0; i < numOfEigenval; ++i) {
//					std::cout << "k^2 " << " = " << eigenvalues[i] << std::endl;
//				}
//				std::cout << "################################################################" << std::endl;

		}catch ( DOpEException &e){
			if (dopedim == dealdim){
				this->GetIntegrator().DeleteDomainData("control");
			}else{
				throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeReducedState");
			}

		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
		 //Reset Values
		GetEigfun().GetSpacialVector() = 0.;
		build_state_matrix_ = true;
		eigenvaluestate_reinit_ = true;
		throw e;
		}
		DeleteUDD();
		if (dopedim == dealdim) {
			this->GetIntegrator().DeleteDomainData("control");
		} else {
			throw DOpEException("dopedim not implemented", "EigenvalueProblem::ComputeReducedState");
		}

		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

		GetEigfun().GetSpacialVector() = eigenfunctions[0];
		eigfun_.GetSpacialVector() = eigenfunctions[0];   //Needed?

		eigenval_ = eigenvalues[0];
		vecOfEigval_.reinit(eigenfunctions[0].size());
		vecOfEigval_ = eigenval_;

		this->GetOutputHandler()->Write(GetEigfun().GetSpacialVector(), "State" + this->GetPostIndex(), problem.GetDoFType());

		}
		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> bool EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedConstraints( const ControlVector<VECTOR> &/*q*/, ConstraintVector<VECTOR> &/*g*/) {
			std::cout << "ComputeReducedConstraints" << std::endl;

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::GetControlBoxConstraints( ControlVector<VECTOR> &/*lb*/, ControlVector<VECTOR> &/*ub*/) { std::cout << "GetControlBoxConstraints" << std::endl;

		}

		/******************************************************/

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
				} else {
					throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
				}
				this->GetIntegrator().AddDomainData("eigenvalue", &(vecOfEigval_));
				this->GetIntegrator().AddDomainData("state",&(GetEigfun().GetSpacialVector()));

				CalculatePreFunctional("cost_functional","_pre",n_pre,0);
				if (dopedim == dealdim) {
					this->GetIntegrator().DeleteDomainData("control");
				}else {
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
		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}

		if (cost_needs_precomputations_ != 0) {
			auto func_vals = GetAuxiliaryParams("cost_functional_pre");
			this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
		}

		this->GetIntegrator().AddDomainData("eigenvalue", &(vecOfEigval_));
		this->GetIntegrator().AddDomainData("state",&(GetEigfun().GetSpacialVector()));
		AddUDD();

		double ret = 0;
		bool found = false;

		if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos) {
			found = true;
			ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos) {
			found = true;
			ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeBoundaryScalar( *(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos) {
			found = true; ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
		}
		if (this->GetProblem()->GetFunctionalType().find("algebraic") != std::string::npos) {
			double ev = vecOfEigval_[0];
			found = true;
			ret += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()),ev);
		}

		if (!found) {
			throw DOpEException( "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(), "StatReducedProblem::ComputeReducedCostFunctional");
		}

		if (dopedim == dealdim) {
			this->GetIntegrator().DeleteDomainData("control");

		} else {
			throw DOpEException("dopedim not implemented", "StatReducedProblem::ComputeReducedCostFunctional");
		}
		if (cost_needs_precomputations_ != 0) {
			this->GetIntegrator().DeleteParamData("cost_functional_pre");
		}
		this->GetIntegrator().DeleteDomainData("state");
		this->GetIntegrator().DeleteDomainData("eigenvalue");

		DeleteUDD();

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

		std::cout << "ComputeReducedFunctionals" << std::endl;

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim, int dealdim> template<class DWRC,
		class PDE> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeRefinementIndicators( const ControlVector<VECTOR> &q, DWRC &dwrc, PDE &pde) {
			std::cout << "ComputeRefinementIndicators" << std::endl;

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedHessianVector( const ControlVector<VECTOR> &/*q*/, const ControlVector<VECTOR> &/*direction*/, ControlVector<VECTOR> &/*hessian_direction*/, ControlVector<VECTOR> &/*hessian_direction_transposed*/) {

		std::cout << "ComputeReducedHessianVector" << std::endl;

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>::ComputeReducedGradientOfGlobalConstraints(unsigned int /*num*/, const ControlVector<VECTOR> &/*q*/, const ConstraintVector<VECTOR> &/*g*/, ControlVector<VECTOR> &/*gradient*/, ControlVector<VECTOR> &/*gradient_transposed*/) {
			std::cout << "ComputeReducedGradientOfGlobalConstraints" << std::endl;

		}

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim,
		dealdim>:: AllocateAuxiliaryParams(std::string name, unsigned int n_components) {
			std::cout << "AllocateAuxiliaryParams" << std::endl;

		std::map<std::string,
		dealii::Vector<double> >::iterator func_vals = auxiliary_params_.find(name);
		if (func_vals != auxiliary_params_.end()) {
			assert(func_vals->second.size() == n_components);
		//already created. Nothing to do
		} else {
			auto ret = auxiliary_params_.emplace(name,dealii::Vector<double>(n_components));
			if (ret.second == false) {
				throw DOpEException("Creation of Storage for Auxiliary time params with name "+name+" failed!", "StatReducedProblem::AllocateAuxiliaryParams"); } } }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim, int dealdim> std::map<std::string,
		dealii::Vector<double> >::iterator EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,
		dopedim, dealdim>:: GetAuxiliaryParams(std::string name)

		{ std::cout << "GetAuxiliaryParams" << std::endl;

		return auxiliary_params_.find(name); }

		/******************************************************/

		template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
		typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
		typename VECTOR, int dopedim,
		int dealdim> void EigenvalueProblem<CONTROLNONLINEARSOLVER,
		NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
CalculatePreFunctional(std::string /*name*/,
		std::string /*postfix*/,
		unsigned int /*n_pre*/,
		unsigned int /*prob_num*/)
{
	std::cout << "CalculatePreFunctional" << std::endl;

}
//////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
