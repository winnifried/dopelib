/**
*
* Copyright (C) 2012 by the DOpElib authors
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

#ifndef _INSTAT_REDUCED_PROBLEM_H_
#define _INSTAT_REDUCED_PROBLEM_H_

#include "reducedprobleminterface.h"
#include "integrator.h"
#include "parameterreader.h"
#include "statevector.h"
#include "solutionextractor.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "instat_step_newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "newtonsolvermixeddims.h"
//#include "integratormixeddims.h"
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "constraintinterface.h"
#include "helper.h"
#include "dwrdatacontainer.h"

#include <base/data_out_base.h>
#include <numerics/data_out.h>
#include <numerics/matrix_tools.h>
#include <numerics/vector_tools.h>
#include <base/function.h>
#include <lac/sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/sparse_direct.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/vector.h>

#include <fstream>

namespace DOpE
{
/**
 * Basic class to solve time dependent PDE- and optimization problems.
 *
 * @tparam <CONTROLNONLINEARSOLVER>    Newton solver for the control variables.
 * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
 * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions..
 * @tparam <INTEGRATOR>                An integrator for the state variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions..
 * @tparam <PROBLEM>                   PDE- or optimization problem under consideration including ts-scheme.
 * @tparam <VECTOR>                    Class in which we want to store the spatial vector 
 *                                     (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
 * @tparam <dopedim>                   The dimension for the control variable.
 * @tparam <dealdim>                   The dimension for the state variable.
 */
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim, int dealdim>
class InstatReducedProblem: public ReducedProblemInterface<PROBLEM, VECTOR>
{
  public:
        /**
         * Constructor for the InstatPDEProblem.
         *
	 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
         *
         * @param OP                Problem is given to the stationary solver.
         * @param state_behavior    Indicates the behavior of the StateVector.
         * @param param_reader      An object which has run time data.
         * @param idc		    An INTETGRATORDATACONT which has all the data needed by the integrator.
	 * @param base_priority     An offset for the priority of the output written to
	 *                          the OutputHandler
	 */
  template<typename INTEGRATORDATACONT>
    InstatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
			 ParameterReader &param_reader,
			 INTEGRATORDATACONT& idc,
			 int base_priority = 0);
  

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
  template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
    InstatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
			 ParameterReader &param_reader, CONTROLINTEGRATORCONT& c_idc,
			 STATEINTEGRATORDATACONT & s_idc, int base_priority = 0);
  
    virtual ~InstatReducedProblem();

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
    bool ComputeReducedConstraints(const ControlVector<VECTOR>& q, ConstraintVector<VECTOR>& g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
      */
    void GetControlBoxConstraints(ControlVector<VECTOR>& lb, ControlVector<VECTOR>& ub);


    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedGradient(const ControlVector<VECTOR>& q, ControlVector<VECTOR>& gradient,
                                ControlVector<VECTOR>& gradient_transposed);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    double ComputeReducedCostFunctional(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedFunctionals(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedHessianVector(const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& direction,
                                     ControlVector<VECTOR>& hessian_direction,
                                     ControlVector<VECTOR>& hessian_direction_transposed);

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
    template<class DWRC>
      void
      ComputeRefinementIndicators(DWRC&)
    {
      throw DOpEException("ExcNotImplemented",
                          "InstatReducedProblem::ComputeRefinementIndicators");
    }

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void StateSizeInfo(std::stringstream& out)
    {
      GetU().PrintInfos(out);
    }

    /******************************************************/

    /**
     *  Here, the given BlockVector<double> v is printed to a file of *.vtk or *.gpl format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The BlockVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk and *.gpl outputs are possible.
     */
    void WriteToFile(const VECTOR &v, std::string name, std::string outfile,
                     std::string dof_type, std::string filetype);

    /******************************************************/

    /**
     *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The Control vector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk or *.gpl outputs are possible.
     */
    void WriteToFile(const ControlVector<VECTOR> &v, std::string name, std::string outfile,
                     std::string dof_type, std::string filetype);

    /******************************************************/

    /**
     *  A std::vector v is printed to a text file.
     *  Note that this assumes that the vector is one entry per time step.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     */
    void WriteToFile(const std::vector<double> &v, std::string outfile);

  protected:
    const StateVector<VECTOR> & GetU() const
    {
      return _u;
    }
    StateVector<VECTOR> & GetU()
    {
      return _u;
    }
    StateVector<VECTOR> & GetZ()
    {
      return _z;
    }
    StateVector<VECTOR> & GetDU()
    {
      return _du;
    }
    StateVector<VECTOR> & GetDZ()
    {
      return _dz;
    }

    NONLINEARSOLVER& GetNonlinearSolver(std::string type);
    CONTROLNONLINEARSOLVER& GetControlNonlinearSolver();
    INTEGRATOR& GetIntegrator()
    {
      return _integrator;
    }
    CONTROLINTEGRATOR& GetControlIntegrator()
    {
      return _control_integrator;
    }

    /******************************************************/

    /**
     * This function computes functionals of interest within
     * a time dependent computation. For instance, drag- and lift values
     * can be computed, as well as deflections, stresses, etc.
     *
     * @param step         The actual time step.
     * @param num_steps    The total number of time steps.
     */
    void ComputeTimeFunctionals(unsigned int step, unsigned int num_steps);
    /**
     * This function is running the time dependent problem for the state variable.
     * There is a loop over all time steps, and in each time step
     * the nonlinear solver is called. The nonlinear solver is even
     * called for linear problems where the solution is computed within one iteration step.
     *
     * @param q            The control vector is given to this function.
     */
    void ComputeReducedState(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function computes the adjoint, i.e., the Lagrange 
     * multiplier to constraint given by the state equation.
     * It is assumed that the state u(q) corresponding to 
     * the argument q is already calculated.
     *
     *
     * @param q            The control vector is given to this function.
     * @param temp_q       A storage vector to hold precomputed values for the gradient 
     *                     of the cost functional.
     */
    void ComputeReducedAdjoint(const ControlVector<VECTOR>& q, ControlVector<VECTOR>& temp_q);

    /******************************************************/

    /**
     * This function does the loop over time.
     * 
     * @param problem      Describes the nonstationary pde to be solved
     * @param q            The given control vector
     * @param outname      The name prefix given to the solution vectors
     *                     if they are written to files, e.g., State, Tangent, ...
     * @param eval_funcs   Decide wether to evaluate the functionals or not.
     *                     Should be true for the primal-problem but false
     *                     for auxilliary forward pdes, like the tangent one.
     */
    template<typename PDE>
      void ForwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname, bool eval_funcs);

    /******************************************************/

    /**
     * This function does the loop over time but in direction -t.
     * 
     * @param problem      Describes the nonstationary pde to be solved
     * @param q            The given control vector
     * @param outname      The name prefix given to the solution vectors
     *                     if they are written to files, e.g., Adjoint, Hessian, ...
     * @param eval_grads   Decide wether to evaluate the gradients of the functionals or not.
     *                     Should be true for the adjoint and dual_hessian-problem but false
     *                     for auxilliary backward pdes.
     */
    template<typename PDE>
      void BackwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, ControlVector<VECTOR>& temp_q, std::string outname, bool eval_grads);

  private:

    StateVector<VECTOR> _u;
    StateVector<VECTOR> _z;
    StateVector<VECTOR> _du;
    StateVector<VECTOR> _dz;

    INTEGRATOR _integrator;
    CONTROLINTEGRATOR _control_integrator;
    NONLINEARSOLVER _nonlinear_state_solver;
    NONLINEARSOLVER _nonlinear_adjoint_solver;
    CONTROLNONLINEARSOLVER _nonlinear_gradient_solver;

    bool _build_state_matrix, _build_adjoint_matrix, _build_control_matrix;
    bool _state_reinit, _adjoint_reinit, _gradient_reinit;

    bool _project_initial_data;

    friend class SolutionExtractor<InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,dopedim, dealdim>,   VECTOR > ;
};

/*************************************************************************/
/*****************************IMPLEMENTATION******************************/
/*************************************************************************/
using namespace dealii;

/******************************************************/
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::declare_params(
                                                                        ParameterReader &param_reader)
{
  NONLINEARSOLVER::declare_params(param_reader);
}
/******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
  template<typename INTEGRATORDATACONT>
      InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::InstatReducedProblem(
          PROBLEM *OP,
          DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader,
          INTEGRATORDATACONT& idc,
          int base_priority) :
           ReducedProblemInterface<PROBLEM, VECTOR> (OP,
                base_priority),
            _u(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _z(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _du(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _dz(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _integrator(idc),
            _control_integrator(idc),
            _nonlinear_state_solver(_integrator, param_reader),
            _nonlinear_adjoint_solver(_integrator, param_reader),
            _nonlinear_gradient_solver(_control_integrator, param_reader)
      {
        //Solvers should be ReInited
          {
            _state_reinit = true;
            _adjoint_reinit = true;
            _gradient_reinit = true;
          }
      }

/******************************************************/
  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
  template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
      InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::InstatReducedProblem(
          PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader,
          CONTROLINTEGRATORCONT& c_idc,
          STATEINTEGRATORDATACONT & s_idc,
          int base_priority) :
           ReducedProblemInterface<PROBLEM, VECTOR> (OP,
                base_priority),
            _u(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _z(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _du(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _dz(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _integrator(s_idc),
            _control_integrator(c_idc),
            _nonlinear_state_solver(_integrator, param_reader),
            _nonlinear_adjoint_solver(_integrator, param_reader),
            _nonlinear_gradient_solver(_control_integrator, param_reader)
      {
        //Solvers should be ReInited
          {
            _state_reinit = true;
            _adjoint_reinit = true;
            _gradient_reinit = true;
          }
      }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::~InstatReducedProblem()
{
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
NONLINEARSOLVER& InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR,
    INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetNonlinearSolver(std::string type)
{
  if ((type == "state") || (type == "tangent"))
  {
    return _nonlinear_state_solver;
  }
  else if ((type == "adjoint") || (type == "adjoint_hessian"))
  {
    return _nonlinear_adjoint_solver;
  }
  else
  {
    throw DOpEException("No Solver for Problem type:`" + type + "' found",
                        "InstatReducedProblem::GetNonlinearSolver");

  }
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
CONTROLNONLINEARSOLVER& InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
    CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
{
  if ((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() == "hessian"))
  {
    return _nonlinear_gradient_solver;
  }
  else
  {
    throw DOpEException("No Solver for Problem type:`" + this->GetProblem()->GetType() + "' found",
                        "InstatReducedProblem::GetControlNonlinearSolver");

  }
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ReInit()
{
 ReducedProblemInterface<PROBLEM, VECTOR>::ReInit();

  //Some Solvers must be reinited when called
  // Better have subproblems, so that solver can be reinited here
  {
    _state_reinit = true;
    _adjoint_reinit = true;
    _gradient_reinit = true;
  }

  _build_state_matrix = true;
  _build_adjoint_matrix = true;

  GetU().ReInit();
  GetZ().ReInit();
  GetDU().ReInit();
  GetDZ().ReInit();

  _build_control_matrix = true;
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedState(const ControlVector<VECTOR>& q)
{
  this->InitializeFunctionalValues(this->GetProblem()->GetNFunctionals() + 1);

  this->GetOutputHandler()->Write("Computing State Solution:", 4 + this->GetBasePriority());

  this->SetProblemType("state");
  auto& problem = this->GetProblem()->GetStateProblem();

  if (_state_reinit == true)
  {
    GetNonlinearSolver("state").ReInit(problem);
    _state_reinit = false;
  }

  this->GetProblem()->AddAuxiliaryControl(&q,"control");
  this->ForwardTimeLoop(problem,this->GetU(),"State",true);
  this->GetProblem()->DeleteAuxiliaryControl("control");
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
bool InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(
                          const ControlVector<VECTOR>& /*q*/,
			  ConstraintVector<VECTOR>& /*g*/)
{
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
  PROBLEM, VECTOR, dopedim, dealdim>::GetControlBoxConstraints(ControlVector<VECTOR>& /*lb*/, ControlVector<VECTOR>& /*ub*/)
{
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedAdjoint(
      const ControlVector<VECTOR>& q, ControlVector<VECTOR>& temp_q)
{
  this->GetOutputHandler()->Write("Computing Adjoint Solution:", 4 + this->GetBasePriority());

  this->SetProblemType("adjoint");
  auto& problem = this->GetProblem()->GetAdjointProblem();
  if (_adjoint_reinit == true)
  {
    GetNonlinearSolver("adjoint").ReInit(problem);
    _adjoint_reinit = false;
  }

  this->GetProblem()->AddAuxiliaryState(&(this->GetU()),"state");
  this->GetProblem()->AddAuxiliaryControl(&q,"control");
  this->BackwardTimeLoop(problem,this->GetZ(),temp_q,"Adjoint",true);
  this->GetProblem()->DeleteAuxiliaryControl("control");
  this->GetProblem()->DeleteAuxiliaryState("state");
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradient(
      const ControlVector<VECTOR>& q,
      ControlVector<VECTOR>& gradient,
      ControlVector<VECTOR>& gradient_transposed)
{
  if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() != DOpEtypes::ControlType::initial)
  {
    gradient = 0.;
  }
  this->ComputeReducedAdjoint(q,gradient);

  this->GetOutputHandler()->Write("Computing Reduced Gradient:",
				  4 + this->GetBasePriority());
  if (this->GetProblem()->HasControlInDirichletData())
  {
    throw DOpEException("Control in Dirichlet Data for instationary problems not yet implemented!"
			,"InstatReducedProblem::ComputeReducedGradient");
  }

  this->SetProblemType("gradient");
  if (_gradient_reinit == true)
  {
    GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
    _state_reinit = false;
  }

  if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
  {
    
    //Set time to initial 
    const std::vector<double> times =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      this->GetProblem()->SetTime(times[local_to_global[0]], it,true);
      GetU().SetTimeDoFNumber(local_to_global[0], it);
      GetZ().SetTimeDoFNumber(local_to_global[0], it);
      q.SetTimeDoFNumber(local_to_global[0]);
      gradient.SetTimeDoFNumber(local_to_global[0]);
      gradient_transposed.SetTimeDoFNumber(local_to_global[0]);
    }
    
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
    
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().AddDomainData("control",
						 &(q.GetSpacialVector()));
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().AddParamData("control",
						&(q.GetSpacialVectorCopy()));
    }
    else
    {
      throw DOpEException("dopedim not implemented",
			  "InstatReducedProblem::ComputeReducedGradient");
    }

    this->GetControlIntegrator().AddDomainData("state",
					       &(GetU().GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("adjoint",
					       &(GetZ().GetSpacialVector()));
    gradient_transposed = 0.;
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().AddDomainData("last_newton_solution",
						 &(gradient_transposed.GetSpacialVector()));
      this->GetControlIntegrator().ComputeNonlinearResidual(
	*(this->GetProblem()), gradient.GetSpacialVector(), true);
      this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().AddParamData("last_newton_solution",
						&(gradient_transposed.GetSpacialVectorCopy()));
      this->GetControlIntegrator().ComputeNonlinearResidual(
	*(this->GetProblem()), gradient.GetSpacialVector(), true);
      
      this->GetControlIntegrator().DeleteParamData("last_newton_solution");
      gradient_transposed.UnLockCopy();
    }

    gradient *= -1.;
    gradient_transposed = gradient;
    
    //Compute l^2 representation of the Gradient
    
    _build_control_matrix = this->GetControlNonlinearSolver().NonlinearSolve(
      *(this->GetProblem()), gradient_transposed.GetSpacialVector(), true,
      _build_control_matrix);
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().DeleteDomainData("control");
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().DeleteParamData("control");
      q.UnLockCopy();
    }
    else
    {
      throw DOpEException("dopedim not implemented",
			  "InstatReducedProblem::ComputeReducedGradient");
    }
    this->GetControlIntegrator().DeleteDomainData("state");
    this->GetControlIntegrator().DeleteDomainData("adjoint");
    
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(
      this->GetControlIntegrator());
    
    this->GetOutputHandler()->Write(gradient,
				    "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
    this->GetOutputHandler()->Write(gradient_transposed,
				    "Gradient_Transposed" + this->GetPostIndex(),
				    this->GetProblem()->GetDoFType());
  }//End initial
  else if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
  {
    
    //Set time to initial 
    const std::vector<double> times =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      this->GetProblem()->SetTime(times[local_to_global[0]], it,true);
      GetU().SetTimeDoFNumber(local_to_global[0], it);
      GetZ().SetTimeDoFNumber(local_to_global[0], it);
      q.SetTimeDoFNumber(local_to_global[0]);
      gradient.SetTimeDoFNumber(local_to_global[0]);
      gradient_transposed.SetTimeDoFNumber(local_to_global[0]);
    }
    //Dupliziere ggf. vorberechnete Werte.
    ControlVector<VECTOR> tmp = gradient;
    tmp.GetSpacialVector() = gradient.GetSpacialVector();
    

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
    
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().AddDomainData("control",
						 &(q.GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("fixed_rhs",&tmp.GetSpacialVector());
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().AddParamData("control",
						&(q.GetSpacialVectorCopy()));
      this->GetControlIntegrator().AddParamData("fixed_rhs",
						&(tmp.GetSpacialVectorCopy()));
    }
    else
    {
      throw DOpEException("dopedim not implemented",
			  "InstatReducedProblem::ComputeReducedGradient");
    }

    this->GetControlIntegrator().AddDomainData("state",
					       &(GetU().GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("adjoint",
					       &(GetZ().GetSpacialVector()));
    gradient_transposed = 0.;
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().AddDomainData("last_newton_solution",
						 &(gradient_transposed.GetSpacialVector()));
      this->GetControlIntegrator().ComputeNonlinearResidual(
	*(this->GetProblem()), gradient.GetSpacialVector(), true);
      this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().AddParamData("last_newton_solution",
						&(gradient_transposed.GetSpacialVectorCopy()));
      this->GetControlIntegrator().ComputeNonlinearResidual(
	*(this->GetProblem()), gradient.GetSpacialVector(), true);
      
      this->GetControlIntegrator().DeleteParamData("last_newton_solution");
      gradient_transposed.UnLockCopy();
    }

    gradient *= -1.;
    gradient_transposed = gradient;
    
    //Compute l^2 representation of the Gradient
    
    _build_control_matrix = this->GetControlNonlinearSolver().NonlinearSolve(
      *(this->GetProblem()), gradient_transposed.GetSpacialVector(), true,
      _build_control_matrix);
    if (dopedim == dealdim)
    {
      this->GetControlIntegrator().DeleteDomainData("control");
      this->GetControlIntegrator().DeleteDomainData("fixed_rhs");
    }
    else if (dopedim == 0)
    {
      this->GetControlIntegrator().DeleteParamData("control");
      q.UnLockCopy();
      this->GetControlIntegrator().DeleteParamData("fixed_rhs");
      tmp.UnLockCopy();
    }
    else
    {
      throw DOpEException("dopedim not implemented",
			  "InstatReducedProblem::ComputeReducedGradient");
    }
    this->GetControlIntegrator().DeleteDomainData("state");
    this->GetControlIntegrator().DeleteDomainData("adjoint");
    
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(
      this->GetControlIntegrator());
    
    this->GetOutputHandler()->Write(gradient,
				    "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
    this->GetOutputHandler()->Write(gradient_transposed,
				    "Gradient_Transposed" + this->GetPostIndex(),
				    this->GetProblem()->GetDoFType());
  }//End stationary
  else
  {
    std::stringstream out;
    out << "Unknown ControlType: "<<DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType());
    throw DOpEException(out.str(), "InstatReducedProblem::ComputeReducedGradient");
  }
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
double InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(
      const ControlVector<VECTOR>& q)
{
  this->ComputeReducedState(q);

  if (this->GetFunctionalValues()[0].size() != 1)
  {
    if (this->GetFunctionalValues()[0].size() == 0)
      throw DOpEException(
                          "Apparently the CostFunctional was never evaluated! \n\tCheck if the return value of `NeedTimes' is set correctly.",
                          "InstatReducedProblem::ComputeReducedCostFunctional");
    else
      throw DOpEException(
                          "The CostFunctional has been evaluated too many times! \n\tCheck if the return value of `NeedTimes' is set correctly.",
                          "InstatReducedProblem::ComputeReducedCostFunctional");
  }
  return this->GetFunctionalValues()[0][0];
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(
      const ControlVector<VECTOR>& /*q*/)
{
  //We dont need q as the values are precomputed during Solve State...
  this->GetOutputHandler()->Write("Computing Functionals:" + this->GetBasePriority(), 4);

  for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
  {
    this->SetProblemType("aux_functional", i);
    if (this->GetProblem()->GetFunctionalType().find("timelocal"))
    {
      if (this->GetFunctionalValues()[i + 1].size() == 1)
      {
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i + 1][0];
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }
      else if (this->GetFunctionalValues()[i + 1].size() > 1)
      {
        if (this->GetFunctionalValues()[i + 1].size()
            == this->GetProblem()->GetSpaceTimeHandler()->GetMaxTimePoint() + 1)
        {
          std::stringstream out;
          this->GetOutputHandler()->InitOut(out);
          out << this->GetProblem()->GetFunctionalName() << " too large. Writing to file instead: ";
          this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
          this->GetOutputHandler()->Write(this->GetFunctionalValues()[i + 1],
                                          this->GetProblem()->GetFunctionalName()
                                              + this->GetPostIndex(), "time");
        }
        else
        {
          std::stringstream out;
          this->GetOutputHandler()->InitOut(out);
          out << this->GetProblem()->GetFunctionalName() << ": ";
          for (unsigned int k = 0; k < this->GetFunctionalValues()[i + 1].size(); k++)
            out << this->GetFunctionalValues()[i + 1][k] << " ";
          this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
        }
      }
      else
      {
        throw DOpEException("Functional: " + this->GetProblem()->GetFunctionalType()
            + " was not evaluated ever!", "InstatReducedProblem::ComputeFunctionals");
      }
    }
    else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
    {
      std::stringstream out;
      this->GetOutputHandler()->InitOut(out);
      out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i + 1][0];
      this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
    }
    else
    {
      throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                          "InstatReducedProblem::ComputeFunctionals");
    }
  }
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(
      const ControlVector<VECTOR>& q,
      const ControlVector<VECTOR>& direction,
      ControlVector<VECTOR>& hessian_direction,
      ControlVector<VECTOR>& hessian_direction_transposed)
{
  this->GetOutputHandler()->Write("Computing ReducedHessianVector:",
				  4 + this->GetBasePriority());
  if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() != DOpEtypes::ControlType::initial)
  {
    hessian_direction = 0.;
  }
  //Solving the Tangent Problem
  {
    this->GetOutputHandler()->Write("\tSolving Tangent:",
				  5 + this->GetBasePriority());
    this->SetProblemType("tangent");
    auto& problem = this->GetProblem()->GetTangentProblem();

    this->GetProblem()->AddAuxiliaryControl(&q,"control");
    this->GetProblem()->AddAuxiliaryControl(&direction,"dq");
    this->GetProblem()->AddAuxiliaryState(&(this->GetU()),"state");
    this->GetProblem()->AddAuxiliaryState(&(this->GetZ()),"adjoint");

    this->ForwardTimeLoop(problem,this->GetDU(),"Tangent",false);

  }
  //Solving the Adjoint-Hessian Problem
  {
    this->GetOutputHandler()->Write("\tSolving Adjoint Hessian:",
          5 + this->GetBasePriority());
    this->SetProblemType("adjoint_hessian");
    auto& problem = this->GetProblem()->GetAdjointHessianProblem();
    
    this->GetProblem()->AddAuxiliaryState(&(this->GetDU()),"tangent");
    
    this->BackwardTimeLoop(problem,this->GetDZ(),hessian_direction,"Hessian",true);
  }
  if (this->GetProblem()->HasControlInDirichletData())
  {
    throw DOpEException("Control in Dirichlet Data for instationary problems not yet implemented!"
			,"InstatReducedProblem::ComputeReducedHessianVector");
  }

  //Computing Hessian times Vector Representation
  {
    this->GetOutputHandler()->Write(
      "\tComputing Representation of the Hessian:",
      5 + this->GetBasePriority());
    this->SetProblemType("hessian");
    
    this->GetProblem()->AddAuxiliaryState(&(this->GetDZ()),"adjoint_hessian");
    
    if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
    {
      //Set time to initial 
      const std::vector<double> times =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
      const unsigned int
	n_dofs_per_interval =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
      std::vector<unsigned int> local_to_global(n_dofs_per_interval);
      {
	TimeIterator it =
	  this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
	it.get_time_dof_indices(local_to_global);
	this->GetProblem()->SetTime(times[local_to_global[0]], it,true);
	GetU().SetTimeDoFNumber(local_to_global[0], it);
	GetZ().SetTimeDoFNumber(local_to_global[0], it);
	GetDU().SetTimeDoFNumber(local_to_global[0], it);
	GetDZ().SetTimeDoFNumber(local_to_global[0], it);
	q.SetTimeDoFNumber(local_to_global[0]);
	hessian_direction.SetTimeDoFNumber(local_to_global[0]);
	hessian_direction_transposed.SetTimeDoFNumber(local_to_global[0]);
      }
      
      this->GetProblem()->AddAuxiliaryToIntegrator(
	this->GetControlIntegrator());

      hessian_direction_transposed = 0.;
      if (dopedim == dealdim)
      {
	this->GetControlIntegrator().AddDomainData("last_newton_solution",
						   &(hessian_direction_transposed.GetSpacialVector()));
	this->GetControlIntegrator().ComputeNonlinearResidual(
	  *(this->GetProblem()), hessian_direction.GetSpacialVector(),
	  true);
	this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
      }
      else if (dopedim == 0)
      {
	this->GetControlIntegrator().AddParamData("last_newton_solution",
						  &(hessian_direction_transposed.GetSpacialVectorCopy()));
	this->GetControlIntegrator().ComputeNonlinearResidual(
	  *(this->GetProblem()), hessian_direction.GetSpacialVector(),
	  true);
	this->GetControlIntegrator().DeleteParamData("last_newton_solution");
	hessian_direction_transposed.UnLockCopy();
      }

      hessian_direction *= -1.;
      hessian_direction_transposed = hessian_direction;
      //Compute l^2 representation of the HessianVector
      //hessian Matrix is the same as control matrix
      _build_control_matrix =
	this->GetControlNonlinearSolver().NonlinearSolve(
	  *(this->GetProblem()),
	  hessian_direction_transposed.GetSpacialVector(), true,
	  _build_control_matrix);
      
      this->GetOutputHandler()->Write(hessian_direction,
				      "HessianDirection" + this->GetPostIndex(),
				      this->GetProblem()->GetDoFType());
      this->GetOutputHandler()->Write(hessian_direction_transposed,
				      "HessianDirection_Transposed" + this->GetPostIndex(),
				      this->GetProblem()->GetDoFType());

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	this->GetControlIntegrator());
    }//Endof the case of control in the initial values
    else if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
    {
      //Set time to initial 
      const std::vector<double> times =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
      const unsigned int
	n_dofs_per_interval =
	this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
      std::vector<unsigned int> local_to_global(n_dofs_per_interval);
      {
	TimeIterator it =
	  this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
	it.get_time_dof_indices(local_to_global);
	this->GetProblem()->SetTime(times[local_to_global[0]], it,true);
	GetU().SetTimeDoFNumber(local_to_global[0], it);
	GetZ().SetTimeDoFNumber(local_to_global[0], it);
	GetDU().SetTimeDoFNumber(local_to_global[0], it);
	GetDZ().SetTimeDoFNumber(local_to_global[0], it);
	q.SetTimeDoFNumber(local_to_global[0]);
	hessian_direction.SetTimeDoFNumber(local_to_global[0]);
	hessian_direction_transposed.SetTimeDoFNumber(local_to_global[0]);
      }
      //Dupliziere ggf. vorberechnete Werte.
      ControlVector<VECTOR> tmp = hessian_direction;
      tmp.GetSpacialVector() = hessian_direction.GetSpacialVector();

      this->GetProblem()->AddAuxiliaryToIntegrator(
	this->GetControlIntegrator());

      hessian_direction_transposed = 0.;
      if (dopedim == dealdim)
      {
	this->GetControlIntegrator().AddDomainData("fixed_rhs",&tmp.GetSpacialVector());
	this->GetControlIntegrator().AddDomainData("last_newton_solution",
						   &(hessian_direction_transposed.GetSpacialVector()));
	this->GetControlIntegrator().ComputeNonlinearResidual(
	  *(this->GetProblem()), hessian_direction.GetSpacialVector(),
	  true);
	this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
	this->GetControlIntegrator().DeleteDomainData("fixed_rhs");
      }
      else if (dopedim == 0)
      {
	this->GetControlIntegrator().AddParamData("fixed_rhs",
						&(tmp.GetSpacialVectorCopy()));
	this->GetControlIntegrator().AddParamData("last_newton_solution",
						  &(hessian_direction_transposed.GetSpacialVectorCopy()));
	this->GetControlIntegrator().ComputeNonlinearResidual(
	  *(this->GetProblem()), hessian_direction.GetSpacialVector(),
	  true);
	this->GetControlIntegrator().DeleteParamData("last_newton_solution");
	hessian_direction_transposed.UnLockCopy();  
	this->GetControlIntegrator().DeleteParamData("fixed_rhs");
	tmp.UnLockCopy();
      }

      hessian_direction *= -1.;
      hessian_direction_transposed = hessian_direction;
      //Compute l^2 representation of the HessianVector
      //hessian Matrix is the same as control matrix
      _build_control_matrix =
	this->GetControlNonlinearSolver().NonlinearSolve(
	  *(this->GetProblem()),
	  hessian_direction_transposed.GetSpacialVector(), true,
	  _build_control_matrix);
      
      this->GetOutputHandler()->Write(hessian_direction,
				      "HessianDirection" + this->GetPostIndex(),
				      this->GetProblem()->GetDoFType());
      this->GetOutputHandler()->Write(hessian_direction_transposed,
				      "HessianDirection_Transposed" + this->GetPostIndex(),
				      this->GetProblem()->GetDoFType());

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	this->GetControlIntegrator());
    }//Endof stationary
    else
    {
      std::stringstream out;
      out << "Unknown ControlType: "<<DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType());
      throw DOpEException(out.str(), "InstatReducedProblem::ComputeReducedHessianVector");
    }
  }//End of HessianVector Repr.

  //Cleaning    
  this->GetProblem()->DeleteAuxiliaryControl("control");
  this->GetProblem()->DeleteAuxiliaryControl("dq");
  this->GetProblem()->DeleteAuxiliaryState("state");
  this->GetProblem()->DeleteAuxiliaryState("adjoint");
  this->GetProblem()->DeleteAuxiliaryState("tangent");
  this->GetProblem()->DeleteAuxiliaryState("adjoint_hessian");
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeTimeFunctionals(unsigned int step, unsigned int num_steps)
{

  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

  this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector()));
  double ret = 0;
  bool found = false;
  {//CostFunctional
    this->SetProblemType("cost_functional");
    if (this->GetProblem()->NeedTimeFunctional())
    {
      if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
      }

      if (!found)
      {
        throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                            "InstatReducedProblem::ComputeTimeFunctionals");
      }
      //Wert speichern
      if (this->GetProblem()->GetFunctionalType().find("timelocal"))
      {
        if (this->GetFunctionalValues()[0].size() != 1)
        {
          this->GetFunctionalValues()[0].resize(1);
	  this->GetFunctionalValues()[0][0] = 0.;
        }
	this->GetFunctionalValues()[0][0] += ret;
      }
      else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
      {//TODO was passiert hier? Vermutlich sollte hier spaeter Zeitintegration durchgefuehrt werden?
        if (this->GetFunctionalValues()[0].size() != 1)
        {
          this->GetFunctionalValues()[0].resize(1);
        }
        double w = 0.;
        if ((step == 0))
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
        }
        else if (step + 1 == num_steps)
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
        }
        else
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
          w += 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
        }
        this->GetFunctionalValues()[0][0] += w * ret;
      }
      else
      {
        throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                            "InstatReducedProblem::ComputeTimeFunctionals");
      }
    }
  }
  {//Aux Functionals
    for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
    {
      ret = 0;
      found = false;
      this->SetProblemType("aux_functional", i);
      if (this->GetProblem()->NeedTimeFunctional())
      {
        if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
        }

        if (!found)
        {
          throw DOpEException(
                              "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                              "InstatReducedProblem::ComputeTimeFunctionals");
        }
        //Wert speichern
        if (this->GetProblem()->GetFunctionalType().find("timelocal"))
        {
          std::stringstream out;
          this->GetOutputHandler()->InitOut(out);
          out << "\t" << this->GetProblem()->GetFunctionalName() << ": " << ret;
          this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());
          this->GetFunctionalValues()[i + 1].push_back(ret);
        }
        else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
        {
          if (this->GetFunctionalValues()[i + 1].size() != 1)
          {
            this->GetFunctionalValues()[i + 1].resize(1);
          }
          double w = 0.;
          if ((step == 0))
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
          }
          else if (step + 1 == num_steps)
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
          }
          else
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
            w += 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
          }
          this->GetFunctionalValues()[i + 1][0] += w * ret;
        }
        else
        {
          throw DOpEException(
                              "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                              "InstatReducedProblem::ComputeTimeFunctionals");
        }
      }
    }
  }
  this->GetIntegrator().DeleteDomainData("state");
  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

}

/******************************************************/
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const VECTOR &v, std::string name, std::string outfile, std::string dof_type, std::string filetype)
{
  if (dof_type == "state")
    {
      auto& data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
      data_out.attach_dof_handler(this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

      data_out.add_data_vector(v, name);
      data_out.build_patches();

      std::ofstream output(outfile.c_str());

      if (filetype == ".vtk")
        {
          data_out.write_vtk(output);
        }
      else if (filetype == ".gpl")
        {
          data_out.write_gnuplot(output);
        }
      else
        {
          throw DOpEException("Don't know how to write filetype `" + filetype + "'!",
              "InstatReducedProblem::WriteToFile");
        }
      data_out.clear();
    }
  else if (dof_type == "control")
    {
#if dope_dimension >0
      auto& data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
      data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

      data_out.add_data_vector (v,name);
      data_out.build_patches ();

      std::ofstream output(outfile.c_str());

      if(filetype == ".vtk")
      {
        data_out.write_vtk (output);
      }
      else if(filetype == ".gpl")
      {
        data_out.write_gnuplot (output);
      }
      else
      {
        throw DOpEException("Don't know how to write filetype `" + filetype + "'!","InstatReducedProblem::WriteToFile");
      }
      data_out.clear();
#else
    std::ofstream output(outfile.c_str());
    Vector<double> off;
    off = v;
    for (unsigned int i = 0; i < off.size(); i++)
    {
      output << off(i) << std::endl;
    }
#endif
  }
  else
  {
    throw DOpEException("No such DoFHandler `" + dof_type + "'!",
                        "InstatReducedProblem::WriteToFile");
  }
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const ControlVector<VECTOR> &v,
                                                                     std::string name,
                                                                     std::string outfile,
                                                                     std::string dof_type,
                                                                     std::string filetype)
{
  WriteToFile(v.GetSpacialVector(), name, outfile, dof_type, filetype);
}

/******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
        const std::vector<double> &v, std::string outfile)
    {
      //TODO This should get timedofhandler later on.
      const std::vector<double>& t =
          this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
      std::ofstream out(outfile.c_str());
      assert( t.size() == v.size());
      assert(out.is_open());

      out << "#Time\tvalue" << std::endl;
      for (unsigned int i = 0; i < v.size(); i++)
        {
          out << t[i] << "\t" << v[i] << std::endl;
        }
      out.close();
    }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
    typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
    typename VECTOR, int dopedim, int dealdim>
    template<typename PDE>
    void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
    CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
    ForwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname, bool eval_funcs)
  {
    VECTOR u_alt;

    unsigned int max_timestep =
      problem.GetSpaceTimeHandler()->GetMaxTimePoint();
    const std::vector<double> times =
      problem.GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(times[local_to_global[0]], it,true);
      sol.SetTimeDoFNumber(local_to_global[0], it);
    }
    //u_alt auf initial_values setzen
    {
      //dazu erstmal gesamt-dof berechnen
      const std::vector<unsigned int>& dofs_per_block =
	this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
      {
	n_dofs += dofs_per_block[i];
      }
      //und dann auf den Helper zuerueckgreifen (wegen Templateisierung)
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_alt);
    }
    
    //Projection der Anfangsdaten
    this->GetOutputHandler()->SetIterationNumber(0, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
          4 + this->GetBasePriority());

      auto& initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      _build_state_matrix = this->GetNonlinearSolver("state").NonlinearSolve_Initial(
          initial_problem, u_alt, true, true);
      _build_state_matrix = true;
      
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      
    }
    sol.GetSpacialVector() = u_alt;
    this->GetOutputHandler()->Write(u_alt, outname + this->GetPostIndex(),
          problem.GetDoFType());
    
    
    if(eval_funcs)
    {//Funktional Auswertung in t_0
      ComputeTimeFunctionals(0,
			     max_timestep);
          this->SetProblemType("state");
    }
    
    
    for (TimeIterator it =
	   problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval(); it
	   != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().after_last_interval(); ++it)
    {
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(times[local_to_global[0]], it);
      sol.SetTimeDoFNumber(local_to_global[0], it);
      //TODO Eventuell waere ein Test mit nicht-gleichmaessigen Zeitschritten sinnvoll!
      
      //we start here at i=1 because we assume that the most
      //left DoF in the actual interval is already computed!
      for (unsigned int i = 1; i < n_dofs_per_interval; i++)
      {
	this->GetOutputHandler()->SetIterationNumber(local_to_global[i],
						     "Time");
	double time = times[local_to_global[i]];

	std::stringstream out;
  this->GetOutputHandler()->InitOut(out);
	out << "\t\t Timestep: " << local_to_global[i] << " ("
	    << times[local_to_global[i - 1]] << " -> " << time
	    << ") using " << problem.GetName();
	problem.GetOutputHandler()->Write(out,
					  4 + this->GetBasePriority());
	
	sol.SetTimeDoFNumber(local_to_global[i], it);
	sol.GetSpacialVector() = 0;
	
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
	
	this->GetNonlinearSolver("state").NonlinearLastTimeEvals(problem,
								 u_alt, sol.GetSpacialVector());

	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	problem.SetTime(time, it);
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
	
	_build_state_matrix
	  = this->GetNonlinearSolver("state").NonlinearSolve(problem,
							     u_alt, sol.GetSpacialVector(), true,
							     _build_state_matrix);

	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	//TODO do a transfer to the next grid for changing spatial meshes!
	u_alt = sol.GetSpacialVector();
	this->GetOutputHandler()->Write(sol.GetSpacialVector(),
					outname + this->GetPostIndex(), problem.GetDoFType());
	if(eval_funcs)
	{//Funktional Auswertung in t_n//if abfrage, welcher typ
	  ComputeTimeFunctionals(local_to_global[i], max_timestep);
	  this->SetProblemType("state");
	}
      }
    }
  }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  template<typename PDE>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
  CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
  BackwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, ControlVector<VECTOR>& temp_q, std::string outname, bool eval_grads)
  {
    VECTOR u_alt;

    unsigned int max_timestep =
      problem.GetSpaceTimeHandler()->GetMaxTimePoint();
    const std::vector<double> times =
      problem.GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	problem.GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval();
      it.get_time_dof_indices(local_to_global);
      //The initial values for the dual problem
      problem.SetTime(times[local_to_global[local_to_global.size()-1]], it);
      sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
    }
    //u_alt auf initial_values setzen
    {
      //dazu erstmal gesamt-dof berechnen
      const std::vector<unsigned int>& dofs_per_block =
	this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
      {
	n_dofs += dofs_per_block[i];
      }
      //und dann auf den Helper zuerueckgreifen (wegen Templateisierung)
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_alt);
    }
    //Projection der Anfangsdaten
    this->GetOutputHandler()->SetIterationNumber(max_timestep, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
          4 + this->GetBasePriority());

      auto& initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      _build_state_matrix = this->GetNonlinearSolver("adjoint").NonlinearSolve_Initial(
          initial_problem, u_alt, true, true);
      _build_state_matrix = true;

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      
    }
    sol.GetSpacialVector() = u_alt;
    this->GetOutputHandler()->Write(u_alt, outname + this->GetPostIndex(),
          problem.GetDoFType());
    
    //TODO: Maybe we should calculate the local gradient computations here
    
    for (TimeIterator it =
	   problem.GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval(); it
	   != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().before_first_interval(); --it)
    {
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(times[local_to_global[local_to_global.size()-1]], it);
      sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
     //TODO Eventuell waere ein Test mit nicht-gleichmaessigen Zeitschritten sinnvoll!
      
      //we start here at i= 1 and transform i -> n_dofs_per_interval-1-i because we assume that the most
      //right DoF in the actual interval is already computed!
      for (unsigned int i = 1; i < n_dofs_per_interval; i++)
      {
	unsigned int j = n_dofs_per_interval-1-i;
	this->GetOutputHandler()->SetIterationNumber(local_to_global[j],
						     "Time");
	double time = times[local_to_global[j]];
	
	std::stringstream out;
  this->GetOutputHandler()->InitOut(out);
	out << "\t\t Timestep: " << local_to_global[j+1] << " ("
	    << times[local_to_global[j + 1]] << " -> " << time
	    << ") using " << problem.GetName();
	problem.GetOutputHandler()->Write(out,
					  4 + this->GetBasePriority());
	
	sol.SetTimeDoFNumber(local_to_global[j], it);
	sol.GetSpacialVector() = 0;
	
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
		
	this->GetNonlinearSolver("adjoint").NonlinearLastTimeEvals(problem,
								   u_alt, sol.GetSpacialVector());
	
	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());

	problem.SetTime(time, it);
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
		
	_build_adjoint_matrix
	  = this->GetNonlinearSolver("adjoint").NonlinearSolve(problem,
							     u_alt, sol.GetSpacialVector(), true,
							     _build_adjoint_matrix);
	
	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	//TODO do a transfer to the next grid for changing spatial meshes!
	u_alt = sol.GetSpacialVector();
	this->GetOutputHandler()->Write(sol.GetSpacialVector(),
					outname + this->GetPostIndex(), problem.GetDoFType());

	//Maybe build local gradient here
	if(eval_grads)
	{
	  if(outname == "Adjoint")
	  {
	    this->SetProblemType("gradient");
	    if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
	    {
	      //Nothing to do
	    }
	    else if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
	    {
	      if(local_to_global[j] != 0)
	      {
		//Update Residual
		//Only if not the initial time: Calculations at initial time are 
		//performed in the ComputeReducedGradient function.
		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
		temp_q.SetTimeDoFNumber(local_to_global[j]);
		this->GetControlIntegrator().AddDomainData("adjoint",&(sol.GetSpacialVector()));
		
		VECTOR tmp = temp_q.GetSpacialVector();
		if (dopedim == dealdim)
		{
		  VECTOR tmp_2 = temp_q.GetSpacialVector();
		  tmp = 0.; 
		  tmp_2 = 0.;
		  this->GetControlIntegrator().AddDomainData("last_newton_solution",&tmp_2);
		  this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp, true);
		  this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
		}
		else if (dopedim == 0)
		{	
		  dealii::Vector<double> tmp_2 = temp_q.GetSpacialVectorCopy();
		  tmp = 0.; 
		  tmp_2 = 0.;
		  this->GetControlIntegrator().AddParamData("last_newton_solution",&tmp_2);
		  this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp, true);
		  this->GetControlIntegrator().DeleteParamData("last_newton_solution");
		  temp_q.UnLockCopy();
		}	
		this->GetControlIntegrator().DeleteDomainData("adjoint");
		temp_q.GetSpacialVector() -= tmp;
		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());	      
	      }
	    }
	    else
	    {
	      throw DOpEException("Unknown ControlType: "+DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType())+". In case Adjoint.", "InstatReducedProblem::BackwardTimeLoop");
	    }
	    this->SetProblemType("adjoint");
	  }//Endof Adjoint case
	  else if (outname == "Hessian")
	  {
	    this->SetProblemType("hessian");
	    if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
	    {
	      //Nothing to do
	    }
	    else if(this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
	    {
	      if(local_to_global[j] != 0)
	      {
		//Update Residual
		//Only if not the initial time: Calculations at initial time are 
		//performed in the ComputeReducedHessian function.
		this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
		temp_q.SetTimeDoFNumber(local_to_global[j]);
		this->GetControlIntegrator().AddDomainData("adjoint_hessian",&(sol.GetSpacialVector()));
		
		VECTOR tmp = temp_q.GetSpacialVector();
		if (dopedim == dealdim)
		{
		  VECTOR tmp_2 = temp_q.GetSpacialVector();
		  tmp = 0.; 
		  tmp_2 = 0.;
		  this->GetControlIntegrator().AddDomainData("last_newton_solution",&tmp_2);
		  this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp, true);
		  this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
		}
		else if (dopedim == 0)
		{	
		  dealii::Vector<double> tmp_2 = temp_q.GetSpacialVectorCopy();
		  tmp = 0.; 
		  tmp_2 = 0.;
		  this->GetControlIntegrator().AddParamData("last_newton_solution",&tmp_2);
		  this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp, true);
		  this->GetControlIntegrator().DeleteParamData("last_newton_solution");
		  temp_q.UnLockCopy();
		}	
		this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
		temp_q.GetSpacialVector() -= tmp;
		this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());	      
	      }
	    }
	    else
	    {
	      throw DOpEException("Unknown ControlType: "+DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType())+". In case Hessian.", "InstatReducedProblem::BackwardTimeLoop");
	    }
	    this->SetProblemType("adjoint_hessian");
	  }//Endof Hessian case
	  else
	  {
	    throw DOpEException("Unknown type "+outname,"InstatReducedProblem::BackwardTimeLoop");
	  }
	}
      }//End interval loop
    }//End time loop
  }
////////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
