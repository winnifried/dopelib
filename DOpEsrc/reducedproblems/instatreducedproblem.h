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

#ifndef INSTAT_REDUCED_PROBLEM_H_
#define INSTAT_REDUCED_PROBLEM_H_

#include <interfaces/reducedprobleminterface.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <include/statevector.h>
#include <include/solutionextractor.h>
#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/dirichletdatainterface.h>
#include <include/dopeexception.h>
#include <templates/instat_step_newtonsolver.h>
#include <templates/fractional_step_theta_step_newtonsolver.h>
#include <templates/newtonsolvermixeddims.h>
//#include <templates/integratormixeddims.h>
#include <templates/cglinearsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/voidlinearsolver.h>
#include <interfaces/constraintinterface.h>
#include <include/helper.h>
#include <container/dwrdatacontainer.h>

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

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <string>

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
     * @param idc       An INTETGRATORDATACONT which has all the data needed by the integrator.
    * @param base_priority     An offset for the priority of the output written to
    *                          the OutputHandler
    */
    template<typename INTEGRATORDATACONT>
    InstatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                         ParameterReader &param_reader,
                         INTEGRATORDATACONT &idc,
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
                         ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc,
                         STATEINTEGRATORDATACONT &s_idc, int base_priority = 0);

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
    bool ComputeReducedConstraints(const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
      */
    void GetControlBoxConstraints(ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub);


    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedGradient(const ControlVector<VECTOR> &q, ControlVector<VECTOR> &gradient,
                                ControlVector<VECTOR> &gradient_transposed);

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
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedHessianVector(const ControlVector<VECTOR> &q, const ControlVector<VECTOR> &direction,
                                     ControlVector<VECTOR> &hessian_direction,
                                     ControlVector<VECTOR> &hessian_direction_transposed);

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
    template<class DWRC, class PDE>
    void
    ComputeRefinementIndicators(const ControlVector<VECTOR> & /*q*/,
                                DWRC & /*dwrc*/, PDE & /*pde*/)
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
    void StateSizeInfo(std::stringstream &out)
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
     *  @param dof_type    Has the DoF type: state or control.
     */
    void WriteToFile(const ControlVector<VECTOR> &v, std::string name, std::string dof_type);

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
    const StateVector<VECTOR> &GetU() const
    {
      return u_;
    }
    StateVector<VECTOR> &GetU()
    {
      return u_;
    }
    StateVector<VECTOR> &GetZ()
    {
      return z_;
    }
    StateVector<VECTOR> &GetDU()
    {
      return du_;
    }
    StateVector<VECTOR> &GetDZ()
    {
      return dz_;
    }

    NONLINEARSOLVER &GetNonlinearSolver(std::string type);
    CONTROLNONLINEARSOLVER &GetControlNonlinearSolver();
    INTEGRATOR &GetIntegrator()
    {
      return integrator_;
    }
    CONTROLINTEGRATOR &GetControlIntegrator()
    {
      return control_integrator_;
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
    void ComputeReducedState(const ControlVector<VECTOR> &q);

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
    void ComputeReducedAdjoint(const ControlVector<VECTOR> &q, ControlVector<VECTOR> &temp_q, ControlVector<VECTOR> &temp_q_trans);

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
    void ForwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, std::string outname, bool eval_funcs);

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
    void BackwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, ControlVector<VECTOR> &temp_q, ControlVector<VECTOR> &temp_q_trans, std::string outname, bool eval_grads);

  private:
    /**
     * This function is used to allocate space for auxiliary time-dependent parameters.
     *
     * @param name         The name under wich the params are stored.
     * @param n_steps      How many time-points are required.
     * @param n_components The number of components needed in the paramerter vector
     *                     at each time-point.
     **/
    void AllocateAuxiliaryTimeParams(std::string name,
                                     unsigned int n_steps,
                                     unsigned int n_components);

    std::map<std::string,std::vector<dealii::Vector<double> >>::iterator
                                                            GetAuxiliaryTimeParams(std::string name);

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
     * @param step        The current time-point number
     * @param n_pre       Number of pre-iteration cycles
     * @param prob_num    The number of the functional (only relevant for aux_functionals)
     *
     * After finishing the problem type is reset to the value of the `name` param
     **/
    void CalculatePreFunctional(std::string name,
                                std::string postfix,
                                unsigned int step,
                                unsigned int n_prem,
                                unsigned int prob_num);

    StateVector<VECTOR> u_;
    StateVector<VECTOR> z_;
    StateVector<VECTOR> du_;
    StateVector<VECTOR> dz_;

    std::map<std::string,std::vector<dealii::Vector<double> >> auxiliary_time_params_;

    INTEGRATOR integrator_;
    CONTROLINTEGRATOR control_integrator_;
    NONLINEARSOLVER nonlinear_state_solver_;
    NONLINEARSOLVER nonlinear_adjoint_solver_;
    CONTROLNONLINEARSOLVER nonlinear_gradient_solver_;

    bool build_state_matrix_, build_adjoint_matrix_, build_control_matrix_;
    bool state_reinit_, adjoint_reinit_, gradient_reinit_;

    bool project_initial_data_;
    unsigned int cost_needs_precomputations_;

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
                         INTEGRATORDATACONT &idc,
                         int base_priority) :
                         ReducedProblemInterface<PROBLEM, VECTOR> (OP,
                             base_priority),
                         u_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         z_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         du_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         dz_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         integrator_(idc),
                         control_integrator_(idc),
                         nonlinear_state_solver_(integrator_, param_reader),
                         nonlinear_adjoint_solver_(integrator_, param_reader),
                         nonlinear_gradient_solver_(control_integrator_, param_reader)
  {
    // Solvers should be ReInited
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
      gradient_reinit_ = true;
    }
    cost_needs_precomputations_=0;
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
                         CONTROLINTEGRATORCONT &c_idc,
                         STATEINTEGRATORDATACONT &s_idc,
                         int base_priority) :
                         ReducedProblemInterface<PROBLEM, VECTOR> (OP,
                             base_priority),
                         u_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         z_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         du_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         dz_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                         integrator_(s_idc),
                         control_integrator_(c_idc),
                         nonlinear_state_solver_(integrator_, param_reader),
                         nonlinear_adjoint_solver_(integrator_, param_reader),
                         nonlinear_gradient_solver_(control_integrator_, param_reader)
  {
    //Solvers should be ReInited
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
      gradient_reinit_ = true;
    }
    cost_needs_precomputations_=0;
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
  NONLINEARSOLVER &InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR,
                  INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetNonlinearSolver(std::string type)
  {
    if ((type == "state") || (type == "tangent"))
      {
        return nonlinear_state_solver_;
      }
    else if ((type == "adjoint") || (type == "adjoint_hessian"))
      {
        return nonlinear_adjoint_solver_;
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
  CONTROLNONLINEARSOLVER &InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                         CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
  {
    if ((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() == "hessian"))
      {
        return nonlinear_gradient_solver_;
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

    // Some Solvers must be reinited when called
    // Better have subproblems, so that solver can be reinited here
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
      gradient_reinit_ = true;
    }

    build_state_matrix_ = true;
    build_adjoint_matrix_ = true;

    GetU().ReInit();
    GetZ().ReInit();
    GetDU().ReInit();
    GetDZ().ReInit();

    // Remove all time-params - they are now obsolete
    auxiliary_time_params_.clear();

    cost_needs_precomputations_=0;

    build_control_matrix_ = true;
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
  typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
  int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
       PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedState(const ControlVector<VECTOR> &q)
  {
    this->InitializeFunctionalValues(this->GetProblem()->GetNFunctionals() + 1);

    this->GetOutputHandler()->Write("Computing State Solution:", 4 + this->GetBasePriority());

    this->SetProblemType("state");
    auto &problem = this->GetProblem()->GetStateProblem();

    if (state_reinit_ == true)
      {
        GetNonlinearSolver("state").ReInit(problem);
        state_reinit_ = false;
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
         const ControlVector<VECTOR> & /*q*/,
         ConstraintVector<VECTOR> & /*g*/)
  {
    abort();
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
  typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
  int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
       PROBLEM, VECTOR, dopedim, dealdim>::GetControlBoxConstraints(ControlVector<VECTOR> & /*lb*/, ControlVector<VECTOR> & /*ub*/)
  {
    abort();
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
  typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
  int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
       PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedAdjoint(
         const ControlVector<VECTOR> &q, ControlVector<VECTOR> &temp_q, ControlVector<VECTOR> &temp_q_trans)
  {
    this->GetOutputHandler()->Write("Computing Adjoint Solution:", 4 + this->GetBasePriority());

    this->SetProblemType("adjoint");
    auto &problem = this->GetProblem()->GetAdjointProblem();
    if (adjoint_reinit_ == true)
      {
        GetNonlinearSolver("adjoint").ReInit(problem);
        adjoint_reinit_ = false;
      }

    this->GetProblem()->AddAuxiliaryState(&(this->GetU()),"state");
    this->GetProblem()->AddAuxiliaryControl(&q,"control");
    this->BackwardTimeLoop(problem,this->GetZ(),temp_q,temp_q_trans,"Adjoint",true);
    this->GetProblem()->DeleteAuxiliaryControl("control");
    this->GetProblem()->DeleteAuxiliaryState("state");
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
  typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
  int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
       PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradient(
         const ControlVector<VECTOR> &q,
         ControlVector<VECTOR> &gradient,
         ControlVector<VECTOR> &gradient_transposed)
  {
    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() != DOpEtypes::ControlType::initial)
      {
        gradient = 0.;
      }
    this->ComputeReducedAdjoint(q,gradient,gradient_transposed);

    this->GetOutputHandler()->Write("Computing Reduced Gradient:",
                                    4 + this->GetBasePriority());
    if (this->GetProblem()->HasControlInDirichletData())
      {
        throw DOpEException("Control in Dirichlet Data for instationary problems not yet implemented!"
                            ,"InstatReducedProblem::ComputeReducedGradient");
      }

    this->SetProblemType("gradient");
    if (gradient_reinit_ == true)
      {
        GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
        gradient_reinit_ = false;
      }

    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
      {

        //Set time to initial
        const std::vector<double> times =
          this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
        const unsigned int
        n_dofs_per_interval =
          this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
        std::vector<unsigned int> local_to_global(n_dofs_per_interval);

        VECTOR *tmp;
        tmp = NULL;
        {
          if (this->GetProblem()->FunctionalNeedFinalValue())
            {
              TimeIterator it_last =
                this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval();
              it_last.get_time_dof_indices(local_to_global);
              GetU().SetTimeDoFNumber(local_to_global[n_dofs_per_interval-1], it_last);
              tmp = new VECTOR;
              *tmp = GetU().GetSpacialVector();
            }

          TimeIterator it =
            this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
          it.get_time_dof_indices(local_to_global);
          this->GetProblem()->SetTime(times[local_to_global[0]],local_to_global[0], it,true);
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

        if (tmp != NULL)
          {
            this->GetControlIntegrator().AddDomainData("state_final",tmp);
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
              *(this->GetProblem()), gradient.GetSpacialVector());
            this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
          }
        else if (dopedim == 0)
          {
            this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                      &(gradient_transposed.GetSpacialVectorCopy()));
            this->GetControlIntegrator().ComputeNonlinearResidual(
              *(this->GetProblem()), gradient.GetSpacialVector());

            this->GetControlIntegrator().DeleteParamData("last_newton_solution");
            gradient_transposed.UnLockCopy();
          }

        gradient *= -1.;
        gradient_transposed = gradient;

        //Compute l^2 representation of the Gradient

        build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve(
                                  *(this->GetProblem()), gradient_transposed.GetSpacialVector(), true,
                                  build_control_matrix_);
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

        if (tmp != NULL)
          {
            this->GetControlIntegrator().DeleteDomainData("state_final");
            delete tmp;
          }
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetControlIntegrator());

        this->GetOutputHandler()->Write(gradient,
                                        "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
        this->GetOutputHandler()->Write(gradient_transposed,
                                        "Gradient_Transposed" + this->GetPostIndex(),
                                        this->GetProblem()->GetDoFType());
      }//End initial
    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
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
          this->GetProblem()->SetTime(times[local_to_global[0]],local_to_global[0], it,true);
          GetU().SetTimeDoFNumber(local_to_global[0], it);
          GetZ().SetTimeDoFNumber(local_to_global[0], it);
          q.SetTimeDoFNumber(local_to_global[0]);
          gradient.SetTimeDoFNumber(local_to_global[0]);
          gradient_transposed.SetTimeDoFNumber(local_to_global[0]);
        }
        // Duplicate possibly already computed values
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
              *(this->GetProblem()), gradient.GetSpacialVector());
            this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
          }
        else if (dopedim == 0)
          {
            this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                      &(gradient_transposed.GetSpacialVectorCopy()));
            this->GetControlIntegrator().ComputeNonlinearResidual(
              *(this->GetProblem()), gradient.GetSpacialVector());

            this->GetControlIntegrator().DeleteParamData("last_newton_solution");
            gradient_transposed.UnLockCopy();
          }

        gradient *= -1.;
        gradient_transposed = gradient;

        //Compute l^2 representation of the Gradient

        build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve(
                                  *(this->GetProblem()), gradient_transposed.GetSpacialVector(), true,
                                  build_control_matrix_);
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
    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::nonstationary)
      {
        //Nothing to do, all in the adjoint calculation
      }
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
           const ControlVector<VECTOR> &q)
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
         const ControlVector<VECTOR> & /*q*/)
  {
    //We dont need q as the values are precomputed during Solve State...
    this->GetOutputHandler()->Write("Computing Functionals:", 4  + this->GetBasePriority());

    for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
      {
        this->SetProblemType("aux_functional", i);
        if (this->GetProblem()->GetFunctionalType().find("timelocal") != std::string::npos)
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
        else if (this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
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
         const ControlVector<VECTOR> &q,
         const ControlVector<VECTOR> &direction,
         ControlVector<VECTOR> &hessian_direction,
         ControlVector<VECTOR> &hessian_direction_transposed)
  {
    this->GetOutputHandler()->Write("Computing ReducedHessianVector:",
                                    4 + this->GetBasePriority());
    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() != DOpEtypes::ControlType::initial)
      {
        hessian_direction = 0.;
      }
    //Solving the Tangent Problem
    {
      this->GetOutputHandler()->Write("\tSolving Tangent:",
                                      5 + this->GetBasePriority());
      this->SetProblemType("tangent");
      auto &problem = this->GetProblem()->GetTangentProblem();

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
      auto &problem = this->GetProblem()->GetAdjointHessianProblem();

      this->GetProblem()->AddAuxiliaryState(&(this->GetDU()),"tangent");

      this->BackwardTimeLoop(problem,this->GetDZ(),hessian_direction,hessian_direction_transposed,"Hessian",true);
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

      if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
        {
          //Set time to initial
          const std::vector<double> times =
            this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
          const unsigned int
          n_dofs_per_interval =
            this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
          std::vector<unsigned int> local_to_global(n_dofs_per_interval);

          VECTOR *tmp_u;
          tmp_u = NULL;
          VECTOR *tmp_du;
          tmp_du = NULL;
          {
            if (this->GetProblem()->FunctionalNeedFinalValue())
              {
                TimeIterator it_last =
                  this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval();
                it_last.get_time_dof_indices(local_to_global);
                GetU().SetTimeDoFNumber(local_to_global[n_dofs_per_interval-1], it_last);
                tmp_u = new VECTOR;
                *tmp_u = GetU().GetSpacialVector();
                GetDU().SetTimeDoFNumber(local_to_global[n_dofs_per_interval-1], it_last);
                tmp_du = new VECTOR;
                *tmp_du = GetDU().GetSpacialVector();
              }

            TimeIterator it =
              this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
            it.get_time_dof_indices(local_to_global);
            this->GetProblem()->SetTime(times[local_to_global[0]],local_to_global[0], it,true);
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
          if (tmp_u != NULL)
            {
              this->GetControlIntegrator().AddDomainData("state_final",tmp_u);
              assert(tmp_du != NULL);
            }
          if (tmp_du != NULL)
            {
              this->GetControlIntegrator().AddDomainData("tangent_final",tmp_du);
              assert(tmp_u != NULL);
            }

          hessian_direction_transposed = 0.;
          if (dopedim == dealdim)
            {
              this->GetControlIntegrator().AddDomainData("last_newton_solution",
                                                         &(hessian_direction_transposed.GetSpacialVector()));
              this->GetControlIntegrator().ComputeNonlinearResidual(
                *(this->GetProblem()), hessian_direction.GetSpacialVector());
              this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
            }
          else if (dopedim == 0)
            {
              this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                        &(hessian_direction_transposed.GetSpacialVectorCopy()));
              this->GetControlIntegrator().ComputeNonlinearResidual(
                *(this->GetProblem()), hessian_direction.GetSpacialVector());
              this->GetControlIntegrator().DeleteParamData("last_newton_solution");
              hessian_direction_transposed.UnLockCopy();
            }

          hessian_direction *= -1.;
          hessian_direction_transposed = hessian_direction;
          //Compute l^2 representation of the HessianVector
          //hessian Matrix is the same as control matrix
          build_control_matrix_ =
            this->GetControlNonlinearSolver().NonlinearSolve(
              *(this->GetProblem()),
              hessian_direction_transposed.GetSpacialVector(), true,
              build_control_matrix_);

          this->GetOutputHandler()->Write(hessian_direction,
                                          "HessianDirection" + this->GetPostIndex(),
                                          this->GetProblem()->GetDoFType());
          this->GetOutputHandler()->Write(hessian_direction_transposed,
                                          "HessianDirection_Transposed" + this->GetPostIndex(),
                                          this->GetProblem()->GetDoFType());

          this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetControlIntegrator());
          if (tmp_u != NULL)
            {
              this->GetControlIntegrator().DeleteDomainData("state_final");
              delete tmp_u;
            }
          if (tmp_du != NULL)
            {
              this->GetControlIntegrator().DeleteDomainData("tangent_final");
              delete tmp_du;
            }
        }//Endof the case of control in the initial values
      else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
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
            this->GetProblem()->SetTime(times[local_to_global[0]],local_to_global[0], it,true);
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
                *(this->GetProblem()), hessian_direction.GetSpacialVector());
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
                *(this->GetProblem()), hessian_direction.GetSpacialVector());
              this->GetControlIntegrator().DeleteParamData("last_newton_solution");
              hessian_direction_transposed.UnLockCopy();
              this->GetControlIntegrator().DeleteParamData("fixed_rhs");
              tmp.UnLockCopy();
            }

          hessian_direction *= -1.;
          hessian_direction_transposed = hessian_direction;
          //Compute l^2 representation of the HessianVector
          //hessian Matrix is the same as control matrix
          build_control_matrix_ =
            this->GetControlNonlinearSolver().NonlinearSolve(
              *(this->GetProblem()),
              hessian_direction_transposed.GetSpacialVector(), true,
              build_control_matrix_);

          this->GetOutputHandler()->Write(hessian_direction,
                                          "HessianDirection" + this->GetPostIndex(),
                                          this->GetProblem()->GetDoFType());
          this->GetOutputHandler()->Write(hessian_direction_transposed,
                                          "HessianDirection_Transposed" + this->GetPostIndex(),
                                          this->GetProblem()->GetDoFType());

          this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetControlIntegrator());
        }//Endof stationary
      else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::nonstationary)
        {
          //Nothing to do
        }
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
    std::stringstream out;
    this->GetOutputHandler()->InitOut(out);
    out << "\t         Precalculating functional values ";
    this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

    this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector()));
    {
      //CostFunctional
      this->SetProblemType("cost_functional");
      cost_needs_precomputations_ = this->GetProblem()->FunctionalNeedPrecomputations();
      if (cost_needs_precomputations_ != 0)
        {
          unsigned int n_pre = cost_needs_precomputations_;
          AllocateAuxiliaryTimeParams("cost_functional_pre",num_steps,n_pre);
          CalculatePreFunctional("cost_functional","_pre",step,n_pre,0);
        }
      //End of Precomputations

      double ret = 0;
      bool found = false;

      if (this->GetProblem()->NeedTimeFunctional())
        {
          if (cost_needs_precomputations_ != 0)
            {
              auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
              this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[step]));
            }

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
          if (this->GetProblem()->GetFunctionalType().find("algebraic")
              != std::string::npos)
            {
              found = true;
              ret += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
            }

          if (!found)
            {
              throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                                  "InstatReducedProblem::ComputeTimeFunctionals");
            }
          if (cost_needs_precomputations_ != 0)
            {
              this->GetIntegrator().DeleteParamData("cost_functional_pre");
            }
          //Check if selection is feasible!
          if (this->GetProblem()->GetFunctionalType().find("timelocal") != std::string::npos
              && this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
            {
              throw DOpEException("A functional may not simultaneously be `timelocal' and `timedistributed'",
                                  "InstatReducedProblem::ComputeTimeFunctionals");
            }
          // Save the value
          if (this->GetProblem()->GetFunctionalType().find("timelocal") != std::string::npos)
            {
              if (this->GetFunctionalValues()[0].size() != 1)
                {
                  this->GetFunctionalValues()[0].resize(1);
                  this->GetFunctionalValues()[0][0] = 0.;
                }
              this->GetFunctionalValues()[0][0] += ret;
            }
          else if (this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
            {
              //TODO double-check! Possibly we need to include time integration later here?!
              if (this->GetFunctionalValues()[0].size() != 1)
                {
                  this->GetFunctionalValues()[0].resize(1);
                  this->GetFunctionalValues()[0][0] = 0.;
                }
              double w = 0.;
              if ((step == 0))
                {
                  w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                             - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
                }
              else if (step == num_steps)
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
    {
      //Aux Functionals
      double ret = 0;
      bool found = false;
      for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
        {
          ret = 0;
          found = false;
          this->SetProblemType("aux_functional", i);
          if (this->GetProblem()->FunctionalNeedPrecomputations() != 0)
            {
              std::stringstream tmp;
              tmp << "aux_functional_"<<i<<"_pre";
              AllocateAuxiliaryTimeParams(tmp.str(),num_steps,this->GetProblem()->FunctionalNeedPrecomputations());
              CalculatePreFunctional("aux_functional","_pre",step,this->GetProblem()->FunctionalNeedPrecomputations(),i);
            }
          if (this->GetProblem()->NeedTimeFunctional())
            {
              if (this->GetProblem()->FunctionalNeedPrecomputations() != 0)
                {
                  std::stringstream tmp;
                  tmp << "aux_functional_"<<i<<"_pre";
                  auto func_vals = GetAuxiliaryTimeParams(tmp.str());
                  this->GetIntegrator().AddParamData(tmp.str(),&(func_vals->second[step]));
                }
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
              if (this->GetProblem()->GetFunctionalType().find("algebraic")
                  != std::string::npos)
                {
                  found = true;
                  ret += this->GetIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
                }
              if (!found)
                {
                  throw DOpEException(
                    "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                    "InstatReducedProblem::ComputeTimeFunctionals");
                }
              if (this->GetProblem()->FunctionalNeedPrecomputations() != 0)
                {
                  std::stringstream tmp;
                  tmp << "aux_functional_"<<i<<"_pre";
                  this->GetIntegrator().DeleteParamData(tmp.str());
                }
              // Save value
              if (this->GetProblem()->GetFunctionalType().find("timelocal") != std::string::npos)
                {
                  std::stringstream out;
                  this->GetOutputHandler()->InitOut(out);
                  out << "\t" << this->GetProblem()->GetFunctionalName() << ": " << ret;
                  this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());
                  this->GetFunctionalValues()[i + 1].push_back(ret);
                }
              else if (this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
                {
                  if (this->GetFunctionalValues()[i + 1].size() != 1)
                    {
                      this->GetFunctionalValues()[i + 1].resize(1);
                      this->GetFunctionalValues()[i + 1][0] = 0.;
                    }
                  double w = 0.;
                  if ((step == 0))
                    {
                      w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                                 - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
                    }
                  else if (step  == num_steps)
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
        auto &data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
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
        auto &data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
        data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

        data_out.add_data_vector (v,name);
        data_out.build_patches ();

        std::ofstream output(outfile.c_str());

        if (filetype == ".vtk")
          {
            data_out.write_vtk (output);
          }
        else if (filetype == ".gpl")
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
                                                       std::string dof_type)
  {
    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial
        || this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
      {
        v.SetTimeDoFNumber(0);
        this->GetOutputHandler()->Write(v.GetSpacialVector(), name, dof_type);
      }
    else if ( this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::nonstationary)
      {
        unsigned int maxt = this->GetProblem()->GetSpaceTimeHandler()->GetMaxTimePoint();

        for (unsigned int i = 0; i <= maxt; i++)
          {
            this->GetOutputHandler()->SetIterationNumber(i, "Time");
            v.SetTimeDoFNumber(i);
            this->GetOutputHandler()->Write(v.GetSpacialVector(), name, dof_type);
          }
      }
    else
      {
        throw DOpEException("Unknown ControlType: "+DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType()), "InstatReducedProblem::WriteToFile");
      }
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
    const std::vector<double> &t =
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
       ForwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, std::string outname, bool eval_funcs)
  {
    VECTOR u_old;

    unsigned int max_timestep =
      problem.GetSpaceTimeHandler()->GetMaxTimePoint();
    const std::vector<double> times =
      problem.GetSpaceTimeHandler()->GetTimes();
    const unsigned int
    n_dofs_per_interval =
      problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    //Storage for pre-calculated tangent functional values
    if (cost_needs_precomputations_ != 0 && outname == "Tangent")
      {
        unsigned int n_pre = cost_needs_precomputations_;
        AllocateAuxiliaryTimeParams("cost_functional_pre_tangent",max_timestep,n_pre);
      }
    {
      TimeIterator it =
        problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(times[local_to_global[0]], local_to_global[0], it,true);
      sol.SetTimeDoFNumber(local_to_global[0], it);
    }
    // Set u_old to initial_values
    {
      // Compute first all dofs
      const std::vector<unsigned int> &dofs_per_block =
        this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
        {
          n_dofs += dofs_per_block[i];
        }
      // ... then use helper because of templates
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_old);
    }

    // Projection of initial data
    this->GetOutputHandler()->SetIterationNumber(0, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
                                      4 + this->GetBasePriority());

      auto &initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      build_state_matrix_ = this->GetNonlinearSolver("state").NonlinearSolve_Initial(
                              initial_problem, u_old, true, true);
      build_state_matrix_ = true;

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    }
    sol.GetSpacialVector() = u_old;
    this->GetOutputHandler()->Write(u_old, outname + this->GetPostIndex(),
                                    problem.GetDoFType());



    if (eval_funcs)
      {
        // Functional evaluation in t_0
        ComputeTimeFunctionals(0,
                               max_timestep);
        this->SetProblemType("state");
      }
    if (cost_needs_precomputations_ != 0 && outname == "Tangent")
      {
        //Precalculate Tangent functional values
        this->GetIntegrator().AddDomainData("tangent",
                                            &(sol.GetSpacialVector()));
        this->GetIntegrator().AddDomainData("state",&(GetU().GetSpacialVector()));
        unsigned int n_pre = cost_needs_precomputations_;

        CalculatePreFunctional("cost_functional","_pre_tangent",0,n_pre,0);

        this->GetIntegrator().DeleteDomainData("tangent");
        this->GetIntegrator().DeleteDomainData("state");
        this->SetProblemType("tangent");
      } // End precomputation of values

    for (TimeIterator it =
           problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval(); it
         != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().after_last_interval(); ++it)
      {
        it.get_time_dof_indices(local_to_global);
        problem.SetTime(times[local_to_global[0]], local_to_global[0], it);
        sol.SetTimeDoFNumber(local_to_global[0], it);
        //TODO Test again with non-uniform time steps.

        //we start here at i=1 because we assume that the most
        //left DoF in the actual interval is already computed!
        for (unsigned int i = 1; i < n_dofs_per_interval; i++)
          {
            this->GetOutputHandler()->SetIterationNumber(local_to_global[i],
                                                         "Time");
            double time = times[local_to_global[i]];

            std::stringstream out;
            this->GetOutputHandler()->InitOut(out);
            out << "\t Timestep: " << local_to_global[i] << " ("
                << times[local_to_global[i - 1]] << " -> " << time
                << ") using " << problem.GetName();
            problem.GetOutputHandler()->Write(out,
                                              4 + this->GetBasePriority());

            sol.SetTimeDoFNumber(local_to_global[i], it);
            sol.GetSpacialVector() = 0;

            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());

            this->GetNonlinearSolver("state").NonlinearLastTimeEvals(problem,
                                                                     u_old, sol.GetSpacialVector());

            this->GetProblem()->DeleteAuxiliaryFromIntegrator(
              this->GetIntegrator());

            problem.SetTime(time, local_to_global[i], it);
            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());
            this->GetProblem()->AddPreviousAuxiliaryToIntegrator(
              this->GetIntegrator());

            build_state_matrix_
              = this->GetNonlinearSolver("state").NonlinearSolve(problem,
                                                                 u_old, sol.GetSpacialVector(), true,
                                                                 build_state_matrix_);

            this->GetProblem()->DeleteAuxiliaryFromIntegrator(
              this->GetIntegrator());
            this->GetProblem()->DeletePreviousAuxiliaryFromIntegrator(
              this->GetIntegrator());

            //TODO do a transfer to the next grid for changing spatial meshes!
            u_old = sol.GetSpacialVector();
            this->GetOutputHandler()->Write(sol.GetSpacialVector(),
                                            outname + this->GetPostIndex(), problem.GetDoFType());
            if (eval_funcs)
              {
                //Functional evaluation in t_n  //if condition to get the type
                ComputeTimeFunctionals(local_to_global[i], max_timestep);
                this->SetProblemType("state");
              }
            if (cost_needs_precomputations_ != 0 && outname == "Tangent")
              {
                //Precalculate Tangent functional values
                this->GetIntegrator().AddDomainData("tangent",
                                                    &(sol.GetSpacialVector()));
                this->GetIntegrator().AddDomainData("state",&(GetU().GetSpacialVector()));
                unsigned int n_pre = cost_needs_precomputations_;

                CalculatePreFunctional("cost_functional","_pre_tangent",local_to_global[i],n_pre,0);
                this->GetIntegrator().DeleteDomainData("tangent");
                this->GetIntegrator().DeleteDomainData("state");
                this->SetProblemType("tangent");
              } // End precomputation of values
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
       BackwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, ControlVector<VECTOR> &temp_q, ControlVector<VECTOR> &temp_q_trans, std::string outname, bool eval_grads)
  {
    VECTOR u_old;

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
      //The initial values for the adjoint problem
      problem.SetTime(times[local_to_global[local_to_global.size()-1]],local_to_global[local_to_global.size()-1], it);
      sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
    }
    // Set u_old to initial_values
    {
      // Compute total dofs first
      const std::vector<unsigned int> &dofs_per_block =
        this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
        {
          n_dofs += dofs_per_block[i];
        }
      // ... and then get helper (because of templates)
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_old);
    }
    // Projection of initial data
    this->GetOutputHandler()->SetIterationNumber(max_timestep, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
                                      4 + this->GetBasePriority());

      auto &initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
      this->GetProblem()->AddPreviousAuxiliaryToIntegrator(this->GetIntegrator());

      if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
        {
          auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
          this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[max_timestep]));
        }
      if (outname == "Hessian" && cost_needs_precomputations_ != 0)
        {
          {
            auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
            this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[max_timestep]));
          }
          {
            auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre_tangent");
            this->GetIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second[max_timestep]));
          }
        }

      //TODO: Possibly another solver for the initial value than for the pde...
      build_state_matrix_ = this->GetNonlinearSolver("adjoint").NonlinearSolve_Initial(
                              initial_problem, u_old, true, true);
      build_state_matrix_ = true;

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      this->GetProblem()->DeletePreviousAuxiliaryFromIntegrator(this->GetIntegrator());
      if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
        {
          this->GetIntegrator().DeleteParamData("cost_functional_pre");
        }
      if (outname == "Hessian" && cost_needs_precomputations_ != 0)
        {
          this->GetIntegrator().DeleteParamData("cost_functional_pre");
          this->GetIntegrator().DeleteParamData("cost_functional_pre_tangent");
        }

    }
    sol.GetSpacialVector() = u_old;
    this->GetOutputHandler()->Write(u_old, outname + this->GetPostIndex(),
                                    problem.GetDoFType());

    //TODO: Maybe we should calculate the local gradient computations here

    for (TimeIterator it =
           problem.GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval(); it
         != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().before_first_interval(); --it)
      {
        it.get_time_dof_indices(local_to_global);
        problem.SetTime(times[local_to_global[local_to_global.size()-1]],local_to_global[local_to_global.size()-1], it);
        sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
        //TODO Add a test with non-uniform time steps to check whether this is correct.

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
            out << "\t Timestep: " << local_to_global[j+1] << " ("
                << times[local_to_global[j + 1]] << " -> " << time
                << ") using " << problem.GetName();
            problem.GetOutputHandler()->Write(out,
                                              4 + this->GetBasePriority());

            sol.SetTimeDoFNumber(local_to_global[j], it);
            sol.GetSpacialVector() = 0;

            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());
            if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
              {
                auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
                this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[local_to_global[j+1]]));
              }
            if (outname == "Hessian" && cost_needs_precomputations_ != 0)
              {
                {
                  auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
                  this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[local_to_global[j+1]]));
                }
                {
                  auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre_tangent");
                  this->GetIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second[local_to_global[j+1]]));
                }
              }

            this->GetNonlinearSolver("adjoint").NonlinearLastTimeEvals(problem,
                                                                       u_old, sol.GetSpacialVector());

            this->GetProblem()->DeleteAuxiliaryFromIntegrator(
              this->GetIntegrator());
            if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
              {
                this->GetIntegrator().DeleteParamData("cost_functional_pre");
              }
            if (outname == "Hessian" && cost_needs_precomputations_ != 0)
              {
                this->GetIntegrator().DeleteParamData("cost_functional_pre");
                this->GetIntegrator().DeleteParamData("cost_functional_pre_tangent");
              }

            problem.SetTime(time,local_to_global[j], it);
            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());
            this->GetProblem()->AddNextAuxiliaryToIntegrator(
              this->GetIntegrator());
            if (local_to_global[j] != 0)
              this->GetProblem()->AddPreviousAuxiliaryToIntegrator(
                this->GetIntegrator());
            if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
              {
                auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
                this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[local_to_global[j]]));
              }
            if (outname == "Hessian" && cost_needs_precomputations_ != 0)
              {
                {
                  auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre");
                  this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second[local_to_global[j]]));
                }
                {
                  auto func_vals = GetAuxiliaryTimeParams("cost_functional_pre_tangent");
                  this->GetIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second[local_to_global[j]]));
                }
              }

            build_adjoint_matrix_
              = this->GetNonlinearSolver("adjoint").NonlinearSolve(problem,
                                                                   u_old, sol.GetSpacialVector(), true,
                                                                   build_adjoint_matrix_);

            this->GetProblem()->DeleteAuxiliaryFromIntegrator(
              this->GetIntegrator());
            this->GetProblem()->DeleteNextAuxiliaryFromIntegrator(
              this->GetIntegrator());
            if (local_to_global[j] != 0)
              this->GetProblem()->DeletePreviousAuxiliaryFromIntegrator(
                this->GetIntegrator());
            if (outname == "Adjoint" && cost_needs_precomputations_ != 0)
              {
                this->GetIntegrator().DeleteParamData("cost_functional_pre");
              }
            if (outname == "Hessian" && cost_needs_precomputations_ != 0)
              {
                this->GetIntegrator().DeleteParamData("cost_functional_pre");
                this->GetIntegrator().DeleteParamData("cost_functional_pre_tangent");
              }

            //TODO do a transfer to the next grid for changing spatial meshes!
            u_old = sol.GetSpacialVector();
            this->GetOutputHandler()->Write(sol.GetSpacialVector(),
                                            outname + this->GetPostIndex(), problem.GetDoFType());

            //Maybe build local gradient here
            if (eval_grads)
              {
                if (outname == "Adjoint")
                  {
                    this->SetProblemType("gradient");
                    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
                      {
                        //Nothing to do
                      }
                    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
                      {
                        std::stringstream out;
                        this->GetOutputHandler()->InitOut(out);
                        out << "\t         Precalculating gradient values ";
                        this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

                        if (local_to_global[j] != 0)
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
                                this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp);
                                this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
                              }
                            else if (dopedim == 0)
                              {
                                dealii::Vector<double> tmp_2 = temp_q.GetSpacialVectorCopy();
                                tmp = 0.;
                                tmp_2 = 0.;
                                this->GetControlIntegrator().AddParamData("last_newton_solution",&tmp_2);
                                this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp);
                                this->GetControlIntegrator().DeleteParamData("last_newton_solution");
                                temp_q.UnLockCopy();
                              }
                            this->GetControlIntegrator().DeleteDomainData("adjoint");
                            temp_q.GetSpacialVector() -= tmp;
                            this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
                          }
                      }//End of type stationary
                    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::nonstationary)
                      {
                        std::stringstream out;
                        this->GetOutputHandler()->InitOut(out);
                        out << "\t         Precalculating gradient values ";
                        this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

                        //Distributed Problem, calculate local Gradient contributions at each time point
                        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
                        temp_q.SetTimeDoFNumber(local_to_global[j]);
                        temp_q_trans.SetTimeDoFNumber(local_to_global[j]);
                        this->GetControlIntegrator().AddDomainData("adjoint",&(sol.GetSpacialVector()));
                        temp_q_trans.GetSpacialVector() = 0.;

                        if (dopedim == dealdim)
                          {
                            this->GetControlIntegrator().AddDomainData("last_newton_solution",
                                                                       &(temp_q_trans.GetSpacialVector()));
                            this->GetControlIntegrator().ComputeNonlinearResidual(
                              *(this->GetProblem()), temp_q.GetSpacialVector());
                            this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
                          }
                        else if (dopedim == 0)
                          {
                            this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                                      &(temp_q_trans.GetSpacialVectorCopy()));
                            this->GetControlIntegrator().ComputeNonlinearResidual(
                              *(this->GetProblem()), temp_q.GetSpacialVector());

                            this->GetControlIntegrator().DeleteParamData("last_newton_solution");
                            temp_q_trans.UnLockCopy();
                          }
                        temp_q.GetSpacialVector() *= -1.;
                        //Prescale with inverse of time step size to anticipate the time-scalar product.
                        temp_q_trans.GetSpacialVector().equ(1./problem.GetSpaceTimeHandler()->GetStepSize(),temp_q.GetSpacialVector());
                        //Compute l^2 representation of the Gradient

                        build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve(
                                                  *(this->GetProblem()), temp_q_trans.GetSpacialVector(), true,
                                                  build_control_matrix_);

                        this->GetControlIntegrator().DeleteDomainData("adjoint");

                        this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());

                        this->GetOutputHandler()->Write(temp_q.GetSpacialVector(),
                                                        "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
                        this->GetOutputHandler()->Write(temp_q_trans.GetSpacialVector(),
                                                        "Gradient_Transposed" + this->GetPostIndex(),
                                                        this->GetProblem()->GetDoFType());
                      }//End of type nonstationary
                    else
                      {
                        throw DOpEException("Unknown ControlType: "+DOpEtypesToString(this->GetProblem()->GetSpaceTimeHandler()->GetControlType())+". In case Adjoint.", "InstatReducedProblem::BackwardTimeLoop");
                      }
                    this->SetProblemType("adjoint");
                  }//Endof Adjoint case
                else if (outname == "Hessian")
                  {
                    this->SetProblemType("hessian");
                    if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::initial)
                      {
                        //Nothing to do
                      }
                    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::stationary)
                      {
                        std::stringstream out;
                        this->GetOutputHandler()->InitOut(out);
                        out << "\t         Precalculating hessian values ";
                        this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

                        if (local_to_global[j] != 0)
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
                                this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp);
                                this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
                              }
                            else if (dopedim == 0)
                              {
                                dealii::Vector<double> tmp_2 = temp_q.GetSpacialVectorCopy();
                                tmp = 0.;
                                tmp_2 = 0.;
                                this->GetControlIntegrator().AddParamData("last_newton_solution",&tmp_2);
                                this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()), tmp);
                                this->GetControlIntegrator().DeleteParamData("last_newton_solution");
                                temp_q.UnLockCopy();
                              }
                            this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
                            temp_q.GetSpacialVector() -= tmp;
                            this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
                          }
                      }//End stationary
                    else if (this->GetProblem()->GetSpaceTimeHandler()->GetControlType() == DOpEtypes::ControlType::nonstationary)
                      {
                        std::stringstream out;
                        this->GetOutputHandler()->InitOut(out);
                        out << "\t         Precalculating hessian values ";
                        this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

                        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
                        temp_q.SetTimeDoFNumber(local_to_global[j]);
                        temp_q_trans.SetTimeDoFNumber(local_to_global[j]);
                        this->GetControlIntegrator().AddDomainData("adjoint_hessian",&(sol.GetSpacialVector()));
                        temp_q_trans.GetSpacialVector() = 0.;

                        if (dopedim == dealdim)
                          {
                            this->GetControlIntegrator().AddDomainData("last_newton_solution",
                                                                       &(temp_q_trans.GetSpacialVector()));
                            this->GetControlIntegrator().ComputeNonlinearResidual(
                              *(this->GetProblem()), temp_q.GetSpacialVector());
                            this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
                          }
                        else if (dopedim == 0)
                          {
                            this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                                      &(temp_q_trans.GetSpacialVectorCopy()));
                            this->GetControlIntegrator().ComputeNonlinearResidual(
                              *(this->GetProblem()), temp_q.GetSpacialVector());
                            this->GetControlIntegrator().DeleteParamData("last_newton_solution");
                            temp_q_trans.UnLockCopy();
                          }

                        temp_q.GetSpacialVector() *= -1.;
                        //Prescale with inverse of time step size to anticipate the time-scalar product.
                        temp_q_trans.GetSpacialVector().equ(1./problem.GetSpaceTimeHandler()->GetStepSize(),temp_q.GetSpacialVector());

                        //Compute l^2 representation of the HessianVector
                        //hessian Matrix is the same as control matrix
                        build_control_matrix_ =
                          this->GetControlNonlinearSolver().NonlinearSolve(
                            *(this->GetProblem()),
                            temp_q_trans.GetSpacialVector(), true,
                            build_control_matrix_);

                        this->GetOutputHandler()->Write(temp_q.GetSpacialVector(),
                                                        "HessianDirection" + this->GetPostIndex(),
                                                        this->GetProblem()->GetDoFType());
                        this->GetOutputHandler()->Write(temp_q_trans.GetSpacialVector(),
                                                        "HessianDirection_Transposed" + this->GetPostIndex(),
                                                        this->GetProblem()->GetDoFType());

                        this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
                        this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
                      }//End nonstationary
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

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
       CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
       AllocateAuxiliaryTimeParams(std::string name,
                                   unsigned int n_steps,
                                   unsigned int n_components)
  {
    std::map<std::string,std::vector<dealii::Vector<double> >>::iterator func_vals = auxiliary_time_params_.find(name);
    if (func_vals != auxiliary_time_params_.end())
      {
        assert(func_vals->second.size() == n_steps+1);
        //already created. Nothing to do
      }
    else
      {
        auto ret = auxiliary_time_params_.emplace(name,std::vector<dealii::Vector<double> >(n_steps+1,dealii::Vector<double>(n_components)));
        if (ret.second == false)
          {
            throw DOpEException("Creation of Storage for Auxiliary time params with name "+name+" failed!",
                                "InstatReducedProblem::AllocateAuxiliaryTimeParams");
          }
      }
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  std::map<std::string,std::vector<dealii::Vector<double> >>::iterator
                                                          InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                                                                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
                                                                               GetAuxiliaryTimeParams(std::string name)

  {
    return auxiliary_time_params_.find(name);
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
       CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
       CalculatePreFunctional(std::string name,
                              std::string postfix,
                              unsigned int step,
                              unsigned int n_pre,
                              unsigned int prob_num)
  {
    //Checking input
    this->SetProblemType(name,prob_num);
    if (this->GetProblem()->GetFunctionalType().find("timedistributed") == std::string::npos)
      {
        throw DOpEException("Functionals need to be timedistributed to use precomputations",
                            "InstatReducedProblem::CalculatePreFunctional");
      }
    if (name != "aux_functional" && name != "cost_functional")
      {
        throw DOpEException("Only valid with name `aux_functional` or `cost_functional` but not: "+name ,
                            "InstatReducedProblem::CalculatePreFunctional");
      }
    if (postfix == "" || postfix == " ")
      {
        throw DOpEException("Postfix needs to be a non-empty string" ,
                            "InstatReducedProblem::CalculatePreFunctional");
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
    auto func_vals = GetAuxiliaryTimeParams(pname);
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
              "InstatReducedProblem::CalculatePreFunctional");
          }
        //Store Precomputed Values
        func_vals->second[step][i] = pre;
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
////////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
