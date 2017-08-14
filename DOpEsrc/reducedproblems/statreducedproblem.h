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

#ifndef STAT_REDUCED_PROBLEM_H_
#define STAT_REDUCED_PROBLEM_H_

#include <interfaces/reducedprobleminterface.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <include/statevector.h>
#include <problemdata/stateproblem.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <container/optproblemcontainer.h>
#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/dirichletdatainterface.h>
#include <include/dopeexception.h>
#include <templates/newtonsolver.h>
#include <templates/newtonsolvermixeddims.h>
#include <templates/cglinearsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/voidlinearsolver.h>
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

#include <fstream>
namespace DOpE
{
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
           typename VECTOR, int dopedim, int dealdim>
  class StatReducedProblem : public ReducedProblemInterface<PROBLEM, VECTOR>
  {
  public:
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
    template<typename INTEGRATORDATACONT>
    StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                       ParameterReader &param_reader, INTEGRATORDATACONT &idc,
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
    template<typename STATEINTEGRATORDATACONT,
             typename CONTROLINTEGRATORCONT>
    StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                       ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc,
                       STATEINTEGRATORDATACONT &s_idc, int base_priority = 0);

    ~StatReducedProblem();

    /******************************************************/

    /**
     * Static member function for run time parameters.
     *
     * @param param_reader      An object which has run time data.
     */
    static void
    declare_params(ParameterReader &param_reader);

    /******************************************************/

    /**
     * This function sets state- and dual vectors to their correct sizes.
     * Further, the flags to build the system matrices are set to true.
     *
     */
    void
    ReInit();

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    bool
    ComputeReducedConstraints(const ControlVector<VECTOR> &q,
                              ConstraintVector<VECTOR> &g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    void
    GetControlBoxConstraints(ControlVector<VECTOR> &lb,
                             ControlVector<VECTOR> &ub);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    void
    ComputeReducedGradient(const ControlVector<VECTOR> &q,
                           ControlVector<VECTOR> &gradient,
                           ControlVector<VECTOR> &gradient_transposed);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    double
    ComputeReducedCostFunctional(const ControlVector<VECTOR> &q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    void
    ComputeReducedFunctionals(const ControlVector<VECTOR> &q);

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
    template<class DWRC,class PDE>
    void
    ComputeRefinementIndicators(const ControlVector<VECTOR> &q,
                                DWRC &dwrc, PDE &pde);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    void
    ComputeReducedHessianVector(const ControlVector<VECTOR> &q,
                                const ControlVector<VECTOR> &direction,
                                ControlVector<VECTOR> &hessian_direction,
                                ControlVector<VECTOR> &hessian_direction_transposed);

    /******************************************************/

    /**
      * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
      *
      */
    void
    ComputeReducedGradientOfGlobalConstraints(unsigned int num,
                                              const ControlVector<VECTOR> &q, const ConstraintVector<VECTOR> &g,
                                              ControlVector<VECTOR> &gradient,
                                              ControlVector<VECTOR> &gradient_transposed);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
    * ReducedProblemInterface
     *
     */
    void
    StateSizeInfo(std::stringstream &out)
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
     *  @param filetype    The filetype. Actually, *.vtk or *.gpl  outputs are possible.
     */
    void
    WriteToFile(const VECTOR &v, std::string name, std::string outfile,
                std::string dof_type, std::string filetype);

    /******************************************************/

    /**
     *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The ControlVector<VECTOR> to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param dof_type    Has the DoF type: state or control.
     */
    void
    WriteToFile(const ControlVector<VECTOR> &v, std::string name, std::string dof_type);

    /**
     * Basic function to write a std::vector to a file.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     *  Doesn't make sense here so aborts if called!
     */
    void
    WriteToFile(const std::vector<double> &/*v*/,
                std::string /*outfile*/)
    {
      abort();
    }

  protected:
    /**
     * This function computes the solution for the state variable.
     * The nonlinear solver is called, even for
     * linear problems where the solution is computed within one iteration step.
     *
     * @param q            The ControlVector<VECTOR> is given to this function.
     */
    void
    ComputeReducedState(const ControlVector<VECTOR> &q);

    /******************************************************/
    /**
     * This function computes the adjoint, i.e., the Lagrange
    * multiplier to constraint given by the state equation.
    * It is assumed that the state u(q) corresponding to
    * the argument q is already calculated.
     *
     * @param q            The ControlVector<VECTOR> is given to this function.
     */
    void
    ComputeReducedAdjoint(const ControlVector<VECTOR> &q);

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
    void
    ComputeDualForErrorEstimation(const ControlVector<VECTOR> &q,
                                  DOpEtypes::WeightComputation weight_comp);

    const StateVector<VECTOR> &
    GetU() const
    {
      return u_;
    }
    StateVector<VECTOR> &
    GetU()
    {
      return u_;
    }
    StateVector<VECTOR> &
    GetZ()
    {
      return z_;
    }
    StateVector<VECTOR> &
    GetDU()
    {
      return du_;
    }
    StateVector<VECTOR> &
    GetDZ()
    {
      return dz_;
    }
    /**
     * Returns the solution of the dual equation for error estimation.
     */
    const StateVector<VECTOR> &
    GetZForEE() const
    {
      return z_for_ee_;
    }
    StateVector<VECTOR> &
    GetZForEE()
    {
      return z_for_ee_;
    }

    NONLINEARSOLVER &
    GetNonlinearSolver(std::string type);
    CONTROLNONLINEARSOLVER &
    GetControlNonlinearSolver();
    INTEGRATOR &
    GetIntegrator()
    {
      return integrator_;
    }
    CONTROLINTEGRATOR &
    GetControlIntegrator()
    {
      return control_integrator_;
    }

  private:
    /**
     * This function is used to allocate space for auxiliary parameters.
     *
     * @param name         The name under wich the params are stored.
     * @param n_components The number of components needed in the paramerter vector
     *                     at each time-point.
     **/
    void AllocateAuxiliaryParams(std::string name,
                                 unsigned int n_components);

    std::map<std::string,dealii::Vector<double> >::iterator
    GetAuxiliaryParams(std::string name);

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
    void CalculatePreFunctional(std::string name,
                                std::string postfix,
                                unsigned int n_pre,
                                unsigned int prob_num);

    StateVector<VECTOR> u_;
    StateVector<VECTOR> z_;
    StateVector<VECTOR> du_;
    StateVector<VECTOR> dz_;
    StateVector<VECTOR> z_for_ee_;

    std::map<std::string,dealii::Vector<double> > auxiliary_params_;

    INTEGRATOR integrator_;
    CONTROLINTEGRATOR control_integrator_;
    NONLINEARSOLVER nonlinear_state_solver_;
    NONLINEARSOLVER nonlinear_adjoint_solver_;
    CONTROLNONLINEARSOLVER nonlinear_gradient_solver_;

    bool build_state_matrix_, build_adjoint_matrix_, build_control_matrix_;
    bool state_reinit_, adjoint_reinit_, gradient_reinit_;
    unsigned int cost_needs_precomputations_;

    friend class SolutionExtractor<
      StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
      CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>,
      VECTOR> ;
  };

  /*************************************************************************/
  /*****************************IMPLEMENTATION******************************/
  /*************************************************************************/
  using namespace dealii;

  /******************************************************/
  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
           typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::declare_params(
                       ParameterReader &param_reader)
{
    NONLINEARSOLVER::declare_params(param_reader);
  }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  template<typename INTEGRATORDATACONT>
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::StatReducedProblem(
                       PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                       ParameterReader &param_reader, INTEGRATORDATACONT &idc,
                       int base_priority)
                       : ReducedProblemInterface<PROBLEM, VECTOR>(OP,
                           base_priority), u_(OP->GetSpaceTimeHandler(), state_behavior,
                                              param_reader), z_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                param_reader), du_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                    param_reader), dz_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                        param_reader), z_for_ee_(OP->GetSpaceTimeHandler(),
                                                                            state_behavior, param_reader), integrator_(idc), control_integrator_(
                         idc), nonlinear_state_solver_(integrator_, param_reader), nonlinear_adjoint_solver_(
                         integrator_, param_reader), nonlinear_gradient_solver_(
                         control_integrator_, param_reader)

  {
    //ReducedProblems should be ReInited
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
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::StatReducedProblem(
                       PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                       ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc,
                       STATEINTEGRATORDATACONT &s_idc, int base_priority)
                       : ReducedProblemInterface<PROBLEM, VECTOR>(OP,
                           base_priority), u_(OP->GetSpaceTimeHandler(), state_behavior,
                                              param_reader), z_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                param_reader), du_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                    param_reader), dz_(OP->GetSpaceTimeHandler(), state_behavior,
                                                                        param_reader), z_for_ee_(OP->GetSpaceTimeHandler(),
                                                                            state_behavior, param_reader), integrator_(s_idc), control_integrator_(
                         c_idc), nonlinear_state_solver_(integrator_, param_reader), nonlinear_adjoint_solver_(
                         integrator_, param_reader), nonlinear_gradient_solver_(
                         control_integrator_, param_reader)

  {
    //ReducedProblems should be ReInited
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
      gradient_reinit_ = true;
    }
    cost_needs_precomputations_ = 0;
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
           typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dopedim, int dealdim>
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::~StatReducedProblem()
  {
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  NONLINEARSOLVER &
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetNonlinearSolver(
                       std::string type)
  {
    if ((type == "state") || (type == "tangent"))
      {
        return nonlinear_state_solver_;
      }
    else if ((type == "adjoint") || (type == "adjoint_hessian")
             || (type == "adjoint_for_ee"))
      {
        return nonlinear_adjoint_solver_;
      }
    else
      {
        throw DOpEException("No Solver for Problem type:`" + type + "' found",
                            "StatReducedProblem::GetNonlinearSolver");

      }
  }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  CONTROLNONLINEARSOLVER &
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
  {
    if ((this->GetProblem()->GetType() == "gradient")
        || (this->GetProblem()->GetType() == "hessian"))
      {
        return nonlinear_gradient_solver_;
      }
    else
      {
        throw DOpEException(
          "No Solver for Problem type:`" + this->GetProblem()->GetType()
          + "' found", "StatReducedProblem::GetControlNonlinearSolver");

      }
  }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ReInit()
  {
    ReducedProblemInterface<PROBLEM, VECTOR>::ReInit();

    //Some Solvers must be reinited when called
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
    GetZForEE().ReInit();

    build_control_matrix_ = true;
    cost_needs_precomputations_ = 0;
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedState(
                       const ControlVector<VECTOR> &q)
  {
    this->InitializeFunctionalValues(
      this->GetProblem()->GetNFunctionals() + 1);

    this->SetProblemType("state");
    auto &problem = this->GetProblem()->GetStateProblem();
    if (state_reinit_ == true)
      {
        GetNonlinearSolver("state").ReInit(problem);
        state_reinit_ = false;

        if (problem.NeedInitialState())
          {
            //Only apply initial state if no previous values are present (i.e., u == 0)
            //thus, we can reuse good values from previous calculations
            if ( GetU().GetSpacialVector().linfty_norm() <= std::numeric_limits<double>::min() )
              {
                this->GetOutputHandler()->Write("Computing Initial Values:",
                                                4 + this->GetBasePriority());

                auto &initial_problem = problem.GetNewtonInitialProblem();
                this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
                if (dopedim == dealdim)
                  {
                    this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
                  }
                else if (dopedim == 0)
                  {
                    this->GetIntegrator().AddParamData("control",
                                                       &(q.GetSpacialVectorCopy()));
                  }
                else
                  {
                    throw DOpEException("dopedim not implemented",
                                        "StatReducedProblem::ComputeReducedState");
                  }

                //TODO: Possibly another solver for the initial value than for the pde...
                build_state_matrix_ = this->GetNonlinearSolver("state").NonlinearSolve(
                                        initial_problem, GetU().GetSpacialVector(), true, true);
                build_state_matrix_ = true;

                if (dopedim == dealdim)
                  {
                    this->GetIntegrator().DeleteDomainData("control");
                  }
                else if (dopedim == 0)
                  {
                    this->GetIntegrator().DeleteParamData("control");
                    q.UnLockCopy();
                  }
                else
                  {
                    throw DOpEException("dopedim not implemented",
                                        "StatReducedProblem::ComputeReducedState");
                  }
                this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

                this->GetOutputHandler()->Write((GetU().GetSpacialVector()),
                                                "Initial_State" + this->GetPostIndex(), problem.GetDoFType());

              }
          }
      }

    this->GetOutputHandler()->Write("Computing State Solution:",
                                    4 + this->GetBasePriority());

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
    if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
                                           &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedState");
      }
    try
      {
        build_state_matrix_ = this->GetNonlinearSolver("state").NonlinearSolve(
                                problem, (GetU().GetSpacialVector()), true, build_state_matrix_);
      }
    catch ( DOpEException &e)
      {
        if (dopedim == dealdim)
          {
            this->GetIntegrator().DeleteDomainData("control");
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().DeleteParamData("control");
            q.UnLockCopy();
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeReducedState");
          }
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
        //Reset Values
        GetU().GetSpacialVector() = 0.;
        build_state_matrix_ = true;
        state_reinit_ = true;
        throw e;
      }

    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedState");
      }
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    this->GetOutputHandler()->Write((GetU().GetSpacialVector()),
                                    "State" + this->GetPostIndex(), problem.GetDoFType());

  }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  bool
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(
                       const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g)
  {
    this->GetOutputHandler()->Write("Evaluating Constraints:",
                                    4 + this->GetBasePriority());

    this->SetProblemType("constraints");

    g = 0;
    //Local constraints
    //  this->GetProblem()->ComputeLocalConstraints(q.GetSpacialVector(), GetU().GetSpacialVector(),
    //                                              g.GetSpacialVector("local"));
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
                            "StatReducedProblem::ComputeReducedConstraints");
      }
    this->GetControlIntegrator().ComputeLocalControlConstraints(
      *(this->GetProblem()), g.GetSpacialVector("local"));
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
                            "StatReducedProblem::ComputeReducedConstraints");
      }
    //Global in Space-Time Constraints
    dealii::Vector<double> &gc = g.GetGlobalConstraints();
    //dealii::Vector<double> global_values(gc.size());

    unsigned int nglobal = gc.size();      //global_values.size();

    if (nglobal > 0)
      {
        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        for (unsigned int i = 0; i < nglobal; i++)
          {
            //this->SetProblemType("local_global_constraints", i);
            this->SetProblemType("global_constraints", i);
            this->GetIntegrator().AddDomainData("state",
                                                &(GetU().GetSpacialVector()));
            if (dopedim == dealdim)
              {
                this->GetIntegrator().AddDomainData("control",
                                                    &(q.GetSpacialVector()));
              }
            else if (dopedim == 0)
              {
                this->GetIntegrator().AddParamData("control",
                                                   &(q.GetSpacialVectorCopy()));
              }
            else
              {
                throw DOpEException("dopedim not implemented",
                                    "StatReducedProblem::ComputeReducedConstraints");
              }

            double ret = 0;
            bool found = false;

            if (this->GetProblem()->GetConstraintType().find("domain")
                != std::string::npos)
              {
                found = true;
                ret += this->GetIntegrator().ComputeDomainScalar(
                         *(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("point")
                != std::string::npos)
              {
                found = true;
                ret += this->GetIntegrator().ComputePointScalar(
                         *(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("boundary")
                != std::string::npos)
              {
                found = true;
                ret += this->GetIntegrator().ComputeBoundaryScalar(
                         *(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("face")
                != std::string::npos)
              {
                found = true;
                ret += this->GetIntegrator().ComputeFaceScalar(
                         *(this->GetProblem()));
              }

            if (!found)
              {
                throw DOpEException(
                  "Unknown Constraint Type: "
                  + this->GetProblem()->GetConstraintType(),
                  "StatReducedProblem::ComputeReducedConstraints");
              }
            //      global_values(i) = ret;
            gc(i) = ret;

            if (dopedim == dealdim)
              {
                this->GetIntegrator().DeleteDomainData("control");
              }
            else if (dopedim == 0)
              {
                this->GetIntegrator().DeleteParamData("control");
                q.UnLockCopy();
              }
            else
              {
                throw DOpEException("dopedim not implemented",
                                    "StatReducedProblem::ComputeReducedConstraints");
              }
            this->GetIntegrator().DeleteDomainData("state");
          }

        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetIntegrator());
        //gc = global_values;
      }

    //Check that no global in space, local in time constraints are given!
    if (g.HasType("local_global_control") || g.HasType("local_global_state"))
      {
        throw DOpEException(
          "There are global in space, local in time constraints given. In the stationary case they should be moved to global in space and time!",
          "StatReducedProblem::ComputeReducedConstraints");
      }

    //this->GetProblem()->PostProcessConstraints(g, true);
    this->GetProblem()->PostProcessConstraints(g);


    return g.IsFeasible();
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlBoxConstraints(
                       ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub)
  {
    this->GetProblem()->GetControlBoxConstraints(lb.GetSpacialVector(),
                                                 ub.GetSpacialVector());
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedAdjoint(
                       const ControlVector<VECTOR> &q)
  {
    this->GetOutputHandler()->Write("Computing Reduced Adjoint:",
                                    4 + this->GetBasePriority());

    this->SetProblemType("adjoint");
    auto &problem = this->GetProblem()->GetAdjointProblem();

    if (adjoint_reinit_ == true)
      {
        GetNonlinearSolver("adjoint").ReInit(problem);
        adjoint_reinit_ = false;
      }

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
    if (cost_needs_precomputations_ != 0)
      {
        auto func_vals = GetAuxiliaryParams("cost_functional_pre");
        this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
      }
    this->GetIntegrator().AddDomainData("state",
                                        &(GetU().GetSpacialVector()));

    if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
                                           &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedAdjoint");
      }

    build_adjoint_matrix_ =
      this->GetNonlinearSolver("adjoint").NonlinearSolve(
        problem, (GetZ().GetSpacialVector()), true,
        build_adjoint_matrix_);

    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedAdjoint");
      }
    if (cost_needs_precomputations_ != 0)
      {
        this->GetIntegrator().DeleteParamData("cost_functional_pre");
      }
    this->GetIntegrator().DeleteDomainData("state");

    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    this->GetOutputHandler()->Write((GetZ().GetSpacialVector()),
                                    "Adjoint" + this->GetPostIndex(), problem.GetDoFType());
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeDualForErrorEstimation(
                       const ControlVector<VECTOR> &q,
                       DOpEtypes::WeightComputation weight_comp)
  {
    this->GetOutputHandler()->Write("Computing Dual for Error Estimation:",
                                    4 + this->GetBasePriority());

    if (weight_comp == DOpEtypes::higher_order_interpolation)
      {
        this->SetProblemType("adjoint_for_ee");
      }
    else
      {
        throw DOpEException("Unknown WeightComputation",
                            "StatPDEProblem::ComputeDualForErrorEstimation");
      }

    auto &problem = this->GetProblem()->GetAdjoint_For_EEProblem();

    if (adjoint_reinit_ == true)
      {
        GetNonlinearSolver("adjoint_for_ee").ReInit(problem);
        adjoint_reinit_ = false;
      }

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

    this->GetIntegrator().AddDomainData("state",
                                        &(GetU().GetSpacialVector())); //&(GetU().GetSpacialVector())

    if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
                                           &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedAdjoint");
      }

    build_adjoint_matrix_ =
      this->GetNonlinearSolver("adjoint_for_ee").NonlinearSolve(problem,
                                                                (GetZForEE().GetSpacialVector()), true, build_adjoint_matrix_);

    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedAdjoint");
      }

    this->GetIntegrator().DeleteDomainData("state");

    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    this->GetOutputHandler()->Write((GetZForEE().GetSpacialVector()),
                                    "Adjoint_for_ee" + this->GetPostIndex(), problem.GetDoFType());

  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradient(
                       const ControlVector<VECTOR> &q, ControlVector<VECTOR> &gradient,
                       ControlVector<VECTOR> &gradient_transposed)
  {
    this->ComputeReducedAdjoint(q);

    this->GetOutputHandler()->Write("Computing Reduced Gradient:",
                                    4 + this->GetBasePriority());

    //Preparations for ControlInTheDirichletData
    VECTOR tmp;
    if (this->GetProblem()->HasControlInDirichletData())
      {
        tmp.reinit(GetU().GetSpacialVector());
        this->SetProblemType("adjoint");
        auto &problem = this->GetProblem()->GetAdjointProblem();

        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        if (dopedim == dealdim)
          {
            this->GetIntegrator().AddDomainData("control",
                                                &(q.GetSpacialVector()));
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().AddParamData("control",
                                               &(q.GetSpacialVectorCopy()));
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeReducedGradient");
          }
        if (cost_needs_precomputations_ != 0)
          {
            auto func_vals = GetAuxiliaryParams("cost_functional_pre");
            this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
          }
        this->GetIntegrator().AddDomainData("state",
                                            &(GetU().GetSpacialVector()));
        this->GetIntegrator().AddDomainData("last_newton_solution",
                                            &(GetZ().GetSpacialVector()));

//        this->GetIntegrator().ComputeNonlinearResidual(problem, tmp, false);
        this->GetIntegrator().ComputeNonlinearResidual(problem, tmp);

        tmp *= -1.;

        if (dopedim == dealdim)
          {
            this->GetIntegrator().DeleteDomainData("control");
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().DeleteParamData("control");
            q.UnLockCopy();
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeReducedGradient");
          }
        if (cost_needs_precomputations_ != 0)
          {
            this->GetIntegrator().DeleteParamData("cost_functional_pre");
          }

        this->GetIntegrator().DeleteDomainData("state");
        this->GetIntegrator().DeleteDomainData("last_newton_solution");
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetIntegrator());
      }
    //Endof Dirichletdata Preparations
    this->SetProblemType("gradient");
    if (gradient_reinit_ == true)
      {
        GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
        gradient_reinit_ = false;
      }

    this->GetProblem()->AddAuxiliaryToIntegrator(
      this->GetControlIntegrator());

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
                            "StatReducedProblem::ComputeReducedGradient");
      }
    if (cost_needs_precomputations_ != 0)
      {
        auto func_vals = GetAuxiliaryParams("cost_functional_pre");
        this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
      }
    this->GetControlIntegrator().AddDomainData("state",
                                               &(GetU().GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("adjoint",
                                               &(GetZ().GetSpacialVector()));
    if (this->GetProblem()->HasControlInDirichletData())
      this->GetControlIntegrator().AddDomainData("adjoint_residual", &tmp);

    gradient_transposed = 0.;
    if (dopedim == dealdim)
      {
        this->GetControlIntegrator().AddDomainData("last_newton_solution",
                                                   &(gradient_transposed.GetSpacialVector()));
//        this->GetControlIntegrator().ComputeNonlinearResidual(
//            *(this->GetProblem()), gradient.GetSpacialVector(), true);
        this->GetControlIntegrator().ComputeNonlinearResidual(
          *(this->GetProblem()), gradient.GetSpacialVector());
        this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                  &(gradient_transposed.GetSpacialVectorCopy()));
//        this->GetControlIntegrator().ComputeNonlinearResidual(
//            *(this->GetProblem()), gradient.GetSpacialVector(), true);
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
                            "StatReducedProblem::ComputeReducedGradient");
      }
    if (cost_needs_precomputations_ != 0)
      {
        this->GetIntegrator().DeleteParamData("cost_functional_pre");
      }
    this->GetControlIntegrator().DeleteDomainData("state");
    this->GetControlIntegrator().DeleteDomainData("adjoint");
    if (this->GetProblem()->HasControlInDirichletData())
      this->GetControlIntegrator().DeleteDomainData("adjoint_residual");

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
  typename VECTOR, int dopedim, int dealdim>
  double
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(
                       const ControlVector<VECTOR> &q)
  {
    this->ComputeReducedState(q);

    this->GetOutputHandler()->Write("Computing Cost Functional:",
                                    4 + this->GetBasePriority());

    this->SetProblemType("cost_functional");
    cost_needs_precomputations_ = this->GetProblem()->FunctionalNeedPrecomputations();
    if (cost_needs_precomputations_ != 0)
      {
        unsigned int n_pre = cost_needs_precomputations_;
        AllocateAuxiliaryParams("cost_functional_pre",n_pre);

        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        if (dopedim == dealdim)
          {
            this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().AddParamData("control",
                                               &(q.GetSpacialVectorCopy()));
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeReducedCostFunctional");
          }
        this->GetIntegrator().AddDomainData("state",
                                            &(GetU().GetSpacialVector()));

        CalculatePreFunctional("cost_functional","_pre",n_pre,0);

        if (dopedim == dealdim)
          {
            this->GetIntegrator().DeleteDomainData("control");
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().DeleteParamData("control");
            q.UnLockCopy();
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeReducedCostFunctional");
          }
        this->GetIntegrator().DeleteDomainData("state");
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
        this->SetProblemType("cost_functional");
      }
    //End of Precomputations

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

    if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
                                           &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedCostFunctional");
      }
    if (cost_needs_precomputations_ != 0)
      {
        auto func_vals = GetAuxiliaryParams("cost_functional_pre");
        this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
      }
    this->GetIntegrator().AddDomainData("state",
                                        &(GetU().GetSpacialVector()));
    double ret = 0;
    bool found = false;


    if (this->GetProblem()->GetFunctionalType().find("domain")
        != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("point")
        != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("boundary")
        != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeBoundaryScalar(
                 *(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("face")
        != std::string::npos)
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
          "Unknown Functional Type: "
          + this->GetProblem()->GetFunctionalType(),
          "StatReducedProblem::ComputeReducedCostFunctional");
      }

    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedCostFunctional");
      }
    if (cost_needs_precomputations_ != 0)
      {
        this->GetIntegrator().DeleteParamData("cost_functional_pre");
      }
    this->GetIntegrator().DeleteDomainData("state");
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    this->GetFunctionalValues()[0].push_back(ret);
    return ret;
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(
                       const ControlVector<VECTOR> &q)
  {
    this->GetOutputHandler()->Write("Computing Functionals:",
                                    4 + this->GetBasePriority());

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

    if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
                                           &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedFunctionals");
      }
    this->GetIntegrator().AddDomainData("state",
                                        &(GetU().GetSpacialVector()));

    for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
      {
        double ret = 0;
        bool found = false;

        this->SetProblemType("aux_functional", i);
        if (this->GetProblem()->FunctionalNeedPrecomputations() != 0)
          {
            std::stringstream tmp;
            tmp << "aux_functional_"<<i<<"_pre";
            AllocateAuxiliaryParams(tmp.str(),this->GetProblem()->FunctionalNeedPrecomputations());
            CalculatePreFunctional("aux_functional","_pre",
                                   this->GetProblem()->FunctionalNeedPrecomputations(),i);
            auto func_vals = GetAuxiliaryParams(tmp.str());
            this->GetIntegrator().AddParamData(tmp.str(),&(func_vals->second));
          }

        if (this->GetProblem()->GetFunctionalType().find("domain")
            != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputeDomainScalar(
                     *(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("point")
            != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputePointScalar(
                     *(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("boundary")
            != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputeBoundaryScalar(
                     *(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("face")
            != std::string::npos)
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
              "Unknown Functional Type: "
              + this->GetProblem()->GetFunctionalType(),
              "StatReducedProblem::ComputeReducedFunctionals");
          }
        if (this->GetProblem()->FunctionalNeedPrecomputations() != 0)
          {
            std::stringstream tmp;
            tmp << "aux_functional_"<<i<<"_pre";
            this->GetIntegrator().DeleteParamData(tmp.str());
          }

        this->GetFunctionalValues()[i + 1].push_back(ret);
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << this->GetProblem()->GetFunctionalName() << ": " << ret;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }

    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedFunctionals");
      }
    this->GetIntegrator().DeleteDomainData("state");
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  template<class DWRC,class PDE>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeRefinementIndicators(
                       const ControlVector<VECTOR> &q, DWRC &dwrc, PDE &pde)
//    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
//        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeRefinementIndicators(
//        const ControlVector<VECTOR>& q, DWRDataContainerBase<VECTOR>& dwrc)
  {
    //Attach the ResidualModifier to the PDE.
    pde.ResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::ResidualModifier),boost::ref(dwrc),_1);
    pde.VectorResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::VectorResidualModifier),boost::ref(dwrc),_1);

    //first we reinit the dwrdatacontainer (this
    //sets the weight-vectors to their correct length)
#if DEAL_II_VERSION_GTE(8,4,0)
    const unsigned int n_elements =
      this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler().get_triangulation().n_active_cells();
#else
    const unsigned int n_elements =
      this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler().get_tria().n_active_cells();
#endif
    dwrc.ReInit(n_elements);

    //Estimation for Costfunctional or if no dual is needed
    if (this->GetProblem()->EEFunctionalIsCost() || !dwrc.NeedDual())
      {
        this->GetOutputHandler()->Write("Computing Error Indicators:",
                                        4 + this->GetBasePriority());
        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        //add the primal and dual solution to the integrator
        this->GetIntegrator().AddDomainData("state",
                                            &(GetU().GetSpacialVector()));
        this->GetIntegrator().AddDomainData("adjoint_for_ee",
                                            &(GetZ().GetSpacialVector()));

        if (dopedim == dealdim)
          {
            this->GetIntegrator().AddDomainData("control",
                                                &(q.GetSpacialVector()));
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().AddParamData("control",
                                               &(q.GetSpacialVectorCopy()));
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeRefinementIndicators");
          }

        this->SetProblemType("error_evaluation");

        //prepare the weights...
        dwrc.PrepareWeights(GetU(), GetZ());
#if dope_dimension > 0
        dwrc.PrepareWeights(q);
#endif
        //now we finally compute the refinement indicators
        this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(),
                                                          dwrc);
        // release the lock on the refinement indicators (see dwrcontainer.h)
        dwrc.ReleaseLock();
        dwrc.ClearWeightData();

        // clear the data
        if (dopedim == dealdim)
          {
            this->GetIntegrator().DeleteDomainData("control");
          }
        else if (dopedim == 0)
          {
            this->GetIntegrator().DeleteParamData("control");
            q.UnLockCopy();
          }
        else
          {
            throw DOpEException("dopedim not implemented",
                                "StatReducedProblem::ComputeRefinementIndicators");
          }
        this->GetIntegrator().DeleteDomainData("state");
        this->GetIntegrator().DeleteDomainData("adjoint_for_ee");
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetIntegrator());
      }
    else //Estimation for other (not the cost) functional
      {
        throw DOpEException("Estimating the error in other functionals than cost is not implemented",
                            "StatReducedProblem::ComputeRefinementIndicators");

      }

    std::stringstream out;
    this->GetOutputHandler()->InitOut(out);
    out << "Error estimate using "<<dwrc.GetName();
    if (dwrc.NeedDual())
      out<<" For the computation of "<<this->GetProblem()->GetFunctionalName();
    out<< ": "<< dwrc.GetError();
    this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(
                       const ControlVector<VECTOR> &q, const ControlVector<VECTOR> &direction,
                       ControlVector<VECTOR> &hessian_direction,
                       ControlVector<VECTOR> &hessian_direction_transposed)
  {
    this->GetOutputHandler()->Write("Computing ReducedHessianVector:",
                                    4 + this->GetBasePriority());
    this->GetOutputHandler()->Write("\tSolving Tangent:",
                                    5 + this->GetBasePriority());

    this->SetProblemType("tangent");
    {
      //Start Tangent Calculatation
      auto &problem = this->GetProblem()->GetTangentProblem();

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
                                          &(GetU().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("state",
                                                 &(GetU().GetSpacialVector()));

      if (dopedim == dealdim)
        {
          this->GetIntegrator().AddDomainData("dq",
                                              &(direction.GetSpacialVector()));
          this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
        }
      else if (dopedim == 0)
        {
          this->GetIntegrator().AddParamData("dq",
                                             &(direction.GetSpacialVectorCopy()));
          this->GetIntegrator().AddParamData("control",
                                             &(q.GetSpacialVectorCopy()));
        }
      else
        {
          throw DOpEException("dopedim not implemented",
                              "StatReducedProblem::ComputeReducedHessianVector");
        }

      //tangent Matrix is the same as state matrix
      build_state_matrix_ = this->GetNonlinearSolver("tangent").NonlinearSolve(
                              problem, (GetDU().GetSpacialVector()), true,
                              build_state_matrix_);

      this->GetOutputHandler()->Write((GetDU().GetSpacialVector()),
                                      "Tangent" + this->GetPostIndex(), problem.GetDoFType());
    }//End Tangent Calculation

    this->GetIntegrator().AddDomainData("adjoint",
                                        &(GetZ().GetSpacialVector()));
    this->GetIntegrator().AddDomainData("tangent",
                                        &(GetDU().GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("adjoint",
                                               &(GetZ().GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("tangent",
                                               &(GetDU().GetSpacialVector()));

    //After the Tangent has been computed, we can precompute
    //cost functional derivative-values (if necessary)
    if (cost_needs_precomputations_ != 0)
      {
        unsigned int n_pre = cost_needs_precomputations_;
        AllocateAuxiliaryParams("cost_functional_pre_tangent",n_pre);
        CalculatePreFunctional("cost_functional","_pre_tangent",n_pre,0);
      } // End precomputation of values

    this->GetOutputHandler()->Write("\tSolving Adjoint Hessian:",
                                    5 + this->GetBasePriority());
    this->SetProblemType("adjoint_hessian");
    {
      //Adjoint_Hessian
      auto &problem = this->GetProblem()->GetAdjoint_HessianProblem();

      if (cost_needs_precomputations_ != 0)
        {
          {
            auto func_vals = GetAuxiliaryParams("cost_functional_pre");
            this->GetIntegrator().AddParamData("cost_functional_pre",&(func_vals->second));
          }
          {
            auto func_vals = GetAuxiliaryParams("cost_functional_pre_tangent");
            this->GetIntegrator().AddParamData("cost_functional_pre_tangent",&(func_vals->second));
          }
        }

      //adjoint_hessian Matrix is the same as adjoint matrix
      build_adjoint_matrix_ =
        this->GetNonlinearSolver("adjoint_hessian").NonlinearSolve(
          problem, (GetDZ().GetSpacialVector()), true,
          build_adjoint_matrix_);

      this->GetOutputHandler()->Write((GetDZ().GetSpacialVector()),
                                      "Hessian" + this->GetPostIndex(), problem.GetDoFType());

      this->GetIntegrator().AddDomainData("adjoint_hessian",
                                          &(GetDZ().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("adjoint_hessian",
                                                 &(GetDZ().GetSpacialVector()));

      this->GetOutputHandler()->Write(
        "\tComputing Representation of the Hessian:",
        5 + this->GetBasePriority());
    }//End Adjoint Hessian
    //Preparations for Control In The Dirichlet Data
    VECTOR tmp;
    VECTOR tmp_second;
    if (this->GetProblem()->HasControlInDirichletData())
      {
        tmp.reinit(GetU().GetSpacialVector());
        tmp_second.reinit(GetU().GetSpacialVector());
        this->SetProblemType("adjoint");
        {
          // Adjoint
          auto &problem = this->GetProblem()->GetAdjointProblem();

          this->GetIntegrator().AddDomainData("last_newton_solution",
                                              &(GetZ().GetSpacialVector()));

//    this->GetIntegrator().ComputeNonlinearResidual(problem, tmp_second, false);
          this->GetIntegrator().ComputeNonlinearResidual(problem, tmp_second);
          tmp_second *= -1.;

          this->GetIntegrator().DeleteDomainData("last_newton_solution");
        }//End Adjoint
        this->SetProblemType("adjoint_hessian");
        {
          //Adjoint_Hessian
          auto &problem = this->GetProblem()->GetAdjoint_HessianProblem();

          this->GetIntegrator().AddDomainData("last_newton_solution",
                                              &(GetDZ().GetSpacialVector()));

//    this->GetIntegrator().ComputeNonlinearResidual(problem, tmp, false);
          this->GetIntegrator().ComputeNonlinearResidual(problem, tmp);
          tmp *= -1.;

          this->GetIntegrator().DeleteDomainData("last_newton_solution");
        }
      }
    //Endof Dirichletdata Preparations
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    this->SetProblemType("hessian");
    this->GetProblem()->AddAuxiliaryToIntegrator(
      this->GetControlIntegrator());
    if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("dq");
        this->GetIntegrator().DeleteDomainData("control");
        this->GetControlIntegrator().AddDomainData("dq",
                                                   &(direction.GetSpacialVector()));
        this->GetControlIntegrator().AddDomainData("control",
                                                   &(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("dq");
        this->GetIntegrator().DeleteParamData("control");
        direction.UnLockCopy();
        q.UnLockCopy();
        this->GetControlIntegrator().AddParamData("dq",
                                                  &(direction.GetSpacialVectorCopy()));
        this->GetControlIntegrator().AddParamData("control",
                                                  &(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedHessianVector");
      }
    if (this->GetProblem()->HasControlInDirichletData())
      {
        this->GetControlIntegrator().AddDomainData("adjoint_residual", &tmp);
        this->GetControlIntegrator().AddDomainData("hessian_residual",
                                                   &tmp_second);
      }

    {
      hessian_direction_transposed = 0.;
      if (dopedim == dealdim)
        {
          this->GetControlIntegrator().AddDomainData("last_newton_solution",
                                                     &(hessian_direction_transposed.GetSpacialVector()));
//          this->GetControlIntegrator().ComputeNonlinearResidual(
//              *(this->GetProblem()), hessian_direction.GetSpacialVector(),
//              true);
          this->GetControlIntegrator().ComputeNonlinearResidual(
            *(this->GetProblem()), hessian_direction.GetSpacialVector());
          this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
        }
      else if (dopedim == 0)
        {
          this->GetControlIntegrator().AddParamData("last_newton_solution",
                                                    &(hessian_direction_transposed.GetSpacialVectorCopy()));
//         this->GetControlIntegrator().ComputeNonlinearResidual(
//             *(this->GetProblem()), hessian_direction.GetSpacialVector(),
//              true);
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
    }

    if (dopedim == dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("dq");
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().DeleteParamData("dq");
        this->GetControlIntegrator().DeleteParamData("control");
        direction.UnLockCopy();
        q.UnLockCopy();
      }
    else
      {
        throw DOpEException("dopedim not implemented",
                            "StatReducedProblem::ComputeReducedHessianVector");
      }
    this->GetIntegrator().DeleteDomainData("state");
    this->GetIntegrator().DeleteDomainData("adjoint");
    this->GetIntegrator().DeleteDomainData("tangent");
    this->GetIntegrator().DeleteDomainData("adjoint_hessian");
    this->GetControlIntegrator().DeleteDomainData("state");
    this->GetControlIntegrator().DeleteDomainData("adjoint");
    this->GetControlIntegrator().DeleteDomainData("tangent");
    this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
    if (this->GetProblem()->HasControlInDirichletData())
      {
        this->GetControlIntegrator().DeleteDomainData("adjoint_residual");
        this->GetControlIntegrator().DeleteDomainData("hessian_residual");
      }
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(
      this->GetControlIntegrator());

    if (cost_needs_precomputations_ != 0)
      {
        this->GetIntegrator().DeleteParamData("cost_functional_pre");
        this->GetIntegrator().DeleteParamData("cost_functional_pre_tangent");
      }

  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradientOfGlobalConstraints(
                       unsigned int num, const ControlVector<VECTOR> &q,
                       const ConstraintVector<VECTOR> &g, ControlVector<VECTOR> &gradient,
                       ControlVector<VECTOR> &gradient_transposed)
  {
    //FIXME: If the global constraints depend on u we need to calculate a corresponding
    //       dual solution before we can calculate the gradient.
    std::stringstream out;
    out << "Computing Reduced Gradient of global constraint " << num << " :";
    this->GetOutputHandler()->Write(out, 4 + this->GetBasePriority());
    //Compute derivatives of global constraints
    this->SetProblemType("global_constraint_gradient", num);

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
                            "StatReducedProblem::ComputeReducedGradient");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(
      this->GetControlIntegrator());
    this->GetControlIntegrator().AddDomainData("constraints_local",
                                               &g.GetSpacialVector("local"));
    this->GetControlIntegrator().AddParamData("constraints_global",
                                              &g.GetGlobalConstraints());

    //Compute
//      this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()), gradient.GetSpacialVector(), true);
    this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()), gradient.GetSpacialVector());
    gradient_transposed = gradient;

    this->GetControlIntegrator().DeleteDomainData("constraints_local");
    this->GetControlIntegrator().DeleteParamData("constraints_global");
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(
      this->GetControlIntegrator());
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
                            "StatReducedProblem::ComputeReducedGradient");
      }
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
                       const VECTOR &v, std::string name, std::string outfile,
                       std::string dof_type, std::string filetype)
  {
    if (dof_type == "state")
      {
        auto &data_out =
          this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
        data_out.attach_dof_handler(
          this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

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
            throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StatReducedProblem::WriteToFile");
          }
        data_out.clear();
      }
    else if (dof_type == "control")
      {
#if dope_dimension >0
        auto &data_out = this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
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
            data_out.write_gnuplot(output);
          }
        else
          {
            throw DOpEException("Don't know how to write filetype `" + filetype + "'!",
                                "StatReducedProblem::WriteToFile");
          }
        data_out.clear();
#else
        if (filetype == ".txt")
          {
            std::ofstream output(outfile.c_str());
            Vector<double> off;
            off = v;
            for (unsigned int i = 0; i < off.size(); i++)
              {
                output << off(i) << std::endl;
              }
          }
        else
          {
            throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StatReducedProblem::WriteToFile");
          }
#endif
      }
    else
      {
        throw DOpEException("No such DoFHandler `" + dof_type + "'!",
                            "StatReducedProblem::WriteToFile");
      }
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
                       const ControlVector<VECTOR> &v, std::string name,
                       std::string dof_type)
  {
    this->GetOutputHandler()->Write(v.GetSpacialVector(), name, dof_type);
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
       CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
       AllocateAuxiliaryParams(std::string name,
                               unsigned int n_components)
  {
    std::map<std::string,dealii::Vector<double> >::iterator func_vals = auxiliary_params_.find(name);
    if (func_vals != auxiliary_params_.end())
      {
        assert(func_vals->second.size() == n_components);
        //already created. Nothing to do
      }
    else
      {
        auto ret = auxiliary_params_.emplace(name,dealii::Vector<double>(n_components));
        if (ret.second == false)
          {
            throw DOpEException("Creation of Storage for Auxiliary time params with name "+name+" failed!",
                                "StatReducedProblem::AllocateAuxiliaryParams");
          }
      }
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  std::map<std::string,dealii::Vector<double> >::iterator
  StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                     CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
                     GetAuxiliaryParams(std::string name)

  {
    return auxiliary_params_.find(name);
  }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  void StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
       CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
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
////////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
