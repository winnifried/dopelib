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

#ifndef INSTAT_PDE_PROBLEM_H_
#define INSTAT_PDE_PROBLEM_H_

#include <interfaces/pdeprobleminterface.h>
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
   * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
   * @tparam <INTEGRATOR>                An integrator for the state variables,
   *                                     e.g, Integrator or IntegratorMixedDimensions..
   * @tparam <PROBLEM>                   PDE- or optimization problem under consideration including ts-scheme.
   * @tparam <VECTOR>                    Class in which we want to store the spatial vector
   *                                     (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <dealdim>                   The dimension for the state variable.
   */
  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  class InstatPDEProblem: public PDEProblemInterface<PROBLEM, VECTOR, dealdim>
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
    InstatPDEProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                     ParameterReader &param_reader,
                     INTEGRATORDATACONT &idc,
                     int base_priority = 0);


    virtual ~InstatPDEProblem();

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
    void ReInit() override;

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * PDEProblemInterface
     *
     */
    void ComputeReducedFunctionals() override;

    /******************************************************/

    /**
     * This function computes the solution for the dual variable
     * for error estimation.
     *
     * I is assumed that the state u(q) corresponding to
     * the argument q is already calculated.
     *
     * @param weight_comp  A flag deciding how the weights should be calculated
     */
    void
    ComputeDualForErrorEstimation(DOpEtypes::WeightComputation)
    {
      throw DOpEException("ExcNotImplemented",
                          "InstatPDEProblem::ComputeDualForErrorEstimation");
    }


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
     * @param dwrc              The data container
     * @param pde               The problem
     *
     */
    template<class DWRC, class PDE>
    void
    ComputeRefinementIndicators(DWRC &dwrc, PDE &pde);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * PDEProblemInterface
     *
     */
    void StateSizeInfo(std::stringstream &out) override
    {
      GetU().PrintInfos(out);
    }

    /******************************************************/

    /**
     *  A std::vector v is printed to a text file.
     *  Note that this assumes that the vector is one entry per time step.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     */
    void WriteToFile(const std::vector<double> &v, std::string outfile) override;

    /**
     * Import overloads from base class.
     */
    using PDEProblemInterface<PROBLEM, VECTOR, dealdim>::WriteToFile;

    /******************************************************/

  protected:
    const StateVector<VECTOR> &GetU() const
    {
      return u_;
    }
    StateVector<VECTOR> &GetU()
    {
      return u_;
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

    NONLINEARSOLVER &GetNonlinearSolver(std::string type);
    INTEGRATOR &GetIntegrator()
    {
      return integrator_;
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
     */
    void ComputeReducedState();

    /******************************************************/

    /**
     * This function does the loop over time.
     *
     * @param problem      Describes the nonstationary pde to be solved
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
     * @param outname      The name prefix given to the solution vectors
     *                     if they are written to files, e.g., Adjoint, Hessian, ...
     * @param eval_grads   Decide wether to evaluate the gradients of the functionals or not.
     *                     Should be true for the adjoint and dual_hessian-problem but false
     *                     for auxilliary backward pdes.
     */
    template<typename PDE>
    void BackwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, std::string outname, bool eval_grads);

  private:
    /**
     * Helper function to prevent code duplicity. Adds the user defined
     * user Data to the Integrator.
     */
    void
    AddUDD(unsigned int dof_number, const TimeIterator &interval)
    {
      for (auto it = this->GetUserDomainData().begin();
           it != this->GetUserDomainData().end(); it++)
        {
          this->GetIntegrator().AddDomainData(it->first, it->second);
        }
      for (auto it = this->GetUserTimeDomainData().begin();
           it != this->GetUserTimeDomainData().end(); it++)
        {
          it->second->SetTimeDoFNumber(dof_number,interval);
          this->GetIntegrator().AddDomainData(it->first, &(it->second->GetSpacialVector()));
        }
    }

    /**
     * Helper function to prevent code duplicity. Deletes the user defined
     * user Data from the Integrator.
     */
    void
    DeleteUDD()
    {
      for (auto it = this->GetUserDomainData().begin();
           it != this->GetUserDomainData().end(); it++)
        {
          this->GetIntegrator().DeleteDomainData(it->first);
        }
      for (auto it = this->GetUserTimeDomainData().begin();
           it != this->GetUserTimeDomainData().end(); it++)
        {
          this->GetIntegrator().DeleteDomainData(it->first);
        }
    }
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
    StateVector<VECTOR> z_for_ee_;

    std::map<std::string,std::vector<dealii::Vector<double> >> auxiliary_time_params_;

    INTEGRATOR integrator_;
    NONLINEARSOLVER nonlinear_state_solver_;
    NONLINEARSOLVER nonlinear_adjoint_solver_;

    bool build_state_matrix_ = false, build_adjoint_matrix_ = false;
    bool state_reinit_ = false, adjoint_reinit_ = false;

    bool project_initial_data_ = false;

    friend class SolutionExtractor<InstatPDEProblem<NONLINEARSOLVER,
             INTEGRATOR, PROBLEM, VECTOR, dealdim>,   VECTOR > ;
  };

  /*************************************************************************/
  /*****************************IMPLEMENTATION******************************/
  /*************************************************************************/
  using namespace dealii;

  /******************************************************/
  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
       PROBLEM, VECTOR, dealdim>::declare_params(
         ParameterReader &param_reader)
  {
    NONLINEARSOLVER::declare_params(param_reader);

    param_reader.SetSubsection("output parameters");
    param_reader.declare_entry("number of patches", "0",
                               Patterns::Integer(0));

  }
  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  template<typename INTEGRATORDATACONT>
  InstatPDEProblem<NONLINEARSOLVER,
                   INTEGRATOR, PROBLEM, VECTOR, dealdim>::InstatPDEProblem(
                     PROBLEM *OP,
                     DOpEtypes::VectorStorageType state_behavior,
                     ParameterReader &param_reader,
                     INTEGRATORDATACONT &idc,
                     int base_priority) :
                     PDEProblemInterface<PROBLEM, VECTOR,dealdim> (OP,
                         base_priority),
                     u_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                     z_for_ee_(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
                     integrator_(idc),
                     nonlinear_state_solver_(integrator_, param_reader),
                     nonlinear_adjoint_solver_(integrator_, param_reader)
  {
    // Solvers should be ReInited
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
    }
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
                   PROBLEM, VECTOR, dealdim>::~InstatPDEProblem()
  {
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  NONLINEARSOLVER &InstatPDEProblem<NONLINEARSOLVER,
                  INTEGRATOR, PROBLEM, VECTOR, dealdim>::GetNonlinearSolver(std::string type)
  {
    if (type == "state")
      {
        return nonlinear_state_solver_;
      }
    else
      {
        throw DOpEException("No Solver for Problem type:`" + type + "' found",
                            "InstatPDEProblem::GetNonlinearSolver");

      }
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
       PROBLEM, VECTOR, dealdim>::ReInit()
  {
    PDEProblemInterface<PROBLEM, VECTOR,dealdim>::ReInit();

    // Some Solvers must be reinited when called
    // Better have subproblems, so that solver can be reinited here
    {
      state_reinit_ = true;
      adjoint_reinit_ = true;
    }

    build_state_matrix_ = true;
    build_adjoint_matrix_ = true;

    GetU().ReInit();

    // Remove all time-params - they are now obsolete
    auxiliary_time_params_.clear();

  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
       PROBLEM, VECTOR, dealdim>::ComputeReducedState()
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

    this->ForwardTimeLoop(problem,this->GetU(),"State",true);

  }
  /******************************************************/

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
       PROBLEM, VECTOR, dealdim>::ComputeReducedFunctionals()
  {
    this->ComputeReducedState();

    this->GetOutputHandler()->Write("Computing Functionals:", 4  + this->GetBasePriority());

    for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
      {
        this->SetProblemType("aux_functional", i);
        if (this->GetProblem()->GetFunctionalType().find("timelocal") != std::string::npos)
          {
            if (this->GetFunctionalValues()[i].size() == 1)
              {
                std::stringstream out;
                this->GetOutputHandler()->InitOut(out);
                out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i][0];
                this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
              }
            else if (this->GetFunctionalValues()[i].size() > 1)
              {
                if (this->GetFunctionalValues()[i].size()
                    == this->GetProblem()->GetSpaceTimeHandler()->GetMaxTimePoint() + 1)
                  {
                    std::stringstream out;
                    this->GetOutputHandler()->InitOut(out);
                    out << this->GetProblem()->GetFunctionalName() << " too large. Writing to file instead: ";
                    this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
                    this->GetOutputHandler()->Write(this->GetFunctionalValues()[i],
                                                    this->GetProblem()->GetFunctionalName()
                                                    + this->GetPostIndex(), "time");
                  }
                else
                  {
                    std::stringstream out;
                    this->GetOutputHandler()->InitOut(out);
                    out << this->GetProblem()->GetFunctionalName() << ": ";
                    for (unsigned int k = 0; k < this->GetFunctionalValues()[i].size(); k++)
                      out << this->GetFunctionalValues()[i][k] << " ";
                    this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
                  }
              }
            else
              {
                throw DOpEException("Functional: " + this->GetProblem()->GetFunctionalType()
                                    + " was not evaluated ever!", "InstatPDEProblem::ComputeFunctionals");
              }
          }
        else if (this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
          {
            std::stringstream out;
            this->GetOutputHandler()->InitOut(out);
            out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i][0];
            this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
          }
        else
          {
            throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                                "InstatPDEProblem::ComputeFunctionals");
          }
      }
  }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  template<class DWRC, class PDE>
  void
  InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeRefinementIndicators(
    DWRC &dwrc, PDE &pde)
  {
    //Attach the ResidualModifier to the PDE.
    pde.ResidualModifier = boost::bind<void>(
                             boost::mem_fn(&DWRC::ResidualModifier), boost::ref(dwrc),boost::placeholders::_1);
    pde.VectorResidualModifier = boost::bind<void>(
                                   boost::mem_fn(&DWRC::VectorResidualModifier), boost::ref(dwrc),boost::placeholders::_1);

    VECTOR u_old;

    const std::vector<double> times =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
    const unsigned int
    n_dofs_per_interval =
      this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();

    //Not implemented for more than 2 dofs per interval.
    assert(n_dofs_per_interval ==2);
    //If we need the dual solution, compute it
    if (dwrc.NeedDual())
      this->ComputeDualForErrorEstimation(dwrc.GetWeightComputation());

    std::vector<unsigned int> local_to_global(n_dofs_per_interval);

    this->GetOutputHandler()->SetIterationNumber(0, "Time");


    //first we reinit the dwrdatacontainer
    dwrc.ReInit();

    unsigned int need_precomputed_nodal_values = dwrc.NPrecomputedNodalValues();
    std::vector<StateVector<VECTOR>* > aux_nodal_values;
    if ( need_precomputed_nodal_values != 0 )
      {
        aux_nodal_values.resize(need_precomputed_nodal_values,NULL);
        for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
          {
            aux_nodal_values[i] = new StateVector<VECTOR>(GetU());
          }
      }
    for (TimeIterator it =
           this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval(); it
         != this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().after_last_interval(); ++it)
      {
        it.get_time_dof_indices(local_to_global);
        this->GetProblem()->SetTime(times[local_to_global[0]], local_to_global[0], it);
        if ( it == this->GetProblem()->GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval())
          {
            //On First intervall, also calculate errors at initial time!
            this->GetProblem()->SetTime(times[local_to_global[0]], local_to_global[0], it);
            dwrc.SetTime(local_to_global[0]);
            this->GetOutputHandler()->SetIterationNumber(local_to_global[0],"Time");

            //some output
            this->GetOutputHandler()->Write("Computing Error Indicators:",
                                            4 + this->GetBasePriority());

            this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

            //add the primal and (if needed) dual solution to the integrator
            this->GetIntegrator().AddDomainData("state",
                                                &(GetU().GetSpacialVector()));
            if (dwrc.NeedDual())
              this->GetIntegrator().AddDomainData("adjoint_for_ee",
                                                  &(GetZForEE().GetSpacialVector()));

            AddUDD(local_to_global[0], it);

            //Check if some nodal values need to be precomputed, e.g., active set indicators
            // for the obstacle problem

            if ( need_precomputed_nodal_values != 0 )
              {
                for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
                  {
                    assert(aux_nodal_values[i] != NULL);
                    aux_nodal_values[i]->SetTimeDoFNumber(local_to_global[0], it);
                    {
                      //some output
                      std::stringstream tmp;
                      tmp << "Precomputation "<<i;
                      this->GetOutputHandler()->Write(tmp.str(),
                                                      4 + this->GetBasePriority());
                    }


                    //Calculate
                    this->SetProblemType("aux_error",i);
                    auto &problem = this->GetProblem()->GetErrorPrecomputations();

                    this->GetIntegrator().ComputeNonlinearRhs(problem, aux_nodal_values[i]->GetSpacialVector());

                    //Distribute for hanging nodes
                    problem.GetDoFConstraints().distribute(aux_nodal_values[i]->GetSpacialVector());
                    //output (in vtk files)
                    this->GetOutputHandler()->Write(aux_nodal_values[i]->GetSpacialVector(),
                                                    "Aux_Error_Indicators_"+i,"state");
                    std::stringstream tmp;
                    tmp << "aux_error_"<<i;
                    this->GetIntegrator().AddDomainData(tmp.str(),&(aux_nodal_values[i]->GetSpacialVector()));

                  }
              }

            this->SetProblemType("error_evaluation");

            //prepare the weights...
            dwrc.PrepareWeights(GetU(), GetZForEE());

            //now we finally compute the refinement indicators
            this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(),
                                                              dwrc);

            // release the lock on the refinement indicators (see dwrcontainer.h)
            dwrc.ReleaseLock();

            const double step_error = dwrc.GetStepError();

            // clear the data
            dwrc.ClearWeightData();
            this->GetIntegrator().DeleteDomainData("state");
            if (dwrc.NeedDual())
              this->GetIntegrator().DeleteDomainData("adjoint_for_ee");
            DeleteUDD();
            this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

            //Cleaning auxiliary nodal variables
            if ( need_precomputed_nodal_values != 0 )
              {
                for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
                  {
                    std::stringstream tmp;
                    tmp << "aux_error_"<<i;
                    this->GetIntegrator().DeleteDomainData(tmp.str());
                  }
              }

            std::stringstream out;
            this->GetOutputHandler()->InitOut(out);
            out << "Error estimate using " << dwrc.GetName();
            if (dwrc.NeedDual())
              out << " for the " << this->GetProblem()->GetFunctionalName();
            out << " at time "<< times[local_to_global[0]];
            out << ": " << step_error;
            this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
            this->GetOutputHandler()->WriteElementwise(dwrc.GetStepErrorIndicators(),
                                                       "Error_Indicators" + this->GetPostIndex(),
                                                       this->GetProblem()->GetDoFType());
          }//End first intervall

        GetU().SetTimeDoFNumber(local_to_global[0], it);
        u_old = GetU().GetSpacialVector();
        //Transfer to current mesh if needed!
        this->GetProblem()->GetSpaceTimeHandler()->TemporalMeshTransferState(u_old, local_to_global[0], local_to_global[1]);

        this->GetIntegrator().AddDomainData("last_time_state",&u_old);

        this->GetProblem()->SetTime(times[local_to_global[1]], local_to_global[1], it);
        GetU().SetTimeDoFNumber(local_to_global[1], it);
        this->GetOutputHandler()->SetIterationNumber(local_to_global[1], "Time");
        //dwrc.ReInit(); // ReInit should be outside of the time loop
        dwrc.SetTime(local_to_global[1]);

        //some output
        this->GetOutputHandler()->Write("Computing Error Indicators:",
                                        4 + this->GetBasePriority());

        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        //add the primal and (if needed) dual solution to the integrator
        this->GetIntegrator().AddDomainData("state",
                                            &(GetU().GetSpacialVector()));
        if (dwrc.NeedDual())
          this->GetIntegrator().AddDomainData("adjoint_for_ee",
                                              &(GetZForEE().GetSpacialVector()));
        AddUDD(local_to_global[1], it);

        //Check if some nodal values need to be precomputed, e.g., active set indicators
        // for the obstacle problem

        if ( need_precomputed_nodal_values != 0 )
          {
            for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
              {
                assert(aux_nodal_values[i] != NULL);
                aux_nodal_values[i]->SetTimeDoFNumber(local_to_global[1], it);
                {
                  //some output
                  std::stringstream tmp;
                  tmp << "Precomputation "<<i;
                  this->GetOutputHandler()->Write(tmp.str(),
                                                  4 + this->GetBasePriority());
                }


                //Calculate
                this->SetProblemType("aux_error",i);
                auto &problem = this->GetProblem()->GetErrorPrecomputations();

                this->GetIntegrator().ComputeNonlinearRhs(problem, aux_nodal_values[i]->GetSpacialVector());

                //Distribute for hanging nodes
                problem.GetDoFConstraints().distribute(aux_nodal_values[i]->GetSpacialVector());
                //output (in vtk files)
                this->GetOutputHandler()->Write(aux_nodal_values[i]->GetSpacialVector(),
                                                "Aux_Error_Indicators_"+i,"state");
                std::stringstream tmp;
                tmp << "aux_error_"<<i;
                this->GetIntegrator().AddDomainData(tmp.str(),&(aux_nodal_values[i]->GetSpacialVector()));

              }
          }

        this->SetProblemType("error_evaluation");

        //prepare the weights...
        dwrc.PrepareWeights(GetU(), GetZForEE());

        //now we finally compute the refinement indicators
        this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(),
                                                          dwrc);

        // release the lock on the refinement indicators (see dwrcontainer.h)
        dwrc.ReleaseLock();

        const double step_error = dwrc.GetStepError();

        // clear the data
        dwrc.ClearWeightData();
        this->GetIntegrator().DeleteDomainData("state");
        if (dwrc.NeedDual())
          this->GetIntegrator().DeleteDomainData("adjoint_for_ee");
        DeleteUDD();
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetIntegrator());
        this->GetIntegrator().DeleteDomainData("last_time_state");


        //Cleaning auxiliary nodal variables
        if ( need_precomputed_nodal_values != 0 )
          {
            for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
              {
                std::stringstream tmp;
                tmp << "aux_error_"<<i;
                this->GetIntegrator().DeleteDomainData(tmp.str());
              }
          }

        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << "Error estimate using " << dwrc.GetName();
        if (dwrc.NeedDual())
          out << " for the " << this->GetProblem()->GetFunctionalName();
        out << " at time "<< times[local_to_global[1]];
        out << ": " << step_error;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
        this->GetOutputHandler()->WriteElementwise(dwrc.GetStepErrorIndicators( ),
                                                   "Error_Indicators" + this->GetPostIndex(),
                                                   this->GetProblem()->GetDoFType());

      }//End of time loop
    //Cleaning auxiliary nodal variables
    if ( need_precomputed_nodal_values != 0 )
      {
        for (unsigned int i = 0; i < need_precomputed_nodal_values; i++)
          {
            delete aux_nodal_values[i];
            aux_nodal_values[i] = NULL;
          }
      }

  }


  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM, typename VECTOR,
           int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR,
       PROBLEM, VECTOR, dealdim>::ComputeTimeFunctionals(unsigned int step, unsigned int num_steps)
  {
    std::stringstream out;
    this->GetOutputHandler()->InitOut(out);
    out << "\t         Precalculating functional values ";
    this->GetOutputHandler()->Write(out, 5 + this->GetBasePriority());

    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

    this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector()));
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
                    "InstatPDEProblem::ComputeTimeFunctionals");
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
                  this->GetFunctionalValues()[i].push_back(ret);
                }
              else if (this->GetProblem()->GetFunctionalType().find("timedistributed") != std::string::npos)
                {
                  if (this->GetFunctionalValues()[i].size() != 1)
                    {
                      this->GetFunctionalValues()[i].resize(1);
                      this->GetFunctionalValues()[i][0] = 0.;
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
                  this->GetFunctionalValues()[i][0] += w * ret;
                }
              else
                {
                  throw DOpEException(
                    "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                    "InstatPDEProblem::ComputeTimeFunctionals");
                }
            }
        }
    }
    this->GetIntegrator().DeleteDomainData("state");
    this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  void
  InstatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::WriteToFile(
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

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  template<typename PDE>
  void InstatPDEProblem<NONLINEARSOLVER,
       INTEGRATOR, PROBLEM, VECTOR, dealdim>::
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
    {
      TimeIterator it =
        problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(times[local_to_global[0]], local_to_global[0], it,true);
      sol.SetTimeDoFNumber(local_to_global[0], it);
    }
    // Set u_old to initial_values
    this->GetProblem()->GetSpaceTimeHandler()->ReinitVector(u_old, DOpEtypes::state);
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
        TimeIterator it =
          problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
        // Functional evaluation in t_0
        AddUDD(local_to_global[0], it);
        ComputeTimeFunctionals(0,
                               max_timestep);
        DeleteUDD();
        this->SetProblemType("state");
      }

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

            //Only set Time for DoFHandler to have the correct unknowns,
            //DO NOT call the set Time method of the problem, to avoid
            //having the wrong function values!
            problem.GetSpaceTimeHandler()->SetInterval(it,local_to_global[i]);
            //Transfer old solution to current mesh
            bool transfer_needed = problem.GetSpaceTimeHandler()->TemporalMeshTransferState(u_old, local_to_global[i-1], local_to_global[i]);

            if (transfer_needed)
              {
                this->GetOutputHandler()->Write(u_old,
                                                "Transfered_" + outname + this->GetPostIndex(), problem.GetDoFType());
                this->GetNonlinearSolver("state").ReInit(problem);
                state_reinit_ = true; //We set reinit=true, so that in the next call,
                //reinit is called before handling the first time step.
                build_state_matrix_ = true;
              }

            std::stringstream out;
            this->GetOutputHandler()->InitOut(out);
            out << "\t Timestep";
            if (transfer_needed)
              {
                out <<" (MT): ";
              }
            else
              {
                out <<": ";
              }
            out << local_to_global[i] << " ("
                << times[local_to_global[i - 1]] << " -> " << time
                << ") using " << problem.GetName();
            problem.GetOutputHandler()->Write(out,
                                              4 + this->GetBasePriority());

            sol.SetTimeDoFNumber(local_to_global[i], it);
            sol.GetSpacialVector() = 0;

            if (transfer_needed)
              {
                //Auxiliary vectors need to be interpolated!
                this->GetProblem()->AddAuxiliaryToIntegratorWithTemporalTransfer(
                  this->GetIntegrator(),local_to_global[i-1], local_to_global[i]);
              }
            else
              {
                this->GetProblem()->AddAuxiliaryToIntegrator(
                  this->GetIntegrator());
              }

            this->GetNonlinearSolver("state").NonlinearLastTimeEvals(problem,
                                                                     u_old, sol.GetSpacialVector());
            this->GetOutputHandler()->Write(sol.GetSpacialVector(),
                                            "LastTimestep_" + outname + this->GetPostIndex(), problem.GetDoFType());
            if (transfer_needed)
              {
                this->GetProblem()->DeleteAuxiliaryFromIntegratorWithTemporalTransfer(
                  this->GetIntegrator());
              }
            else
              {
                this->GetProblem()->DeleteAuxiliaryFromIntegrator(
                  this->GetIntegrator());
              }

            problem.SetTime(time, local_to_global[i], it);

            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());
            if (transfer_needed)
              {
                this->GetProblem()->AddPreviousAuxiliaryToIntegratorWithTemporalTransfer(
                  this->GetIntegrator(),local_to_global[i-1], local_to_global[i]);
              }
            else
              {
                this->GetProblem()->AddPreviousAuxiliaryToIntegrator(
                  this->GetIntegrator());
              }
            //Also rebuild matrix if a mesh transfer happend.
            build_state_matrix_
              = this->GetNonlinearSolver("state").NonlinearSolve(problem,
                                                                 u_old, sol.GetSpacialVector(), true,
                                                                 build_state_matrix_);

            this->GetProblem()->DeleteAuxiliaryFromIntegrator(
              this->GetIntegrator());
            if (transfer_needed)
              {
                this->GetProblem()->DeletePreviousAuxiliaryFromIntegratorWithTemporalTransfer(
                  this->GetIntegrator());
              }
            else
              {
                this->GetProblem()->DeletePreviousAuxiliaryFromIntegrator(
                  this->GetIntegrator());
              }
            u_old = sol.GetSpacialVector();
            this->GetOutputHandler()->Write(sol.GetSpacialVector(),
                                            outname + this->GetPostIndex(), problem.GetDoFType());
            if (eval_funcs)
              {
                //Functional evaluation in t_n  //if condition to get the type
                AddUDD(local_to_global[i], it);
                ComputeTimeFunctionals(local_to_global[i], max_timestep);
                DeleteUDD();
                this->SetProblemType("state");
              }
          }
      }
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  template<typename PDE>
  void InstatPDEProblem<NONLINEARSOLVER,
       INTEGRATOR, PROBLEM, VECTOR, dealdim>::
       BackwardTimeLoop(PDE &problem, StateVector<VECTOR> &sol, std::string outname, bool eval_grads)
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
    this->GetProblem()->GetSpaceTimeHandler()->ReinitVector(u_old, DOpEtypes::state);
    // Projection of initial data
    this->GetOutputHandler()->SetIterationNumber(max_timestep, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
                                      4 + this->GetBasePriority());

      auto &initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
      this->GetProblem()->AddPreviousAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      build_state_matrix_ = this->GetNonlinearSolver("adjoint").NonlinearSolve_Initial(
                              initial_problem, u_old, true, true);
      build_state_matrix_ = true;

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      this->GetProblem()->DeletePreviousAuxiliaryFromIntegrator(this->GetIntegrator());
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

            //Only set Time for DoFHandler to have the correct unknowns,
            //DO NOT call the set Time method of the problem, to avoid
            //having the wrong function values!
            problem.GetSpaceTimeHandler()->SetInterval(it,local_to_global[j]);
            //Transfer old solution to current mesh
            bool transfer_needed = problem.GetSpaceTimeHandler()->TemporalMeshTransferState(u_old, local_to_global[j+1], local_to_global[j]);

            if (transfer_needed)
              {
                this->GetOutputHandler()->Write(u_old,
                                                "Transfered_" + outname + this->GetPostIndex(), problem.GetDoFType());
                this->GetNonlinearSolver("adjoint").ReInit(problem);
                adjoint_reinit_ = true; //We set reinit=true, so that in the next call,
                //reinit is called before handling the first time step.
                build_adjoint_matrix_ = true;
              }

            std::stringstream out;
            this->GetOutputHandler()->InitOut(out);
            out << "\t Timestep";
            if (transfer_needed)
              {
                out <<" (MT): ";
              }
            else
              {
                out <<": ";
              }
            out << local_to_global[j+1] << " ("
                << times[local_to_global[j + 1]] << " -> " << time
                << ") using " << problem.GetName();
            problem.GetOutputHandler()->Write(out,
                                              4 + this->GetBasePriority());

            sol.SetTimeDoFNumber(local_to_global[j], it);
            sol.GetSpacialVector() = 0;


            if (transfer_needed)
              {
                //Auxiliary vectors need to be interpolated!
                this->GetProblem()->AddAuxiliaryToIntegratorWithTemporalTransfer(
                  this->GetIntegrator(),local_to_global[j+1], local_to_global[j]);
              }
            else
              {
                this->GetProblem()->AddAuxiliaryToIntegrator(
                  this->GetIntegrator());
              }

            this->GetNonlinearSolver("adjoint").NonlinearLastTimeEvals(problem,
                                                                       u_old, sol.GetSpacialVector());

            if (transfer_needed)
              {
                this->GetProblem()->DeleteAuxiliaryFromIntegratorWithTemporalTransfer(
                  this->GetIntegrator());
              }
            else
              {
                this->GetProblem()->DeleteAuxiliaryFromIntegrator(
                  this->GetIntegrator());
              }

            problem.SetTime(time,local_to_global[j], it);

            this->GetProblem()->AddAuxiliaryToIntegrator(
              this->GetIntegrator());
            this->GetProblem()->AddNextAuxiliaryToIntegrator(
              this->GetIntegrator());
            if (local_to_global[j] != 0)
              this->GetProblem()->AddPreviousAuxiliaryToIntegrator(
                this->GetIntegrator());

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

            u_old = sol.GetSpacialVector();
            this->GetOutputHandler()->Write(sol.GetSpacialVector(),
                                            outname + this->GetPostIndex(), problem.GetDoFType());

          }//End interval loop
      }//End time loop
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER,
       INTEGRATOR, PROBLEM, VECTOR, dealdim>::
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
                                "InstatPDEProblem::AllocateAuxiliaryTimeParams");
          }
      }
  }

  /******************************************************/

  template<typename NONLINEARSOLVER,
           typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  std::map<std::string,std::vector<dealii::Vector<double> >>::iterator
                                                          InstatPDEProblem<NONLINEARSOLVER,
                                                                           INTEGRATOR, PROBLEM, VECTOR, dealdim>::
                                                                           GetAuxiliaryTimeParams(std::string name)

  {
    return auxiliary_time_params_.find(name);
  }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
           typename VECTOR, int dealdim>
  void InstatPDEProblem<NONLINEARSOLVER,
       INTEGRATOR, PROBLEM, VECTOR, dealdim>::
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
                            "InstatPDEProblem::CalculatePreFunctional");
      }
    if (name != "aux_functional")
      {
        throw DOpEException("Only valid with name `aux_functional` but not: "+name,
                            "InstatPDEProblem::CalculatePreFunctional");
      }
    if (postfix == "" || postfix == " ")
      {
        throw DOpEException("Postfix needs to be a non-empty string",
                            "InstatPDEProblem::CalculatePreFunctional");
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
          assert(false);
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
              "InstatPDEProblem::CalculatePreFunctional");
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
