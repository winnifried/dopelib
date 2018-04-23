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

#ifndef REDUCED_IPOPT__ALGORITHM_H_
#define REDUCED_IPOPT__ALGORITHM_H_

#include <opt_algorithms/reducedalgorithm.h>
#include <include/parameterreader.h>
#include <reducedproblems/ipopt_problem.h>

#ifdef DOPELIB_WITH_IPOPT
//Make shure the unused variable warnings from ipopt don't bother us
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "IpIpoptApplication.hpp"
#pragma GCC diagnostic pop
#endif

#include <iostream>
#include <assert.h>
#include <iomanip>

namespace DOpE
{
  /**
   * @class Reduced_IpoptAlgorithm
   *
   * This class provides a solver for constrained optimization
   * problems in reduced form, i.e., the dependent variable
   * given by an equality constraint is  assumed to be eliminated
   * by solving the equation. I.e.,
   * we solve the problem min j(q) s.t., a \le q \le b, g(q) \le 0
   *
   * The solution is done by interfacing to the IPOPT library.
   *
   * @tparam <PROBLEM>    The problem container. See, e.g., OptProblemContainer
   * @tparam <VECTOR>     The vector type of the solution.
   */
  template <typename PROBLEM, typename VECTOR>
  class Reduced_IpoptAlgorithm : public ReducedAlgorithm<PROBLEM, VECTOR>
  {
  public:
    /**
     * The constructor for the algorithm
     *
     * @param OP              A pointer to the problem container
     * @param S               The reduced problem. This object handles the equality
     *                        constraint. For the interface see ReducedProblemInterface.
     * @param param_reader    A parameter reader to access user given runtime parameters.
     * @param Except          The DOpEExceptionHandler. This is used to handle the output
     *                        by all exception.
     * @param Output          The DOpEOutputHandler. This takes care of all output
     *                        generated by the problem.
     * @param base_priority   An offset for the priority of the output generated by the algorithm.
     */
    Reduced_IpoptAlgorithm (PROBLEM *OP,
                            ReducedProblemInterface<PROBLEM, VECTOR> *S,
                            DOpEtypes::VectorStorageType vector_behavior,
                            ParameterReader &param_reader,
                            DOpEExceptionHandler<VECTOR> *Except = NULL,
                            DOpEOutputHandler<VECTOR> *Output = NULL,
                            int base_priority = 0);
    ~Reduced_IpoptAlgorithm ();

    /**
     * Used to declare run time parameters. This is needed to declare all
     * parameters a startup without the need for an object to be already
     * declared.
     */
    static void
    declare_params (ParameterReader &param_reader);

    /**
     * This solves an Optimizationproblem in only the control variable
     * using the commercial optimization library ipopt.
     * To use it you need to define the compiler flag
     * DOPELIB_WITH_IPOPT and ensure that all required ipopt headers and
     * libraries are within the path or otherwise known.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int
    Solve (ControlVector<VECTOR> &q,
           double global_tol = -1.);

  protected:

  private:
    std::string postindex_;
    DOpEtypes::VectorStorageType vector_behavior_;

    double tol_;
    bool capture_out_;
    std::string lin_solve_;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  void
  Reduced_IpoptAlgorithm<PROBLEM, VECTOR>::declare_params (ParameterReader &param_reader)
  {
    param_reader.SetSubsection ("reduced_ipoptalgorithm parameters");
    param_reader.declare_entry ("tol", "1.e-5", Patterns::Double (0, 1),
                                "Tolerance");
    param_reader.declare_entry ("capture ipopt output", "true",
                                Patterns::Bool (),
                                "Select if the ipopt output should be stored in log file");
    param_reader.declare_entry ("ipopt linsolve", "ma27",
                                Patterns::Selection ("ma27|ma57|ma77|ma86|pardiso|wsmp|mumps"),
                                "Linear Solver to be used in ipopt.");
    ReducedAlgorithm<PROBLEM, VECTOR>::declare_params (param_reader);
  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  Reduced_IpoptAlgorithm<PROBLEM, VECTOR>::Reduced_IpoptAlgorithm (PROBLEM *OP,
      ReducedProblemInterface<
      PROBLEM,
      VECTOR> *S,
      DOpEtypes::VectorStorageType vector_behavior,
      ParameterReader &param_reader,
      DOpEExceptionHandler<
      VECTOR> *Except,
      DOpEOutputHandler<
      VECTOR> *Output,
      int base_priority)
    : ReducedAlgorithm<PROBLEM, VECTOR> (OP, S, param_reader, Except,
                                         Output, base_priority)
  {

    param_reader.SetSubsection ("reduced_ipoptalgorithm parameters");

    tol_ = param_reader.get_double ("tol");
    capture_out_ = param_reader.get_bool ("capture ipopt output");
    lin_solve_ = param_reader.get_string ("ipopt linsolve");

    vector_behavior_ = vector_behavior;
    postindex_ = "_" + this->GetProblem ()->GetName ();

    DOpEtypes::ControlType ct =
      S->GetProblem ()->GetSpaceTimeHandler ()->GetControlType ();
    if ((ct != DOpEtypes::ControlType::initial) && (ct
                                                    != DOpEtypes::ControlType::stationary))
      {
        throw DOpEException (
          "The ControlType: " + DOpEtypesToString (ct)
          + " is not supported.",
          "Reduced_IpoptAlgorithm::Reduced_IpoptAlgorithm");
      }
  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  Reduced_IpoptAlgorithm<PROBLEM, VECTOR>::~Reduced_IpoptAlgorithm ()
  {

  }

  /******************************************************/

  template <typename PROBLEM, typename VECTOR>
  int
  Reduced_IpoptAlgorithm<PROBLEM, VECTOR>::Solve (ControlVector<VECTOR> &q,
                                                  double global_tol)
  {
#ifndef DOPELIB_WITH_IPOPT
    throw DOpEException (
      "To use this algorithm you need to have IPOPT installed! To use this set the DOPELIB_WITH_IPOPT CompilerFlag.",
      "Reduced_IpoptAlgorithm::Solve");
#else
    q.ReInit();

    ControlVector<VECTOR> dq(q);
    ControlVector<VECTOR> q_min(q), q_max(q);
    this->GetReducedProblem()->GetControlBoxConstraints(q_min,q_max);

    ConstraintVector<VECTOR> constraints(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),vector_behavior_);

    unsigned int iter=0;
    double cost=0.;
    double cost_start=0.;
    std::stringstream out;
    this->GetOutputHandler()->InitNewtonOut(out);
    global_tol = std::max(tol_,global_tol);

    out << "**************************************************\n";
    out << "*        Starting Solution using IPOPT           *\n";
    out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t*\n";
    out << "*  CDoFs : ";
    q.PrintInfos(out);
    out << "*  SDoFs : ";
    this->GetReducedProblem()->StateSizeInfo(out);
    out << "*  Constraints : ";
    constraints.PrintInfos(out);
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);

    this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Ipopt"+postindex_);

    this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");

    try
      {
        cost_start = cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_IpoptAlgorithm::Solve");
      }

    this->GetOutputHandler()->InitOut(out);
    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
    this->GetOutputHandler()->InitNewtonOut(out);

    /////////////////////////////////DO SOMETHING to Solve.../////////////////////////
    out<<"************************************************\n";
    out<<"*               Calling IPOPT                  *\n";
    if (capture_out_)
      out<<"*  output will be written to logfile only!     *\n";
    else
      out<<"*  output will not be written to logfile!      *\n";
    out<<"************************************************\n\n";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority());

    this->GetOutputHandler()->DisallowAllOutput();
    if (capture_out_)
      this->GetOutputHandler()->StartSaveCTypeOutputToLog();

    int ret_val = -1;
    {
      // Create a new instance of your nlp
      //  (use a SmartPtr, not raw)
      //Ipopt_Problem<ReducedProblemInterface<PROBLEM,VECTOR, dopedim,dealdim>,VECTOR>
      //  ip_prob((this->GetReducedProblem()),q,&q_min,&q_max,constraints);
      Ipopt::SmartPtr<Ipopt::TNLP> mynlp = new
      Ipopt_Problem<ReducedProblemInterface<PROBLEM,VECTOR>,VECTOR>(
        ret_val, this->GetReducedProblem(),q,&q_min,&q_max,constraints);

      //Ipopt::IpoptApplication app;
      Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
      // Change some options
      // Note: The following choices are only examples, they might not be
      //       suitable for your optimization problem.
      app->Options()->SetNumericValue("tol", tol_);
      app->Options()->SetStringValue("mu_strategy", "adaptive");
      app->Options()->SetStringValue("output_file", this->GetOutputHandler()->GetResultsDir()+"ipopt.out");
      app->Options()->SetStringValue("linear_solver", lin_solve_);
      app->Options()->SetStringValue("hessian_approximation","limited-memory");

      // Intialize the IpoptApplication and process the options
      Ipopt::ApplicationReturnStatus status;
      status = app->Initialize();
      if (status != Ipopt::Solve_Succeeded)
        {
          this->GetOutputHandler()->Write("\n\n*** Error during initialization!\n",1+this->GetBasePriority());
          abort();
        }

      // Ask Ipopt to solve the problem
      status = app->OptimizeTNLP(mynlp);
    }

    if (capture_out_)
      this->GetOutputHandler()->StopSaveCTypeOutputToLog();
    this->GetOutputHandler()->ResumeOutput();
    out<<"\n************************************************\n";
    out<<"*               IPOPT Finished                 *\n";
    out<<"*          with Exit Code: "<<std::setw(3)<<ret_val;
    if (ret_val == 1)
      out<<" (success)       *\n";
    else
      out<<" (unknown error: "<<ret_val<<") *\n";
    out<<"************************************************\n";
    //FIXME What is the result...
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority());

    iter++;
    this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Ipopt"+postindex_);

    this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");
    try
      {
        cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_IpoptAlgorithm::Solve");
      }
    //We are done write total evaluation
    this->GetOutputHandler()->InitOut(out);
    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
    this->GetOutputHandler()->InitNewtonOut(out);
    try
      {
        this->GetReducedProblem()->ComputeReducedFunctionals(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_IpoptAlgorithm::Solve");
      }

    out << "**************************************************\n";
    out << "*        Stopping Solution Using IPOPT           *\n";
    out << "*             Relative reduction in cost functional:"<<std::scientific << std::setw(11) << this->GetOutputHandler()->ZeroTolerance((cost-cost_start)/fabs(0.5*(cost_start+cost)),1.0) <<"          *\n";
    out.precision(7);
    out << "*             Final value: "<<cost<<"                                     *\n";
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
    return iter;
#endif //Endof ifdef DOPELIB_WITH_IPOPT
  }

  /*****************************END_OF_NAMESPACE_DOpE*********************************/
}
#endif
