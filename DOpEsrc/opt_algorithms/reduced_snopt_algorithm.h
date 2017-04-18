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

#ifndef REDUCED_SNOPT__ALGORITHM_H_
#define REDUCED_SNOPT__ALGORITHM_H_

#include <opt_algorithms/reducedalgorithm.h>
#include <include/parameterreader.h>
#include <basic/dopetypes.h>

#include <wrapper/snopt_wrapper.h>

#include <iostream>
#include <assert.h>
#include <iomanip>

namespace DOpE
{
  /**
   * @class Reduced_SnoptAlgorithm
   *
   * This class provides a solver for constrained optimization
   * problems in reduced form, i.e., the dependent variable
   * given by an equality constraint is  assumed to be eliminated
   * by solving the equation. I.e.,
   * we solve the problem min j(q) s.t., a \le q \le b, g(q) \le 0
   *
   * The solution is done by interfacing to the SNOPT library.
   *
   * @tparam <PROBLEM>    The problem container. See, e.g., OptProblemContainer
   * @tparam <VECTOR>     The vector type of the solution.
   */
  template<typename PROBLEM, typename VECTOR>
  class Reduced_SnoptAlgorithm : public ReducedAlgorithm<PROBLEM, VECTOR>
  {
  public:
    Reduced_SnoptAlgorithm(PROBLEM *OP,
                           ReducedProblemInterface<PROBLEM, VECTOR> *S,
                           DOpEtypes::VectorStorageType vector_behavior, ParameterReader &param_reader,
                           DOpEExceptionHandler<VECTOR> *Except = NULL,
                           DOpEOutputHandler<VECTOR> *Output = NULL, int base_priority = 0);
    ~Reduced_SnoptAlgorithm();

    /**
     * Used to declare run time parameters. This is needed to declare all
     * parameters a startup without the need for an object to be already
     * declared.
     */
    static void
    declare_params(ParameterReader &param_reader);

    /**
     * This solves an Optimizationproblem in only the control variable
     * using the commercial optimization library snopt.
     * To use it you need to define the compiler flag
     * DOPELIB_WITH_SNOPT and ensure that all required snopt headers and
     * libraries are within the path or otherwise known.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int
    Solve(ControlVector<VECTOR> &q, double global_tol = -1.);

  protected:

  private:
#ifdef DOPELIB_WITH_SNOPT
    int rsa_func_(DOpEWrapper::SNOPT_FUNC_DATA &data);
#endif
    std::string postindex_;
    DOpEtypes::VectorStorageType vector_behavior_;

    double func_prec_, feas_tol_, opt_tol_;
    int max_inner_iter_, max_outer_iter_;
    bool capture_out_;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

  template<typename PROBLEM, typename VECTOR>
  void
  Reduced_SnoptAlgorithm<PROBLEM, VECTOR>::declare_params(
    ParameterReader &param_reader)
  {
    param_reader.SetSubsection("reduced_snoptalgorithm parameters");
    param_reader.declare_entry("function precision", "1.e-6",
                               Patterns::Double(0, 1),
                               "Declares how many digits we assume to have computed correctly, this should correspond to the tolerance used for the PDE solve");
    param_reader.declare_entry("feasibility tol", "1.e-5",
                               Patterns::Double(0, 1),
                               "Tolerance with respect to the feasibility of the constraints.");
    param_reader.declare_entry("optimality tol", "1.e-5",
                               Patterns::Double(0, 1),
                               "Tolerance with respect to the optimality condition.");
    param_reader.declare_entry("max inner iterations", "500",
                               Patterns::Integer(0),
                               "Maximal allowed number of inner iterations over all outer iterations");
    param_reader.declare_entry("max iterations", "1000", Patterns::Integer(0),
                               "Maximal allowed number of outer iterations over all outer iterations");
    param_reader.declare_entry("capture snopt output", "true",
                               Patterns::Bool(),
                               "Select if the snopt output should be stored in log file");
    ReducedAlgorithm<PROBLEM, VECTOR>::declare_params(param_reader);
  }

  /******************************************************/

  template<typename PROBLEM, typename VECTOR>
  Reduced_SnoptAlgorithm<PROBLEM, VECTOR>::Reduced_SnoptAlgorithm(PROBLEM *OP,
      ReducedProblemInterface<PROBLEM, VECTOR> *S,
      DOpEtypes::VectorStorageType vector_behavior, ParameterReader &param_reader,
      DOpEExceptionHandler<VECTOR> *Except, DOpEOutputHandler<VECTOR> *Output,
      int base_priority) :
    ReducedAlgorithm<PROBLEM, VECTOR>(OP, S, param_reader, Except, Output,
                                      base_priority)
  {

    param_reader.SetSubsection("reduced_snoptalgorithm parameters");

    func_prec_ = param_reader.get_double("function precision");
    feas_tol_ = param_reader.get_double("feasibility tol");
    opt_tol_ = param_reader.get_double("optimality tol");
    max_inner_iter_ = param_reader.get_integer("max inner iterations");
    max_outer_iter_ = param_reader.get_integer("max iterations");
    capture_out_ = param_reader.get_bool("capture snopt output");

    vector_behavior_ = vector_behavior;
    postindex_ = "_" + this->GetProblem()->GetName();

    DOpEtypes::ControlType ct = S->GetProblem()->GetSpaceTimeHandler()->GetControlType();
    if ((ct != DOpEtypes::ControlType::initial) && (ct != DOpEtypes::ControlType::stationary))
      {
        throw DOpEException("The ControlType: "+ DOpEtypesToString(ct) + " is not supported.",
                            "Reduced_SnoptAlgorithm::Reduced_SnoptAlgorithm");
      }
  }

  /******************************************************/

  template<typename PROBLEM, typename VECTOR>
  Reduced_SnoptAlgorithm<PROBLEM, VECTOR>::~Reduced_SnoptAlgorithm()
  {

  }

  /******************************************************/

  template<typename PROBLEM, typename VECTOR>
  int
  Reduced_SnoptAlgorithm<PROBLEM, VECTOR>::Solve(ControlVector<VECTOR> &q,
                                                 double global_tol)
  {
#ifndef DOPELIB_WITH_SNOPT
    throw DOpEException("To use this algorithm you need to have SNOPT installed! To use this set the DOPELIB_WITH_SNOPT CompilerFlag.","Reduced_SnoptAlgorithm::Solve");
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
    global_tol = std::max(opt_tol_,global_tol);

    out << "**************************************************\n";
    out << "*        Starting Solution using SNOPT           *\n";
    out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t*\n";
    out << "*  CDoFs : ";
    q.PrintInfos(out);
    out << "*  SDoFs : ";
    this->GetReducedProblem()->StateSizeInfo(out);
    out << "*  Constraints : ";
    constraints.PrintInfos(out);
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);

    this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Snopt"+postindex_);

    this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");

    try
      {
        cost_start = cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_SnoptAlgorithm::Solve");
      }

    this->GetOutputHandler()->InitOut(out);
    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
    this->GetOutputHandler()->InitNewtonOut(out);

    /////////////////////////////////DO SOMETHING to Solve.../////////////////////////
    out<<"************************************************\n";
    out<<"*               Calling SNOPT                  *\n";
    if (capture_out_)
      out<<"*  output will be written to logfile only!     *\n";
    else
      out<<"*  output will not be written to logfile!      *\n";
    out<<"************************************************\n\n";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority());

    this->GetOutputHandler()->DisallowAllOutput();
    if (capture_out_)
      this->GetOutputHandler()->StartSaveCTypeOutputToLog();

    int ret_val;
    {
      DOpEWrapper::SNOPT_Problem RSAProb;

      integer n=q.GetSpacialVector().size();
      integer neF =1+constraints.GetGlobalConstraints().size();
      integer lenA=0;
      integer neA=0;
      integer *iAfun = NULL;
      integer *jAvar = NULL;
      doublereal *A = NULL;

      integer lenG = n*(neF); //We have that the derivative of J and the global constraints
      //are nonzero w.r.t. all components.
      integer neG = lenG;     //Predefine the number of valid entries in iGfun and jGvar will be lenG.
      integer *iGfun = new integer[lenG];
      integer *jGvar = new integer[lenG];

      doublereal *x = new doublereal[n];
      doublereal *xlow = new doublereal[n];
      doublereal *xupp = new doublereal[n];
      doublereal *xmul = new doublereal[n];
      integer *xstate = new integer[n];

      doublereal *F = new doublereal[neF];
      doublereal *Flow = new doublereal[neF];
      doublereal *Fupp = new doublereal[neF];
      doublereal *Fmul = new doublereal[neF];
      integer *Fstate = new integer[neF];

      integer nxnames = 1;
      integer nFnames = 1;
      char *xnames = new char[nxnames*8];
      char *Fnames = new char[nFnames*8];

      integer ObjRow = 0;
      doublereal ObjAdd = 0;

      {
        const VECTOR &gv_q = q.GetSpacialVector();
        const VECTOR &gv_qmin = q_min.GetSpacialVector();
        const VECTOR &gv_qmax = q_max.GetSpacialVector();

        for (unsigned int i=0; i < n; i++)
          {
            x[i] = gv_q(i);
            xlow[i] = gv_qmin(i);
            xupp[i] = gv_qmax(i);
            xstate[i] = 0;
            iGfun[i] = 0;
            jGvar[i] = i;
          }
        Flow[0] = -1.e+20;
        Fupp[0] = 1.e+20;
        Fstate[0] = 0;
        for (unsigned int j = 1; j<neF; j++)
          {
            //Global constraints are to be given by the user such that
            //The feasible region is given by <= 0
            //TODO this should be defineable by the user...
            Flow[j] = -1.e+20;
            Fupp[j] = 0.;
            Fstate[j] = 0;
            for (unsigned int i=0; i < n; i++)
              {
                iGfun[n*j+i] = j;
                jGvar[n*j+i] = i;
              }
          }
      }

      //RSAProb.setPrintFile  ( "RSA.out" );
      RSAProb.setProblemSize( n, neF );
      RSAProb.setObjective ( ObjRow, ObjAdd );
      RSAProb.setA ( lenA, iAfun, jAvar, A );
      RSAProb.setG ( lenG, iGfun, jGvar );
      RSAProb.setX ( x, xlow, xupp, xmul, xstate );
      RSAProb.setF ( F, Flow, Fupp, Fmul, Fstate );
      RSAProb.setXNames ( xnames, nxnames );
      RSAProb.setFNames ( Fnames, nFnames );
      RSAProb.setProbName ( "RSA" );
      RSAProb.setNeA ( neA );
      RSAProb.setNeG ( neG );

      DOpEWrapper::SNOPT_A_userfunc_interface = boost::bind<int>(boost::mem_fn(&Reduced_SnoptAlgorithm<PROBLEM, VECTOR>::rsa_func_),boost::ref(*this),_1);
      RSAProb.setUserFun ( DOpEWrapper::SNOPT_A_userfunc_ );

      RSAProb.setIntParameter( "Derivative option", 1 );
      RSAProb.setRealParameter( "Function precison", func_prec_);
      RSAProb.setRealParameter( "Major optimality tolerance", global_tol);
      RSAProb.setRealParameter( "Major feasibility tolerance", feas_tol_);
      RSAProb.setRealParameter( "Minor feasibility tolerance", feas_tol_);
      RSAProb.setIntParameter( "Minor iterations limit", max_inner_iter_);
      RSAProb.setIntParameter( "Major iterations limit", max_outer_iter_);
      RSAProb.solve(2);
      ret_val = RSAProb.GetReturnStatus();
      {
        VECTOR &gv_q = q.GetSpacialVector();
        for (unsigned int i=0; i < gv_q.size(); i++)
          gv_q(i) = x[i];
      }
      if (iAfun != NULL)
        delete []iAfun;
      if (jAvar != NULL)
        delete []jAvar;
      if (A != NULL)
        delete []A;
      delete []iGfun;
      delete []jGvar;

      delete []x;
      delete []xlow;
      delete []xupp;
      delete []xmul;
      delete []xstate;

      delete []F;
      delete []Flow;
      delete []Fupp;
      delete []Fmul;
      delete []Fstate;

      delete []xnames;
      delete []Fnames;

    }

    if (capture_out_)
      this->GetOutputHandler()->StopSaveCTypeOutputToLog();
    this->GetOutputHandler()->ResumeOutput();
    out<<"\n************************************************\n";
    out<<"*               SNOPT Finished                 *\n";
    out<<"*          with Exit Code: "<<std::setw(3)<<ret_val;
    if (ret_val == 1)
      out<<" (success)       *\n";
    else
      out<<" (unknown error) *\n";
    out<<"************************************************\n";
    //FIXME What is the result...
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority());

    iter++;
    this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Snopt"+postindex_);

    this->GetOutputHandler()->Write(q,"Control"+postindex_,"control");
    try
      {
        cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      }
    catch (DOpEException &e)
      {
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_SnoptAlgorithm::Solve");
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
        this->GetExceptionHandler()->HandleCriticalException(e,"Reduced_SnoptAlgorithm::Solve");
      }

    out << "**************************************************\n";
    out << "*        Stopping Solution Using SNOPT           *\n";
    out << "*             Relative reduction in cost functional:"<<std::scientific << std::setw(11) << this->GetOutputHandler()->ZeroTolerance((cost-cost_start)/fabs(0.5*(cost_start+cost)),1.0) <<"          *\n";
    out.precision(7);
    out << "*             Final value: "<<cost<<"                                     *\n";
    out << "**************************************************";
    this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
    return iter;
#endif //Endof ifdef WITHSNOPT
  }

  /******************************************************/

#ifdef DOPELIB_WITH_SNOPT
  template <typename PROBLEM, typename VECTOR>
  int Reduced_SnoptAlgorithm<PROBLEM, VECTOR>
  ::rsa_func_(DOpEWrapper::SNOPT_FUNC_DATA &data )
  {
    //Needs to implement the evaluation of j and its derivative using the
    //Interface required by SNOPT
    ConstraintVector<VECTOR> constraints(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),vector_behavior_);
    ControlVector<VECTOR> tmp(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),vector_behavior_);
    VECTOR &ref_x = tmp.GetSpacialVector();
    assert((int) ref_x.size() ==  *(data.n));
    for (unsigned int i=0; i < ref_x.size(); i++)
      ref_x(i) = (data.x)[i];

    assert(*(data.needF) > 0);
    try
      {
        (data.F)[0] = this->GetReducedProblem()->ComputeReducedCostFunctional(tmp);
      }
    catch (DOpEException &e)
      {
        *(data.Status) = -1;
        this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
        return -1;
      }

    if (*(data.neF) > 1)
      {
        try
          {
            this->GetReducedProblem()->ComputeReducedConstraints(tmp,constraints);
          }
        catch (DOpEException &e)
          {
            *(data.Status) = -1;
            this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
            return -1;
          }
        const dealii::Vector<double> &gc = constraints.GetGlobalConstraints();
        assert(*(data.neF) == (int) gc.size()+1);
        for (unsigned int i=0; i < gc.size(); i++)
          {
            (data.F)[i+1] = gc(i);
          }
      }

    if (*(data.needG) > 0)
      {
        ControlVector<VECTOR> gradient(tmp);
        ControlVector<VECTOR> gradient_transposed(tmp);

        try
          {
            this->GetReducedProblem()->ComputeReducedGradient(tmp,gradient,gradient_transposed);
          }
        catch (DOpEException &e)
          {
            *(data.Status) = -2;
            this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
            return -1;
          }
        assert(*(data.neG) == *(data.n) **(data.neF));
        const VECTOR &ref_g = gradient_transposed.GetSpacialVector();
        for (unsigned int i=0; i < *(data.n); i++)
          {
            (data.G)[i] = ref_g(i);
          }
        //Evaluate global constraint gradients
        const dealii::Vector<double> &gc = constraints.GetGlobalConstraints();
        for (unsigned int j=0; j < gc.size(); j++)
          {
            try
              {
                this->GetReducedProblem()->ComputeReducedGradientOfGlobalConstraints(j,tmp,constraints,gradient,gradient_transposed);
              }
            catch (DOpEException &e)
              {
                *(data.Status) = -2;
                this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
                return -1;
              }
            const VECTOR &ref_g = gradient_transposed.GetSpacialVector();
            for (unsigned int i=0; i < *(data.n); i++)
              {
                (data.G)[*(data.n)*(j+1)+i] = ref_g(i);
              }
          }
      }

    *(data.Status) = 0;
    return 0;
  }
#endif //Endof ifdef WITHSNOPT
  /*****************************END_OF_NAMESPACE_DOpE*********************************/
}
#endif
