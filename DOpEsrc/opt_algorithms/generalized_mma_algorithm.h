#ifndef _GENERALIZED_MMA__ALGORITHM2_H_
#define _GENERALIZED_MMA__ALGORITHM2_H_

#include "reducedalgorithm.h"
#include "parameterreader.h"
#include "constraintvector.h"
#include "augmentedlagrangianproblem.h"
#include "reducednewtonalgorithmwithinverse.h"
#include "reducedtrustregionnewton.h"

#include <iostream>
#include <assert.h>
#include <iomanip>

namespace DOpE
{
  /**
   * This implements a generalized version of the MMA method as proposed in
   * Stingl, Kocvara, Leugering:  `A sequential convex semidefinite programming 
   * algorithm with an application to multiple-load free material optimization',
   * SIAM J. Optim. 20(1) (2009)
   * localdim corresponds to the size of the matrices that are localy constraint
   *
   * with Additional rules for parameter updates from the PENNON Algorithm
   * Given in the dissertation of M. Stingl
   * 'On the Solution of Nonlinear Semidefinite Programs by Augmented Lagrangian 
   *  Methods' 
   * The update for the asymptotes follows See K. Svanberg, MMA and GCMMA, 2007
   */
  template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR, typename SOLVER, int dopedim,  int dealdim, int localdim>
    class GeneralizedMMAAlgorithm : public ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>
  {
  public:
    GeneralizedMMAAlgorithm(PROBLEM* OP,
			     CONSTRAINTACCESSOR* CA, 
			     ReducedProblemInterface<PROBLEM,VECTOR,dopedim,dealdim>* S,
			     std::string vector_behavior, 
			     ParameterReader &param_reader,
			     INTEGRATORDATACONT& idc,
			     DOpEExceptionHandler<VECTOR>* Except=NULL,
			     DOpEOutputHandler<VECTOR>* Output=NULL);
    ~GeneralizedMMAAlgorithm() {}

    static void declare_params(ParameterReader &param_reader);

    void ReInit() 
    {  
      ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>::ReInit();
      _sub_problem_opt_alg.ReInit();
    }


    /**
     * This applies Alg 4.3 of 
     * Stingl, Kocvara, Leugering:  `A sequential convex semidefinite programming 
     * algorithm with an application to multiple-load free material optimization',
     * SIAM J. Optim. 20(1) (2009)
     */
    int Solve(ControlVector<VECTOR>& q, double global_tol =-1.);
   
  protected:
    /**
     * Solves the MMA-Subproblem using an augmented Lagrangian Method
     * with given bounds and asymptotes
     *
     */
    int SolveMMASubProblem(ControlVector<VECTOR>& dq, 
			   const ControlVector<VECTOR>& q,
			   const ControlVector<VECTOR>& lb,
			   const ControlVector<VECTOR>& ub,
			   const ControlVector<VECTOR>& q_min,
			   const ControlVector<VECTOR>& q_max,
			   const ControlVector<VECTOR>& gradient,
			   ConstraintVector<VECTOR>& mult,
			   double J,
			   double& prediction,
			   double global_tol);

    /**
     * This solves the seperable hyperbolic aopproximation by Alg 5.1 of 
     * Stingl, Kocvara, Leugering:  `A sequential convex semidefinite programming 
     * algorithm with an application to multiple-load free material optimization',
     * SIAM J. Optim. 20(1) (2009)
     * with Additional rules for parameter updates from the PENNON Algorithm
     * Given in the dissertation of M. Stingl
     * 'On the Solution of Nonlinear Semidefinite Programs by Augmented Lagrangian 
     *  Methods' 
     */
    int SolveSCSDPSubProblem(const ControlVector<VECTOR>& q,
			     const ControlVector<VECTOR>& gradient,
			     const ControlVector<VECTOR>& lb,
			     const ControlVector<VECTOR>& ub,
			     const ControlVector<VECTOR>& q_min,
			     const ControlVector<VECTOR>& q_max,
			     ControlVector<VECTOR>& dq,
			     ConstraintVector<VECTOR>& dm,
			     ConstraintVector<VECTOR>& constraints,
			     double alpha,
			     double J);
    int FindStationaryPointOfAugmentedLagrangian(const ControlVector<VECTOR>& q,
						 const ControlVector<VECTOR>& gradient,
						 const ControlVector<VECTOR>& lb,
						 const ControlVector<VECTOR>& ub,
						 const ControlVector<VECTOR>& q_min,
						 const ControlVector<VECTOR>& q_max,
						 const ConstraintVector<VECTOR>& dm,
						 ControlVector<VECTOR>& dq,
						 double alpha,
						 double J);
    
    double AugmentedLagrangianResidual(const ControlVector<VECTOR>& q,
				       const ControlVector<VECTOR>& gradient,
				       const ControlVector<VECTOR>& lb,
				       const ControlVector<VECTOR>& ub,
				       const ControlVector<VECTOR>& q_min,
				       const ControlVector<VECTOR>& q_max,
				       const ConstraintVector<VECTOR>& dm,
				       const ControlVector<VECTOR>& dq,
				       double  J);

    int OuterLineSearch(const ControlVector<VECTOR>& dq,
			double& cost, 
			ControlVector<VECTOR>& q,
			const ControlVector<VECTOR>& model_q, 
			const ControlVector<VECTOR>& lb,
			const ControlVector<VECTOR>& ub,
			const ControlVector<VECTOR>& q_min,
			const ControlVector<VECTOR>& q_max,
			const ControlVector<VECTOR>& gradient,
			const ConstraintVector<VECTOR>& dm,
			ConstraintVector<VECTOR>& constraints,
			double J);
    int MultiplierLineSearch(const ConstraintVector<VECTOR>& dm,
			     ConstraintVector<VECTOR>& m,
                             double scale); 

    double ComputeModelValue(const ControlVector<VECTOR>& dq,//The current Value
			     const ControlVector<VECTOR>& q, //All else defines the model...
			     const ControlVector<VECTOR>& gradient,
			     const ControlVector<VECTOR>& lb,
			     const ControlVector<VECTOR>& ub,
			     const ControlVector<VECTOR>& q_min,
			     const ControlVector<VECTOR>& q_max,
			     const ConstraintVector<VECTOR>& dm,
			     const ConstraintVector<VECTOR>& constraints,
			     double  J);
  private:
    
     unsigned int _line_maxiter, _mma_outer_maxiter, _mma_inner_maxiter;
     double       _linesearch_rho, _linesearch_c, _mma_global_tol, _ndofs;
     double _merit_multiplier, _rho, _p, _initial_lagrange_mult_scale;

     ConstraintVector<VECTOR> _mma_constraints, _mma_multiplier;
     CONSTRAINTACCESSOR& _CA;
     AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH, PROBLEM,dopedim,dealdim,localdim> _augmented_lagrangian_problem;
     SOLVER _augmented_lagrangian_solver;
     ReducedNewtonAlgorithmWithInverse<AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH, PROBLEM,dopedim,dealdim,localdim>,VECTOR,dopedim,dealdim> _sub_problem_opt_alg;
  };

/**************************************************Implementation*********************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM, typename VECTOR, typename SOLVER,int dopedim,int dealdim, int localdim>
void GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR,SOLVER,dopedim, dealdim, localdim>::declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("MMAalgorithm parameters");
  param_reader.declare_entry("line_maxiter", "4",Patterns::Integer(0));
  param_reader.declare_entry("linesearch_rho", "0.9",Patterns::Double(0));
  param_reader.declare_entry("linesearch_c", "0.1",Patterns::Double(0));

  param_reader.declare_entry("mma_outer_maxiter", "20", Patterns::Integer(0)); 
  param_reader.declare_entry("mma_inner_maxiter", "20", Patterns::Integer(0)); 
  param_reader.declare_entry("mma_global_tol", "1.e-8", Patterns::Double(0)); 

  ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>::declare_params(param_reader);
  ReducedNewtonAlgorithmWithInverse<AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH, PROBLEM,dopedim,dealdim,localdim>,VECTOR,dopedim,dealdim>::declare_params(param_reader);
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
  ::GeneralizedMMAAlgorithm(PROBLEM* OP, 
			    CONSTRAINTACCESSOR* CA, 
			    ReducedProblemInterface<PROBLEM,VECTOR,dopedim,dealdim>* S,
			    std::string vector_behavior,
			    ParameterReader &param_reader,
			    INTEGRATORDATACONT& idc,
			     DOpEExceptionHandler<VECTOR>* Except,
			    DOpEOutputHandler<VECTOR>* Output)
  : ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>(OP,S,param_reader,Except,Output),
  _mma_constraints(OP->GetSpaceTimeHandler(),vector_behavior),
  _mma_multiplier(OP->GetSpaceTimeHandler(),vector_behavior),
  _CA(*CA),
  _augmented_lagrangian_problem(*OP,*CA),
  _augmented_lagrangian_solver(&_augmented_lagrangian_problem,vector_behavior,param_reader,idc,5),
  _sub_problem_opt_alg(&_augmented_lagrangian_problem,&_augmented_lagrangian_solver,param_reader,this->GetExceptionHandler(),this->GetOutputHandler(),5)
{
  param_reader.SetSubsection("MMAalgorithm parameters");
  _line_maxiter         = param_reader.get_integer ("line_maxiter");
  _linesearch_rho       = param_reader.get_double ("linesearch_rho");
  _linesearch_c         = param_reader.get_double ("linesearch_c");

  _mma_outer_maxiter    = param_reader.get_integer ("mma_outer_maxiter");
  _mma_inner_maxiter    = param_reader.get_integer ("mma_inner_maxiter");
  _mma_global_tol = param_reader.get_double  ("mma_global_tol");

  _augmented_lagrangian_problem.SetValue(1.e-1,"p");

  _merit_multiplier = 1.;
  _initial_lagrange_mult_scale = 1.;
  _rho = 1.e-8;
}
/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim,localdim>::Solve(ControlVector<VECTOR>& q, double /*global_tol*/)
{
  _mma_constraints.ReInit();
  _mma_multiplier.ReInit();
  _mma_multiplier = 0; 

  _p = 0.1;

  _augmented_lagrangian_problem.SetValue(_p,"p");

  q.ReInit();

  ControlVector<VECTOR> dq(q), gradient(q), gradient_transposed(q), q_pre(q), q_pre_pre(q);
  ConstraintVector<VECTOR> dm(_mma_multiplier);
  ControlVector<VECTOR> lb(q), ub(q),q_min_recent(q), q_max_recent(q),q_min(q), q_max(q), sigma(q);
  ControlVector<VECTOR> tmp(q);

  this->GetReducedProblem()->GetControlBoxConstraints(q_min,q_max);

  q_pre = q;
  q_pre_pre = q;

  double cost=0.;
  double cost_alt=0.;
  double cost_start=0.;
  std::stringstream out;
  this->GetOutputHandler()->InitOut(out);

  bool feasible = true;
  double rho;

  dq = 1.;
  _ndofs = dq.Norm("l1");

  out << "**************************************************\n";
  out << "*        Starting MMA Algorithm       *\n";
  out << "*  CDoFs       : ";
  q.PrintInfos(out);
  out << "*  SDoFs       : ";
  this->GetReducedProblem()->StateSizeInfo(out);
  out << "*  Constraints : ";
  _mma_constraints.PrintInfos(out);
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1,1,1);
  
  try
  {
     cost_start=cost_alt=cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
  }
  out<< "CostFunctional: " << cost;
  this->GetOutputHandler()->Write(out,2);

  try
  {
    this->GetReducedProblem()->ComputeReducedFunctionals(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
  }

  try
  {
    this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
  } 

  try
  {
    this->GetReducedProblem()->ComputeReducedConstraints(q,_mma_constraints);
    //if(!this->GetReducedProblem()->ComputeReducedConstraints(q,_constraints))
      //throw DOpEException("Infeasible starting value!","GeneralizedMMAAlgorithm::Solve");
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
  }

  //Init Local Bounds and Asymptotes
  { // See K. Svanberg, MMA and GCMMA, 2007
    sigma = q_max;
    sigma.add(-1.,q_min);
    sigma *= 0.5;
    lb = q;
    lb.add(-1.,sigma);
    ub = q;
    ub.add(1.,sigma);
    
    q_min_recent.equ(1.,q);
    q_min_recent.add(-0.9,sigma);
    q_max_recent.equ(1.,q);
    q_max_recent.add(0.9,sigma);
    q_min_recent.max(q_min);
    q_max_recent.min(q_max);
  }

  _augmented_lagrangian_problem.InitMultiplier(_mma_multiplier,gradient);
  //All values initialized.
  unsigned int iter=0;
  unsigned int n_inner_loops=0;
  unsigned int n_total_inner_iters=0;
  
  //Compute the initial Residual. 
  double kkt_error = 0.;
  double kkt_error_initial = 0.;
  double accuracy = 0.;
  {//Compute _rho
    rho = 1.;
    _augmented_lagrangian_problem.SetValue(rho,"rho");
  }
  {
    double complementarity_error = this->GetReducedProblem()->Complementarity(_mma_multiplier,_mma_constraints);
    complementarity_error *= 1./_ndofs;
    double stationarity_error = AugmentedLagrangianResidual(q,gradient,lb,ub,q_min,q_max,_mma_multiplier,q,cost);
    double feasibility_error = this->GetReducedProblem()->GetMaxViolation(_mma_constraints);
    out << "MMA-Outer (0) - [NA/NA/NA] - CE: "<<complementarity_error<< "\tSE: ";
    out<< stationarity_error<<"\t FE: " << feasibility_error<<"\t CostFunctional: "<<cost;
    out<<"\t Step: NA      ";
    out<<"\t p: "<<_p<<"\t acc: NA      "<<"\t rho: "<<rho;
    this->GetOutputHandler()->Write(out,2);
    
    //and error to check for the convergence of the asymptotes is missing!
    //FIXME: Check convergence of Asymptotes?

    accuracy = stationarity_error*0.1;
    kkt_error = std::max(std::max(complementarity_error,stationarity_error),feasibility_error);
    kkt_error_initial = kkt_error;
  }
  
  bool retry = false;
  int inner_iters = 0;
  int n_rho_updates = 0;
  double length = 0.;
  double prediction = 0.;
  double p_init = _p;
  bool inner_iter_succeeded = true;

  while( kkt_error > _mma_global_tol)
  {
    iter++;
    this->GetOutputHandler()->SetIterationNumber(iter,"MMA-Outer");

    //Solve the inner problem
    try
    {    
      inner_iter_succeeded = true;
      n_inner_loops++;
      cost_alt = cost+2.*_mma_multiplier.Norm("infty")*_mma_constraints.Norm("l1","positive");
  
      inner_iters += SolveMMASubProblem(dq,q,lb,ub,q_min_recent,q_max_recent,gradient,_mma_multiplier,cost,prediction,accuracy);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleException(e,"GeneralizedMMAAlgorithm::Solve");
      inner_iters += _mma_inner_maxiter;
      //std::cout<<"HOHOHO Inner Iteration failed"<<prediction<<" -- "<<cost<<std::endl;
      inner_iter_succeeded = false;
    }

    //New iteration update q, and asymptotes
    try
    {
      dq.min(q_max);
      dq.max(q_min);//Make the control feasible w.r.t. the local bounds.

      cost = this->GetReducedProblem()->ComputeReducedCostFunctional(dq);
      this->GetReducedProblem()->ComputeReducedGradient(dq,gradient,gradient_transposed);
      feasible = this->GetReducedProblem()->ComputeReducedConstraints(dq,_mma_constraints);

     }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
    }
    
    if(prediction < cost-_mma_global_tol && rho < 1.e+8)
    { 
      //increase conservativity
      double delta = cost-prediction;
      dq.add(-1.,q);
      dq.comp_mult(dq);
      tmp.equ(1.,sigma);
      tmp.comp_mult(sigma);
      tmp.add(-1.,dq);
      tmp.comp_invert();
      delta /= 0.5*(dq*tmp);
      
      dq = q;
      
      rho = std::min(10*rho,1.1*(rho+delta));

      n_rho_updates++;
      out<<"Bad Model in Minimizer: Making Model more conservative: New Rho: "<<rho;
      this->GetOutputHandler()->Write(out,3);
      _p = p_init;
      _augmented_lagrangian_problem.SetValue(_p,"p");
      _augmented_lagrangian_problem.SetValue(rho,"rho");
      
      cost = this->GetReducedProblem()->ComputeReducedCostFunctional(dq);
      this->GetReducedProblem()->ComputeReducedGradient(dq,gradient,gradient_transposed);
      feasible = this->GetReducedProblem()->ComputeReducedConstraints(dq,_mma_constraints);
      retry = true;
    }
    else if(cost + 2.*_mma_multiplier.Norm("infty")*_mma_constraints.Norm("l1","positive")< cost_alt)
    {
      q_pre_pre = q_pre;
      q_pre = q;
      {
	q.add(-1.,dq);
	length = sqrt(1./_ndofs*(q*q));
	q = dq;
      }
 
      retry = false;
    }
    else
    {    
      if(inner_iter_succeeded&& accuracy > _mma_global_tol*0.01)
      {
	//dq = q;
	cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
	this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
	feasible = this->GetReducedProblem()->ComputeReducedConstraints(q,_mma_constraints);
	retry = true;
	accuracy *= 0.1;
	{
	  out<<"The minimizer doesnot give sufficient descend. Increasing accuracy requirements: New accuracy :"<<accuracy;
	}
	this->GetOutputHandler()->Write(out,3);
      }
      else if(rho < 1.e+8)
      {
	//increase conservativity
	double delta = fabs(cost-prediction);
	dq.add(-1.,q);
	dq.comp_mult(dq);
	tmp.equ(1.,sigma);
	tmp.comp_mult(sigma);
	tmp.add(-1.,dq);
	tmp.comp_invert();
	delta /= 0.5*(dq*tmp);
	if(std::isnan(delta) || std::isinf(delta))
	{
	  delta = 9. *rho;
	}
	dq = q;
	
	rho = std::min(10*rho,1.1*(rho+delta));
	
	n_rho_updates++;
	out<<"Iteration failed, trying again with more convex function: New Rho: "<<rho;
	this->GetOutputHandler()->Write(out,3);
	_p = p_init;
	_augmented_lagrangian_problem.SetValue(_p,"p");
	_augmented_lagrangian_problem.SetValue(rho,"rho");
	
	cost = this->GetReducedProblem()->ComputeReducedCostFunctional(dq);
	this->GetReducedProblem()->ComputeReducedGradient(dq,gradient,gradient_transposed);
	feasible = this->GetReducedProblem()->ComputeReducedConstraints(dq,_mma_constraints);
	retry = true;
      }
      else
      {
	throw DOpEException("We failed to solve the problem!","GeneralizedMMAAlgorithm::Solve");
      }
    }

    if(!retry)
    {
      this->GetOutputHandler()->Write(q,"Control","control");

      //All problem dependend values are up-to-date
      //Change asymptotes
      if(iter < 3)
      {
	lb = q;
	lb.add(-1.,sigma);
	ub = q;
	ub.add(1.,sigma);
      }
      else
      {
	//Only update the distance of the asymptotes if we have made some serious step.
	//if(length > 1.e-6)
	{
	  //Update according to K. Svanberg 2007
	  dq.equ(1.,q);
	  dq.add(-1.,q_pre);
	  q_max_recent.equ(1.,q_pre);
	  q_max_recent.add(-1.,q_pre_pre);
	  dq.comp_mult(q_max_recent); //Contains the sign vector.
	  dq.init_by_sign(0.7,1.2,1.,1.e-10);//New scaling
	  
	  sigma.comp_mult(dq);
	  //Check safeties
	  dq.equ(1.,q_max);
	  dq.add(-1.,q_min);
	  dq *= 10.;
	  sigma.min(dq);
	  dq *= 0.001;
	  sigma.max(dq);  // 0.01 (q_max-q_min) \le sigma \le 10 (q_max - q_min)
	}

	lb = q;
	lb.add(-1.,sigma);
	ub = q;
	ub.add(1.,sigma);

      } 
      q_min_recent.equ(1.,q);
      q_min_recent.add(-0.9,sigma);
      q_max_recent.equ(1.,q);
      q_max_recent.add(0.9,sigma);
      q_min_recent.max(q_min);
      q_max_recent.min(q_max);

//      this->GetOutputHandler()->Write(lb,"LowerAsy","control");
//      this->GetOutputHandler()->Write(q_min_recent,"LowerBd","control");
//      this->GetOutputHandler()->Write(ub,"UpperAsy","control");
//      this->GetOutputHandler()->Write(q_max_recent,"UpperBd","control");

      //Update the Error-Measure and accuracy requirements.
      {
	double complementarity_error = this->GetReducedProblem()->Complementarity(_mma_multiplier,_mma_constraints);
	complementarity_error *= 1./_ndofs;
	double stationarity_error = AugmentedLagrangianResidual(q,gradient,lb,ub,q_min,q_max,_mma_multiplier,q,cost);
	double feasibility_error = this->GetReducedProblem()->GetMaxViolation(_mma_constraints);
	out << "MMA-Outer ("<<iter<<") - ["<<inner_iters<<"/"<<n_inner_loops<<"/"<<n_rho_updates<<"] - CE: "<<this->GetOutputHandler()->ZeroTolerance(complementarity_error,kkt_error_initial)<< "\tSE: ";
	out<< this->GetOutputHandler()->ZeroTolerance(stationarity_error,kkt_error_initial)<<"\t FE: " << this->GetOutputHandler()->ZeroTolerance(feasibility_error,kkt_error_initial)<<"\t CostFunctional: "<<cost;
	out<<"\t Step: "<<this->GetOutputHandler()->ZeroTolerance(length,1.);
        out<<"\t p: "<<this->GetOutputHandler()->ZeroTolerance(_p,1.)<<"\t acc: "<<this->GetOutputHandler()->ZeroTolerance(accuracy,1.)<<"\t rho: "<<rho;
	this->GetOutputHandler()->Write(out,2);
	
	//and error to check for the convergence of the asymptotes is missing!
	//FIXME Check convergence of Asymptotes?
	accuracy = stationarity_error*0.1;
	//accuracy = std::min(stationarity_error*0.1, accuracy);
	kkt_error = std::max(std::max(complementarity_error,stationarity_error),feasibility_error);
      }
      
      //update rho
      //Compute new rho
      //rho = std::max(0.1*rho,1.e-5);
      rho = std::max(0.6*rho,1.e-5);
      
      n_rho_updates = 0;
      _augmented_lagrangian_problem.SetValue(rho,"rho");
      cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
      this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
      feasible = this->GetReducedProblem()->ComputeReducedConstraints(q,_mma_constraints);
      p_init = _p;
      
      n_total_inner_iters += inner_iters;
      n_inner_loops = 0;
      inner_iters = 0;
    }//Endof Not Retry
    else
    {
      iter--; //We don't iterate...
    }
  }
  try
  {
     cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"GeneralizedMMAAlgorithm::Solve");
  }
  out<< "CostFunctional: " << cost;
  this->GetOutputHandler()->Write(out,2);

  out << "**************************************************\n";
  out << "*        Stopping MMA Algorithm       *\n";
  out << "*             after "<<std::setw(6)<<iter<<"  Iterations and "<<n_total_inner_iters<<" Inner-Iterations    *\n";
  out << "*             with KKT-Error "<<std::scientific << std::setw(11) << kkt_error<<"          *\n";
  out << "*             Relative reduction in cost functional:"<<std::scientific << std::setw(11) << (cost-cost_start)/fabs(0.5*(cost_start+cost)) <<"          *\n";
  out.precision(7);
  out << "*             Final value: "<<cost<<"                                     *\n";             
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1,1,1);
  return iter;
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim,localdim>
  ::SolveMMASubProblem(ControlVector<VECTOR>& dq, 
		       const ControlVector<VECTOR>& q, 
		       const ControlVector<VECTOR>& lb,
		       const ControlVector<VECTOR>& ub,
		       const ControlVector<VECTOR>& q_min,
		       const ControlVector<VECTOR>& q_max,
		       const ControlVector<VECTOR>& gradient,
		       ConstraintVector<VECTOR>& mult,
		       double J,
		       double& prediction,
		       double global_tol)

{
  ConstraintVector<VECTOR> dm(mult);
  ControlVector<VECTOR> delta_q(q), last_good_control(q);
  
  dq = q;
  delta_q = q;
  dm = mult;

  unsigned int iter=0;
  //unsigned int n_restart=0;
  double cost=J;
  double cost_alt=J;
  double cost_start=J;
  //double mult_scale = 1.;
  bool has_last_good=false;
  std::stringstream out;
  this->GetOutputHandler()->InitOut(out);

  out << "\t**************************************************\n";
  out << "\t*        Starting AugLag Algorithm       *\n";
  out << "\t*  CDoFs       : ";
  dq.PrintInfos(out);
  out << "\t*  Constraints : ";
  dm.PrintInfos(out);
  out << "\t**************************************************";
  this->GetOutputHandler()->Write(out,4,1,1);

  this->GetOutputHandler()->SetIterationNumber(iter,"AugLag-Outer");

  this->GetOutputHandler()->Write(dq,"ControlUpdate","control");
   
  int lineiter =0;
  int m_lineiter =0;
  int nl_iter=0;
  {
    bool run=true;
    int piter=0;
    _augmented_lagrangian_problem.SetValue(_p,"p");
    double alpha = sqrt(gradient*gradient)*0.5;
    
    ConstraintVector<VECTOR> constraints(dm);
    ConstraintVector<VECTOR> real_constraints(dm);
 
 
    
    bool feasible = false;
    //compute KKT Error
    feasible = this->GetReducedProblem()->ComputeReducedConstraints(dq,real_constraints);
    if(feasible)
    {
      has_last_good = true;
      last_good_control = dq;
    }
    _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
    _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
    feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(dq,constraints);
    if(!feasible)
    { 
      _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
      _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");

      throw DOpEException("Infeasible iterate!","GeneralizedMMAAlgorithm::SolveMMASubProblem");
    }  
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
   

    cost = ComputeModelValue(dq,q,gradient,lb,ub,q_min,q_max,dm,constraints,J);

    double complementarity_error = 0.;
    complementarity_error = this->GetReducedProblem()->Complementarity(mult,real_constraints);
    complementarity_error *= 1./_ndofs;
    double stationarity_error = AugmentedLagrangianResidual(q,gradient,lb,ub,q_min,q_max,mult,dq,J);
    double feasibility_error = this->GetReducedProblem()->GetMaxViolation(real_constraints);
    prediction = cost - mult*constraints;

    out << "\tAugLag-Outer (0) - CE: "<<complementarity_error<< "\tSE: ";
    out<< stationarity_error<<"\t FE: " << feasibility_error<<"\t CostFunctional: "<<cost<<"\t Prediction: "<<prediction<<"\t Multiplier: "<<mult.Norm("infty");
    this->GetOutputHandler()->Write(out,4);
    out<<"\tInitialize Alpha: "<<alpha;
    this->GetOutputHandler()->Write(out,5);

    double kkt_error = std::max(std::max(complementarity_error,stationarity_error),feasibility_error);
    double kkt_error_last = 0.;

    double step_length =  0.;
    double rstep_length =  0.;
    bool retry= false;

    while(run)
    {
      iter++;
      if(iter > _mma_inner_maxiter)
      {
	throw DOpEIterationException("Iteration count exceeded bounds!","GeneralizedMMAAlgorithm::SolveMMASubProblem");
      }
      this->GetOutputHandler()->SetIterationNumber(iter,"AugLag-Outer");
        
      delta_q = dq;
      dm = mult;
      //Step 2 of Alg 9.4.1
      try{
	nl_iter = this->SolveSCSDPSubProblem(q,gradient,lb,ub,q_min,q_max,delta_q,dm,constraints,alpha,J); 
      }
      catch(DOpEException& e)
      {
	//Try again with new multiplier
	double scale = mult.Norm("infty");
	_augmented_lagrangian_problem.InitMultiplier(mult,gradient);
	mult *= scale*2.;

	//If the prediced value is better than the last value we stop here.
	if(prediction < J)
	{ 
	  //  std::cout<<"\nHI "<< prediction<<" -- "<<J<<" -- "<<global_tol <<"\n"<<std::endl;
	  this->GetOutputHandler()->Write(dq,"ControlUpdate","control");
	  
	  out << "\t**************************************************\n";
	  out << "\t*        Stopping AugLag Algorithm       *\n";
	  out << "\t*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
	  out.precision(4);
	  out << "\t*             with KKT-Error "<<std::scientific << std::setw(11) << kkt_error<<"          *\n";
	  out << "\t*              Relative reduction in cost functional:"<<std::scientific << std::setw(11) << (cost-cost_start)/fabs(0.5*(cost_start+cost)) <<"          *\n";
	  out.precision(10);
	  out << "\t**************************************************";
	  this->GetOutputHandler()->Write(out,4,1,1);
	  return iter;
	}

	delta_q = dq;
	dm = mult;

//	std::cout<<"Reinit ... ! "<<kkt_error<<" -- "<<scale<<" --- "<<prediction<<std::endl;
//	std::cout<< "\tReinit ("<<iter<<") -  a="<<alpha<<" p="<<_p<<" - CE: "<<complementarity_error<< "\tSE: ";
//	std::cout<< stationarity_error <<"\t FE: " << feasibility_error<<"\t CostFunctional: "<<cost<<"\t Prediction: "<<prediction<<"\t Multiplier: "<<mult.Norm("infty")<<std::endl;
	
	try{
	  out << "*************************************************************\n";
	  out << "*     AugLag Algorithm Inner Iteration ("<<std::setw(6)<<iter<<") Failed!      *\n";
	  out << "*            Restarting Inner Iteration with new Multiplier.*\n";
	  out << "*************************************************************";
	  this->GetOutputHandler()->Write(out,4,1,1);
	  nl_iter = this->SolveSCSDPSubProblem(q,gradient,lb,ub,q_min,q_max,delta_q,dm,constraints,alpha,J); 
	}
	catch(DOpEException& e)
	{
	  throw DOpEException("Inner Iteration failed ...","GeneralizedMMAAlgorithm::SolveMMASubProblem");
	}
      }
 
      {
	//Linesearch Set 3 of Alg 9.4.1
	cost_alt = cost;
	delta_q.add(-1.,dq);	
	step_length =sqrt(delta_q*delta_q); 
	rstep_length = step_length/sqrt(dq*dq);
	
	if(rstep_length > 1.e-12)
	{	
	  try
	  {
	    lineiter = OuterLineSearch(delta_q,cost,dq,q,lb,ub,q_min,q_max,gradient,dm,constraints,J);
	  } 
	  catch(DOpEIterationException& e)
	  {
	    lineiter = -1;
	  }
	  catch(DOpEException& e)
	  {
	    //this->GetExceptionHandler()->HandleCriticalException(e);
	    _augmented_lagrangian_problem.InitMultiplier(mult,gradient);

	    if(has_last_good)
	    {
	      dq = last_good_control;
	    }
	    else
	    { 
	      dq = q;
	    }
	    retry = true;
//	    std::cout<<"Retry ... !"<<std::endl;
	  }
	}
	this->GetOutputHandler()->Write(dq,"ControlUpdate","control");
	
	_augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
	_augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
	feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(dq,constraints);
	this->GetReducedProblem()->ComputeReducedConstraints(dq,real_constraints);
	if(!feasible)
	{
	  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
	  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
	  throw DOpEException("Infeasible iterate!","GeneralizedMMAAlgorithm::SolveMMASubProblem");
	} 
	_augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
	_augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
	
	if(!retry)
	{
	  complementarity_error = this->GetReducedProblem()->Complementarity(mult,real_constraints);
	  _augmented_lagrangian_solver.ComputeReducedConstraintGradient(mult,constraints,dm);
	  
	  //Update Multiplier
	  dm.add(-1.,mult);
	  try
	  {
	    m_lineiter = 0;
	    m_lineiter = MultiplierLineSearch(dm,mult,1.); 
	  }
	  catch(DOpEException& e)
	  {
	    m_lineiter=-1;
	    //Ignore, because we know that the value is most likely infeasible...
	  }
	}
	//compute cost again.
	cost = ComputeModelValue(dq,q,gradient,lb,ub,q_min,q_max,dm,constraints,J);
	prediction = cost - mult*constraints;
      }//Endof the Update
      
      if(!retry)
      {
	//Check stopping Criterium Step 4 of Alg 9.4.1
	{
	  complementarity_error = this->GetReducedProblem()->Complementarity(mult,real_constraints);
	  complementarity_error *= 1./_ndofs;
	  stationarity_error = AugmentedLagrangianResidual(q,gradient,lb,ub,q_min,q_max,mult,dq,J);
	  feasibility_error = this->GetReducedProblem()->GetMaxViolation(real_constraints);
	  //out << "\tAugLag-Outer ("<<iter<<") - ls["<<nl_iter<<"/"<<lineiter<<"/"<<m_lineiter<<"] a="<<alpha<<" p="<<p<<" - Complementarity Error: "<<complementarity_error<< "\tStationarity Violation: ";
	  out << "\tAugLag-Outer ("<<iter<<") - ls["<<nl_iter<<"/"<<lineiter<<"/"<<m_lineiter<<"] a="<<alpha<<" p="<<_p<<" - CE: "<<complementarity_error<< "\tSE: ";
	  out<< stationarity_error <<"\t FE: " << feasibility_error<<"\t CostFunctional: "<<cost<<"\t Prediction: "<<prediction<<"\t Multiplier: "<<mult.Norm("infty");
	  this->GetOutputHandler()->Write(out,4);
	  
	  if(feasibility_error <= 0.)
	  {
	    has_last_good = true;
	    last_good_control = dq;
	  }

	  kkt_error_last = kkt_error;
	  kkt_error = std::max(std::max(complementarity_error,stationarity_error),feasibility_error);
	  //check stopping criterion
	  if(kkt_error < std::max(global_tol,_mma_global_tol*0.01))
	  {
	    run = false;
	    break;
	  }
	}
	if(run)
	{
	  //update p Step 5 of Alg 9.4.1
//	  if((((std::max(std::max(complementarity_error,stationarity_error),feasibility_error) >= kkt_error_last)|| feasibility_error > 0.1*stationarity_error)|| ((-1 == lineiter))) && _p > 1.e-6 )
	  if((((std::max(std::max(complementarity_error,stationarity_error),feasibility_error) >= kkt_error_last)|| feasibility_error > 0.1*stationarity_error)|| ((-1 == lineiter))) && _p > _mma_global_tol )
	  {
	    double gamma = 0.5;
	    
	    if(!this->GetReducedProblem()->IsEpsilonFeasible(real_constraints,gamma*_p))
	    {
	      if(piter < 30)
	      {
		piter++;
		gamma = (this->GetReducedProblem()->GetMaxViolation(real_constraints)+_p)/(2.*_p);
	      }
	      else
	      {
		if(has_last_good)
		{
		  bool l_run = true;
		  while(l_run)
		  {
		    this->GetReducedProblem()->FeasibilityShift(last_good_control, dq,gamma*0.5);
		    this->GetReducedProblem()->ComputeReducedConstraints(dq,real_constraints);
		    l_run = !this->GetReducedProblem()->IsEpsilonFeasible(real_constraints,gamma*_p);
		  }
		}
		else
		{
		  double scale = mult.Norm("infty");
		  _augmented_lagrangian_problem.InitMultiplier(mult,gradient);
		  mult *= scale*2.;
		  gamma = 1.;
		}
		piter = 0;
	      }
	    }
	    else
	    {
	      piter = 0;
	    }
	    out<<"\tUpdate barrier parameter: "<<_p<<" -> ";
	    _p *=gamma;
	    out<<_p;
	    this->GetOutputHandler()->Write(out,5);
	    _augmented_lagrangian_problem.SetValue(_p,"p");

	    cost = ComputeModelValue(dq,q,gradient,lb,ub,q_min,q_max,dm,constraints,J);
	    
	    complementarity_error = this->GetReducedProblem()->Complementarity(mult,real_constraints);
	    complementarity_error *= 1./_ndofs;
	    stationarity_error = AugmentedLagrangianResidual(q,gradient,lb,ub,q_min,q_max,mult,dq,J);
	    feasibility_error = this->GetReducedProblem()->GetMaxViolation(real_constraints);
	    
	    _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
	    _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
	    feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(dq,constraints);
	    this->GetReducedProblem()->ComputeReducedConstraints(dq,real_constraints);
	    if(!feasible)
	    {
	      throw DOpEException("Infeasible iterate!","GeneralizedMMAAlgorithm::SolveMMASubProblem");
	    } 
	    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
	    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");

	  }
	  kkt_error = std::max(std::max(complementarity_error,stationarity_error),feasibility_error);
	  
	  //update alpha Step 6 of Alg 9.4.1
	
	  if(fabs(cost-cost_alt) < alpha * (1+fabs(cost)))
	  {
	    out<<"\tUpdate Alpha: "<<alpha<<" -> ";
	    if(alpha > kkt_error*0.5)
	    {
	      alpha = stationarity_error*0.5;
	    }
	    else
	    {
	      alpha = std::max(stationarity_error*0.01,alpha*0.5);
	    }
	    out<<alpha;
	    this->GetOutputHandler()->Write(out,5);
	  }	
	}//Endof Update for p and alpha 
      }//Endof if !retry 
      else
      {
      	iter--;
      	retry = false;
      }
    }
    this->GetOutputHandler()->Write(dq,"ControlUpdate","control");

    out << "\t**************************************************\n";
    out << "\t*        Stopping AugLag Algorithm       *\n";
    out << "\t*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
    out.precision(4);
    out << "\t*             with KKT-Error "<<std::scientific << std::setw(11) << kkt_error<<"          *\n";
    out << "\t*              Relative reduction in cost functional:"<<std::scientific << std::setw(11) << (cost-cost_start)/fabs(0.5*(cost_start+cost)) <<"          *\n";
    out.precision(10);
    out << "\t**************************************************";
    this->GetOutputHandler()->Write(out,4,1,1);
  }
  return iter;
}
/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
  ::SolveSCSDPSubProblem(const ControlVector<VECTOR>& q,
			 const ControlVector<VECTOR>& gradient,
			 const ControlVector<VECTOR>& lb,
			 const ControlVector<VECTOR>& ub,
			 const ControlVector<VECTOR>& q_min,
			 const ControlVector<VECTOR>& q_max,
			 ControlVector<VECTOR>& dq,
			 ConstraintVector<VECTOR>& dm,
			 ConstraintVector<VECTOR>& constraints,
			 double alpha,
                         double J)
{
  bool feasible = true;
  
  //Solve the unconstraint minimization of the augmented lagrangian
  int ret = FindStationaryPointOfAugmentedLagrangian(q,gradient,lb,ub,q_min,q_max,dm,dq,alpha,J);
  
  //update multiplier and p, alpha 
  
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");

  feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(dq,constraints);
  if(!feasible)
  { 
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");

    throw DOpEException("Too Infeasible iterate!","GeneralizedMMAAlgorithm::SolveSCSDPSubProblem");
  }  
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");

  return ret;
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  double GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
  ::AugmentedLagrangianResidual(const ControlVector<VECTOR>& q,
				const ControlVector<VECTOR>& gradient,
				const ControlVector<VECTOR>& lb,
				const ControlVector<VECTOR>& ub,
				const ControlVector<VECTOR>& q_min,
				const ControlVector<VECTOR>& q_max,
				const ConstraintVector<VECTOR>& dm,
				const ControlVector<VECTOR>& dq,
				double  J)
{
  _augmented_lagrangian_problem.SetValue(J,"mma_functional");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q,"mma_control");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&lb,"mma_lower_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&ub,"mma_upper_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&gradient,"mma_functional_gradient");
  _augmented_lagrangian_problem.AddAuxiliaryConstraint(&dm,"mma_multiplier");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
 
  double ret = _sub_problem_opt_alg.NewtonResidual(dq);

  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_control");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_functional_gradient");
  _augmented_lagrangian_problem.DeleteAuxiliaryConstraint("mma_multiplier");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
  return ret;
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
  ::FindStationaryPointOfAugmentedLagrangian(const ControlVector<VECTOR>& q,
					     const ControlVector<VECTOR>& gradient,
					     const ControlVector<VECTOR>& lb,
					     const ControlVector<VECTOR>& ub,
					     const ControlVector<VECTOR>& q_min,
					     const ControlVector<VECTOR>& q_max,
					     const ConstraintVector<VECTOR>& dm,
					     ControlVector<VECTOR>& dq,
					     double alpha,
                                             double  J)
{
  _augmented_lagrangian_problem.SetValue(J,"mma_functional");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q,"mma_control");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&lb,"mma_lower_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&ub,"mma_upper_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&gradient,"mma_functional_gradient");
  _augmented_lagrangian_problem.AddAuxiliaryConstraint(&dm,"mma_multiplier");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
  int ret = 0;
  try
  {
    ret = _sub_problem_opt_alg.Solve(dq,alpha);
  }
  catch(DOpEIterationException& e)
  {
    this->GetExceptionHandler()->HandleException(e,"GeneralizedMMAAlgorithm::FindStationaryPointOfAugmentedLagrangian");
    ret = -1;
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleException(e,"GeneralizedMMAAlgorithm::FindStationaryPointOfAugmentedLagrangian");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_control");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_asymptote");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_asymptote");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_functional_gradient");
    _augmented_lagrangian_problem.DeleteAuxiliaryConstraint("mma_multiplier");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
    throw e;
  }
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_control");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_functional_gradient");
  _augmented_lagrangian_problem.DeleteAuxiliaryConstraint("mma_multiplier");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
  return ret;
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
    ::OuterLineSearch(const ControlVector<VECTOR>& dq,
		      double& cost, 
		      ControlVector<VECTOR>& q,
		      const ControlVector<VECTOR>& model_q, 
		      const ControlVector<VECTOR>& lb,
		      const ControlVector<VECTOR>& ub,
		      const ControlVector<VECTOR>& q_min,
		      const ControlVector<VECTOR>& q_max,
		      const ControlVector<VECTOR>& gradient,
		      const ConstraintVector<VECTOR>& dm,
		      ConstraintVector<VECTOR>& constraints,
		      double J)
{
  double rho = _linesearch_rho;
  //double c   = _linesearch_c;

  double costnew = 0.;
  bool force_linesearch = false;
  double alpha = 1.;
  
  q+=dq;

  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
  bool feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(q,constraints);
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
  if(!feasible)
  {
    force_linesearch = true;
  }
  else
  {
    try
    {
      costnew = ComputeModelValue(q,model_q,gradient,lb,ub,q_min,q_max,dm,constraints,J); 
    }
    catch(DOpEException& e)
    {
      force_linesearch = true;
    }
  }
  unsigned int iter =0;
  if(_line_maxiter > 0)
  {
    if(std::isinf(costnew) ||std::isnan(costnew) || (costnew > cost) || force_linesearch)
    {
      this->GetOutputHandler()->Write("\t linesearch ",4+this->GetBasePriority());
      while(std::isinf(costnew) ||std::isnan(costnew) || (costnew > cost) || force_linesearch)
      {
	iter++; 
	if(iter > _line_maxiter)
	{
	  cost = costnew;
	  if(force_linesearch)
	  {
	    throw DOpEException("Iteration count exceeded bounds while unable to compute the CostFunctional!","GeneralizedMMAAlgorithm::OuterLineSearch");
	  }
	  else
	  {
	    throw DOpEIterationException("Iteration count exceeded bounds!","GeneralizedMMAAlgorithm::OuterLineSearch");
	  }
	}
	force_linesearch = false;
	q.add(alpha*(rho-1.),dq);
	alpha *= rho;
	
	_augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
	_augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
	feasible = _augmented_lagrangian_solver.ComputeReducedConstraints(q,constraints);
	_augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
	_augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
	if(!feasible)
	{
	  force_linesearch = true;
	}
	else
	{
	  try
	  {
	    costnew = ComputeModelValue(dq,model_q,gradient,lb,ub,q_min,q_max,dm,constraints,J); 
	  }
	  catch(DOpEException& e)
	  {
	    force_linesearch = true;
	  }
	}
      }
    }
  }
  cost = costnew;
 
  return iter;

}

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  int GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
    ::MultiplierLineSearch(const ConstraintVector<VECTOR>& dm,
			   ConstraintVector<VECTOR>& m,
                           double scale)
{
  double norm_update = sqrt(dm*dm);
  double norm_m = sqrt(m*m);

  double lambda = std::min(0.5*norm_m/norm_update,scale);
    
  m.add(lambda,dm);

  if(lambda == 1.)
    return 0;
  return 1;
}

/******************************************************/

template <typename CONSTRAINTACCESSOR,typename INTEGRATORDATACONT, typename STH, typename PROBLEM,typename VECTOR,typename SOLVER,int dopedim,int dealdim, int localdim>
  double GeneralizedMMAAlgorithm<CONSTRAINTACCESSOR, INTEGRATORDATACONT, STH, PROBLEM,VECTOR, SOLVER,dopedim, dealdim, localdim>
  ::ComputeModelValue(const ControlVector<VECTOR>& dq,//The current Value
		      const ControlVector<VECTOR>& q, //All else defines the model...
		      const ControlVector<VECTOR>& gradient,
		      const ControlVector<VECTOR>& lb,
		      const ControlVector<VECTOR>& ub,
		      const ControlVector<VECTOR>& q_min,
		      const ControlVector<VECTOR>& q_max,
		      const ConstraintVector<VECTOR>& dm,
		      const ConstraintVector<VECTOR>& /*constraints*/,
		      double  J)
{
  _augmented_lagrangian_problem.SetValue(J,"mma_functional");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q,"mma_control");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&gradient,"mma_functional_gradient");
  _augmented_lagrangian_problem.AddAuxiliaryConstraint(&dm,"mma_multiplier");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&lb,"mma_lower_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&ub,"mma_upper_asymptote");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_min,"mma_lower_bound");
  _augmented_lagrangian_problem.AddAuxiliaryControl(&q_max,"mma_upper_bound");
  double ret = 0.;
  try 
  {
    ret = _augmented_lagrangian_solver.ComputeReducedCostFunctional(dq);
  }
  catch( DOpEException& e)
  {
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_control");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_functional_gradient");
    _augmented_lagrangian_problem.DeleteAuxiliaryConstraint("mma_multiplier");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_asymptote");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_asymptote");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
    _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
    throw e;
  }
  //ret -=dm*constraints;
  
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_control");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_functional_gradient");
  _augmented_lagrangian_problem.DeleteAuxiliaryConstraint("mma_multiplier");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_asymptote");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_lower_bound");
  _augmented_lagrangian_problem.DeleteAuxiliaryControl("mma_upper_bound");
  return ret;
}

/******************************************************/
} //Endof  Namespace DOpE
#endif
