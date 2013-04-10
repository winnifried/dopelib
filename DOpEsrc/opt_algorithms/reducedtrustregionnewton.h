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

#ifndef _REDUCEDTRUSTREGION_NEWTON__ALGORITHM_H_
#define _REDUCEDTRUSTREGION_NEWTON__ALGORITHM_H_

#include "reducedalgorithm.h"
#include "parameterreader.h"

#include <iostream>
#include <assert.h>
#include <iomanip>
namespace DOpE
{

  template <typename PROBLEM, typename VECTOR, int dopedim,  int dealdim>
    class ReducedTrustregion_NewtonAlgorithm : public ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>
  {
  public:
    ReducedTrustregion_NewtonAlgorithm(PROBLEM* OP, 
				       ReducedProblemInterface<PROBLEM,VECTOR,dopedim,dealdim>* S,
				       ParameterReader &param_reader,
				       DOpEExceptionHandler<VECTOR>* Except=NULL,
				       DOpEOutputHandler<VECTOR>* Output=NULL,
				       int base_priority=0);
    ~ReducedTrustregion_NewtonAlgorithm();
    
    static void declare_params(ParameterReader &param_reader);


    /**
     * This solves an Optimizationproblem in only the control variable
     * by a trustregion_newtons method.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int Solve(ControlVector<VECTOR>& q,double global_tol=-1.);
    double NewtonResidual(const ControlVector<VECTOR>& q);
  protected:
    bool ComputeTRModelMinimizer(const ControlVector<VECTOR>&  q,  
				 const ControlVector<VECTOR>& gradient,
				 const ControlVector<VECTOR>& gradient_transposed,
				 ControlVector<VECTOR>& hessian,
				 ControlVector<VECTOR>& hessian_transposed,
				 ControlVector<VECTOR>& p_u,
				 ControlVector<VECTOR>& p_b,
				 ControlVector<VECTOR> & min,
				 double tr_delta,
                                 double& cost,
                                 double& model,
                                 double& expand,
				 int& liniter);

    virtual int SolveReducedLinearSystem(const ControlVector<VECTOR>& q,
					 const ControlVector<VECTOR>& gradient,
					 const ControlVector<VECTOR>& gradient_transposed,
					 ControlVector<VECTOR>& dq);
    
    virtual double Residual(const ControlVector<VECTOR>& gradient,
			    const ControlVector<VECTOR>& gradient_transposed)
                            {return  gradient*gradient_transposed;}
  private:
    unsigned int _nonlinear_maxiter;
    double       _nonlinear_tol, _nonlinear_global_tol,_tr_delta_max,_tr_delta_null,_tr_delta_eta;
    unsigned int _linear_maxiter;
    double       _linear_tol, _linear_global_tol;
    std::string _postindex;
    std::string _tr_method;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
void ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("reducedtrustregionnewtonalgorithm parameters");
    param_reader.declare_entry("nonlinear_maxiter", "10",Patterns::Integer(0));
    param_reader.declare_entry("nonlinear_tol", "1.e-7",Patterns::Double(0));
    param_reader.declare_entry("nonlinear_global_tol", "1.e-11",Patterns::Double(0));
    param_reader.declare_entry("tr_delta_max", "1.",Patterns::Double(0));
    param_reader.declare_entry("tr_delta_null", "0.25",Patterns::Double(0));
    param_reader.declare_entry("tr_delta_eta", "0.01",Patterns::Double(0,0.25));

    param_reader.declare_entry("linear_maxiter", "40",Patterns::Integer(0));
    param_reader.declare_entry("linear_tol", "1.e-10",Patterns::Double(0));
    param_reader.declare_entry("linear_global_tol", "1.e-12",Patterns::Double(0));
    
    param_reader.declare_entry("tr_method", "dogleg",Patterns::Selection("dogleg|exact|steinhaug"));
    
    ReducedAlgorithm<PROBLEM,VECTOR,dopedim,dealdim>::declare_params(param_reader);
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>
  ::ReducedTrustregion_NewtonAlgorithm(PROBLEM* OP, 
				       ReducedProblemInterface<PROBLEM,VECTOR,dopedim, dealdim>* S,
				       ParameterReader &param_reader,
				       DOpEExceptionHandler<VECTOR>* Except,
				       DOpEOutputHandler<VECTOR>* Output,
				       int base_priority) 
  : ReducedAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>(OP,S,param_reader,Except,Output,base_priority)
  {
    param_reader.SetSubsection("reducedtrustregionnewtonalgorithm parameters");
    _nonlinear_maxiter    = param_reader.get_integer ("nonlinear_maxiter");
    _nonlinear_tol        = param_reader.get_double ("nonlinear_tol");
    _nonlinear_global_tol = param_reader.get_double ("nonlinear_global_tol");
    _tr_delta_max         = param_reader.get_double("tr_delta_max");
    _tr_delta_null        = param_reader.get_double("tr_delta_null");
    _tr_delta_eta         = param_reader.get_double("tr_delta_eta");

    assert(_tr_delta_eta < 0.25);
    assert(_tr_delta_null<_tr_delta_max);

    _linear_maxiter       = param_reader.get_integer ("linear_maxiter");
    _linear_tol           = param_reader.get_double ("linear_tol");
    _linear_global_tol    = param_reader.get_double ("linear_global_tol");

    _tr_method = param_reader.get_string("tr_method");

    _postindex = "_"+this->GetProblem()->GetName();
  }

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>::~ReducedTrustregion_NewtonAlgorithm()
  {
    
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
double ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>::NewtonResidual(const ControlVector<VECTOR>& q) 
{
  //Solve j'(q) = 0
  ControlVector<VECTOR> gradient(q), gradient_transposed(q);
 
  try
  {
    this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e);
  }
  
  try
  {
    this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e);
  }
    
  return sqrt(Residual(gradient,gradient_transposed));
}
/******************************************************/
/**
 * Implements the Trust Region Algorithm from Nocedal-Wright Alg 4.1
 */
template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>::Solve(ControlVector<VECTOR>& q,double global_tol)
{
 
  q.ReInit();
  //Solve j'(q) = 0
  ControlVector<VECTOR> p_u(q), p_b(q), gradient(q), gradient_transposed(q), hessian(q), hessian_transposed(q),dq(q);
 
  unsigned int iter=0;
  double cost=0.;
  std::stringstream out;

  unsigned int n_good  =0;
  unsigned int n_bad  =0;
  this->GetOutputHandler()->InitNewtonOut(out);

  out << "**************************************************************\n";
  out << "*        Starting Reduced Trustregion_Newton Algorithm       *\n";
  out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t\t\t*\n";
  out << "*  CDoFs : ";
  q.PrintInfos(out);
  out << "*  SDoFs : ";
  this->GetReducedProblem()->StateSizeInfo(out);
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);

  this->GetOutputHandler()->SetIterationNumber(iter,"OptNewton"+_postindex);

  this->GetOutputHandler()->Write(q,"Control"+_postindex,"control");
  
  try
  {
     cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e);
  }
  
  this->GetOutputHandler()->InitOut(out);
  out<< "CostFunctional: " << cost;
  this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
  this->GetOutputHandler()->InitNewtonOut(out);

 //try
 //{
 //  this->GetReducedProblem()->ComputeReducedFunctionals(q);
 //}
 //catch(DOpEException& e)
 //{
 //  this->GetExceptionHandler()->HandleCriticalException(e);
 //}

  try
  {
    this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e);
  }

  double res = Residual(gradient,gradient_transposed);
  double firstres = res; 

  double tr_delta_max = _tr_delta_max;
  double tr_delta = _tr_delta_null;
  double tr_eta = 0.01;
  double tr_rho = 0.;
  double tr_model  = 0.;
  double point_norm = sqrt(q*q);

  assert(res >= 0);

  this->GetOutputHandler()->Write(gradient,"NewtonResidual"+_postindex,"control");
  out<< "\t Newton step: " <<iter<<"\t Residual (abs.): "<<sqrt(res)<<"\n";
  out<< "\t Newton step: " <<iter<<"\t Residual (rel.): "<<std::scientific<<sqrt(res)/sqrt(res)<<"\n";
  this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
  int liniter = 0;
  global_tol =  std::max(_nonlinear_global_tol,global_tol);
  //while( (res >= global_tol*global_tol) && (res >= _nonlinear_tol*_nonlinear_tol*firstres) )
  while( iter==0 ||  iter ==1 || ((res >= global_tol*global_tol) && (res >= _nonlinear_tol*_nonlinear_tol*firstres) ))
  {
    this->GetOutputHandler()->SetIterationNumber(iter,"OptNewton"+_postindex);
    tr_model = 0.;
    if(iter > _nonlinear_maxiter)
    {
      out << "**************************************************\n";
      out << "*        Aborting Reduced Trustregion_Newton Algorithm       *\n";
      out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
      out.precision(4);
      out << "*             with Residual "<<std::scientific << std::setw(11) << sqrt(res)<<"          *\n";
      out << "*             with Cost "<<std::scientific << std::setw(11) << cost<<"          *\n";
      out << "*             n_good: "<<n_good<<" n_bad: "<<n_bad<<"          *\n";
      out.precision(10);
      out << "**************************************************";
      this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
      throw DOpEIterationException("Iteration count exceeded bounds!","ReducedTrustregion_NewtonAlgorithm::Solve");
    }
    if(tr_delta <= 1.e-8*point_norm)
    {
      out << "**************************************************\n";
      out << "*        Aborting Reduced Trustregion_Newton Algorithm       *\n";
      out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
      out.precision(4);
      out << "*             with Residual "<<std::scientific << std::setw(11) << sqrt(res)<<"          *\n";
      out << "*             with Cost "<<std::scientific << std::setw(11) << cost<<"          *\n";
      out << "*             n_good: "<<n_good<<" n_bad: "<<n_bad<<"          *\n";
      out.precision(10);
      out << "**************************************************";
      this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
      throw DOpEIterationException("Iteration aborted due to too small Trustregion radius!","ReducedTrustregion_NewtonAlgorithm::Solve");
    }

    iter++;
         
    //Compute Minimizer p of the Model on the set \|p\| \le tr_delta
    double last_cost = cost;
    double expand = 2.;
    bool good = ComputeTRModelMinimizer(q,gradient,gradient_transposed,hessian,hessian_transposed,p_u,p_b,dq,tr_delta,cost,tr_model,expand,liniter);
      
    if(tr_model != 0.)
    {
      double loc_delta = 1.e-8*std::max(1.,fabs(last_cost));
      if(std::max(fabs(last_cost-cost),fabs(tr_model)) < loc_delta)
      {
	tr_rho = 1.;
      }
      else
	tr_rho = (last_cost-cost-loc_delta)/(0.-tr_model-loc_delta);
    }
    else
      tr_rho = 0.;

    double norm = sqrt(dq*dq);
    if(norm <= 1.e-8*point_norm  && good)
    {
      out << "**************************************************\n";
      out << "*        Aborting Reduced Trustregion_Newton Algorithm       *\n";
      out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
      out.precision(4);
      out << "*             with Residual "<<std::scientific << std::setw(11) << sqrt(res)<<"          *\n";
      out << "*             with Cost "<<std::scientific << std::setw(11) << cost<<"          *\n";
      out << "*             n_good: "<<n_good<<" n_bad: "<<n_bad<<"          *\n";
      out.precision(10);
      out << "**************************************************";
      this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
      throw DOpEIterationException("Iteration aborted due to too small update!","ReducedTrustregion_NewtonAlgorithm::Solve");
    }
    
    out<<"TR-Newton Predicted Reduction: "<<-tr_model<<" Actual Reduction: "<<last_cost-cost<<" rho: "<<tr_rho<<" where TR-Minimizer is "<<good<<" with lenght: "<<norm;
    this->GetOutputHandler()->Write(out,4+this->GetBasePriority());
    out<<"\t TR-Newton step: " <<iter<<"\t";
    out<<"delta: "<<tr_delta<<"->";
    if((tr_rho < 0.01) ||  !good)
    {
      tr_delta=std::max(0.5*norm,0.125*tr_delta);
    }
    else
    {
      if((tr_rho > 0.9))
      {
	tr_delta = std::min(expand*tr_delta,tr_delta_max); 
      }
      
      //else tr_delta = tr_delta;
    }
    out<<tr_delta;
    
    if(tr_rho > tr_eta)
    {
      out<<"  accepting step!";
      q.add(1.,dq);
      point_norm = sqrt(q*q);
      n_good++;
    }
    else
    {
      out<<"  rejecting step!";
      n_bad++;
    }
    //Compute all values
    try
    {
      cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }
    try
    {
      this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }
    
    res = Residual(gradient,gradient_transposed);
    
    out<<"\t Residual: "<<sqrt(res)<<"\t LinearIters ["<<liniter<<"]";
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
    
    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
    
    this->GetOutputHandler()->Write(q,"Control"+_postindex,"control");
    this->GetOutputHandler()->Write(gradient,"NewtonResidual"+_postindex,"control");
  }

  //We are done write total evaluation
  out<< "CostFunctional: " << cost;
  this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
  try
    {
      this->GetReducedProblem()->ComputeReducedFunctionals(q);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }
  
  out << "**************************************************\n";
  out << "*        Stopping Reduced Trustregion_Newton Algorithm       *\n";
  out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
  out.precision(4);
  out << "*             with rel. Residual "<<std::scientific << std::setw(11) << this->GetOutputHandler()->ZeroTolerance(sqrt(res)/sqrt(firstres),1.0)<<"          *\n";
  out << "*             with Cost "<<std::scientific << std::setw(11) << cost<<"          *\n";
  out << "*             n_good: "<<n_good<<" n_bad: "<<n_bad<<"          *\n";
  out.precision(10);
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
  
  return iter;
}
/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  bool ReducedTrustregion_NewtonAlgorithm<PROBLEM,VECTOR,dopedim, dealdim>::
  ComputeTRModelMinimizer(const ControlVector<VECTOR>&  q,  
			  const ControlVector<VECTOR>& gradient,
			  const ControlVector<VECTOR>& gradient_transposed,
			  ControlVector<VECTOR>& hessian,
			  ControlVector<VECTOR>& hessian_transposed,
			  ControlVector<VECTOR>& p_u,
			  ControlVector<VECTOR>& p_b,
			  ControlVector<VECTOR> & min,
			  double tr_delta,
			  double& cost,
			  double& model,
                          double& expand,
                          int& liniter)
{

  bool ret = true;
  if("dogleg" == _tr_method)
  { 
    //Compute the unconstraint model minimizer
    try
    {
      liniter = SolveReducedLinearSystem(q,gradient,gradient_transposed,p_b);
    }
    catch(DOpEIterationException& e)
    {
      //Seems uncritical too many linear solves, it'll probably work
      //So only write a warning, and continue.
      this->GetExceptionHandler()->HandleException(e);
      liniter = -1;
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }
    //compute the stepest descend direction...
    try{
      this->GetReducedProblem()->ComputeReducedHessianVector(q,gradient,hessian,hessian_transposed);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }
    {
      double scale = (gradient*gradient_transposed)/(gradient*hessian_transposed);
      p_u.equ(-1.*scale,gradient);
    }

    ControlVector<VECTOR> tmp1(q), tmp2(q);
    
    //Check if p_b is feasible
    double n_b = sqrt(p_b*p_b);

    if(n_b <= tr_delta)
    {
      min = p_b;
    }
    else
    {
      //compute the relaxation factor...
      //This is such that for a quadratic functional after one iteration
      //the solution is within the TR-radius if not prevented by the maximal radius.
      expand = std::max(n_b/tr_delta,2.);
      //but shouldnot grow too fast
      expand = std::min(expand, 10.);
      
      //Check if p_u is feasible
      double  n_u = sqrt(p_u*p_u);
//      assert(n_u < n_b);
      
      if(n_u <= tr_delta)
      {
	//solution between  p_u and p_b
	double a = n_u*n_u;
	double b = n_b*n_b;
	double c = p_u*p_b;
	//l solves l^2(a+b-2c) + l(2c-2a) = tr_delta^2-a
	double d  = (c-a)/(a+b-2.*c);
	double e = (tr_delta*tr_delta-a)/(a+b-2.*c);
	assert(e+d*d >= 0.);
	double l = sqrt(e+d*d) - d;
	assert(l >=  0.);
	assert(l <= 1.);
	min.equ(l,p_b);
	min.add(1-l,p_u);
      }
      else
      {
	//solution between  0 und p_u
	min.equ(1./n_u*tr_delta,p_u);
      }
    }
    //Check if the choice  is in the domain of definition of f!
    tmp1 = q;
    tmp1 += min;
    
    try
    {
      cost = this->GetReducedProblem()->ComputeReducedCostFunctional(tmp1);
    }
    catch(DOpEException& e)
    {
      //this failed... we need to move closer to q!
      ret = false;
      min = 0.;
    }
    //reset Precomputations
    this->GetReducedProblem()->ComputeReducedCostFunctional(q);
    if(ret)
    {
      //Evaluate the model.
      model = gradient*min;
      //second order term
      try{
	this->GetReducedProblem()->ComputeReducedHessianVector(q,min,tmp1,tmp2);
      }
      catch(DOpEException& e)
      {
	this->GetExceptionHandler()->HandleCriticalException(e);
      }
      model += 0.5*(tmp1*min);
    }
    else
      model = 0.;
  }
  else if("exact" == _tr_method)
  {
    throw DOpEException("Method not yet implemented: "+_tr_method,"ReducedTrustregion_NewtonAlgorithm::ComputeTRModelMinimizer");
  }
  else if("steinhaug" == _tr_method)
  {
    throw DOpEException("Method not yet implemented: "+_tr_method,"ReducedTrustregion_NewtonAlgorithm::ComputeTRModelMinimizer");
  }
  else
  {
    throw DOpEException("Unknown Method: "+_tr_method,"ReducedTrustregion_NewtonAlgorithm::ComputeTRModelMinimizer");
  }
  
  return ret;
}

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  int ReducedTrustregion_NewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>
  ::SolveReducedLinearSystem(const ControlVector<VECTOR>& q,
			     const ControlVector<VECTOR>& gradient,
			     const ControlVector<VECTOR>& gradient_transposed,
			     ControlVector<VECTOR>& dq)
{
  std::stringstream out;
  dq = 0.;
  ControlVector<VECTOR> r(q), r_transposed(q),  d(q), Hd(q), Hd_transposed(q);

  r            = gradient;
  r_transposed = gradient_transposed;
  d = gradient_transposed;

  double res = Residual(r,r_transposed);//r*r_transposed;
  double firstres = res;

  assert(res >= 0.);

  out << "Starting Reduced Linear Solver with Residual: "<<sqrt(res);
  this->GetOutputHandler()->Write(out,5+this->GetBasePriority());

  unsigned int iter = 0;
  double cgalpha, cgbeta, oldres;

  this->GetOutputHandler()->SetIterationNumber(iter,"OptNewtonCg"+_postindex);

  //while(res>=_linear_tol*_linear_tol*firstres && res>=_linear_global_tol*_linear_global_tol)
  //using Algorithm 6.1 from Nocedal Wright
  while(res>= std::min(0.25,sqrt(firstres))*firstres && res>=_linear_global_tol*_linear_global_tol)
  {
    iter++;
    this->GetOutputHandler()->SetIterationNumber(iter,"OptNewtonCg"+_postindex);
    if(iter > _linear_maxiter)
    {
      throw DOpEIterationException("Iteration count exceeded bounds!","ReducedNewtonAlgorithm::SolveReducedLinearSystem");
    }

    try
    {
      this->GetReducedProblem()->ComputeReducedHessianVector(q,d,Hd,Hd_transposed);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e);
    }

    cgalpha = res / (Hd*d);

    if(cgalpha < 0)
    {
      if(iter==1)
      {
	dq.add(cgalpha,d);
      }
      throw DOpENegativeCurvatureException("Negative curvature detected!","ReducedNewtonAlgorithm::SolveReducedLinearSystem");
    }

    dq.add(cgalpha,d);
    r.add(cgalpha,Hd);
    r_transposed.add(cgalpha,Hd_transposed);

    oldres = res;
    res = Residual(r,r_transposed);//r*r_transposed;
    if(res < 0.)
    {
      //something is broken, maybe don't use update formula and 
      //calculate res from scratch. 
      try
      {
	this->GetReducedProblem()->ComputeReducedHessianVector(q,dq,Hd,Hd_transposed);
      }
      catch(DOpEException& e)
      {
	this->GetExceptionHandler()->HandleCriticalException(e);
      }
      r = gradient;			      
      r_transposed = gradient_transposed;
      r.add(1.,Hd);
      r_transposed.add(1.,Hd_transposed);
      res = Residual(r,r_transposed);
    }
    assert(res >= 0.);
    out<<"\t Cg step: " <<iter<<"\t Residual: "<<sqrt(res);
    this->GetOutputHandler()->Write(out,5+this->GetBasePriority());

    cgbeta = res / oldres; //Fletcher-Reeves
    d*= cgbeta;
    d.equ(-1,r_transposed);
  }
  return iter;
}

}
#endif
