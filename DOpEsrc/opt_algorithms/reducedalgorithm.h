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

#ifndef _REDUCED_ALGORITHM_H_
#define _REDUCED_ALGORITHM_H_

#include "optproblemcontainer.h"
#include "reducedprobleminterface.h"
#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "controlvector.h"

#include <lac/vector.h>

#include <iostream>
#include <assert.h>
#include <iomanip>

namespace DOpE
{

  template <typename PROBLEM, typename VECTOR, int dopedim, int dealdim>
  class ReducedAlgorithm
  {
  public:
    ReducedAlgorithm(PROBLEM* OP, ReducedProblemInterface<PROBLEM,VECTOR, dopedim,dealdim>* S, ParameterReader &param_reader,DOpEExceptionHandler<VECTOR>* Except=NULL,DOpEOutputHandler<VECTOR>* Output=NULL,int base_priority=0);
    ~ReducedAlgorithm();

    static void declare_params(ParameterReader &param_reader);

    virtual void ReInit() { this->GetReducedProblem()->ReInit(); if(_rem_output){this->GetOutputHandler()->ReInit();} }

    virtual int Solve(ControlVector<VECTOR>& q,double global_tol=-1.)=0;

    virtual void SolveForward(ControlVector<VECTOR>& q);

    virtual void CheckGrads(double c,ControlVector<VECTOR>& q,ControlVector<VECTOR>& dq, unsigned int niter=1, double eps=1.);
    virtual void FirstDifferenceQuotient(double exact, double eps, const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& dq);
    virtual void CheckHessian(double c,ControlVector<VECTOR>& q,ControlVector<VECTOR>& dq, unsigned int niter=1, double eps=1.);
    virtual void SecondDifferenceQuotient(double exact, double eps, const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& dq);

    DOpEExceptionHandler<VECTOR>* GetExceptionHandler() { return _ExceptionHandler; }
    DOpEOutputHandler<VECTOR>* GetOutputHandler() { return _OutputHandler; }
  protected:
    PROBLEM* GetProblem() { return _OP; }
    const PROBLEM* GetProblem() const { return _OP; }
    const ReducedProblemInterface<PROBLEM, VECTOR, dopedim,dealdim>* GetReducedProblem() const { return _Solver; }
    ReducedProblemInterface<PROBLEM, VECTOR, dopedim,dealdim>* GetReducedProblem() { return _Solver; }

    int GetBasePriority() const { return _base_priority; }

  private:
    PROBLEM* _OP;
    ReducedProblemInterface<PROBLEM, VECTOR, dopedim,dealdim>* _Solver;
    DOpEExceptionHandler<VECTOR>* _ExceptionHandler;
    DOpEOutputHandler<VECTOR>* _OutputHandler;
    bool _rem_exception;
    bool _rem_output;
    int _base_priority;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
    using namespace dealii;

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::declare_params(ParameterReader &param_reader)
{
  DOpEOutputHandler<VECTOR>::declare_params(param_reader);
}

/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
    ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::ReducedAlgorithm(PROBLEM* OP,
								 ReducedProblemInterface<PROBLEM, VECTOR, dopedim, dealdim>* S,
								 ParameterReader &param_reader,
								 DOpEExceptionHandler<VECTOR>* Except,
								 DOpEOutputHandler<VECTOR>* Output,
								 int base_priority)
  {
    assert(OP);
    assert(S);

    _OP = OP;
    _Solver = S;
    if(Output==NULL)
    {
      _OutputHandler = new DOpEOutputHandler<VECTOR>(S,param_reader);
      _rem_output = true;
    }
    else
    {
      _OutputHandler = Output;
      _rem_output = false;
    }
    if(Except == NULL)
    {
      _ExceptionHandler = new DOpEExceptionHandler<VECTOR>(_OutputHandler);
      _rem_exception = true;
    }
    else
    {
      _ExceptionHandler = Except;
      _rem_exception  = false;
    }
    _OP->RegisterOutputHandler(_OutputHandler);
    _OP->RegisterExceptionHandler(_ExceptionHandler);
    _Solver->RegisterOutputHandler(_OutputHandler);
    _Solver->RegisterExceptionHandler(_ExceptionHandler);

    _base_priority = base_priority;
  }

/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::~ReducedAlgorithm()
  {
    if(_ExceptionHandler&&_rem_exception)
    {
      delete _ExceptionHandler;
    }
    if(_OutputHandler&&_rem_output)
    {
      delete _OutputHandler;
    }
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::SolveForward(ControlVector<VECTOR>& q)
{
  q.ReInit();

  //Solve j'(q) = 0
  double cost=0.;
  std::stringstream out;

  out << "**************************************************\n";
  out << "*             Starting Forward Solver            *\n";
  out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t*\n";
  out << "*  CDoFs : ";
  q.PrintInfos(out);
  out << "*  SDoFs : ";
  this->GetReducedProblem()->StateSizeInfo(out);
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);

  try
  {
     cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e);
  }

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

}
/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::CheckGrads(double c,ControlVector<VECTOR>& q,ControlVector<VECTOR>& dq, unsigned int niter, double eps)
  {
    q.ReInit();
    dq.ReInit();

    dq = c;

    ControlVector<VECTOR> point(q);
    point = q;
    std::stringstream out;

    ControlVector<VECTOR> gradient(q), gradient_transposed(q);

    this->GetReducedProblem()->ComputeReducedCostFunctional(point);
    this->GetReducedProblem()->ComputeReducedGradient(point,gradient,gradient_transposed);
    double  cost_diff = gradient*dq;
    out<<"Checking Gradients...."<<std::endl;
    out<<" Epsilon \t Exact \t Diff.Quot. \t Rel. Error ";
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());

    for(unsigned int i = 0; i < niter; i++)
    {
      FirstDifferenceQuotient(cost_diff,eps,q,dq);
      eps /= 10.;
    }
  }
/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::FirstDifferenceQuotient(double exact, double eps, const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& dq)
  {
    ControlVector<VECTOR> point(q);
    point = q;

    std::stringstream out;

    point.add(eps,dq);

    double cost_right=0.;
    //Differenzenquotient
    cost_right = this->GetReducedProblem()->ComputeReducedCostFunctional(point);

    point.add(-2.*eps,dq);

    double cost_left=0.;
    //Differenzenquotient
    cost_left = this->GetReducedProblem()->ComputeReducedCostFunctional(point);

    double diffquot = (cost_right - cost_left)/(2.*eps);
    out<<eps<<"\t"<<exact<<"\t"<<diffquot<<"\t"<<(exact-diffquot)/exact<<std::endl;
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
  }

/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::CheckHessian(double c,ControlVector<VECTOR>& q,ControlVector<VECTOR>& dq, unsigned int niter, double eps)
  {
    q.ReInit();
    dq.ReInit();

    dq = c;

    ControlVector<VECTOR> point(q);
    point=q;
    std::stringstream out;

    ControlVector<VECTOR> gradient(q), gradient_transposed(q),hessian(q), hessian_transposed(q);

    this->GetReducedProblem()->ComputeReducedCostFunctional(point);
    this->GetReducedProblem()->ComputeReducedGradient(point,gradient,gradient_transposed);

    this->GetReducedProblem()->ComputeReducedHessianVector(point,dq,hessian,hessian_transposed);

    double  cost_diff = hessian*dq;
    out<<"Checking Hessian...."<<std::endl;
    out<<" Epsilon \t Exact \t Diff.Quot. \t Rel. Error ";
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());

    for(unsigned int i = 0; i < niter; i++)
    {
      SecondDifferenceQuotient(cost_diff,eps,q,dq);
      eps /= 10.;
    }
  }
/******************************************************/

  template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
  void ReducedAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::SecondDifferenceQuotient(double exact, double eps, const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& dq)
  {
    ControlVector<VECTOR> point(q);
    point=q;
    std::stringstream out;

    double cost_mid=0.;
    //Differenzenquotient
    cost_mid = this->GetReducedProblem()->ComputeReducedCostFunctional(point);

    point.add(eps,dq);

    double cost_right=0.;
    //Differenzenquotient
    cost_right = this->GetReducedProblem()->ComputeReducedCostFunctional(point);

    point.add(-2.*eps,dq);

    double cost_left=0.;
    //Differenzenquotient
    cost_left = this->GetReducedProblem()->ComputeReducedCostFunctional(point);

    double diffquot = (cost_left - 2.*cost_mid + cost_right)/(eps*eps);

    out<<eps<<"\t"<<exact<<"\t"<<diffquot<<"\t"<<(exact-diffquot)/exact<<std::endl;
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
  }

/******************************************************/

}
#endif
