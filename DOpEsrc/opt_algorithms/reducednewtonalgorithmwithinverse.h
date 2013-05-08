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

#ifndef _REDUCEDNEWTON__ALGORITHM_INVERSE_H_
#define _REDUCEDNEWTON__ALGORITHM_INVERSE_H_

#include "reducednewtonalgorithm.h"
#include "parameterreader.h"

#include <iostream>
#include <assert.h>
#include <iomanip>
namespace DOpE
{
  /**
   * This class implements a linesearch newton algorithm where the linear system is solved
   * exactly. This requires that <PROBLEM> posseses a working method 
   * PROBLEM::ComputeReducedHessianInverseVector which is reasonable only if the problem
   * has a very simple structure.
   *
   * @tparam <PROBLEM>     The problem to deal with.
   * @tparam <VECTOR>      The type of Vector used in the ControlVectors
   * @tparam <dopedim>     The dimension for the control variable.
   * @tparam <dealdim>     The dimension of the state variable.
   *
   */
  template <typename PROBLEM, typename VECTOR,int dopedim,  int dealdim>
    class ReducedNewtonAlgorithmWithInverse : public ReducedNewtonAlgorithm<PROBLEM, VECTOR>
  {
  public:
    ReducedNewtonAlgorithmWithInverse(PROBLEM* OP, 
				      ReducedProblemInterface<PROBLEM, VECTOR>* S,
				      ParameterReader &param_reader,
				      DOpEExceptionHandler<VECTOR>* Except=NULL,
				      DOpEOutputHandler<VECTOR>* Output=NULL,
				      int base_priority=0);
    ~ReducedNewtonAlgorithmWithInverse();
    
    static void declare_params(ParameterReader &param_reader);


    /**
     * This solves an Optimizationproblem in only the control variable
     * by a newtons method.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int Solve(ControlVector<VECTOR>& q,double global_tol=-1.);

  protected:
    int SolveReducedLinearSystem(const ControlVector<VECTOR>& q, 
				 const ControlVector<VECTOR>& gradient,
				 const ControlVector<VECTOR>& gradient_transposed, 
				 ControlVector<VECTOR>& dq);
    
    double Residual(const ControlVector<VECTOR>& gradient,
		    const ControlVector<VECTOR>& /*gradient_transposed*/)
    {return  gradient*gradient;}
  private:
    unsigned int _line_maxiter;
    double       _linesearch_rho, _linesearch_c;
    
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void ReducedNewtonAlgorithmWithInverse<PROBLEM, VECTOR,dopedim, dealdim>::declare_params(ParameterReader &param_reader)
  {
    ReducedNewtonAlgorithm<PROBLEM, VECTOR>::declare_params(param_reader);
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
ReducedNewtonAlgorithmWithInverse<PROBLEM, VECTOR,dopedim, dealdim>::ReducedNewtonAlgorithmWithInverse(PROBLEM* OP, 
											       ReducedProblemInterface<PROBLEM, VECTOR>* S,
											       ParameterReader &param_reader,
											       DOpEExceptionHandler<VECTOR>* Except,
											       DOpEOutputHandler<VECTOR>* Output,
											       int base_priority) 
  : ReducedNewtonAlgorithm<PROBLEM, VECTOR>(OP,S,param_reader,Except,Output,base_priority)
  {
    param_reader.SetSubsection("reducednewtonalgorithm parameters");
    _line_maxiter         = param_reader.get_integer ("line_maxiter");
    _linesearch_rho       = param_reader.get_double ("linesearch_rho");
    _linesearch_c         = param_reader.get_double ("linesearch_c");
  }

/******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
ReducedNewtonAlgorithmWithInverse<PROBLEM, VECTOR,dopedim, dealdim>::~ReducedNewtonAlgorithmWithInverse()
  {
    
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
int ReducedNewtonAlgorithmWithInverse<PROBLEM, VECTOR,dopedim, dealdim>::Solve(ControlVector<VECTOR>& q,double global_tol)
{
  return ReducedNewtonAlgorithm<PROBLEM, VECTOR>::Solve(q,global_tol);
}

/******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
int ReducedNewtonAlgorithmWithInverse<PROBLEM, VECTOR,dopedim, dealdim>::SolveReducedLinearSystem(const ControlVector<VECTOR>& q, 
											  const ControlVector<VECTOR>& gradient,
											  const ControlVector<VECTOR>& gradient_transposed __attribute__((unused)), 
											  ControlVector<VECTOR>& dq)
{
  int iter = 0;
  this->GetReducedProblem()->ComputeReducedHessianInverseVector(q,gradient,dq);
//  {
//    //Check...
//    ControlVector<VECTOR> Hd(q), Hd_transposed(q);
//    this->GetReducedProblem()->ComputeReducedHessianVector(q,dq,Hd,Hd_transposed);
//    Hd.add(-1.,gradient);
//    std::cout<<" Linear residual "<<Hd*Hd<<std::endl;
//  }
  return iter;
}


/******************************************************/
}
#endif
