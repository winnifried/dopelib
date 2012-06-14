#ifndef _REDUCEDNEWTON__ALGORITHM_H_
#define _REDUCEDNEWTON__ALGORITHM_H_

#include "reducedalgorithm.h"
#include "parameterreader.h"

#include <iostream>
#include <assert.h>
#include <iomanip>
namespace DOpE
{

  template <typename PROBLEM, typename VECTOR, int dopedim,  int dealdim>
    class ReducedNewtonAlgorithm : public ReducedAlgorithm<PROBLEM, VECTOR, dopedim,dealdim>
  {
  public:
    ReducedNewtonAlgorithm(PROBLEM* OP,
			   ReducedProblemInterface<PROBLEM, VECTOR,dopedim,dealdim>* S,
			   ParameterReader &param_reader,
			   DOpEExceptionHandler<VECTOR>* Except=NULL,
			   DOpEOutputHandler<VECTOR>* Output=NULL,
			   int base_priority=0);
    ~ReducedNewtonAlgorithm();

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
    double NewtonResidual(const ControlVector<VECTOR>& q);

  protected:
    virtual int SolveReducedLinearSystem(const ControlVector<VECTOR>& q,
				 const ControlVector<VECTOR>& gradient,
				 const ControlVector<VECTOR>& gradient_transposed,
				 ControlVector<VECTOR>& dq);

    virtual int ReducedNewtonLineSearch(const ControlVector<VECTOR>& dq,
				const ControlVector<VECTOR>&  gradient,
				double& cost,
				ControlVector<VECTOR>& q);
    virtual double Residual(const ControlVector<VECTOR>& gradient,
			    const ControlVector<VECTOR>& gradient_transposed)
                            {return  gradient*gradient_transposed;}
  private:
    unsigned int _nonlinear_maxiter, _linear_maxiter, _line_maxiter;
    double       _nonlinear_tol, _nonlinear_global_tol, _linear_tol, _linear_global_tol, _linesearch_rho, _linesearch_c;
    bool         _compute_functionals_in_every_step;
    std::string _postindex;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;
  
  /******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
void ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("reducednewtonalgorithm parameters");
    param_reader.declare_entry("nonlinear_maxiter", "10",Patterns::Integer(0));
    param_reader.declare_entry("nonlinear_tol", "1.e-7",Patterns::Double(0));
    param_reader.declare_entry("nonlinear_global_tol", "1.e-11",Patterns::Double(0));

    param_reader.declare_entry("linear_maxiter", "40",Patterns::Integer(0));
    param_reader.declare_entry("linear_tol", "1.e-10",Patterns::Double(0));
    param_reader.declare_entry("linear_global_tol", "1.e-12",Patterns::Double(0));

    param_reader.declare_entry("line_maxiter", "4",Patterns::Integer(0));
    param_reader.declare_entry("linesearch_rho", "0.9",Patterns::Double(0));
    param_reader.declare_entry("linesearch_c", "0.1",Patterns::Double(0));

    param_reader.declare_entry("compute_functionals_in_every_step", "false",Patterns::Bool());

    ReducedAlgorithm<PROBLEM, VECTOR, dopedim,dealdim>::declare_params(param_reader);
  }
/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::ReducedNewtonAlgorithm(PROBLEM* OP,
									 ReducedProblemInterface<PROBLEM, VECTOR,dopedim, dealdim>* S,
									 ParameterReader &param_reader,
									 DOpEExceptionHandler<VECTOR>* Except,
									 DOpEOutputHandler<VECTOR>* Output,
									 int base_priority)
  : ReducedAlgorithm<PROBLEM, VECTOR,dopedim, dealdim>(OP,S,param_reader,Except,Output,base_priority)
  {

    param_reader.SetSubsection("reducednewtonalgorithm parameters");
    _nonlinear_maxiter    = param_reader.get_integer ("nonlinear_maxiter");
    _nonlinear_tol        = param_reader.get_double ("nonlinear_tol");
    _nonlinear_global_tol = param_reader.get_double ("nonlinear_global_tol");

    _linear_maxiter       = param_reader.get_integer ("linear_maxiter");
    _linear_tol           = param_reader.get_double ("linear_tol");
    _linear_global_tol    = param_reader.get_double ("linear_global_tol");

    _line_maxiter         = param_reader.get_integer ("line_maxiter");
    _linesearch_rho       = param_reader.get_double ("linesearch_rho");
    _linesearch_c         = param_reader.get_double ("linesearch_c");

    _compute_functionals_in_every_step  = param_reader.get_bool ("compute_functionals_in_every_step");

    _postindex = "_"+this->GetProblem()->GetName();
  }

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::~ReducedNewtonAlgorithm()
  {

  }

/******************************************************/

template <typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  double ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::NewtonResidual(const ControlVector<VECTOR>& q) 
{
  //Solve j'(q) = 0
  ControlVector<VECTOR> gradient(q), gradient_transposed(q);
 
  try
  {
    this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::NewtonResidual");
  }
  
  try
  {
    this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::NewtonResidual");
  }
  
  return sqrt(Residual(gradient,gradient_transposed));
}

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::Solve(ControlVector<VECTOR>& q,double global_tol)
{

  q.ReInit();
  //Solve j'(q) = 0
  ControlVector<VECTOR> dq(q), gradient(q), gradient_transposed(q);

  unsigned int iter=0;
  double cost=0.;
  std::stringstream out;
  this->GetOutputHandler()->InitNewtonOut(out);

  out << "**************************************************\n";
  out << "*        Starting Reduced Newton Algorithm       *\n";
  out << "*   Solving : "<<this->GetProblem()->GetName()<<"\t*\n";
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
    this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::Solve");
  }

  out<< "CostFunctional: " << cost;
  this->GetOutputHandler()->Write(out,2+this->GetBasePriority());

  if (_compute_functionals_in_every_step == true)
    {
      try
	{
	  this->GetReducedProblem()->ComputeReducedFunctionals(q);
	}
      catch(DOpEException& e)
	{
	  this->GetExceptionHandler()->HandleCriticalException(e);
	}
    }

  try
  {
    this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
  }
  catch(DOpEException& e)
  {
    this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::Solve");
  }

  double res = Residual(gradient,gradient_transposed);//gradient*gradient_transposed;
  double firstres = res;

  assert(res >= 0);

  this->GetOutputHandler()->Write(gradient,"NewtonResidual"+_postindex,"control");
  out<< "\t Newton step: " <<iter<<"\t Residual (abs.): "<<sqrt(res)<<"\n";
  out<< "\t Newton step: " <<iter<<"\t Residual (rel.): "<<std::scientific<<sqrt(res)/sqrt(res)<<"\n";
  this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
  int liniter = 0;
  int lineiter =0;
  unsigned int miniter = 0;
  if(global_tol > 0.)
    miniter = 1;

  global_tol =  std::max(_nonlinear_global_tol,global_tol);
  while(( (res >= global_tol*global_tol) && (res >= _nonlinear_tol*_nonlinear_tol*firstres) ) ||  iter < miniter )
  {
    iter++;
    this->GetOutputHandler()->SetIterationNumber(iter,"OptNewton"+_postindex);

    if(iter > _nonlinear_maxiter)
    {
      throw DOpEIterationException("Iteration count exceeded bounds!","ReducedNewtonAlgorithm::Solve");
    }

    //Compute a search direction
    try
    {
      liniter = SolveReducedLinearSystem(q,gradient,gradient_transposed, dq);
    }
    catch(DOpEIterationException& e)
    {
      //Seems uncritical too many linear solves, it'll probably work
      //So only write a warning, and continue.
      this->GetExceptionHandler()->HandleException(e,"ReducedNewtonAlgorithm::Solve");
      liniter = -1;
    }
    catch(DOpENegativeCurvatureException& e)
    {
      this->GetExceptionHandler()->HandleException(e,"ReducedNewtonAlgorithm::Solve");
      lineiter = -2;
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::Solve");
    }
    //Linesearch
    try
    {
      lineiter = ReducedNewtonLineSearch(dq,gradient,cost,q);
    }
    catch(DOpEIterationException& e)
    {
      //Seems uncritical too many line search steps, it'll probably work
      //So only write a warning, and continue.
      this->GetExceptionHandler()->HandleException(e,"ReducedNewtonAlgorithm::Solve");
      lineiter = -1;
    }
    //catch(DOpEException& e)
    //{
    //  this->GetExceptionHandler()->HandleCriticalException(e);
    //}

    out<< "CostFunctional: " << cost;
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());

    if (_compute_functionals_in_every_step == true)
      {
	try
	  {
	    this->GetReducedProblem()->ComputeReducedFunctionals(q);
	  }
	catch(DOpEException& e)
	  {
	    this->GetExceptionHandler()->HandleCriticalException(e);
	  }
      }


    //Prepare the next Iteration
    try
    {
      this->GetReducedProblem()->ComputeReducedGradient(q,gradient,gradient_transposed);
    }
    catch(DOpEException& e)
    {
      this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::Solve");
    }

    this->GetOutputHandler()->Write(q,"Control"+_postindex,"control");
    this->GetOutputHandler()->Write(gradient,"NewtonResidual"+_postindex,"control");

    res = Residual(gradient,gradient_transposed);//gradient*gradient_transposed;

    out<<"\t Newton step: " <<iter<<"\t Residual (rel.): "<<this->GetOutputHandler()->ZeroTolerance(sqrt(res)/sqrt(firstres),1.0)<< "\t LinearIters ["<<liniter<<"]\t LineSearch {"<<lineiter<<"} ";
    this->GetOutputHandler()->Write(out,3+this->GetBasePriority());
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
      this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::Solve");
    }

  out << "**************************************************\n";
  out << "*        Stopping Reduced Newton Algorithm       *\n";
  out << "*             after "<<std::setw(6)<<iter<<"  Iterations           *\n";
  out.precision(4);
  out << "*             with rel. Residual "<<std::scientific << std::setw(11) << this->GetOutputHandler()->ZeroTolerance(sqrt(res)/sqrt(firstres),1.0)<<"          *\n";
  out.precision(10);
  out << "**************************************************";
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority(),1,1);
  return iter;
}

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::SolveReducedLinearSystem(const ControlVector<VECTOR>& q,
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
  this->GetOutputHandler()->Write(out,4+this->GetBasePriority());

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
      this->GetExceptionHandler()->HandleCriticalException(e,"ReducedNewtonAlgorithm::SolveReducedLinearSystem");
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

    assert(res >= 0.);
    out<<"\t Cg step: " <<iter<<"\t Residual: "<<sqrt(res);
    this->GetOutputHandler()->Write(out,4+this->GetBasePriority());

    cgbeta = res / oldres; //Fletcher-Reeves
    d*= cgbeta;
    d.equ(-1,r_transposed);
  }
  return iter;
}


/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int ReducedNewtonAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::ReducedNewtonLineSearch(const ControlVector<VECTOR>& dq,
							     const ControlVector<VECTOR>&  gradient,
							     double& cost,
							     ControlVector<VECTOR>& q)
{
  double rho = _linesearch_rho;
  double c   = _linesearch_c;

  double costnew = 0.;
  bool force_linesearch = false;

  q+=dq;
  try
  {
     costnew = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
  {
//    this->GetExceptionHandler()->HandleException(e);
    force_linesearch = true;   
    this->GetOutputHandler()->Write("Computing Cost Failed",4+this->GetBasePriority());
  }

  double alpha=1;
  unsigned int iter =0;

  double reduction = gradient*dq;
  if(reduction > 0)
  {
    this->GetOutputHandler()->WriteError("Waring: computed direction doesn't seem to be a descend direction!");
    reduction = 0;
  }

  if(_line_maxiter > 0)
  {
    if(fabs(reduction) < 1.e-10*cost)
      reduction = 0.;
    if(std::isinf(costnew) || std::isnan(costnew) || (costnew >= cost + c*alpha*reduction) || force_linesearch)
    {
      this->GetOutputHandler()->Write("\t linesearch ",4+this->GetBasePriority());
      while(std::isinf(costnew) || std::isnan(costnew) || (costnew >= cost + c*alpha*reduction) || force_linesearch)
      {
	iter++;
	if(iter > _line_maxiter)
	{ 
	  if(force_linesearch)
	  {
	    throw DOpEException("Iteration count exceeded bounds while unable to compute the CostFunctional!","ReducedNewtonAlgorithm::ReducedNewtonLineSearch");
	  }
	  else
	  {
	    cost = costnew;
	    throw DOpEIterationException("Iteration count exceeded bounds!","ReducedNewtonAlgorithm::ReducedNewtonLineSearch");
	  }
	}
	force_linesearch = false;
	q.add(alpha*(rho-1.),dq);
	alpha *= rho;

	try
	{
	  costnew = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
	}
	catch(DOpEException& e)
	{
	  //this->GetExceptionHandler()->HandleException(e);
	  force_linesearch = true;
	  this->GetOutputHandler()->Write("Computing Cost Failed",4+this->GetBasePriority());
	}
      }
    }
    cost = costnew;
  }

  return iter;

}


}
#endif
