#ifndef _REDUCED_SNOPT__ALGORITHM_H_
#define _REDUCED_SNOPT__ALGORITHM_H_

#include "reducedalgorithm.h"
#include "parameterreader.h"

#include "snopt_wrapper.h"

#include <iostream>
#include <assert.h>
#include <iomanip>

namespace DOpE
{

  template <typename PROBLEM, typename VECTOR, int dopedim,  int dealdim>
    class Reduced_SnoptAlgorithm : public ReducedAlgorithm<PROBLEM, VECTOR, dopedim,dealdim>
  {
  public:
    Reduced_SnoptAlgorithm(PROBLEM* OP,
			   ReducedProblemInterface<PROBLEM, VECTOR,dopedim,dealdim>* S,
			   std::string vector_behavior,
			   ParameterReader &param_reader,
			   DOpEExceptionHandler<VECTOR>* Except=NULL,
			   DOpEOutputHandler<VECTOR>* Output=NULL,
			   int base_priority=0);
    ~Reduced_SnoptAlgorithm();

    static void declare_params(ParameterReader &param_reader);

    /**
     * This solves an Optimizationproblem in only the control variable
     * using the commercial optimization library snopt. 
     * To use it you need to define the compiler flag 
     * WITH_SNOPT and ensure that all required snopt headers and 
     * libraries are within the path or otherwise known.
     *
     * @param q           The initial point.
     * @param global_tol  An optional parameter specifying the required  tolerance.
     *                    The actual tolerance is the maximum of this and the one specified in the param
     *                    file. Its default value is negative, so that it has no influence if not specified.
     */
    virtual int Solve(ControlVector<VECTOR>& q,double global_tol=-1.);

  protected:

  private:
#ifdef WITH_SNOPT
    int rsa_func_(DOpEWrapper::SNOPT_FUNC_DATA& data);
#endif
    std::string _postindex;
    std::string _vector_behavior;

    double _func_prec, _feas_tol, _opt_tol;
    int _max_inner_iter, _max_outer_iter;
    bool _capture_out;
  };

  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  using namespace dealii;

  /******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
void Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("reduced_snoptalgorithm parameters");
    param_reader.declare_entry("function precision","1.e-6",Patterns::Double(0,1),"Declares how many digits we assume to have computed correctly, this should correspond to the tolerance used for the PDE solve");
    param_reader.declare_entry("feasibility tol","1.e-5",Patterns::Double(0,1),"Tolerance with respect to the feasibility of the constraints."); 
    param_reader.declare_entry("optimality tol","1.e-5",Patterns::Double(0,1),"Tolerance with respect to the optimality condition.");
    param_reader.declare_entry("max inner iterations","500",Patterns::Integer(0),"Maximal allowed number of inner iterations over all outer iterations");
    param_reader.declare_entry("max iterations","1000",Patterns::Integer(0),"Maximal allowed number of outer iterations over all outer iterations");
    param_reader.declare_entry("capture snopt output","true",Patterns::Bool(),"Select if the snopt output should be stored in log file");
    ReducedAlgorithm<PROBLEM, VECTOR, dopedim,dealdim>::declare_params(param_reader);
  }

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::Reduced_SnoptAlgorithm(PROBLEM* OP,
										  ReducedProblemInterface<PROBLEM, VECTOR,dopedim, dealdim>* S,
										  std::string vector_behavior,
										  ParameterReader &param_reader,
										  DOpEExceptionHandler<VECTOR>* Except,
										  DOpEOutputHandler<VECTOR>* Output,
										  int base_priority)
  : ReducedAlgorithm<PROBLEM, VECTOR,dopedim, dealdim>(OP,S,param_reader,Except,Output,base_priority)
  {

    param_reader.SetSubsection("reduced_snoptalgorithm parameters");

    _func_prec      = param_reader.get_double("function precision");
    _feas_tol       = param_reader.get_double("feasibility tol");
    _opt_tol        = param_reader.get_double("optimality tol");
    _max_inner_iter = param_reader.get_integer("max inner iterations");
    _max_outer_iter = param_reader.get_integer("max iterations");
    _capture_out   = param_reader.get_bool("capture snopt output");
    
    _vector_behavior = vector_behavior;
    _postindex = "_"+this->GetProblem()->GetName();

  }

/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::~Reduced_SnoptAlgorithm()
  {

  }


/******************************************************/

template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::Solve(ControlVector<VECTOR>& q,double global_tol)
{
#ifndef WITH_SNOPT
  throw DOpEException("To use this algorithm you need to have SNOPT installed! To use this set the WITH_SNOP CompilerFlag.","Reduced_SnoptAlgorithm::Solve");
#else 
  q.ReInit();
  
  ControlVector<VECTOR> dq(q);
  ControlVector<VECTOR> q_min(q), q_max(q);
  this->GetReducedProblem()->GetControlBoxConstraints(q_min,q_max);

  ConstraintVector<VECTOR> constraints(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),_vector_behavior);
  
  unsigned int iter=0;
  double cost=0.;
  double cost_start=0.;
  std::stringstream out;
  this->GetOutputHandler()->InitNewtonOut(out);
  global_tol =  std::max(_opt_tol,global_tol);

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

  this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Snopt"+_postindex);

  this->GetOutputHandler()->Write(q,"Control"+_postindex,"control");

  try
  {
     cost_start = cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
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
  if(_capture_out)
    out<<"*  output will be written to logfile only!     *\n";
  else
    out<<"*  output will not be written to logfile!      *\n";
  out<<"************************************************\n\n";
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority());
  
  this->GetOutputHandler()->DisallowAllOutput();
  if(_capture_out)
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
    doublereal *A  = NULL;

    integer lenG   = n*(neF);//We have that the derivative of J and the global constraints
                             //are nonzero w.r.t. all components. 
    integer neG    = lenG;//Predefine the number of valid entries in iGfun and jGvar will be lenG.
    integer *iGfun = new integer[lenG];
    integer *jGvar = new integer[lenG];

    doublereal *x      = new doublereal[n];
    doublereal *xlow   = new doublereal[n];
    doublereal *xupp   = new doublereal[n];
    doublereal *xmul   = new doublereal[n];
    integer    *xstate = new    integer[n];
    
    doublereal *F      = new doublereal[neF];
    doublereal *Flow   = new doublereal[neF];
    doublereal *Fupp   = new doublereal[neF];
    doublereal *Fmul   = new doublereal[neF];
    integer    *Fstate = new integer[neF];

    integer nxnames = 1;
    integer nFnames = 1;
    char *xnames = new char[nxnames*8];
    char *Fnames = new char[nFnames*8];
    
    integer    ObjRow = 0;
    doublereal ObjAdd = 0;
    
    {
      const VECTOR& gv_q = q.GetSpacialVector();
      const VECTOR& gv_qmin = q_min.GetSpacialVector();
      const VECTOR& gv_qmax = q_max.GetSpacialVector();
      
      for(unsigned int i=0; i < n; i++)
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
      for(unsigned int j = 1; j<neF; j++)
      { //Global constraints are to be given by the user such that 
	//The feasible region is given by <= 0
	//TODO this should be defineable by the user...
	Flow[j] = -1.e+20; 
	Fupp[j] = 0.;
	Fstate[j] = 0;
	for(unsigned int i=0; i < n; i++)
	{
	  iGfun[n*j+i] = j;
	  jGvar[n*j+i] = i;
	}
      }
    }

    //RSAProb.setPrintFile  ( "RSA.out" );
    RSAProb.setProblemSize( n, neF );
    RSAProb.setObjective  ( ObjRow, ObjAdd );
    RSAProb.setA          ( lenA, iAfun, jAvar, A );
    RSAProb.setG          ( lenG, iGfun, jGvar );
    RSAProb.setX          ( x, xlow, xupp, xmul, xstate );
    RSAProb.setF          ( F, Flow, Fupp, Fmul, Fstate );
    RSAProb.setXNames     ( xnames, nxnames );
    RSAProb.setFNames     ( Fnames, nFnames );
    RSAProb.setProbName   ( "RSA" );
    RSAProb.setNeA         ( neA );
    RSAProb.setNeG         ( neG );
  
    DOpEWrapper::SNOPT_A_userfunc_interface = boost::bind<int>(boost::mem_fn(&Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>::rsa_func_),boost::ref(*this),_1);
    RSAProb.setUserFun    ( DOpEWrapper::SNOPT_A_userfunc_ );
 
    RSAProb.setIntParameter( "Derivative option", 1 );
    RSAProb.setRealParameter( "Function precison", _func_prec);
    RSAProb.setRealParameter( "Major optimality tolerance", global_tol);
    RSAProb.setRealParameter( "Major feasibility tolerance", _feas_tol);
    RSAProb.setRealParameter( "Minor feasibility tolerance", _feas_tol);
    RSAProb.setIntParameter( "Minor iterations limit", _max_inner_iter);
    RSAProb.setIntParameter( "Major iterations limit", _max_outer_iter);
    RSAProb.solve(2);
    ret_val = RSAProb.GetReturnStatus();
    {
      VECTOR& gv_q = q.GetSpacialVector();
      for(unsigned int i=0; i < gv_q.size(); i++)
	gv_q(i) = x[i];
    }
    if(iAfun != NULL)
      delete []iAfun;  
    if(jAvar != NULL)
      delete []jAvar;  
    if(A != NULL)
      delete []A;
    delete []iGfun;  delete []jGvar;
    
    delete []x;      delete []xlow;   delete []xupp;
    delete []xmul;   delete []xstate;
    
    delete []F;      delete []Flow;   delete []Fupp;
    delete []Fmul;   delete []Fstate;
    
    delete []xnames; delete []Fnames;
 
  } 
  
  if(_capture_out)
    this->GetOutputHandler()->StopSaveCTypeOutputToLog();
  this->GetOutputHandler()->ResumeOutput();
  out<<"\n************************************************\n";
  out<<"*               SNOPT Finished                 *\n";
  out<<"*          with Exit Code: "<<std::setw(3)<<ret_val;
  if(ret_val == 1)
    out<<" (success)       *\n";
  else
    out<<" (unknown error) *\n";
  out<<"************************************************\n";
  //FIXME What is the result...
  this->GetOutputHandler()->Write(out,1+this->GetBasePriority());

  iter++;
  this->GetOutputHandler()->SetIterationNumber(iter,"Opt_Snopt"+_postindex);

  this->GetOutputHandler()->Write(q,"Control"+_postindex,"control");
  try
  {
     cost = this->GetReducedProblem()->ComputeReducedCostFunctional(q);
  }
  catch(DOpEException& e)
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
    catch(DOpEException& e)
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

#ifdef WITH_SNOPT
template <typename PROBLEM, typename VECTOR, int dopedim,int dealdim>
int Reduced_SnoptAlgorithm<PROBLEM, VECTOR, dopedim, dealdim>
  ::rsa_func_(DOpEWrapper::SNOPT_FUNC_DATA& data )
{
  //Needs to implement the evaluation of j and its derivative using the 
  //Interface required by SNOPT
  ConstraintVector<VECTOR>* constraints = NULL;
  ControlVector<VECTOR> tmp(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),_vector_behavior);
  VECTOR& ref_x = tmp.GetSpacialVector();
  assert(ref_x.size() == *(data.n));
  for(unsigned int i=0; i < ref_x.size(); i++)
    ref_x(i) = (data.x)[i];

  assert(*(data.needF) > 0);
  try
  {
    (data.F)[0] = this->GetReducedProblem()->ComputeReducedCostFunctional(tmp);
  }
  catch(DOpEException& e)
  {
    *(data.Status) = -1;
    this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
    return -1;
  }

  if(*(data.neF) > 1)
  {
    constraints = new ConstraintVector<VECTOR>(this->GetReducedProblem()->GetProblem()->GetSpaceTimeHandler(),_vector_behavior);
    try
    {
      this->GetReducedProblem()->ComputeReducedConstraints(tmp,*constraints);
    }
    catch(DOpEException& e)
    {
      *(data.Status) = -1;
      this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
    
      if(constraints != NULL)
	delete constraints;
      return -1;
    }
    const dealii::Vector<double>& gc = constraints->GetGlobalConstraints();
    assert(*(data.neF) == gc.size()+1);
    for(unsigned int i=0; i < gc.size(); i++)
    {
      (data.F)[i+1] = gc(i);
    }
  }
  

  if(*(data.needG) > 0)
  {
    ControlVector<VECTOR> gradient(tmp);
    ControlVector<VECTOR> gradient_transposed(tmp);
    
    try
    {
      this->GetReducedProblem()->ComputeReducedGradient(tmp,gradient,gradient_transposed);
    }
    catch(DOpEException& e)
    {
      *(data.Status) = -2;
      this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
    
      if(constraints != NULL)
	delete constraints;
      return -1;
    }
    assert(*(data.neG) == *(data.n)**(data.neF));
    const VECTOR& ref_g = gradient_transposed.GetSpacialVector();
    for(unsigned int i=0; i < *(data.n); i++)
    {
      (data.G)[i] = ref_g(i);
    }
    if(constraints != NULL)
    {
      //Evaluate global constraint gradients
      const dealii::Vector<double>& gc = constraints->GetGlobalConstraints();
      for(unsigned int j=0; j < gc.size(); j++)
      {
	try
	{
	  this->GetReducedProblem()->ComputeReducedGradientOfGlobalConstraints(j,tmp,*constraints,gradient,gradient_transposed);
	}
	catch(DOpEException& e)
	{
	  *(data.Status) = -2;
	  this->GetExceptionHandler()->HandleException(e,"Reduced_SnoptAlgorithm::rsa_func_");
    
	  if(constraints != NULL)
	    delete constraints;
	  return -1;
	}
	const VECTOR& ref_g = gradient_transposed.GetSpacialVector();
	for(unsigned int i=0; i < *(data.n); i++)
	{
	  (data.G)[*(data.n)*(j+1)+i] = ref_g(i);
	}
      }
    }
  }


  if(constraints != NULL)
    delete constraints;
  *(data.Status) = 0;
  return 0;
}
#endif //Endof ifdef WITHSNOPT

/*****************************END_OF_NAMESPACE_DOpE*********************************/
}
#endif
