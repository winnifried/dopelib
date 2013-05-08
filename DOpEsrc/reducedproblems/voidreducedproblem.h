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

#ifndef _VOID_REDUCED_PROBLEM_H_
#define _VOID_REDUCED_PROBLEM_H_

#include "reducedprobleminterface.h"
#include "integrator.h"
#include "parameterreader.h"
#include "statevector.h"
#include "constraintvector.h"

#include <lac/vector.h>

#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include "optproblemcontainer.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "newtonsolver.h"
#include "newtonsolvermixeddims.h"
#include "integratormixeddims.h"
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "constraintinterface.h"

#include <base/data_out_base.h>
#include <numerics/data_out.h>
#include <numerics/matrix_tools.h>
#include <numerics/vector_tools.h>
#include <base/function.h>
#include <lac/sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/sparse_direct.h>

#include <fstream>
namespace DOpE
{
   /**
   * Basic class to reduce an optimization problem min J(q,u) s.t. a(q,u)=0
   * to a problem in q alone. Here we consider the case when there is no equation
   * to be solved, i.e., we solve min J(q).
   *
   * @tparam <CONTROLNONLINEARSOLVER>    Newton solver for the control variables.
   * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,
   *                                     e.g, Integrator or IntegratorMixedDimensions.
   * @tparam <PROBLEM>                   PDE- or optimization problem under consideration.
   * @tparam <dopedim>                   The dimension for the control variable.
   * @tparam <dealdim>                   The dimension for the state variable.
   */
  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM,
    typename VECTOR, int dopedim, int dealdim>
    class VoidReducedProblem : public ReducedProblemInterface<PROBLEM,VECTOR,dealdim>
  {
  public:
    /**
     * Constructur for the VoidReducedProblem.
     *
     * @param OP                Problem is given to the stationary solver.
     * @param param_reader      An object which has run time data.
     */
  template<typename INTEGRATORDATACONT>
    VoidReducedProblem(PROBLEM *OP,
		       std::string vector_behavior,
		       ParameterReader &param_reader,
		       INTEGRATORDATACONT& idc,
		       int base_priority=0);
    ~VoidReducedProblem();

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
     * @param algo_type         A string which indicates the type of the
     *                          algorithm. Actual, there is one possibility to
     *                          use the `reduced' algorithm.
     */
    void ReInit();

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     *
     */
    bool ComputeReducedConstraints(const ControlVector<VECTOR>& q, ConstraintVector<VECTOR>& g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     *
     */
    void GetControlBoxConstraints(ControlVector<VECTOR>& lb, ControlVector<VECTOR>& ub) ;

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     *
     * @param q            The ControlVector is given to this function.
     */
    void ComputeReducedGradient(const ControlVector<VECTOR>& q,
				ControlVector<VECTOR>& gradient,
				ControlVector<VECTOR>& gradient_transposed);


    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     *
     * @param q            The ControlVector is given to this function.
     */
    void ComputeReducedConstraintGradient(const ConstraintVector<VECTOR>& direction,
					  const ConstraintVector<VECTOR>& constraints,
					  ConstraintVector<VECTOR>& gradient);

    /******************************************************/

    /**
     * Returns the functional values to compute.
     *
     * @param q            The ControlVector is given to this function.
     *
     * @return             Returns a double values of the computed functional.
     */
    double ComputeReducedCostFunctional(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function computes reduced functionals of interest within
     * a time dependent computation.
     *
     * @param q            The ControlVector is given to this function.
     */
    void ComputeReducedFunctionals(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class.
     * We assume that adjoint state z(u(q)) is already computed.
     *
     * @param q                             The ControlVector is given to this function.
     * @param direction                     Documentation will follow later.
     * @param hessian_direction             Documentation will follow later.
     * @paramhessian_direction_transposed   Documentation will follow later.
     */
    void ComputeReducedHessianVector(const ControlVector<VECTOR>& q,
				     const ControlVector<VECTOR>& direction,
				     ControlVector<VECTOR>& hessian_direction,
				     ControlVector<VECTOR>& hessian_direction_transposed);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class.
     * We assume that adjoint state z(u(q)) is already computed.
     *
     * @param q                             The ControlVector is given to this function.
     * @param direction                     Documentation will follow later.
     * @param hessian_direction             Documentation will follow later.
     * @paramhessian_direction_transposed   Documentation will follow later.
     */
    void ComputeReducedHessianInverseVector(const ControlVector<VECTOR>& q,
					    const ControlVector<VECTOR>& direction,
					    ControlVector<VECTOR>& hessian_direction);

    /******************************************************/

//    /**
//     * Function does not make sense when we do not have a state variable.
//     * Just needed for compatibility issues.
//     */
//    template<class DWRCONTAINER>
//    void
//    ComputeRefinementIndicators(DWRCONTAINER& dwrc);

    /******************************************************/

    /**
     *  This function calls GetU().PrintInfos(out) function which
     *  prints information on this vector into the given stream.
     *
     *  @param out    The output stream.
     */
    void StateSizeInfo(std::stringstream& out) { out<<"Not Present"<<std::endl; }

    /******************************************************/

    /**
     *  Here, the given BlockVector<double> v is printed to a file of *.vtk format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The BlockVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     */
    void WriteToFile(const dealii::BlockVector<double> &v,
		     std::string name,
		     std::string outfile,
		     std::string dof_type,
		     std::string filetype);

    /******************************************************/

    /**
     *  Here, the given ControlVector v is printed to a file of *.vtk format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The ControlVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     */
    void WriteToFile(const ControlVector<VECTOR> &v,
		     std::string name,
		     std::string outfile,
		     std::string dof_type,
		     std::string filetype);

    /**
     * Basic function to write a std::vector to a file.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     *  Doesn't make sense here so aborts if called!
     */
    void WriteToFile(const std::vector<double> &v __attribute__((unused)),
		     std::string outfile __attribute__((unused)))  { abort(); }

    /**
     * Function that returns a name of this class to allow differentiation in the output
     *
     */
    std::string GetName() const
    {
      return "VoidReducedProblem";
    }
  protected:
    CONTROLNONLINEARSOLVER& GetControlNonlinearSolver();
    CONTROLINTEGRATOR& GetControlIntegrator() { return _control_integrator; }

    //Solves the Equation H(q) sol = rhs
    void H_inverse(const ControlVector<VECTOR>& q,
		   const ControlVector<VECTOR>& rhs,
		   ControlVector<VECTOR>& sol);

  private:
    CONTROLINTEGRATOR _control_integrator;
    CONTROLNONLINEARSOLVER _nonlinear_gradient_solver;

    bool _build_control_matrix;
    bool  _gradient_reinit;
    std::vector<ControlVector<VECTOR>*> _constraint_gradient;
    std::string _vector_behavior;

    ConstraintVector<VECTOR> _constraints;

  };

/*************************************************************************/
/*****************************IMPLEMENTATION******************************/
/*************************************************************************/
  using namespace dealii;

/******************************************************/
template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
 void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
  ::declare_params(ParameterReader &param_reader)
{
  CONTROLNONLINEARSOLVER::declare_params(param_reader);
}
/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  template<typename INTEGRATORDATACONT>
 VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
    ::VoidReducedProblem(PROBLEM *OP,
			 std::string vector_behavior,
			 ParameterReader &param_reader,
			 INTEGRATORDATACONT& idc,
			 int base_priority)
  : ReducedProblemInterface<PROBLEM, VECTOR,  dealdim>(OP,base_priority),
  _control_integrator(idc),
  _nonlinear_gradient_solver(_control_integrator, param_reader),
  _constraints(OP->GetSpaceTimeHandler(),vector_behavior)
{
      //ReducedProblems should be ReInited
      {
	_gradient_reinit = true;
      }
      _vector_behavior = vector_behavior;
}

/******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::~VoidReducedProblem()
{
  for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
  {
    if(_constraint_gradient[i] != NULL)
    {
      delete _constraint_gradient[i];
    }
  }
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  CONTROLNONLINEARSOLVER& VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
  {
    if((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() == "hessian"))
    {
      return _nonlinear_gradient_solver;
    }
    else
    {
      throw DOpEException("No Solver for Problem type:`"+this->GetProblem()->GetType() +"' found","VoidReducedProblem::GetControlNonlinearSolver");

    }
  }
/******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ReInit()
{
  ReducedProblemInterface<PROBLEM, VECTOR, dealdim>::ReInit();

  //Some Solvers must be reinited when called
  // Better have subproblems, so that solver can be reinited here
  {
    _gradient_reinit = true;
  }
  _build_control_matrix = true;

  _constraints.ReInit();
  _constraint_gradient.resize(_constraints.GetGlobalConstraints().size(), NULL);
  for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
  {
    if(_constraint_gradient[i] == NULL)
    {
      _constraint_gradient[i] = new ControlVector<VECTOR>(this->GetProblem()->GetSpaceTimeHandler(),_vector_behavior);
    }
    _constraint_gradient[i]->ReInit();
  }
}

/******************************************************/


template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
bool VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(const ControlVector<VECTOR>& q, ConstraintVector<VECTOR>& g)
{
  this->GetOutputHandler()->Write(this->GetName()+"->Evaluating Constraints:",4+this->GetBasePriority());
  this->SetProblemType("constraints");

  g=0;
  //Local constraints
  //this->GetProblem()->ComputeLocalControlConstraints(q.GetSpacialVector(),g.GetSpacialVector("local"));
  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedConstraints");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

  this->GetControlIntegrator().ComputeLocalControlConstraints(*(this->GetProblem()),g.GetSpacialVector("local"));

  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("control");
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedConstraints");
  }

  //Global in Space Constraints
  unsigned int nglobal = g.GetGlobalConstraints().size();

  if(nglobal > 0)
  {
    for(unsigned int i = 0; i < nglobal; i++)
    {
       this->SetProblemType("global_constraints",i);
       this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

       if(dopedim==dealdim)
       {
	 this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
       }
       else if(dopedim == 0)
       {
	 this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
       }
       else
       {
	 throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedConstraints");
       }

       double ret = 0;
       bool found=false;

       if(this->GetProblem()->GetConstraintType().find("domain") != std::string::npos)
       {
	 found = true;
	 ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
       }
       if(this->GetProblem()->GetConstraintType().find("point") != std::string::npos)
       {
	 found = true;
	 ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
       }
       if(this->GetProblem()->GetConstraintType().find("boundary") != std::string::npos)
       {
	 found = true;
	 ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
       }
       if(this->GetProblem()->GetConstraintType().find("face") != std::string::npos)
       {
	 found = true;
	 ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
       }

       if(!found)
       {
	 throw DOpEException("Unknown Constraint Type: " + this->GetProblem()->GetConstraintType(),"VoidReducedProblem::ComputeReducedConstraints");
       }
       g.GetGlobalConstraints()(i)=ret;

       if(dopedim==dealdim)
       {
	 this->GetControlIntegrator().DeleteDomainData("control");
       }
       else if(dopedim==0)
       {
	 this->GetControlIntegrator().DeleteParamData("control");
	 q.UnLockCopy();
       }
       else
       {
	 throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedConstraints");
       }
       this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());

    }
  }

  this->GetProblem()->PostProcessConstraints(g);

  return this->GetProblem()->IsFeasible(g);
}
/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
::GetControlBoxConstraints(ControlVector<VECTOR>& lb, ControlVector<VECTOR>& ub)
{
  this->GetProblem()->GetControlBoxConstraints(lb.GetSpacialVector(),ub.GetSpacialVector());
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
::ComputeReducedGradient(const ControlVector<VECTOR>& q,
			 ControlVector<VECTOR>& gradient,
			 ControlVector<VECTOR>& gradient_transposed)
{
  this->GetOutputHandler()->Write(this->GetName()+"->Computing Reduced Gradient:",4+this->GetBasePriority());

  //Preparations for ControlInTheDirichletData
  if(this->GetProblem()->HasControlInDirichletData())
  {
    throw DOpEException("Control in Dirichlet-Data not implemented","VoidReducedProblem::ComputeReducedGradient");
  }

  this->SetProblemType("gradient");
  if(_gradient_reinit ==  true)
  {
    GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
    _gradient_reinit = false;
  }

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedGradient");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().AddDomainData("constraints_local",&_constraints.GetSpacialVector("local"));
  this->GetControlIntegrator().AddParamData("constraints_global",&_constraints.GetGlobalConstraints());

  gradient_transposed = 0.;
  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("last_newton_solution",&(gradient_transposed.GetSpacialVector()));
    this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),gradient.GetSpacialVector(),true);
    this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("last_newton_solution",&(gradient_transposed.GetSpacialVectorCopy()));
    this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),gradient.GetSpacialVector(),true);

    this->GetControlIntegrator().DeleteParamData("last_newton_solution");
    gradient_transposed.UnLockCopy();
  }
  gradient *= -1.;
  if(this->GetProblem()->GetFunctionalType() == "algebraic")
  {
    dealii::BlockVector<double> tmp = gradient.GetSpacialVector();
    this->GetControlIntegrator().ComputeNonlinearAlgebraicResidual(*(this->GetProblem()),tmp);
    gradient.GetSpacialVector() += tmp;
  }


  this->GetControlIntegrator().DeleteDomainData("constraints_local");
  this->GetControlIntegrator().DeleteParamData("constraints_global");


  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());

  this->GetOutputHandler()->Write (gradient, "Gradient"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
  this->GetOutputHandler()->Write (gradient_transposed, "Gradient_Transposed"+this->GetPostIndex(), this->GetProblem()->GetDoFType());

  //Compute derivatives of global constraints
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Computing Constraint Gradient:",4+this->GetBasePriority());

    for(unsigned int i = 0; i < _constraints.GetGlobalConstraints().size(); i++)
    {
      //this->SetProblemType("local_global_constraint_gradient",i);
      this->SetProblemType("global_constraint_gradient",i);
      this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()),_constraint_gradient[i]->GetSpacialVector(),true);
    }
  }
  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("control");
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedGradient");
  }
  gradient_transposed = gradient;
  gradient_transposed*= 1./gradient.GetSpacialVector().size();
}
/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
::ComputeReducedConstraintGradient(const ConstraintVector<VECTOR>& direction,
				   const ConstraintVector<VECTOR>& constraints,
				   ConstraintVector<VECTOR>& gradient)
{
  this->GetProblem()->ComputeReducedConstraintGradient(direction,constraints,gradient);
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
double VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(const ControlVector<VECTOR>& q)
{
  double ret = 0;
  bool found=false;
  //Functional may depend on constraints
  if(!ComputeReducedConstraints(q,_constraints))
  {
    //Infeasible q!
    throw DOpEException("Infeasible value!","VoidReducedProblem::ComputeReducedCostFunctional");
  }

  this->GetOutputHandler()->Write(this->GetName()+"->Computing Cost Functional:",4+this->GetBasePriority());

  this->SetProblemType("cost_functional");

  this->GetControlIntegrator().AddDomainData("constraints_local",&_constraints.GetSpacialVector("local"));
  this->GetControlIntegrator().AddParamData("constraints_global",&_constraints.GetGlobalConstraints());

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedCostFunctional");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

  if(this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
  {
    found = true;
    ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
  }
  if(this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
  {
    found = true;
    ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
  }
  if(this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
  {
    found = true;
    ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
  }
  if(this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
  {
    found = true;
    ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
  }
  if(this->GetProblem()->GetFunctionalType().find("algebraic") != std::string::npos)
  {
    found = true;
    ret += this->GetControlIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
  }
  if(!found)
  {
      throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),"VoidReducedProblem::ComputeReducedCostFunctional");
  }

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("control");
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedCostFunctional");
  }
  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().DeleteDomainData("constraints_local");
  this->GetControlIntegrator().DeleteParamData("constraints_global");
  return ret;
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(const ControlVector<VECTOR>& q)
{
  this->GetOutputHandler()->Write(this->GetName()+"->Computing Functionals:",4+this->GetBasePriority());

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedFunctionals");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

  for(unsigned int i=0; i<this->GetProblem()->GetNFunctionals(); i++)
  {
    double ret = 0;
    bool found=false;

    this->SetProblemType("aux_functional",i);



    if(this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
    {
      found = true;
      ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
    }
    if(this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
    {
      found = true;
      ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
    }
    if(this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
    {
      found = true;
      ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
    }
    if(this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
    {
      found = true;
      ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
    }

    if(!found)
    {
      throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),"VoidReducedProblem::ComputeReducedFunctionals");
    }
    std::stringstream out;
    this->GetOutputHandler()->InitOut(out);
    out<<this->GetName()<<"->"<< this->GetProblem()->GetFunctionalName() <<": "<< ret;
    this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
  }

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("control");
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedFunctionals");
  }
  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());

}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(const ControlVector<VECTOR>& q,
						      const ControlVector<VECTOR>& direction,
						      ControlVector<VECTOR>& hessian_direction,
						      ControlVector<VECTOR>& hessian_direction_transposed)
{
  this->GetOutputHandler()->Write(this->GetName()+"->Computing ReducedHessianVector:",4+this->GetBasePriority());

  this->GetOutputHandler()->Write("\t"+this->GetName()+"->Computing Representation of the Hessian:",5+this->GetBasePriority());
  //Preparations for Control In The Dirichlet Data
  if(this->GetProblem()->HasControlInDirichletData())
  {
   throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
  }
   //Endof Dirichletdata Preparations

  this->SetProblemType("hessian");

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("dq",&(direction.GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("dq",&(direction.GetSpacialVectorCopy()));
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().AddDomainData("constraints_local",&_constraints.GetSpacialVector("local"));
  this->GetControlIntegrator().AddParamData("constraints_global",&_constraints.GetGlobalConstraints());

  {
    hessian_direction_transposed = 0.;
    if(dopedim == dealdim)
    {
      this->GetControlIntegrator().AddDomainData("last_newton_solution",&(hessian_direction_transposed.GetSpacialVector()));
      this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),hessian_direction.GetSpacialVector(),true);
      this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
    }
    else if(dopedim == 0)
    {
      this->GetControlIntegrator().AddParamData("last_newton_solution",&(hessian_direction_transposed.GetSpacialVectorCopy()));
      this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),hessian_direction.GetSpacialVector(),true);
      this->GetControlIntegrator().DeleteParamData("last_newton_solution");
      hessian_direction_transposed.UnLockCopy();
    }
    hessian_direction *= -1.;
    hessian_direction_transposed=hessian_direction;
    //Compute l^2 representation of the HessianVector
    //hessian Matrix is the same as control matrix
    _build_control_matrix = this->GetControlNonlinearSolver().NonlinearSolve(*(this->GetProblem()),hessian_direction_transposed.GetSpacialVector(),true,_build_control_matrix,this->GetBasePriority());

    //Add constraint Gradients

    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      std::stringstream name;
      name << "constraint_gradient_"<<i;
      this->GetControlIntegrator().AddDomainData(name.str(),&(_constraint_gradient[i]->GetSpacialVector()));
    }
    if(this->GetProblem()->GetFunctionalType() == "algebraic")
    {
      dealii::BlockVector<double> tmp = hessian_direction.GetSpacialVector();
      this->GetControlIntegrator().ComputeNonlinearAlgebraicResidual(*(this->GetProblem()),tmp);
      hessian_direction.GetSpacialVector() += tmp;
      hessian_direction_transposed.GetSpacialVector() += tmp;
    }
    //Delete constraint Gradients
    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      std::stringstream name;
      name << "constraint_gradient_"<<i;
      this->GetControlIntegrator().DeleteDomainData(name.str());
    }

    this->GetOutputHandler()->Write (hessian_direction, "HessianDirection"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
    this->GetOutputHandler()->Write (hessian_direction_transposed, "HessianDirection_Transposed"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
  }

  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().DeleteDomainData("constraints_local");
  this->GetControlIntegrator().DeleteParamData("constraints_global");

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("dq");
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("dq");
    this->GetControlIntegrator().DeleteParamData("control");
    direction.UnLockCopy();
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
  }
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianInverseVector(const ControlVector<VECTOR>& q,
																 const ControlVector<VECTOR>& direction,
																 ControlVector<VECTOR>& hessian_direction)
{
  this->GetOutputHandler()->Write(this->GetName()+"->Computing ReducedHessianInverseVector:",4+this->GetBasePriority());

  //Solve (A^T H^(-1)A + diag(beta)) y = -A^TH^(-1) direction
  Vector<double> y(_constraint_gradient.size());
  ControlVector<VECTOR> tmp(q);
  {
    Vector<double> rhs(_constraint_gradient.size());
    Vector<double> beta(_constraint_gradient.size());
    H_inverse(q,direction,tmp);
    tmp *=-1.;
    //store H^{-1}direction
    hessian_direction.equ(1.,tmp);


    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      rhs(i) = (*(_constraint_gradient[i]))*tmp;
    }

    dealii::FullMatrix<double> X(_constraint_gradient.size(),_constraint_gradient.size());
    dealii::FullMatrix<double> Y(_constraint_gradient.size(),_constraint_gradient.size());
    dealii::FullMatrix<double> Z(_constraint_gradient.size(),_constraint_gradient.size());
    Y = 0.;
    //Build Matrix -A^TH^{-1}A
    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      H_inverse(q,*(_constraint_gradient[i]),tmp);

      this->GetOutputHandler()->Write (*(_constraint_gradient[i]), "ConstraintGradient"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
      this->GetOutputHandler()->Write (tmp, "HessianInverseConstraintGradient"+this->GetPostIndex(), this->GetProblem()->GetDoFType());

      for(unsigned int j = 0; j < _constraint_gradient.size(); j++)
      {
	X(i,j) = ((*(_constraint_gradient[j]))*tmp);
      }
    }

    //Build diag(beta)
    const ConstraintVector<VECTOR>* multiplier = this->GetProblem()->GetAuxiliaryConstraint("mma_multiplier");
    const dealii::Vector<double>& lambda = multiplier->GetGlobalConstraints();

    assert(lambda.size()  == _constraint_gradient.size());

    this->GetProblem()->ComputeReducedGlobalConstraintHessian(_constraints,beta);

    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      Y(i,i) = 1./(lambda(i)*beta(i));
      assert(lambda(i) > 0.);
      assert(beta(i) > 0.);
    }
    X.add(1.,Y);
    Y.invert(X);
    Y.vmult(y,rhs);
  }
  {
    //Solve H hessian_direction = -Ay+direction
    //Da hessian_direction = H^{-1}direction
    //fehlt nur noch hessian_direction -= H^{-1}Ay
    for(unsigned int i = 0; i < _constraint_gradient.size(); i++)
    {
      H_inverse(q,*(_constraint_gradient[i]),tmp);
      hessian_direction.add(-1.*y(i),tmp);
    }
  }
  this->GetOutputHandler()->Write (hessian_direction, "HessianInverseDirection"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
}
/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::H_inverse(const ControlVector<VECTOR>& q,
													const ControlVector<VECTOR>& rhs,
													ControlVector<VECTOR>& sol)
{
  sol = 0.;
  this->SetProblemType("hessian_inverse");
  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().AddDomainData("dq",&(rhs.GetSpacialVector()));
    this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
  }
  else if(dopedim == 0)
  {
    this->GetControlIntegrator().AddParamData("dq",&(rhs.GetSpacialVectorCopy()));
    this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::H_inverse");
  }
  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().AddDomainData("constraints_local",&_constraints.GetSpacialVector("local"));
  this->GetControlIntegrator().AddParamData("constraints_global",&_constraints.GetGlobalConstraints());

  if(this->GetProblem()->GetFunctionalType() == "algebraic")
  {
    this->GetControlIntegrator().ComputeNonlinearAlgebraicResidual(*(this->GetProblem()),sol.GetSpacialVector());
  }
  else
  {
    throw DOpEException("Wrong FunctionalType "+this->GetProblem()->GetFunctionalType(),"VoidReducedProblem::H_inverse");
  }

  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetControlIntegrator());
  this->GetControlIntegrator().DeleteDomainData("constraints_local");
  this->GetControlIntegrator().DeleteParamData("constraints_global");

  if(dopedim==dealdim)
  {
    this->GetControlIntegrator().DeleteDomainData("dq");
    this->GetControlIntegrator().DeleteDomainData("control");
  }
  else if(dopedim==0)
  {
    this->GetControlIntegrator().DeleteParamData("dq");
    this->GetControlIntegrator().DeleteParamData("control");
    rhs.UnLockCopy();
    q.UnLockCopy();
  }
  else
  {
    throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
  }
}
/******************************************************/
template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const dealii::BlockVector<double> &v,
						       std::string  name, std::string outfile,
						       std::string dof_type,
						       std::string filetype)
{
  if(dof_type == "control")
  {
#if dope_dimension >0
    DataOut<dopedim> data_out;
    data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

    data_out.add_data_vector (v,name);
    data_out.build_patches ();

    std::ofstream output(outfile.c_str());

    if(filetype == ".vtk")
    {
      data_out.write_vtk (output);
    }
    else
    {
      throw DOpEException("Don't know how to write filetype `" + filetype + "'!","VoidReducedProblem::WriteToFile");
    }
#else
    if(filetype == ".txt")
      {
	std::ofstream output(outfile.c_str());
	Vector<double> off;
	off = v;
	for(unsigned int i =0; i<off.size(); i++)
	{
	  output << off(i) <<std::endl;
	}
      }
    else
    {
      throw DOpEException("Don't know how to write filetype `" + filetype + "'!","VoidReducedProblem::WriteToFile");
    }
#endif
  }
  else
  {
    throw DOpEException("No such DoFHandler `" + dof_type + "'!","VoidReducedProblem::WriteToFile");
  }
}

/******************************************************/

template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const ControlVector<VECTOR> &v,
						       std::string  name, std::string outfile,
						       std::string dof_type,
						       std::string filetype)
{
  WriteToFile(v.GetSpacialVector(), name, outfile,dof_type, filetype);
}
/******************************************************/

//  template<typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR,
//      typename PROBLEM, typename VECTOR, int dopedim, int dealdim>
//    template<class DWRCONTAINER>
//      void
//      VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM,
//          VECTOR, dopedim, dealdim>::ComputeRefinementIndicators(
//          DWRCONTAINER& dwrc)
//
//      {
//        throw DOpEException(
//            "Function makes no sense in this context because we have no state variable.",
//            "VoidReducedProblem::ComputeRefinementIndicators");
//        return 0;
//      }
}
#endif
