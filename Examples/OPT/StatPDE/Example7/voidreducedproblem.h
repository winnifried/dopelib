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

#ifndef VOID_REDUCED_PROBLEM_H_
#define VOID_REDUCED_PROBLEM_H_

#include <interfaces/reducedprobleminterface.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <include/statevector.h>
#include <include/constraintvector.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <container/optproblemcontainer.h>
#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/dirichletdatainterface.h>
#include <include/dopeexception.h>
#include <templates/newtonsolver.h>
#include <templates/newtonsolvermixeddims.h>
#include <templates/integratormixeddims.h>
#include <templates/cglinearsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/voidlinearsolver.h>
#include <interfaces/constraintinterface.h>

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
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

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
  class VoidReducedProblem : public ReducedProblemInterface<PROBLEM,VECTOR>
  {
  public:
    /**
     * Constructur for the VoidReducedProblem.
     *
     * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
     *
     * @param OP                ProblemContainer.
     * @param vector_behavior   A string indicating how vectors should be stored
     * @param param_reader      An object which has run time data.
     * @param idc               The InegratorDataContainer
     * @param base_priority     An offset for the priority of the output written to
     *                          the OutputHandler
     */
    template<typename INTEGRATORDATACONT>
    VoidReducedProblem(PROBLEM *OP,
                       DOpEtypes::VectorStorageType vector_behavior,
                       ParameterReader &param_reader,
                       INTEGRATORDATACONT &idc,
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
     * Reinitialization after mesh changes to adjust sizes ...
     */
    void ReInit();

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    bool ComputeReducedConstraints(const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void GetControlBoxConstraints(ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub) ;

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedGradient(const ControlVector<VECTOR> &q,
                                ControlVector<VECTOR> &gradient,
                                ControlVector<VECTOR> &gradient_transposed);


    /******************************************************/

    /**
     * Calculates the derivatives of the constraints at the point
     * constraints in the given direction (in the constraint space).
     * Note that this is only different from the identity if
     * the constraints are transformed again, as it happens for instance
     * with the AugmentedLagrangianProblem.
     */
    void ComputeReducedConstraintGradient(const ConstraintVector<VECTOR> &direction,
                                          const ConstraintVector<VECTOR> &constraints,
                                          ConstraintVector<VECTOR> &gradient);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    double ComputeReducedCostFunctional(const ControlVector<VECTOR> &q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedFunctionals(const ControlVector<VECTOR> &q);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedHessianVector(const ControlVector<VECTOR> &q,
                                     const ControlVector<VECTOR> &direction,
                                     ControlVector<VECTOR> &hessian_direction,
                                     ControlVector<VECTOR> &hessian_direction_transposed);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void ComputeReducedHessianInverseVector(const ControlVector<VECTOR> &q,
                                            const ControlVector<VECTOR> &direction,
                                            ControlVector<VECTOR> &hessian_direction);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     * ReducedProblemInterface
     *
     */
    void StateSizeInfo(std::stringstream &out)
    {
      out<<"Not Present"<<std::endl;
    }

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
     *  @param dof_type    Has the DoF type: state or control.
     */
    void WriteToFile(const ControlVector<VECTOR> &v,
                     std::string name,
                     std::string dof_type);

    /**
     * Basic function to write a std::vector to a file.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     *  Doesn't make sense here so aborts if called!
     */
    void WriteToFile(const std::vector<double> &/*v*/,
                     std::string /*outfile*/)
    {
      abort();
    }

    /**
     * Function that returns a name of this class to allow differentiation in the output
     *
     */
    std::string GetName() const
    {
      return "VoidReducedProblem";
    }

    /**
     * Sets the value of the Augmented Lagrangian Value
     * It is assumed that p is positive.
     */
    void SetValue(double p, std::string name)
    {
      if ("p" == name)
        {
          assert(p > 0.);
          p_ = p;
        }
      else
        {
          throw DOpEException("Unknown value " + name,
                              "AumentedLagrangianProblem::SetType");
        }
    }
  protected:
    CONTROLNONLINEARSOLVER &GetControlNonlinearSolver();
    CONTROLINTEGRATOR &GetControlIntegrator()
    {
      return control_integrator_;
    }

    /**
     * Solves the Equation H(q) sol = rhs
     * Works only if we can reasonably build the inverse of the hessian.
     * To do so the ProblemContainer needs to provide a problem
     * type "hessian_inverse"
     *
     * @param q      The point at which the hessian is considered
     * @param rhs    The right hand side
     * @param sol    The solution of the problem
     */
    void H_inverse(const ControlVector<VECTOR> &q,
                   const ControlVector<VECTOR> &rhs,
                   ControlVector<VECTOR> &sol);

  private:
    CONTROLINTEGRATOR control_integrator_;
    CONTROLNONLINEARSOLVER nonlinear_gradient_solver_;

    bool build_control_matrix_;
    bool  gradient_reinit_;
    std::vector<ControlVector<VECTOR>*> constraint_gradient_;
    DOpEtypes::VectorStorageType vector_behavior_;

    ConstraintVector<VECTOR> constraints_;
    double p_;
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
                       DOpEtypes::VectorStorageType vector_behavior,
                       ParameterReader &param_reader,
                       INTEGRATORDATACONT &idc,
                       int base_priority)
    : ReducedProblemInterface<PROBLEM, VECTOR>(OP,base_priority),
      control_integrator_(idc),
      nonlinear_gradient_solver_(control_integrator_, param_reader),
      constraints_(OP->GetSpaceTimeHandler(),vector_behavior)
  {
    //ReducedProblems should be ReInited
    {
      gradient_reinit_ = true;
    }
    vector_behavior_ = vector_behavior;
  }

  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::~VoidReducedProblem()
  {
    for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
      {
        if (constraint_gradient_[i] != NULL)
          {
            delete constraint_gradient_[i];
          }
      }
  }

  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  CONTROLNONLINEARSOLVER &VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
  {
    if ((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() == "hessian"))
      {
        return nonlinear_gradient_solver_;
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
    ReducedProblemInterface<PROBLEM, VECTOR>::ReInit();

    //Some Solvers must be reinited when called
    // Better have subproblems, so that solver can be reinited here
    {
      gradient_reinit_ = true;
    }
    build_control_matrix_ = true;

    constraints_.ReInit();
    constraint_gradient_.resize(constraints_.GetGlobalConstraints().size(), NULL);
    for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
      {
        if (constraint_gradient_[i] == NULL)
          {
            constraint_gradient_[i] = new ControlVector<VECTOR>(this->GetProblem()->GetSpaceTimeHandler(),vector_behavior_);
          }
        constraint_gradient_[i]->ReInit();
      }
  }

  /******************************************************/


  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  bool VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(const ControlVector<VECTOR> &q, ConstraintVector<VECTOR> &g)
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Evaluating Constraints:",4+this->GetBasePriority());
    this->SetProblemType("constraints");

    g=0;
    //Local constraints
    //this->GetProblem()->ComputeLocalControlConstraints(q.GetSpacialVector(),g.GetSpacialVector("local"));
    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
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
    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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

    if (nglobal > 0)
      {
        for (unsigned int i = 0; i < nglobal; i++)
          {
            this->SetProblemType("global_constraints",i);
            this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

            if (dopedim==dealdim)
              {
                this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
              }
            else if (dopedim == 0)
              {
                this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
              }
            else
              {
                throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedConstraints");
              }

            double ret = 0;
            bool found=false;

            if (this->GetProblem()->GetConstraintType().find("domain") != std::string::npos)
              {
                found = true;
                ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("point") != std::string::npos)
              {
                found = true;
                ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("boundary") != std::string::npos)
              {
                found = true;
                ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
              }
            if (this->GetProblem()->GetConstraintType().find("face") != std::string::npos)
              {
                found = true;
                ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
              }

            if (!found)
              {
                throw DOpEException("Unknown Constraint Type: " + this->GetProblem()->GetConstraintType(),"VoidReducedProblem::ComputeReducedConstraints");
              }
            g.GetGlobalConstraints()(i)=ret;

            if (dopedim==dealdim)
              {
                this->GetControlIntegrator().DeleteDomainData("control");
              }
            else if (dopedim==0)
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

    return g.IsFeasible();
  }
  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
  ::GetControlBoxConstraints(ControlVector<VECTOR> &lb, ControlVector<VECTOR> &ub)
  {
    this->GetProblem()->GetControlBoxConstraints(lb.GetSpacialVector(),ub.GetSpacialVector());
  }

  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>
  ::ComputeReducedGradient(const ControlVector<VECTOR> &q,
                           ControlVector<VECTOR> &gradient,
                           ControlVector<VECTOR> &gradient_transposed)
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Computing Reduced Gradient:",4+this->GetBasePriority());

    //Preparations for ControlInTheDirichletData
    if (this->GetProblem()->HasControlInDirichletData())
      {
        throw DOpEException("Control in Dirichlet-Data not implemented","VoidReducedProblem::ComputeReducedGradient");
      }

    this->SetProblemType("gradient");
    if (gradient_reinit_ ==  true)
      {
        GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
        gradient_reinit_ = false;
      }

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedGradient");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
    this->GetControlIntegrator().AddDomainData("constraints_local",&constraints_.GetSpacialVector("local"));
    this->GetControlIntegrator().AddParamData("constraints_global",&constraints_.GetGlobalConstraints());

    gradient_transposed = 0.;
    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("last_newton_solution",&(gradient_transposed.GetSpacialVector()));
        this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),gradient.GetSpacialVector());
        this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("last_newton_solution",&(gradient_transposed.GetSpacialVectorCopy()));
        this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),gradient.GetSpacialVector());

        this->GetControlIntegrator().DeleteParamData("last_newton_solution");
        gradient_transposed.UnLockCopy();
      }
    gradient *= -1.;
    if (this->GetProblem()->GetFunctionalType() == "algebraic")
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

      for (unsigned int i = 0; i < constraints_.GetGlobalConstraints().size(); i++)
        {
          //this->SetProblemType("local_global_constraint_gradient",i);
          this->SetProblemType("global_constraint_gradient",i);
          this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()),constraint_gradient_[i]->GetSpacialVector());
        }
    }
    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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
  ::ComputeReducedConstraintGradient(const ConstraintVector<VECTOR> &direction,
                                     const ConstraintVector<VECTOR> &constraints,
                                     ConstraintVector<VECTOR> &gradient)
  {
    this->GetProblem()->ComputeReducedConstraintGradient(direction,constraints,gradient);
  }

  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  double VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(const ControlVector<VECTOR> &q)
  {
    double ret = 0;
    bool found=false;
    //Functional may depend on constraints
    ComputeReducedConstraints(q,constraints_);
    //Only trouble if too small values! This is
    //a special adaptation for the Augmented Lagrangian!
    if (!constraints_.IsLargerThan(-p_))
      {
        //Infeasible q!
        throw DOpEException("Infeasible value!","VoidReducedProblem::ComputeReducedCostFunctional");
      }

    this->GetOutputHandler()->Write(this->GetName()+"->Computing Cost Functional:",4+this->GetBasePriority());

    this->SetProblemType("cost_functional");

    this->GetControlIntegrator().AddDomainData("constraints_local",&constraints_.GetSpacialVector("local"));
    this->GetControlIntegrator().AddParamData("constraints_global",&constraints_.GetGlobalConstraints());

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedCostFunctional");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

    if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
      {
        found = true;
        ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
      {
        found = true;
        ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
      {
        found = true;
        ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
      {
        found = true;
        ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
      }
    if (this->GetProblem()->GetFunctionalType().find("algebraic") != std::string::npos)
      {
        found = true;
        ret += this->GetControlIntegrator().ComputeAlgebraicScalar(*(this->GetProblem()));
      }
    if (!found)
      {
        throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),"VoidReducedProblem::ComputeReducedCostFunctional");
      }

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(const ControlVector<VECTOR> &q)
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Computing Functionals:",4+this->GetBasePriority());

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedFunctionals");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());

    for (unsigned int i=0; i<this->GetProblem()->GetNFunctionals(); i++)
      {
        double ret = 0;
        bool found=false;

        this->SetProblemType("aux_functional",i);



        if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
          {
            found = true;
            ret += this->GetControlIntegrator().ComputeDomainScalar(*(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
          {
            found = true;
            ret += this->GetControlIntegrator().ComputePointScalar(*(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
          {
            found = true;
            ret += this->GetControlIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
          }
        if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
          {
            found = true;
            ret += this->GetControlIntegrator().ComputeFaceScalar(*(this->GetProblem()));
          }

        if (!found)
          {
            throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),"VoidReducedProblem::ComputeReducedFunctionals");
          }
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out<<this->GetName()<<"->"<< this->GetProblem()->GetFunctionalName() <<": "<< ret;
        this->GetOutputHandler()->Write(out,2+this->GetBasePriority());
      }

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(const ControlVector<VECTOR> &q,
      const ControlVector<VECTOR> &direction,
      ControlVector<VECTOR> &hessian_direction,
      ControlVector<VECTOR> &hessian_direction_transposed)
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Computing ReducedHessianVector:",4+this->GetBasePriority());

    this->GetOutputHandler()->Write("\t"+this->GetName()+"->Computing Representation of the Hessian:",5+this->GetBasePriority());
    //Preparations for Control In The Dirichlet Data
    if (this->GetProblem()->HasControlInDirichletData())
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
      }
    //Endof Dirichletdata Preparations

    this->SetProblemType("hessian");

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("dq",&(direction.GetSpacialVector()));
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("dq",&(direction.GetSpacialVectorCopy()));
        this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::ComputeReducedHessianVector");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
    this->GetControlIntegrator().AddDomainData("constraints_local",&constraints_.GetSpacialVector("local"));
    this->GetControlIntegrator().AddParamData("constraints_global",&constraints_.GetGlobalConstraints());

    {
      hessian_direction_transposed = 0.;
      if (dopedim == dealdim)
        {
          this->GetControlIntegrator().AddDomainData("last_newton_solution",&(hessian_direction_transposed.GetSpacialVector()));
          this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),hessian_direction.GetSpacialVector());
          this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
        }
      else if (dopedim == 0)
        {
          this->GetControlIntegrator().AddParamData("last_newton_solution",&(hessian_direction_transposed.GetSpacialVectorCopy()));
          this->GetControlIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),hessian_direction.GetSpacialVector());
          this->GetControlIntegrator().DeleteParamData("last_newton_solution");
          hessian_direction_transposed.UnLockCopy();
        }
      hessian_direction *= -1.;
      hessian_direction_transposed=hessian_direction;
      //Compute l^2 representation of the HessianVector
      //hessian Matrix is the same as control matrix
      build_control_matrix_ = this->GetControlNonlinearSolver().NonlinearSolve(*(this->GetProblem()),hessian_direction_transposed.GetSpacialVector(),true,build_control_matrix_,this->GetBasePriority());

      //Add constraint Gradients

      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
        {
          std::stringstream name;
          name << "constraint_gradient_"<<i;
          this->GetControlIntegrator().AddDomainData(name.str(),&(constraint_gradient_[i]->GetSpacialVector()));
        }
      if (this->GetProblem()->GetFunctionalType() == "algebraic")
        {
          dealii::BlockVector<double> tmp = hessian_direction.GetSpacialVector();
          this->GetControlIntegrator().ComputeNonlinearAlgebraicResidual(*(this->GetProblem()),tmp);
          hessian_direction.GetSpacialVector() += tmp;
          hessian_direction_transposed.GetSpacialVector() += tmp;
        }
      //Delete constraint Gradients
      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
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

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("dq");
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianInverseVector(const ControlVector<VECTOR> &q,
      const ControlVector<VECTOR> &direction,
      ControlVector<VECTOR> &hessian_direction)
  {
    this->GetOutputHandler()->Write(this->GetName()+"->Computing ReducedHessianInverseVector:",4+this->GetBasePriority());

    //Solve (A^T H^(-1)A + diag(beta)) y = -A^TH^(-1) direction
    Vector<double> y(constraint_gradient_.size());
    ControlVector<VECTOR> tmp(q);
    {
      Vector<double> rhs(constraint_gradient_.size());
      Vector<double> beta(constraint_gradient_.size());
      H_inverse(q,direction,tmp);
      tmp *=-1.;
      //store H^{-1}direction
      hessian_direction.equ(1.,tmp);


      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
        {
          rhs(i) = (*(constraint_gradient_[i]))*tmp;
        }

      dealii::FullMatrix<double> X(constraint_gradient_.size(),constraint_gradient_.size());
      dealii::FullMatrix<double> Y(constraint_gradient_.size(),constraint_gradient_.size());
      dealii::FullMatrix<double> Z(constraint_gradient_.size(),constraint_gradient_.size());
      Y = 0.;
      //Build Matrix -A^TH^{-1}A
      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
        {
          H_inverse(q,*(constraint_gradient_[i]),tmp);

          this->GetOutputHandler()->Write (*(constraint_gradient_[i]), "ConstraintGradient"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
          this->GetOutputHandler()->Write (tmp, "HessianInverseConstraintGradient"+this->GetPostIndex(), this->GetProblem()->GetDoFType());

          for (unsigned int j = 0; j < constraint_gradient_.size(); j++)
            {
              X(i,j) = ((*(constraint_gradient_[j]))*tmp);
            }
        }

      //Build diag(beta)
      const ConstraintVector<VECTOR> *multiplier = this->GetProblem()->GetAuxiliaryConstraint("mma_multiplier");
      const dealii::Vector<double> &lambda = multiplier->GetGlobalConstraints();

      assert(lambda.size()  == constraint_gradient_.size());

      this->GetProblem()->ComputeReducedGlobalConstraintHessian(constraints_,beta);

      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
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
      for (unsigned int i = 0; i < constraint_gradient_.size(); i++)
        {
          H_inverse(q,*(constraint_gradient_[i]),tmp);
          hessian_direction.add(-1.*y(i),tmp);
        }
    }
    this->GetOutputHandler()->Write (hessian_direction, "HessianInverseDirection"+this->GetPostIndex(), this->GetProblem()->GetDoFType());
  }
  /******************************************************/

  template <typename CONTROLNONLINEARSOLVER, typename CONTROLINTEGRATOR, typename PROBLEM, typename VECTOR,int dopedim,int dealdim>
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::H_inverse(const ControlVector<VECTOR> &q,
      const ControlVector<VECTOR> &rhs,
      ControlVector<VECTOR> &sol)
  {
    sol = 0.;
    this->SetProblemType("hessian_inverse");
    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().AddDomainData("dq",&(rhs.GetSpacialVector()));
        this->GetControlIntegrator().AddDomainData("control",&(q.GetSpacialVector()));
      }
    else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("dq",&(rhs.GetSpacialVectorCopy()));
        this->GetControlIntegrator().AddParamData("control",&(q.GetSpacialVectorCopy()));
      }
    else
      {
        throw DOpEException("dopedim not implemented","VoidReducedProblem::H_inverse");
      }
    this->GetProblem()->AddAuxiliaryToIntegrator(this->GetControlIntegrator());
    this->GetControlIntegrator().AddDomainData("constraints_local",&constraints_.GetSpacialVector("local"));
    this->GetControlIntegrator().AddParamData("constraints_global",&constraints_.GetGlobalConstraints());

    if (this->GetProblem()->GetFunctionalType() == "algebraic")
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

    if (dopedim==dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("dq");
        this->GetControlIntegrator().DeleteDomainData("control");
      }
    else if (dopedim==0)
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
    if (dof_type == "control")
      {
#if dope_dimension >0
        DataOut<dopedim> data_out;
        data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

        data_out.add_data_vector (v,name);
        data_out.build_patches ();

        std::ofstream output(outfile.c_str());

        if (filetype == ".vtk")
          {
            data_out.write_vtk (output);
          }
        else
          {
            throw DOpEException("Don't know how to write filetype `" + filetype + "'!","VoidReducedProblem::WriteToFile");
          }
#else
        if (filetype == ".txt")
          {
            std::ofstream output(outfile.c_str());
            Vector<double> off;
            off = v;
            for (unsigned int i =0; i<off.size(); i++)
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
  void VoidReducedProblem<CONTROLNONLINEARSOLVER, CONTROLINTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const ControlVector<VECTOR> &v, std::string  name, std::string dof_type)
  {
    this->GetOutputHandler()->Write(v.GetSpacialVector(), name, dof_type);
  }
  /******************************************************/

}
#endif
