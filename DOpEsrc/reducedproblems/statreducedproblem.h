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

#ifndef _STAT_REDUCED_PROBLEM_H_
#define _STAT_REDUCED_PROBLEM_H_

#include "reducedprobleminterface.h"
#include "integrator.h"
#include "parameterreader.h"
#include "statevector.h"
#include "stateproblem.h"

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
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "constraintinterface.h"
#include "solutionextractor.h"

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
   * Basic class to solve stationary PDE- and optimization problems.
   *
   * @tparam <CONTROLNONLINEARSOLVER>    Newton solver for the control variables.
   * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
   * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,
   *                                     e.g, Integrator or IntegratorMixedDimensions.
   * @tparam <INTEGRATOR>                An integrator for the state variables,
   *                                     e.g, Integrator or IntegratorMixedDimensions.
   * @tparam <PROBLEM>                   PDE- or optimization problem under consideration.
   * @tparam <VECTOR>                    Class in which we want to store the spatial vector 
   *                                     (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <dopedim>                   The dimension for the control variable.
   * @tparam <dealdim>                   The dimension for the state variable.
   */
  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    class StatReducedProblem : public ReducedProblemInterface<PROBLEM, VECTOR>
    {
      public:
        /**
         * Constructor for the StatReducedProblem.
         *
	 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
         *
	 * @param OP                Problem is given to the stationary solver.
         * @param state_behavior    Indicates the behavior of the StateVector.
         * @param param_reader      An object which has run time data.
	 * @param idc               The InegratorDataContainer for state and control integration
	 * @param base_priority     An offset for the priority of the output written to
	 *                          the OutputHandler
         */
        template<typename INTEGRATORDATACONT>
          StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
              ParameterReader &param_reader, INTEGRATORDATACONT& idc,
              int base_priority = 0);

        /**
         * Constructor for the StatReducedProblem.
         *
	 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
         *
	 * @param OP                Problem is given to the stationary solver.
         * @param state_behavior    Indicates the behavior of the StateVector.
         * @param param_reader      An object which has run time data.
	 * @param c_idc             The InegratorDataContainer for control integration
	 * @param s_idc             The InegratorDataContainer for state integration
	 * @param base_priority     An offset for the priority of the output written to
	 *                          the OutputHandler
         */
        template<typename STATEINTEGRATORDATACONT,
            typename CONTROLINTEGRATORCONT>
          StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
              ParameterReader &param_reader, CONTROLINTEGRATORCONT& c_idc,
              STATEINTEGRATORDATACONT & s_idc, int base_priority = 0);

        ~StatReducedProblem();

        /******************************************************/

        /**
         * Static member function for run time parameters.
         *
         * @param param_reader      An object which has run time data.
         */
        static void
        declare_params(ParameterReader &param_reader);

        /******************************************************/

        /**
         * This function sets state- and dual vectors to their correct sizes.
         * Further, the flags to build the system matrices are set to true.
         *
         */
        void
        ReInit();

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        bool
        ComputeReducedConstraints(const ControlVector<VECTOR>& q,
            ConstraintVector<VECTOR>& g);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        void
        GetControlBoxConstraints(ControlVector<VECTOR>& lb,
            ControlVector<VECTOR>& ub);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        void
        ComputeReducedGradient(const ControlVector<VECTOR>& q,
            ControlVector<VECTOR>& gradient,
            ControlVector<VECTOR>& gradient_transposed);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        double
        ComputeReducedCostFunctional(const ControlVector<VECTOR>& q);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
       void
        ComputeReducedFunctionals(const ControlVector<VECTOR>& q);

        /******************************************************/

        /**
         * Computes the error indicators for the error of a previosly
         * specified functional. Assumes that the primal state solution
         * is already computed and the functional is specified (see
         * problem::SetFunctionalForErrorEstimation).
         *
         * Everything else is determined by the DWRDataContainer
         * you use (represented by the template parameter DWRC).
         *
	 * @tparam <DWRC>           A container for the refinement indicators
	 *                          See, e.g., DWRDataContainer
	 * @tparam <PDE>            The problem contrainer
	 *
         * @param q                 The ControlVector at which the indicators 
	 *                          are to be evaluated. 
	 * @param dwrc              The data container
	 * @param pde               The problem
         *
         */
       template<class DWRC,class PDE>
          void
          ComputeRefinementIndicators(const ControlVector<VECTOR>& q,
              DWRC& dwrc, PDE& pde);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        void
        ComputeReducedHessianVector(const ControlVector<VECTOR>& q,
            const ControlVector<VECTOR>& direction,
            ControlVector<VECTOR>& hessian_direction,
            ControlVector<VECTOR>& hessian_direction_transposed);

        /******************************************************/

       /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
       void
        ComputeReducedGradientOfGlobalConstraints(unsigned int num,
            const ControlVector<VECTOR>& q, const ConstraintVector<VECTOR>& g,
            ControlVector<VECTOR>& gradient,
            ControlVector<VECTOR>& gradient_transposed);

        /******************************************************/

        /**
         * Implementation of Virtual Method in Base Class
	 * ReducedProblemInterface
         *
         */
        void
        StateSizeInfo(std::stringstream& out)
        {
          GetU().PrintInfos(out);
        }

        /******************************************************/

        /**
         *  Here, the given BlockVector<double> v is printed to a file of *.vtk or *.gpl format.
         *  However, in later implementations other file formats will be available.
         *
         *  @param v           The BlockVector to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk or *.gpl  outputs are possible.
         */
        void
        WriteToFile(const VECTOR &v, std::string name, std::string outfile,
            std::string dof_type, std::string filetype);

        /******************************************************/

        /**
         *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
         *  However, in later implementations other file formats will be available.
         *
         *  @param v           The ControlVector<VECTOR> to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk or *.gpl outputs are possible.
         */
        void
        WriteToFile(const ControlVector<VECTOR> &v, std::string name,
            std::string outfile, std::string dof_type, std::string filetype);

        /**
         * Basic function to write a std::vector to a file.
         *
         *  @param v           A std::vector to write to a file.
         *  @param outfile     The basic name for the output file to print.
         *  Doesn't make sense here so aborts if called!
         */
        void
	  WriteToFile(const std::vector<double> &/*v*/,
		      std::string /*outfile*/)
        {
          abort();
        }

      protected:
        /**
         * This function computes the solution for the state variable.
         * The nonlinear solver is called, even for
         * linear problems where the solution is computed within one iteration step.
         *
         * @param q            The ControlVector<VECTOR> is given to this function.
         */
        void
        ComputeReducedState(const ControlVector<VECTOR>& q);

        /******************************************************/
        /**
         * This function computes the adjoint, i.e., the Lagrange 
	 * multiplier to constraint given by the state equation.
	 * It is assumed that the state u(q) corresponding to 
	 * the argument q is already calculated.
         *
         * @param q            The ControlVector<VECTOR> is given to this function.
         */
        void
        ComputeReducedAdjoint(const ControlVector<VECTOR>& q);

        /******************************************************/

        /**
         * This function computes the solution for the dual variable
         * for error estimation.
	 *
	 * I is assumed that the state u(q) corresponding to 
	 * the argument q is already calculated.
         *
         * @param q            The ControlVector<VECTOR> is given to this function.
	 * @param weight_comp  A flag deciding how the weights should be calculated
         */
        void
        ComputeDualForErrorEstimation(const ControlVector<VECTOR>& q,
            DOpEtypes::WeightComputation weight_comp);

        const StateVector<VECTOR> &
        GetU() const
        {
          return _u;
        }
        StateVector<VECTOR> &
        GetU()
        {
          return _u;
        }
        StateVector<VECTOR> &
        GetZ()
        {
          return _z;
        }
        StateVector<VECTOR> &
        GetDU()
        {
          return _du;
        }
        StateVector<VECTOR> &
        GetDZ()
        {
          return _dz;
        }
        /**
         * Returns the solution of the dual equation for error estimation.
         */
        const StateVector<VECTOR> &
        GetZForEE() const
        {
          return _z_for_ee;
        }
        StateVector<VECTOR> &
        GetZForEE()
        {
          return _z_for_ee;
        }

        NONLINEARSOLVER&
        GetNonlinearSolver(std::string type);
        CONTROLNONLINEARSOLVER&
        GetControlNonlinearSolver();
        INTEGRATOR&
        GetIntegrator()
        {
          return _integrator;
        }
        CONTROLINTEGRATOR&
        GetControlIntegrator()
        {
          return _control_integrator;
        }

      private:
        StateVector<VECTOR> _u;
        StateVector<VECTOR> _z;
        StateVector<VECTOR> _du;
        StateVector<VECTOR> _dz;
        StateVector<VECTOR> _z_for_ee;

        INTEGRATOR _integrator;
        CONTROLINTEGRATOR _control_integrator;
        NONLINEARSOLVER _nonlinear_state_solver;
        NONLINEARSOLVER _nonlinear_adjoint_solver;
        CONTROLNONLINEARSOLVER _nonlinear_gradient_solver;

        bool _build_state_matrix, _build_adjoint_matrix, _build_control_matrix;
        bool _state_reinit, _adjoint_reinit, _gradient_reinit;

        friend class SolutionExtractor<
            StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>,
            VECTOR> ;
    };

  /*************************************************************************/
  /*****************************IMPLEMENTATION******************************/
  /*************************************************************************/
  using namespace dealii;

  /******************************************************/
  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::declare_params(
        ParameterReader &param_reader)
    {
      NONLINEARSOLVER::declare_params(param_reader);
    }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    template<typename INTEGRATORDATACONT>
      StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::StatReducedProblem(
          PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader, INTEGRATORDATACONT& idc,
          int base_priority)
          : ReducedProblemInterface<PROBLEM, VECTOR>(OP,
              base_priority), _u(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _z(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _du(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _dz(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _z_for_ee(OP->GetSpaceTimeHandler(),
              state_behavior, param_reader), _integrator(idc), _control_integrator(
              idc), _nonlinear_state_solver(_integrator, param_reader), _nonlinear_adjoint_solver(
              _integrator, param_reader), _nonlinear_gradient_solver(
              _control_integrator, param_reader)

      {
        //ReducedProblems should be ReInited
        {
          _state_reinit = true;
          _adjoint_reinit = true;
          _gradient_reinit = true;
        }
      }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
      StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::StatReducedProblem(
          PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader, CONTROLINTEGRATORCONT& c_idc,
          STATEINTEGRATORDATACONT & s_idc, int base_priority)
          : ReducedProblemInterface<PROBLEM, VECTOR>(OP,
              base_priority), _u(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _z(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _du(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _dz(OP->GetSpaceTimeHandler(), state_behavior,
              param_reader), _z_for_ee(OP->GetSpaceTimeHandler(),
              state_behavior, param_reader), _integrator(s_idc), _control_integrator(
              c_idc), _nonlinear_state_solver(_integrator, param_reader), _nonlinear_adjoint_solver(
              _integrator, param_reader), _nonlinear_gradient_solver(
              _control_integrator, param_reader)

      {
        //ReducedProblems should be ReInited
        {
          _state_reinit = true;
          _adjoint_reinit = true;
          _gradient_reinit = true;
        }
      }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::~StatReducedProblem()
    {
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    NONLINEARSOLVER&
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetNonlinearSolver(
        std::string type)
    {
      if ((type == "state") || (type == "tangent"))
      {
        return _nonlinear_state_solver;
      }
      else if ((type == "adjoint") || (type == "adjoint_hessian")
          || (type == "adjoint_for_ee"))
      {
        return _nonlinear_adjoint_solver;
      }
      else
      {
        throw DOpEException("No Solver for Problem type:`" + type + "' found",
            "StatReducedProblem::GetNonlinearSolver");

      }
    }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    CONTROLNONLINEARSOLVER&
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
    {
      if ((this->GetProblem()->GetType() == "gradient")
          || (this->GetProblem()->GetType() == "hessian"))
      {
        return _nonlinear_gradient_solver;
      }
      else
      {
        throw DOpEException(
            "No Solver for Problem type:`" + this->GetProblem()->GetType()
                + "' found", "StatReducedProblem::GetControlNonlinearSolver");

      }
    }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ReInit()
    {
      ReducedProblemInterface<PROBLEM, VECTOR>::ReInit();

      //Some Solvers must be reinited when called
      // Better have subproblems, so that solver can be reinited here
      {
        _state_reinit = true;
        _adjoint_reinit = true;
        _gradient_reinit = true;
      }

      _build_state_matrix = true;
      _build_adjoint_matrix = true;

      GetU().ReInit();
      GetZ().ReInit();
      GetDU().ReInit();
      GetDZ().ReInit();
      GetZForEE().ReInit();

      _build_control_matrix = true;
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedState(
        const ControlVector<VECTOR>& q)
    {
      this->InitializeFunctionalValues(
          this->GetProblem()->GetNFunctionals() + 1);

      this->GetOutputHandler()->Write("Computing State Solution:",
          4 + this->GetBasePriority());

      this->SetProblemType("state");
      auto& problem = this->GetProblem()->GetStateProblem();
      if (_state_reinit == true)
      {
        GetNonlinearSolver("state").ReInit(problem);
        _state_reinit = false;
      }

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedState");
      }

      _build_state_matrix = this->GetNonlinearSolver("state").NonlinearSolve(
          problem, (GetU().GetSpacialVector()), true, _build_state_matrix);

      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedState");
      }
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetOutputHandler()->Write((GetU().GetSpacialVector()),
          "State" + this->GetPostIndex(), problem.GetDoFType());

    }
  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    bool
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(
        const ControlVector<VECTOR>& q, ConstraintVector<VECTOR>& g)
    {
      this->GetOutputHandler()->Write("Evaluating Constraints:",
          4 + this->GetBasePriority());

      this->SetProblemType("constraints");

      g = 0;
      //Local constraints
      //  this->GetProblem()->ComputeLocalConstraints(q.GetSpacialVector(), GetU().GetSpacialVector(),
      //                                              g.GetSpacialVector("local"));
      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",
            &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedConstraints");
      }
      this->GetControlIntegrator().ComputeLocalControlConstraints(
          *(this->GetProblem()), g.GetSpacialVector("local"));
      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedConstraints");
      }
      //Global in Space-Time Constraints
      dealii::Vector<double>& gc = g.GetGlobalConstraints();
      //dealii::Vector<double> global_values(gc.size());

      unsigned int nglobal = gc.size();      //global_values.size();

      if (nglobal > 0)
      {
        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        for (unsigned int i = 0; i < nglobal; i++)
        {
          //this->SetProblemType("local_global_constraints", i);
          this->SetProblemType("global_constraints", i);
          this->GetIntegrator().AddDomainData("state",
              &(GetU().GetSpacialVector()));
          if (dopedim == dealdim)
          {
            this->GetIntegrator().AddDomainData("control",
                &(q.GetSpacialVector()));
          }
          else if (dopedim == 0)
          {
            this->GetIntegrator().AddParamData("control",
                &(q.GetSpacialVectorCopy()));
          }
          else
          {
            throw DOpEException("dopedim not implemented",
                "StatReducedProblem::ComputeReducedConstraints");
          }

          double ret = 0;
          bool found = false;

          if (this->GetProblem()->GetConstraintType().find("domain")
              != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputeDomainScalar(
                *(this->GetProblem()));
          }
          if (this->GetProblem()->GetConstraintType().find("point")
              != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputePointScalar(
                *(this->GetProblem()));
          }
          if (this->GetProblem()->GetConstraintType().find("boundary")
              != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputeBoundaryScalar(
                *(this->GetProblem()));
          }
          if (this->GetProblem()->GetConstraintType().find("face")
              != std::string::npos)
          {
            found = true;
            ret += this->GetIntegrator().ComputeFaceScalar(
                *(this->GetProblem()));
          }

          if (!found)
          {
            throw DOpEException(
                "Unknown Constraint Type: "
                    + this->GetProblem()->GetConstraintType(),
                "StatReducedProblem::ComputeReducedConstraints");
          }
          //      global_values(i) = ret;
          gc(i) = ret;

          if (dopedim == dealdim)
          {
            this->GetIntegrator().DeleteDomainData("control");
          }
          else if (dopedim == 0)
          {
            this->GetIntegrator().DeleteParamData("control");
            q.UnLockCopy();
          }
          else
          {
            throw DOpEException("dopedim not implemented",
                "StatReducedProblem::ComputeReducedConstraints");
          }
          this->GetIntegrator().DeleteDomainData("state");
        }

        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetIntegrator());
        //gc = global_values;
      }

      //Check that no global in space, local in time constraints are given!
      if (g.HasType("local_global_control") || g.HasType("local_global_state"))
      {
        throw DOpEException(
            "There are global in space, local in time constraints given. In the stationary case they should be moved to global in space and time!",
            "StatReducedProblem::ComputeReducedConstraints");
      }

      //this->GetProblem()->PostProcessConstraints(g, true);
      this->GetProblem()->PostProcessConstraints(g);

      
      return g.IsFeasible();
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlBoxConstraints(
        ControlVector<VECTOR>& lb, ControlVector<VECTOR>& ub)
    {
      this->GetProblem()->GetControlBoxConstraints(lb.GetSpacialVector(),
          ub.GetSpacialVector());
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedAdjoint(
        const ControlVector<VECTOR>& q)
    {
      this->GetOutputHandler()->Write("Computing Reduced Adjoint:",
          4 + this->GetBasePriority());

      this->SetProblemType("adjoint");
      if (_adjoint_reinit == true)
      {
        GetNonlinearSolver("adjoint").ReInit(*(this->GetProblem()));
        _adjoint_reinit = false;
      }

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector())); //&(GetU().GetSpacialVector())

      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedAdjoint");
      }

      _build_adjoint_matrix =
          this->GetNonlinearSolver("adjoint").NonlinearSolve(
              *(this->GetProblem()), (GetZ().GetSpacialVector()), true,
              _build_adjoint_matrix);

      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedAdjoint");
      }
      this->GetIntegrator().DeleteDomainData("state");

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetOutputHandler()->Write((GetZ().GetSpacialVector()),
          "Adjoint" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeDualForErrorEstimation(
        const ControlVector<VECTOR>& q,
        DOpEtypes::WeightComputation weight_comp)
    {
      this->GetOutputHandler()->Write("Computing Dual for Error Estimation:",
          4 + this->GetBasePriority());

      if (weight_comp == DOpEtypes::higher_order_interpolation)
      {
        this->SetProblemType("adjoint_for_ee");
      }
      else
      {
        throw DOpEException("Unknown WeightComputation",
            "StatPDEProblem::ComputeDualForErrorEstimation");
      }

      //      auto& problem = this->GetProblem()->GetStateProblem();//Hier ist adjoint problem einzufuegen
      auto& problem = *(this->GetProblem());
      if (_adjoint_reinit == true)
      {
        GetNonlinearSolver("adjoint_for_ee").ReInit(problem);
        _adjoint_reinit = false;
      }

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector())); //&(GetU().GetSpacialVector())

      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedAdjoint");
      }

      _build_adjoint_matrix =
          this->GetNonlinearSolver("adjoint_for_ee").NonlinearSolve(problem,
              (GetZForEE().GetSpacialVector()), true, _build_adjoint_matrix);

      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedAdjoint");
      }

      this->GetIntegrator().DeleteDomainData("state");

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetOutputHandler()->Write((GetZForEE().GetSpacialVector()),
          "Adjoint_for_ee" + this->GetPostIndex(), problem.GetDoFType());

    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradient(
        const ControlVector<VECTOR>& q, ControlVector<VECTOR>& gradient,
        ControlVector<VECTOR>& gradient_transposed)
    {
      this->ComputeReducedAdjoint(q);

      this->GetOutputHandler()->Write("Computing Reduced Gradient:",
          4 + this->GetBasePriority());

      //Preparations for ControlInTheDirichletData
      VECTOR tmp;
      if (this->GetProblem()->HasControlInDirichletData())
      {
        tmp.reinit(GetU().GetSpacialVector());
        this->SetProblemType("adjoint");
        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        if (dopedim == dealdim)
        {
          this->GetIntegrator().AddDomainData("control",
              &(q.GetSpacialVector()));
        }
        else if (dopedim == 0)
        {
          this->GetIntegrator().AddParamData("control",
              &(q.GetSpacialVectorCopy()));
        }
        else
        {
          throw DOpEException("dopedim not implemented",
              "StatReducedProblem::ComputeReducedGradient");
        }

        this->GetIntegrator().AddDomainData("state",
            &(GetU().GetSpacialVector()));
        this->GetIntegrator().AddDomainData("last_newton_solution",
            &(GetZ().GetSpacialVector()));

        this->GetIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),
            tmp, false);
        tmp *= -1.;

        if (dopedim == dealdim)
        {
          this->GetIntegrator().DeleteDomainData("control");
        }
        else if (dopedim == 0)
        {
          this->GetIntegrator().DeleteParamData("control");
          q.UnLockCopy();
        }
        else
        {
          throw DOpEException("dopedim not implemented",
              "StatReducedProblem::ComputeReducedGradient");
        }

        this->GetIntegrator().DeleteDomainData("state");
        this->GetIntegrator().DeleteDomainData("last_newton_solution");
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetIntegrator());
      }
      //Endof Dirichletdata Preparations
      this->SetProblemType("gradient");
      if (_gradient_reinit == true)
      {
        GetControlNonlinearSolver().ReInit(*(this->GetProblem()));
        _gradient_reinit = false;
      }

      this->GetProblem()->AddAuxiliaryToIntegrator(
          this->GetControlIntegrator());

      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",
            &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedGradient");
      }

      this->GetControlIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("adjoint",
          &(GetZ().GetSpacialVector()));
      if (this->GetProblem()->HasControlInDirichletData())
        this->GetControlIntegrator().AddDomainData("adjoint_residual", &tmp);

      gradient_transposed = 0.;
      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().AddDomainData("last_newton_solution",
            &(gradient_transposed.GetSpacialVector()));
        this->GetControlIntegrator().ComputeNonlinearResidual(
            *(this->GetProblem()), gradient.GetSpacialVector(), true);
        this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("last_newton_solution",
            &(gradient_transposed.GetSpacialVectorCopy()));
        this->GetControlIntegrator().ComputeNonlinearResidual(
            *(this->GetProblem()), gradient.GetSpacialVector(), true);

        this->GetControlIntegrator().DeleteParamData("last_newton_solution");
        gradient_transposed.UnLockCopy();

      }

      gradient *= -1.;
      gradient_transposed = gradient;

      //Compute l^2 representation of the Gradient

      _build_control_matrix = this->GetControlNonlinearSolver().NonlinearSolve(
          *(this->GetProblem()), gradient_transposed.GetSpacialVector(), true,
          _build_control_matrix);
      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedGradient");
      }
      this->GetControlIntegrator().DeleteDomainData("state");
      this->GetControlIntegrator().DeleteDomainData("adjoint");
      if (this->GetProblem()->HasControlInDirichletData())
        this->GetControlIntegrator().DeleteDomainData("adjoint_residual");

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetControlIntegrator());

      this->GetOutputHandler()->Write(gradient,
          "Gradient" + this->GetPostIndex(), this->GetProblem()->GetDoFType());
      this->GetOutputHandler()->Write(gradient_transposed,
          "Gradient_Transposed" + this->GetPostIndex(),
          this->GetProblem()->GetDoFType());

    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    double
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(
        const ControlVector<VECTOR>& q)
    {
      this->ComputeReducedState(q);

      double ret = 0;
      bool found = false;

      this->GetOutputHandler()->Write("Computing Cost Functional:",
          4 + this->GetBasePriority());

      this->SetProblemType("cost_functional");

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedCostFunctional");
      }
      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));

      if (this->GetProblem()->GetFunctionalType().find("domain")
          != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("point")
          != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("boundary")
          != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeBoundaryScalar(
            *(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("face")
          != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
      }

      if (!found)
      {
        throw DOpEException(
            "Unknown Functional Type: "
                + this->GetProblem()->GetFunctionalType(),
            "StatReducedProblem::ComputeReducedCostFunctional");
      }

      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedCostFunctional");
      }
      this->GetIntegrator().DeleteDomainData("state");
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetFunctionalValues()[0].push_back(ret);
      return ret;
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(
        const ControlVector<VECTOR>& q)
    {
      this->GetOutputHandler()->Write("Computing Functionals:",
          4 + this->GetBasePriority());

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedFunctionals");
      }
      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));

      for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
      {
        double ret = 0;
        bool found = false;

        this->SetProblemType("aux_functional", i);
        if (this->GetProblem()->GetFunctionalType().find("domain")
            != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeDomainScalar(
              *(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("point")
            != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputePointScalar(
              *(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("boundary")
            != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeBoundaryScalar(
              *(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("face")
            != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
        }

        if (!found)
        {
          throw DOpEException(
              "Unknown Functional Type: "
                  + this->GetProblem()->GetFunctionalType(),
              "StatReducedProblem::ComputeReducedFunctionals");
        }
        this->GetFunctionalValues()[i + 1].push_back(ret);
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << this->GetProblem()->GetFunctionalName() << ": " << ret;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }

      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedFunctionals");
      }
      this->GetIntegrator().DeleteDomainData("state");
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    template<class DWRC,class PDE>
      void
      StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeRefinementIndicators(
          const ControlVector<VECTOR>& q, DWRC& dwrc, PDE& pde)
//    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
//        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeRefinementIndicators(
//        const ControlVector<VECTOR>& q, DWRDataContainerBase<VECTOR>& dwrc)
      {
	//Attach the ResidualModifier to the PDE.
	pde.ResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::ResidualModifier),boost::ref(dwrc),_1);
	pde.VectorResidualModifier = boost::bind<void>(boost::mem_fn(&DWRC::VectorResidualModifier),boost::ref(dwrc),_1);
	
        //first we reinit the dwrdatacontainer (this
        //sets the weight-vectors to their correct length)
        const unsigned int n_elements =
            this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler().get_tria().n_active_cells();
        dwrc.ReInit(n_elements);

	//Estimation for Costfunctional or if no dual is needed
	if(this->GetProblem()->EEFunctionalIsCost() || !dwrc.NeedDual())
	{
	  this->GetOutputHandler()->Write("Computing Error Indicators:",
					4 + this->GetBasePriority());
	  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());
          
          //add the primal and dual solution to the integrator
	  this->GetIntegrator().AddDomainData("state",
					      &(GetU().GetSpacialVector()));
	  this->GetIntegrator().AddDomainData("adjoint_for_ee",
					      &(GetZ().GetSpacialVector()));

	  if (dopedim == dealdim)
	  {
	    this->GetIntegrator().AddDomainData("control",
						&(q.GetSpacialVector()));
	  }
	  else if (dopedim == 0)
	  {
	    this->GetIntegrator().AddParamData("control",
					       &(q.GetSpacialVectorCopy()));
	  }
	  else
	  {
	    throw DOpEException("dopedim not implemented",
              "StatReducedProblem::ComputeRefinementIndicators");
	  }

	  this->SetProblemType("error_evaluation");

	  //prepare the weights...
	  dwrc.PrepareWeights(GetU(), GetZ());
	  dwrc.PrepareWeights(q);

	  //now we finally compute the refinement indicators
	  this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(),
							    dwrc);
	  // release the lock on the refinement indicators (see dwrcontainer.h)
	  dwrc.ReleaseLock();
	  dwrc.ClearWeightData();

	  // clear the data
	  if (dopedim == dealdim)
	  {
	    this->GetIntegrator().DeleteDomainData("control");
	  }
	  else if (dopedim == 0)
	  {
	    this->GetIntegrator().DeleteParamData("control");
	    q.UnLockCopy();
	  }
	  else
	  {
	    throw DOpEException("dopedim not implemented",
				"StatReducedProblem::ComputeRefinementIndicators");
	  }
	  this->GetIntegrator().DeleteDomainData("state");
	  this->GetIntegrator().DeleteDomainData("adjoint_for_ee");
	  this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetIntegrator());
	}
	else //Estimation for other (not the cost) functional
	{
	  throw DOpEException("Estimating the error in other functionals than cost is not implemented",
                  "StatReducedProblem::ComputeRefinementIndicators"); 
	    
	}

        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << "Error estimate using "<<dwrc.GetName();
	if(dwrc.NeedDual())
	  out<<" For the computation of "<<this->GetProblem()->GetFunctionalName();
	out<< ": "<< dwrc.GetError();
	this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(
        const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& direction,
        ControlVector<VECTOR>& hessian_direction,
        ControlVector<VECTOR>& hessian_direction_transposed)
    {
      this->GetOutputHandler()->Write("Computing ReducedHessianVector:",
          4 + this->GetBasePriority());
      this->GetOutputHandler()->Write("\tSolving Tangent:",
          5 + this->GetBasePriority());

      this->SetProblemType("tangent");

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));

      if (dopedim == dealdim)
      {
        this->GetIntegrator().AddDomainData("dq",
            &(direction.GetSpacialVector()));
        this->GetIntegrator().AddDomainData("control", &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().AddParamData("dq",
            &(direction.GetSpacialVectorCopy()));
        this->GetIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedHessianVector");
      }

      //tangent Matrix is the same as state matrix
      _build_state_matrix = this->GetNonlinearSolver("tangent").NonlinearSolve(
          *(this->GetProblem()), (GetDU().GetSpacialVector()), true,
          _build_state_matrix);

      this->GetOutputHandler()->Write((GetDU().GetSpacialVector()),
          "Tangent" + this->GetPostIndex(), this->GetProblem()->GetDoFType());

      this->GetOutputHandler()->Write("\tSolving Adjoint Hessian:",
          5 + this->GetBasePriority());
      this->SetProblemType("adjoint_hessian");
      this->GetIntegrator().AddDomainData("adjoint",
          &(GetZ().GetSpacialVector()));
      this->GetIntegrator().AddDomainData("tangent",
          &(GetDU().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("adjoint",
          &(GetZ().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("tangent",
          &(GetDU().GetSpacialVector()));

      //adjoint_hessian Matrix is the same as adjoint matrix
      _build_adjoint_matrix =
          this->GetNonlinearSolver("adjoint_hessian").NonlinearSolve(
              *(this->GetProblem()), (GetDZ().GetSpacialVector()), true,
              _build_adjoint_matrix);

      this->GetOutputHandler()->Write((GetDZ().GetSpacialVector()),
          "Hessian" + this->GetPostIndex(), this->GetProblem()->GetDoFType());

      this->GetIntegrator().AddDomainData("adjoint_hessian",
          &(GetDZ().GetSpacialVector()));
      this->GetControlIntegrator().AddDomainData("adjoint_hessian",
          &(GetDZ().GetSpacialVector()));

      this->GetOutputHandler()->Write(
          "\tComputing Representation of the Hessian:",
          5 + this->GetBasePriority());
      //Preparations for Control In The Dirichlet Data
      VECTOR tmp;
      VECTOR tmp_second;
      if (this->GetProblem()->HasControlInDirichletData())
      {
        tmp.reinit(GetU().GetSpacialVector());
        tmp_second.reinit(GetU().GetSpacialVector());
        this->SetProblemType("adjoint");
        this->GetIntegrator().AddDomainData("last_newton_solution",
            &(GetZ().GetSpacialVector()));

        this->GetIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),
            tmp_second, false);
        tmp_second *= -1.;

        this->GetIntegrator().DeleteDomainData("last_newton_solution");
        this->SetProblemType("adjoint_hessian");
        this->GetIntegrator().AddDomainData("last_newton_solution",
            &(GetDZ().GetSpacialVector()));

        this->GetIntegrator().ComputeNonlinearResidual(*(this->GetProblem()),
            tmp, false);
        tmp *= -1.;

        this->GetIntegrator().DeleteDomainData("last_newton_solution");
      }
      //Endof Dirichletdata Preparations
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->SetProblemType("hessian");
      this->GetProblem()->AddAuxiliaryToIntegrator(
          this->GetControlIntegrator());
      if (dopedim == dealdim)
      {
        this->GetIntegrator().DeleteDomainData("dq");
        this->GetIntegrator().DeleteDomainData("control");
        this->GetControlIntegrator().AddDomainData("dq",
            &(direction.GetSpacialVector()));
        this->GetControlIntegrator().AddDomainData("control",
            &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetIntegrator().DeleteParamData("dq");
        this->GetIntegrator().DeleteParamData("control");
        direction.UnLockCopy();
        q.UnLockCopy();
        this->GetControlIntegrator().AddParamData("dq",
            &(direction.GetSpacialVectorCopy()));
        this->GetControlIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedHessianVector");
      }
      if (this->GetProblem()->HasControlInDirichletData())
      {
        this->GetControlIntegrator().AddDomainData("adjoint_residual", &tmp);
        this->GetControlIntegrator().AddDomainData("hessian_residual",
            &tmp_second);
      }

      {
        hessian_direction_transposed = 0.;
        if (dopedim == dealdim)
        {
          this->GetControlIntegrator().AddDomainData("last_newton_solution",
              &(hessian_direction_transposed.GetSpacialVector()));
          this->GetControlIntegrator().ComputeNonlinearResidual(
              *(this->GetProblem()), hessian_direction.GetSpacialVector(),
              true);
          this->GetControlIntegrator().DeleteDomainData("last_newton_solution");
        }
        else if (dopedim == 0)
        {
          this->GetControlIntegrator().AddParamData("last_newton_solution",
              &(hessian_direction_transposed.GetSpacialVectorCopy()));
          this->GetControlIntegrator().ComputeNonlinearResidual(
              *(this->GetProblem()), hessian_direction.GetSpacialVector(),
              true);
          this->GetControlIntegrator().DeleteParamData("last_newton_solution");
          hessian_direction_transposed.UnLockCopy();
        }
        hessian_direction *= -1.;
        hessian_direction_transposed = hessian_direction;
        //Compute l^2 representation of the HessianVector
        //hessian Matrix is the same as control matrix
        _build_control_matrix =
            this->GetControlNonlinearSolver().NonlinearSolve(
                *(this->GetProblem()),
                hessian_direction_transposed.GetSpacialVector(), true,
                _build_control_matrix);

        this->GetOutputHandler()->Write(hessian_direction,
            "HessianDirection" + this->GetPostIndex(),
            this->GetProblem()->GetDoFType());
        this->GetOutputHandler()->Write(hessian_direction_transposed,
            "HessianDirection_Transposed" + this->GetPostIndex(),
            this->GetProblem()->GetDoFType());
      }

      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("dq");
        this->GetControlIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().DeleteParamData("dq");
        this->GetControlIntegrator().DeleteParamData("control");
        direction.UnLockCopy();
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedHessianVector");
      }
      this->GetIntegrator().DeleteDomainData("state");
      this->GetIntegrator().DeleteDomainData("adjoint");
      this->GetIntegrator().DeleteDomainData("tangent");
      this->GetIntegrator().DeleteDomainData("adjoint_hessian");
      this->GetControlIntegrator().DeleteDomainData("state");
      this->GetControlIntegrator().DeleteDomainData("adjoint");
      this->GetControlIntegrator().DeleteDomainData("tangent");
      this->GetControlIntegrator().DeleteDomainData("adjoint_hessian");
      if (this->GetProblem()->HasControlInDirichletData())
      {
        this->GetControlIntegrator().DeleteDomainData("adjoint_residual");
        this->GetControlIntegrator().DeleteDomainData("hessian_residual");
      }
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetControlIntegrator());

    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradientOfGlobalConstraints(
        unsigned int num, const ControlVector<VECTOR>& q,
        const ConstraintVector<VECTOR>& g, ControlVector<VECTOR>& gradient,
        ControlVector<VECTOR>& gradient_transposed)
    {
      //FIXME: If the global constraints depend on u we need to calculate a corresponding
      //       dual solution before we can calculate the gradient.
      std::stringstream out;
      out << "Computing Reduced Gradient of global constraint " << num << " :";
      this->GetOutputHandler()->Write(out, 4 + this->GetBasePriority());
      //Compute derivatives of global constraints
      this->SetProblemType("global_constraint_gradient", num);

      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().AddDomainData("control",
            &(q.GetSpacialVector()));
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().AddParamData("control",
            &(q.GetSpacialVectorCopy()));
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedGradient");
      }
      this->GetProblem()->AddAuxiliaryToIntegrator(
          this->GetControlIntegrator());
      this->GetControlIntegrator().AddDomainData("constraints_local",
          &g.GetSpacialVector("local"));
      this->GetControlIntegrator().AddParamData("constraints_global",
          &g.GetGlobalConstraints());

      //Compute
      this->GetControlIntegrator().ComputeNonlinearRhs(*(this->GetProblem()),
          gradient.GetSpacialVector(), true);
      gradient_transposed = gradient;

      this->GetControlIntegrator().DeleteDomainData("constraints_local");
      this->GetControlIntegrator().DeleteParamData("constraints_global");
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(
          this->GetControlIntegrator());
      if (dopedim == dealdim)
      {
        this->GetControlIntegrator().DeleteDomainData("control");
      }
      else if (dopedim == 0)
      {
        this->GetControlIntegrator().DeleteParamData("control");
        q.UnLockCopy();
      }
      else
      {
        throw DOpEException("dopedim not implemented",
            "StatReducedProblem::ComputeReducedGradient");
      }
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
        const VECTOR &v, std::string name, std::string outfile,
        std::string dof_type, std::string filetype)
    {
      if (dof_type == "state")
      {
        auto& data_out =
            this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
        data_out.attach_dof_handler(
            this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

        data_out.add_data_vector(v, name);
        data_out.build_patches();

        std::ofstream output(outfile.c_str());

        if (filetype == ".vtk")
        {
          data_out.write_vtk(output);
        }
        else if (filetype == ".gpl")
        {
          data_out.write_gnuplot(output);
        }
        else
        {
          throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StatReducedProblem::WriteToFile");
        }
        data_out.clear();
      }
      else if (dof_type == "control")
      {
#if dope_dimension >0
        auto& data_out = this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
        data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

        data_out.add_data_vector (v,name);
        data_out.build_patches ();

        std::ofstream output(outfile.c_str());

        if(filetype == ".vtk")
        {
          data_out.write_vtk (output);
        }
        else if (filetype == ".gpl")
        {
          data_out.write_gnuplot(output);
        }
        else
        {
          throw DOpEException("Don't know how to write filetype `" + filetype + "'!",
              "StatReducedProblem::WriteToFile");
        }
        data_out.clear();
#else
        if (filetype == ".txt")
        {
          std::ofstream output(outfile.c_str());
          Vector<double> off;
          off = v;
          for (unsigned int i = 0; i < off.size(); i++)
          {
            output << off(i) << std::endl;
          }
        }
        else
        {
          throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StatReducedProblem::WriteToFile");
        }
#endif
      }
      else
      {
        throw DOpEException("No such DoFHandler `" + dof_type + "'!",
            "StatReducedProblem::WriteToFile");
      }
    }

  /******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
        const ControlVector<VECTOR> &v, std::string name, std::string outfile,
        std::string dof_type, std::string filetype)
    {
      WriteToFile(v.GetSpacialVector(), name, outfile, dof_type, filetype);
    }

/******************************************************/
}
#endif
