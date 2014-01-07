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

#ifndef _STAT_PDE_PROBLEM_H_
#define _STAT_PDE_PROBLEM_H_

#include "pdeprobleminterface.h"
#include "integrator.h"
#include "parameterreader.h"
#include "statevector.h"
#include "stateproblem.h"

#include <lac/vector.h>

#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include "pdeproblemcontainer.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "directlinearsolver.h"
#include "solutionextractor.h"

#include <base/data_out_base.h>
#include <base/utilities.h>
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
   // * Basic class to solve stationary PDE-problems. 
   * In contrast to StatReducedProblem no control variable is present. 
   * This allows to avoid initialization of the related and objects 
   * which will not be used.
   *
   * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
   * @tparam <INTEGRATOR>                An integrator for the state variables,
   *                                     e.g, Integrator
   * @tparam <PROBLEM>                   PDE- or optimization problem under consideration.
   * @tparam <VECTOR>                    Class in which we want to store the spatial vector (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <dopedim>                   The dimension for the control variable.
   * @tparam <dealdim>                   The dimension for the state variable.
   */
  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    class StatPDEProblem : public PDEProblemInterface<PROBLEM, VECTOR, dealdim>
    {
      public:
        /**
         * Constructor for the StatPDEProblem.
         *
	 * @tparam <INTEGRATORDATACONT> An IntegratorDataContainer
         *
         * @param OP                Problem is given to the stationary solver.
         * @param state_behavior    Indicates the behavior of the StateVector.
         * @param param_reader      An object which has run time data.
         * @param idc		    An INTETGRATORDATACONT which has all the data needed by the integrator.
	 * @param base_priority     An offset for the priority of the output written to
	 *                          the OutputHandler
	 */
        template<typename INTEGRATORDATACONT>
          StatPDEProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
              ParameterReader &param_reader, INTEGRATORDATACONT& idc,
              int base_priority = 0);

        /**
         * Constructor for the StatPDEProblem.
         *
         * @param OP                Problem is given to the stationary solver.
         * @param state_behavior    Indicates the behavior of the StateVector.
         * @param param_reader      An object which has run time data.
         * @param idc               An INTETGRATORDATACONT which has all the data needed by the integrator.
         * @param idc2              An INTETGRATORDATACONT which is used by the integrator to evalueate the
         *                          functionals.
	 * @param base_priority     An offset for the priority of the output written to
	 *                          the OutputHandler
         */
        template<typename INTEGRATORDATACONT>
          StatPDEProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
              ParameterReader &param_reader, INTEGRATORDATACONT& idc1,
              INTEGRATORDATACONT& idc2, int base_priority = 0);

        virtual
        ~StatPDEProblem();

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
        void
        ComputeReducedFunctionals();

        /******************************************************/

        /**
         * This function evaluates reduced functionals of interest
         * for the given statevector st_u.
         *
         */
        void
        ComputeReducedFunctionals(const StateVector<VECTOR>& st_u);

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
	 * @param dwrc              The data container
	 * @param pde               The problem
         *
         */
        template<class DWRC, class PDE>
          void
          ComputeRefinementIndicators(DWRC& dwrc, PDE& pde);

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
         *  Here, the given VECTOR v is printed to a file of *.vtk format.
         *  However, in later implementations other file formats will be available.
         *
         *  @param v           The VECTOR to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
         */
        void
        WriteToFile(const VECTOR &v, std::string name, std::string outfile,
            std::string dof_type, std::string filetype);

        /******************************************************/

        /**
         *  Here, the given Vector v (containing element-related data) is printed to
         *  a file of *.vtk format. However, in later implementations other
         *  file formats will be available.
         *
         *  @param v           The Vector to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
         *  @param type        How to interprete the given data, i.e. does v contain nodal-related or
         *                     element-related data.
         */
        void
        WriteToFileElementwise(const Vector<double> &v, std::string name,
            std::string outfile, std::string dof_type, std::string filetype);

        /******************************************************/

        /**
         *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk format.
         *  However, in later implementations other file formats will be available.
         *
         *  @param v           The ControlVector<VECTOR> to write to a file.
         *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
         *  @param outfile     The basic name for the output file to print.
         *  @param dof_type    Has the DoF type: state or control.
         *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
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
         */
        void
        ComputeReducedState();

        /******************************************************/

        /**
         * This function computes the solution for the dual variable
         * for error estimation.
	 *
	 * I is assumed that the state u(q) corresponding to 
	 * the argument q is already calculated.
         *
         * @param weight_comp  A flag deciding how the weights should be calculated
         */
        void
        ComputeDualForErrorEstimation(DOpEtypes::WeightComputation);

        /******************************************************/
        /**
         * Returns the solution of the state-equation.
         */
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
        INTEGRATOR&
        GetIntegrator()
        {
          return _integrator;
        }

        const bool&
        GetBuildStateMatrix() const
        {
          return _build_state_matrix;
        }

        const bool&
        GetBuildAdjointMatrix() const
        {
          return _build_adjoint_matrix;
        }

        const bool&
        GetStateReinit() const
        {
          return _state_reinit;
        }

        const bool&
        GetAdjointReinit() const
        {
          return _adjoint_reinit;
        }


        bool&
        GetBuildStateMatrix()
        {
          return _build_state_matrix;
        }

        bool&
        GetBuildAdjointMatrix()
        {
          return _build_adjoint_matrix;
        }

        bool&
        GetStateReinit()
        {
          return _state_reinit;
        }

        bool&
        GetAdjointReinit()
        {
          return _adjoint_reinit;
        }


      private:
        /**
         * Helper function to prevent code duplicity. Adds the user defined
         * user Data to the Integrator.
         */
        void
        AddUDD()
        {
          for (auto it = this->GetUserDomainData().begin();
              it != this->GetUserDomainData().end(); it++)
              {
            this->GetIntegrator().AddDomainData(it->first, it->second);
          }
        }

        /**
         * Helper function to prevent code duplicity. Deletes the user defined
         * user Data from the Integrator.
         */
        void
        DeleteUDD()
        {
          for (auto it = this->GetUserDomainData().begin();
              it != this->GetUserDomainData().end(); it++)
              {
            this->GetIntegrator().DeleteDomainData(it->first);
          }
        }

        StateVector<VECTOR> _u;
        StateVector<VECTOR> _z_for_ee;

        INTEGRATOR _integrator;
        NONLINEARSOLVER _nonlinear_state_solver;
        NONLINEARSOLVER _nonlinear_adjoint_solver;

        bool _build_state_matrix;
        bool _build_adjoint_matrix;
        bool _state_reinit;
        bool _adjoint_reinit;

        int _n_patches;

        friend class SolutionExtractor<
            StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>,
            VECTOR> ;
    };

  /*************************************************************************/
  /*****************************IMPLEMENTATION******************************/
  /*************************************************************************/
  using namespace dealii;

  /******************************************************/
  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::declare_params(
        ParameterReader &param_reader)
    {
      NONLINEARSOLVER::declare_params(param_reader);
      param_reader.SetSubsection("output parameters");
      param_reader.declare_entry("number of patches", "0",
          Patterns::Integer(0));

    }
  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    template<typename INTEGRATORDATACONT>
      StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::StatPDEProblem(
          PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader, INTEGRATORDATACONT& idc,
          int base_priority)
          : PDEProblemInterface<PROBLEM, VECTOR, dealdim>(OP, base_priority), _u(
              OP->GetSpaceTimeHandler(), state_behavior, param_reader), _z_for_ee(
              OP->GetSpaceTimeHandler(), state_behavior, param_reader), _integrator(
              idc), _nonlinear_state_solver(_integrator, param_reader), _nonlinear_adjoint_solver(
              _integrator, param_reader)
      {
        param_reader.SetSubsection("output parameters");
        _n_patches = param_reader.get_integer("number of patches");
        //PDEProblems should be ReInited
        {
          _state_reinit = true;
          _adjoint_reinit = true;
        }
      }

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    template<typename INTEGRATORDATACONT>
      StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::StatPDEProblem(
          PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
          ParameterReader &param_reader, INTEGRATORDATACONT& idc,
          INTEGRATORDATACONT& idc2, int base_priority)
          : PDEProblemInterface<PROBLEM, VECTOR, dealdim>(OP, base_priority), _u(
              OP->GetSpaceTimeHandler(), state_behavior, param_reader), _z_for_ee(
              OP->GetSpaceTimeHandler(), state_behavior, param_reader), _integrator(
              idc, idc2), _nonlinear_state_solver(_integrator, param_reader), _nonlinear_adjoint_solver(
              _integrator, param_reader)
      {
        //PDEProblems should be ReInited
        {
          _state_reinit = true;
          _adjoint_reinit = true;
        }
      }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::~StatPDEProblem()
    {
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    NONLINEARSOLVER&
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::GetNonlinearSolver(
        std::string type)
    {
      if (type == "state")
      {
        return _nonlinear_state_solver;
      }
      else if (type == "adjoint_for_ee")
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

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ReInit()
    {
      PDEProblemInterface < PROBLEM, VECTOR, dealdim > ::ReInit();

      //Some Solvers must be reinited when called
      // Better we have subproblems, so that solver can be reinited here
      {
        _state_reinit = true;
        _adjoint_reinit = true;
      }

      _build_state_matrix = true;
      _build_adjoint_matrix = true;

      GetU().ReInit();
      GetZForEE().ReInit();
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeReducedState()
    {
      this->InitializeFunctionalValues(this->GetProblem()->GetNFunctionals());
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
      AddUDD();
      _build_state_matrix = this->GetNonlinearSolver("state").NonlinearSolve(
          problem, (GetU().GetSpacialVector()), true, _build_state_matrix);
      DeleteUDD();
      
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetOutputHandler()->Write((GetU().GetSpacialVector()),
      "State" + this->GetPostIndex(), problem.GetDoFType());

    }

      /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeDualForErrorEstimation(
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
          &(GetU().GetSpacialVector()));
      AddUDD();

      _build_adjoint_matrix =
          this->GetNonlinearSolver("adjoint_for_ee").NonlinearSolve(problem,
              (GetZForEE().GetSpacialVector()), true, _build_adjoint_matrix);

      this->GetIntegrator().DeleteDomainData("state");
      DeleteUDD();

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

      this->GetOutputHandler()->Write((GetZForEE().GetSpacialVector()),
      "Adjoint_for_ee" + this->GetPostIndex(), problem.GetDoFType());

    }

      /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeReducedFunctionals()
    {
      this->ComputeReducedState();

      this->GetOutputHandler()->Write("Computing Functionals:",
          4 + this->GetBasePriority());

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));
      AddUDD();

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
              "StatPDEProblem::ComputeReducedFunctionals");
        }
        this->GetFunctionalValues()[i].push_back(ret);
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << this->GetProblem()->GetFunctionalName() << ": " << ret;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }

      this->GetIntegrator().DeleteDomainData("state");
      DeleteUDD();

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeReducedFunctionals(
        const StateVector<VECTOR>& st_u)
    {
      this->InitializeFunctionalValues(this->GetProblem()->GetNFunctionals());
      GetU() = st_u;

      this->GetOutputHandler()->Write("Computing Functionals:",
          4 + this->GetBasePriority());

      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      this->GetIntegrator().AddDomainData("state",
          &(GetU().GetSpacialVector()));
      AddUDD();

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
              "StatPDEProblem::ComputeReducedFunctionals");
        }
        this->GetFunctionalValues()[i].push_back(ret);
        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << this->GetProblem()->GetFunctionalName() << ": " << ret;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }

      this->GetIntegrator().DeleteDomainData("state");
      DeleteUDD();

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    template<class DWRC, class PDE>
      void
      StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::ComputeRefinementIndicators(
          DWRC& dwrc, PDE& pde)
      {
        //Attach the ResidualModifier to the PDE.
        pde.ResidualModifier = boost::bind<void>(
            boost::mem_fn(&DWRC::ResidualModifier), boost::ref(dwrc), _1);
        pde.VectorResidualModifier = boost::bind<void>(
            boost::mem_fn(&DWRC::VectorResidualModifier), boost::ref(dwrc), _1);
        //first we reinit the dwrdatacontainer (this
        //sets the weight-vectors to their correct length)
        const unsigned int n_elements =
            this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler().get_tria().n_active_cells();
        dwrc.ReInit(n_elements);
        //If we need the dual solution, compute it
        if (dwrc.NeedDual())
          this->ComputeDualForErrorEstimation(dwrc.GetWeightComputation());

        //some output
        this->GetOutputHandler()->Write("Computing Error Indicators:",
            4 + this->GetBasePriority());

        this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

        //add the primal and (if needed) dual solution to the integrator
        this->GetIntegrator().AddDomainData("state",
            &(GetU().GetSpacialVector()));
        if (dwrc.NeedDual())
          this->GetIntegrator().AddDomainData("adjoint_for_ee",
              &(GetZForEE().GetSpacialVector()));
        AddUDD();

        this->SetProblemType("error_evaluation");

        //prepare the weights...
        dwrc.PrepareWeights(GetU(), GetZForEE());

        //now we finally compute the refinement indicators
        this->GetIntegrator().ComputeRefinementIndicators(*this->GetProblem(),
            dwrc);

        // release the lock on the refinement indicators (see dwrcontainer.h)
        dwrc.ReleaseLock();

        const float error = dwrc.GetError();

        // clear the data
        dwrc.ClearWeightData();
        this->GetIntegrator().DeleteDomainData("state");
        if (dwrc.NeedDual())
          this->GetIntegrator().DeleteDomainData("adjoint_for_ee");
        DeleteUDD();
        this->GetProblem()->DeleteAuxiliaryFromIntegrator(
            this->GetIntegrator());

        std::stringstream out;
        this->GetOutputHandler()->InitOut(out);
        out << "Error estimate using " << dwrc.GetName();
        if (dwrc.NeedDual())
          out << " for the " << this->GetProblem()->GetFunctionalName();
        out << ": " << error;
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
        this->GetOutputHandler()->WriteElementwise(dwrc.GetErrorIndicators(),
            "Error_Indicators" + this->GetPostIndex(),
            this->GetProblem()->GetDoFType());

      }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::WriteToFileElementwise(
        const Vector<double> &v, std::string name, std::string outfile,
        std::string dof_type, std::string filetype)
    {
      if (dof_type == "state")
      {
        auto& data_out =
            this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
        data_out.attach_dof_handler(
            this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

        data_out.add_data_vector(v, name);
        data_out.build_patches(_n_patches);

        std::ofstream output(outfile.c_str());

        if (filetype == ".vtk")
        {
          data_out.write_vtk(output);
        }
        else
        {
          throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StatPDEProblem::WriteToFileElementwise");
        }
        data_out.clear();
      }
      else
      {
        throw DOpEException("No such DoFHandler `" + dof_type + "'!",
            "StatPDEProblem::WriteToFileElementwise");
      }
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::WriteToFile(
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
        //TODO: mapping[0] is a workaround, as deal does not support interpolate
        // boundary_values with a mapping collection at this point.
        data_out.build_patches(
            this->GetProblem()->GetSpaceTimeHandler()->GetMapping()[0]);

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
              "StatPDEProblem::WriteToFile");
        }
        data_out.clear();
      }
      else
      {
        throw DOpEException("No such DoFHandler `" + dof_type + "'!",
            "StatPDEProblem::WriteToFile");
      }
    }

  /******************************************************/

  template<typename NONLINEARSOLVER, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dealdim>
    void
    StatPDEProblem<NONLINEARSOLVER, INTEGRATOR, PROBLEM, VECTOR, dealdim>::WriteToFile(
        const ControlVector<VECTOR> &v, std::string name, std::string outfile,
        std::string dof_type, std::string filetype)
    {
      WriteToFile(v.GetSpacialVector(), name, outfile, dof_type, filetype);
    }
}
#endif
