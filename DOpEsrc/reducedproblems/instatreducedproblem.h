#ifndef _INSTAT_REDUCED_PROBLEM_H_
#define _INSTAT_REDUCED_PROBLEM_H_

#include "reducedprobleminterface.h"
#include "integrator.h"
#include "parameterreader.h"
#include "statevector.h"
#include "solutionextractor.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "instat_step_newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "newtonsolvermixeddims.h"
//#include "integratormixeddims.h"
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "constraintinterface.h"
#include "helper.h"
#include "dwrdatacontainer.h"

#include <base/data_out_base.h>
#include <numerics/data_out.h>
#include <numerics/vectors.h>
#include <numerics/matrices.h>
#include <base/function.h>
#include <lac/sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/sparse_direct.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/vector.h>

#include <fstream>

namespace DOpE
{
/**
 * Basic class to solve time dependent PDE- and optimization problems.
 *
 * @tparam <CONTROLNONLINEARSOLVER>    Newton solver for the control variables.
 * @tparam <NONLINEARSOLVER>           Newton solver for the state variables.
 * @tparam <CONTROLINTEGRATOR>         An integrator for the control variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions..
 * @tparam <INTEGRATOR>                An integrator for the state variables,
 *                                     e.g, Integrator or IntegratorMixedDimensions..
 * @tparam <PROBLEM>                   PDE- or optimization problem under consideration including ts-scheme.
 * @tparam <VECTOR>                    Class in which we want to store the spatial vector (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
 * @tparam <dopedim>                   The dimension for the control variable.
 * @tparam <dealdim>                   The dimension for the state variable.
 */
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim, int dealdim>
class InstatReducedProblem: public ReducedProblemInterface<PROBLEM, VECTOR, dopedim, dealdim>
{
  public:
    /**
     * Constructur for the InstatReducedProblem.
     *
     * @param OP                Problem is given to the time stepping scheme.
     * @param state_behavior    Indicates the behavior of the StateVector. Please see, documentation of
     StateVector in `statevector.h'.
     * @param param_reader      An object which has run time data.
     * @param quad_rule					Quadrature-Rule, which is given to the integrators.
     * @param face_quad_rule		FaceQuadrature-Rule, which is given to the integrators.
     */
  template<typename INTEGRATORDATACONT>
          InstatReducedProblem(PROBLEM *OP, std::string state_behavior,
              ParameterReader &param_reader,
              INTEGRATORDATACONT& idc,
              int base_priority = 0);


    /**
     * Constructor for the InstatReducedProblem.
     *
     * @param OP                			Problem is given to the time stepping scheme.
     * @param state_behavior    			Indicates the behavior of the StateVector. Please see, documentation of
     StateVector in `statevector.h'.
     * @param param_reader      			An object which has run time data.
     * @param control_quad_rule				Quadrature-Rule, which is given to the control_integrator.
     * @param control_face_quad_rule	FaceQuadrature-Rule, which is given to the control_integrator.
     * @param state_quad_rule					Quadrature-Rule, which is given to the state_integrator.
     * @param state_face_quad_rule		FaceQuadrature-Rule, which is given to the state_integrator.
     */

  template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
          InstatReducedProblem(PROBLEM *OP, std::string state_behavior,
              ParameterReader &param_reader, CONTROLINTEGRATORCONT& c_idc,
              STATEINTEGRATORDATACONT & s_idc, int base_priority = 0);

    virtual ~InstatReducedProblem();

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
     */
    void ReInit();

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     */
    bool ComputeReducedConstraints(const ControlVector<VECTOR>& q, ConstraintVector<VECTOR>& g);

    /******************************************************/

    /**
     * Implementation of Virtual Method in Base Class
     */
    void GetControlBoxConstraints(ControlVector<VECTOR>& lb, ControlVector<VECTOR>& ub);


    /******************************************************/

    /**
     * This function is not implemented so far and aborts computation if
     * called.
     *
     * @param q            The control vector is given to this function.
     */
    void ComputeReducedGradient(const ControlVector<VECTOR>& q, ControlVector<VECTOR>& gradient,
                                ControlVector<VECTOR>& gradient_transposed);

    /******************************************************/

    /**
     * Returns the functional values to compute.
     *
     * @param q            The control vector is given to this function.
     *
     * @return             Returns a double values of the computed functional.
     */
    double ComputeReducedCostFunctional(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function computes reduced functionals of interest within
     * a time dependent computation.
     *
     * @param q            The control vector is given to this function.
     */
    void ComputeReducedFunctionals(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function is not implemented so far and aborts computation if
     * called. We assume that adjoint state z(u(q)) is already computed.
     *
     * @param q                             The control vector is given to this function.
     * @param direction                     Documentation will follow later.
     * @param hessian_direction             Documentation will follow later.
     * @paramhessian_direction_transposed   Documentation will follow later.
     */
    void ComputeReducedHessianVector(const ControlVector<VECTOR>& q, const ControlVector<VECTOR>& direction,
                                     ControlVector<VECTOR>& hessian_direction,
                                     ControlVector<VECTOR>& hessian_direction_transposed);

    /******************************************************/

    /**
     * Computes the error indicators for the error of a previosly
     * specified functional. Assumes that the primal state solution
     * is already computed and the functional is specified (see
     * problem::SetFunctionalForErrorEstimation).
     *
     * Everything else is determined by the DWRDataContainer
     * you use (represented by the template parameter DWRC).
     */
    template<class DWRC>
    void
    ComputeRefinementIndicators(DWRC&)
    {
      throw DOpEException("ExcNotImplemented",
                          "InstatReducedProblem::ComputeRefinementIndicators");
    }

    /******************************************************/

    /**
     *  This function calls GetU().PrintInfos(out) function which
     *  prints information on this vector into the given stream.
     *
     *  @param out    The output stream.
     */
    void StateSizeInfo(std::stringstream& out)
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
     *  @param filetype    The filetype. Actually, *.vtk and *.gpl outputs are possible.
     */
    void WriteToFile(const VECTOR &v, std::string name, std::string outfile,
                     std::string dof_type, std::string filetype);

    /******************************************************/

    /**
     *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
     *  However, in later implementations other file formats will be available.
     *
     *  @param v           The Control vector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk or *.gpl outputs are possible.
     */
    void WriteToFile(const ControlVector<VECTOR> &v, std::string name, std::string outfile,
                     std::string dof_type, std::string filetype);

    /******************************************************/

    /**
     *  A std::vector v is printed to a text file.
     *  Note that this assumes that the vector is one entry per time step.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     */
    void WriteToFile(const std::vector<double> &v, std::string outfile);

  protected:
    const StateVector<VECTOR> & GetU() const
    {
      return _u;
    }
    StateVector<VECTOR> & GetU()
    {
      return _u;
    }
    StateVector<VECTOR> & GetZ()
    {
      return _z;
    }
    StateVector<VECTOR> & GetDU()
    {
      return _du;
    }
    StateVector<VECTOR> & GetDZ()
    {
      return _dz;
    }

    bool& GetBuildStateMatrix()
    {
      return _build_state_matrix;
    }

    NONLINEARSOLVER& GetNonlinearSolver(std::string type);
    CONTROLNONLINEARSOLVER& GetControlNonlinearSolver();
    INTEGRATOR& GetIntegrator()
    {
      return _integrator;
    }
    CONTROLINTEGRATOR& GetControlIntegrator()
    {
      return _control_integrator;
    }

    /******************************************************/

    /**
     * This function computes functionals of interest within
     * a time dependent computation. For instance, drag- and lift values
     * can be computed, as well as deflections, stresses, etc.
     *
     * @param step         The actual time step.
     * @param num_steps    The total number of time steps.
     */
    void ComputeTimeFunctionals(unsigned int step, unsigned int num_steps);
    /**
     * This function is running the time dependent problem for the state variable.
     * There is a loop over all time steps, and in each time step
     * the nonlinear solver is called. The nonlinear solver is even
     * called for linear problems where the solution is computed within one iteration step.
     *
     * @param q            The control vector is given to this function.
     */
    void ComputeReducedState(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function is not implemented so far and aborts computation if
     * called.
     *
     * @param q            The control vector is given to this function.
     */
    void ComputeReducedAdjoint(const ControlVector<VECTOR>& q);

    /******************************************************/

    /**
     * This function does the loop over time.
     * 
     * @param problem      Describes the nonstationary pde to be solved
     * @param q            The given control vector
     * @param eval_funcs   Decide wether to evaluate the functionals or not.
     *                     Should be true for the primal-problem but false
     *                     for auxilliary forward pdes, like the tangent one.
     */
    template<typename PDE>
      void ForwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname, bool eval_funcs);

    /******************************************************/

    /**
     * This function does the loop over time but in direction -t.
     * 
     * @param problem      Describes the nonstationary pde to be solved
     * @param q            The given control vector
     */
    template<typename PDE>
      void BackwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname);

  private:

    StateVector<VECTOR> _u;
    StateVector<VECTOR> _z;
    StateVector<VECTOR> _du;
    StateVector<VECTOR> _dz;

    INTEGRATOR _integrator;
    CONTROLINTEGRATOR _control_integrator;
    NONLINEARSOLVER _nonlinear_state_solver;
    NONLINEARSOLVER _nonlinear_adjoint_solver;
    CONTROLNONLINEARSOLVER _nonlinear_gradient_solver;

    bool _build_state_matrix, _build_adjoint_matrix, _build_control_matrix;
    bool _state_reinit, _adjoint_reinit, _gradient_reinit;

    bool _project_initial_data;

    friend class SolutionExtractor<InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR,dopedim, dealdim>,   VECTOR > ;
};

/*************************************************************************/
/*****************************IMPLEMENTATION******************************/
/*************************************************************************/
using namespace dealii;

/******************************************************/
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::declare_params(
                                                                        ParameterReader &param_reader)
{
  NONLINEARSOLVER::declare_params(param_reader);
}
/******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
  template<typename INTEGRATORDATACONT>
      InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::InstatReducedProblem(
          PROBLEM *OP,
          std::string state_behavior,
          ParameterReader &param_reader,
          INTEGRATORDATACONT& idc,
          int base_priority) :
            ReducedProblemInterface<PROBLEM, VECTOR, dopedim, dealdim> (OP,
                base_priority),
            _u(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _z(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _du(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _dz(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _integrator(idc),
            _control_integrator(idc),
            _nonlinear_state_solver(_integrator, param_reader),
            _nonlinear_adjoint_solver(_integrator, param_reader),
            _nonlinear_gradient_solver(_control_integrator, param_reader)
      {
        //Solvers should be ReInited
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
      InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
          CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::InstatReducedProblem(
          PROBLEM *OP, std::string state_behavior,
          ParameterReader &param_reader,
          CONTROLINTEGRATORCONT& c_idc,
          STATEINTEGRATORDATACONT & s_idc,
          int base_priority) :
            ReducedProblemInterface<PROBLEM, VECTOR, dopedim, dealdim> (OP,
                base_priority),
            _u(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _z(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _du(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _dz(OP->GetSpaceTimeHandler(), state_behavior, param_reader),
            _integrator(s_idc),
            _control_integrator(c_idc),
            _nonlinear_state_solver(_integrator, param_reader),
            _nonlinear_adjoint_solver(_integrator, param_reader),
            _nonlinear_gradient_solver(_control_integrator, param_reader)
      {
        //Solvers should be ReInited
          {
            _state_reinit = true;
            _adjoint_reinit = true;
            _gradient_reinit = true;
          }
      }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::~InstatReducedProblem()
{
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
NONLINEARSOLVER& InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR,
    INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetNonlinearSolver(std::string type)
{
  if ((type == "state") || (type == "tangent"))
  {
    return _nonlinear_state_solver;
  }
  else if ((type == "adjoint") || (type == "adjoint_hessian"))
  {
    return _nonlinear_adjoint_solver;
  }
  else
  {
    throw DOpEException("No Solver for Problem type:`" + type + "' found",
                        "InstatReducedProblem::GetNonlinearSolver");

  }
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
CONTROLNONLINEARSOLVER& InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
    CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::GetControlNonlinearSolver()
{
  if ((this->GetProblem()->GetType() == "gradient") || (this->GetProblem()->GetType() == "hessian"))
  {
    return _nonlinear_gradient_solver;
  }
  else
  {
    throw DOpEException("No Solver for Problem type:`" + this->GetProblem()->GetType() + "' found",
                        "InstatReducedProblem::GetControlNonlinearSolver");

  }
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ReInit()
{
  ReducedProblemInterface<PROBLEM, VECTOR, dopedim, dealdim>::ReInit();

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

  _build_control_matrix = true;
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedState(const ControlVector<VECTOR>& q)
{
  this->InitializeFunctionalValues(this->GetProblem()->GetNFunctionals() + 1);

  this->GetOutputHandler()->Write("Computing State Solution:", 4 + this->GetBasePriority());

  this->SetProblemType("state");
  auto& problem = this->GetProblem()->GetStateProblem();

  if (_state_reinit == true)
  {
    GetNonlinearSolver("state").ReInit(problem);
    _state_reinit = false;
  }

  this->GetProblem()->AddAuxiliaryControl(&q,"control");
  this->ForwardTimeLoop(problem,this->GetU(),"State",true);
  this->GetProblem()->DeleteAuxiliaryControl("control");
}
/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
bool InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedConstraints(
                          const ControlVector<VECTOR>& /*q*/,
			  ConstraintVector<VECTOR>& /*g*/)
{
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
  PROBLEM, VECTOR, dopedim, dealdim>::GetControlBoxConstraints(ControlVector<VECTOR>& /*lb*/, ControlVector<VECTOR>& /*ub*/)
{
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedAdjoint(
      const ControlVector<VECTOR>& q)
{
  this->GetOutputHandler()->Write("Computing Adjoint Solution:", 4 + this->GetBasePriority());

  this->SetProblemType("adjoint");
  auto& problem = this->GetProblem()->GetAdjointProblem();
  if (_adjoint_reinit == true)
  {
    GetNonlinearSolver("adjoint").ReInit(problem);
    _adjoint_reinit = false;
  }

  this->GetProblem()->AddAuxiliaryState(&(this->GetU()),"state");
  this->GetProblem()->AddAuxiliaryControl(&q,"control");
  this->BackwardTimeLoop(problem,this->GetZ(),"Adjoint");
  this->GetProblem()->DeleteAuxiliaryControl("control");
  this->GetProblem()->DeleteAuxiliaryState("state");
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedGradient(
      const ControlVector<VECTOR>& q,
      ControlVector<VECTOR>& /*gradient*/,
      ControlVector<VECTOR>& /*gradient_transposed*/)
{
  this->ComputeReducedAdjoint(q);

  this->GetOutputHandler()->Write("Computing Reduced Gradient:",
				  4 + this->GetBasePriority());
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
double InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedCostFunctional(
      const ControlVector<VECTOR>& q)
{
  this->ComputeReducedState(q);

  if (this->GetFunctionalValues()[0].size() != 1)
  {
    if (this->GetFunctionalValues()[0].size() == 0)
      throw DOpEException(
                          "Apparently the CostFunctional was never evaluated! \n\tCheck if the return value of `NeedTimes' is set correctly.",
                          "InstatReducedProblem::ComputeReducedCostFunctional");
    else
      throw DOpEException(
                          "The CostFunctional has been evaluated too many times! \n\tCheck if the return value of `NeedTimes' is set correctly.",
                          "InstatReducedProblem::ComputeReducedCostFunctional");
  }
  return this->GetFunctionalValues()[0][0];
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedFunctionals(
      const ControlVector<VECTOR>& /*q*/)
{
  //We dont need q as the values are precomputed during Solve State...
  this->GetOutputHandler()->Write("Computing Functionals:" + this->GetBasePriority(), 4);

  for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
  {
    this->SetProblemType("aux_functional", i);
    if (this->GetProblem()->GetFunctionalType().find("timelocal"))
    {
      if (this->GetFunctionalValues()[i + 1].size() == 1)
      {
        std::stringstream out;
        out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i + 1][0];
        this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
      }
      else if (this->GetFunctionalValues()[i + 1].size() > 1)
      {
        if (this->GetFunctionalValues()[i + 1].size()
            == this->GetProblem()->GetSpaceTimeHandler()->GetMaxTimePoint() + 1)
        {
          std::stringstream out;
          out << this->GetProblem()->GetFunctionalName() << " too large. Writing to file instead: ";
          this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
          this->GetOutputHandler()->Write(this->GetFunctionalValues()[i + 1],
                                          this->GetProblem()->GetFunctionalName()
                                              + this->GetPostIndex(), "time");
        }
        else
        {
          std::stringstream out;
          out << this->GetProblem()->GetFunctionalName() << ": ";
          for (unsigned int k = 0; k < this->GetFunctionalValues()[i + 1].size(); k++)
            out << this->GetFunctionalValues()[i + 1][k] << " ";
          this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
        }
      }
      else
      {
        throw DOpEException("Functional: " + this->GetProblem()->GetFunctionalType()
            + " was not evaluated ever!", "InstatReducedProblem::ComputeFunctionals");
      }
    }
    else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
    {
      std::stringstream out;
      out << this->GetProblem()->GetFunctionalName() << ": " << this->GetFunctionalValues()[i + 1][0];
      this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
    }
    else
    {
      throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                          "InstatReducedProblem::ComputeFunctionals");
    }
  }
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeReducedHessianVector(
      const ControlVector<VECTOR>& /*q*/,
      const ControlVector<VECTOR>& /*direction*/,
      ControlVector<VECTOR>& /*hessian_direction*/,
      ControlVector<VECTOR>& /*hessian_direction_transposed*/)
{
  abort();
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::ComputeTimeFunctionals(unsigned int step, unsigned int num_steps)
{

  this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

  this->GetIntegrator().AddDomainData("state", &(GetU().GetSpacialVector()));
  double ret = 0;
  bool found = false;
  {//CostFunctional
    this->SetProblemType("cost_functional");
    if (this->GetProblem()->NeedTimeFunctional())
    {
      if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
      }
      if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
      {
        found = true;
        ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
      }

      if (!found)
      {
        throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                            "InstatReducedProblem::ComputeTimeFunctionals");
      }
      //Wert speichern
      if (this->GetProblem()->GetFunctionalType().find("timelocal"))
      {
        if (this->GetFunctionalValues()[0].size() != 0)
        {
          throw DOpEException("Too many evaluations of CostFunctional: "
              + this->GetProblem()->GetFunctionalType(),
                              "InstatReducedProblem::ComputeTimeFunctionals");
        }
        this->GetFunctionalValues()[0].push_back(ret);
      }
      else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
      {//TODO was passiert hier? Vermutlich sollte hier spaeter Zeitintegration durchgefuehrt werden?
        if (this->GetFunctionalValues()[0].size() != 1)
        {
          this->GetFunctionalValues()[0].resize(1);
        }
        double w = 0.;
        if ((step == 0))
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
        }
        else if (step + 1 == num_steps)
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
        }
        else
        {
          w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
          w += 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
              - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
        }
        this->GetFunctionalValues()[0][0] += w * ret;
      }
      else
      {
        throw DOpEException("Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                            "InstatReducedProblem::ComputeTimeFunctionals");
      }
    }
  }
  {//Aux Functionals
    for (unsigned int i = 0; i < this->GetProblem()->GetNFunctionals(); i++)
    {
      ret = 0;
      found = false;
      this->SetProblemType("aux_functional", i);
      if (this->GetProblem()->NeedTimeFunctional())
      {
        if (this->GetProblem()->GetFunctionalType().find("domain") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeDomainScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("point") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputePointScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("boundary") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeBoundaryScalar(*(this->GetProblem()));
        }
        if (this->GetProblem()->GetFunctionalType().find("face") != std::string::npos)
        {
          found = true;
          ret += this->GetIntegrator().ComputeFaceScalar(*(this->GetProblem()));
        }

        if (!found)
        {
          throw DOpEException(
                              "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                              "InstatReducedProblem::ComputeTimeFunctionals");
        }
        //Wert speichern
        if (this->GetProblem()->GetFunctionalType().find("timelocal"))
        {
          std::stringstream out;
          out << "\t" << this->GetProblem()->GetFunctionalName() << ": " << ret;
          this->GetOutputHandler()->Write(out, 2 + this->GetBasePriority());
          this->GetFunctionalValues()[i + 1].push_back(ret);
        }
        else if (this->GetProblem()->GetFunctionalType().find("timedistributed"))
        {
          if (this->GetFunctionalValues()[i + 1].size() != 1)
          {
            this->GetFunctionalValues()[i + 1].resize(1);
          }
          double w = 0.;
          if ((step == 0))
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
          }
          else if (step + 1 == num_steps)
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
          }
          else
          {
            w = 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step + 1)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step));
            w += 0.5 * (this->GetProblem()->GetSpaceTimeHandler()->GetTime(step)
                - this->GetProblem()->GetSpaceTimeHandler()->GetTime(step - 1));
          }
          this->GetFunctionalValues()[i + 1][0] += w * ret;
        }
        else
        {
          throw DOpEException(
                              "Unknown Functional Type: " + this->GetProblem()->GetFunctionalType(),
                              "InstatReducedProblem::ComputeTimeFunctionals");
        }
      }
    }
  }
  this->GetIntegrator().DeleteDomainData("state");
  this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());

}

/******************************************************/
template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const VECTOR &v, std::string name, std::string outfile, std::string dof_type, std::string filetype)
{
  if (dof_type == "state")
    {
      auto& data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
      data_out.attach_dof_handler(this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

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
          throw DOpEException("Don't know how to write filetype `" + filetype + "'!",
              "InstatReducedProblem::WriteToFile");
        }
      data_out.clear();
    }
  else if (dof_type == "control")
    {
#if dope_dimension >0
      auto& data_out =  this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
      data_out.attach_dof_handler (this->GetProblem()->GetSpaceTimeHandler()->GetControlDoFHandler());

      data_out.add_data_vector (v,name);
      data_out.build_patches ();

      std::ofstream output(outfile.c_str());

      if(filetype == ".vtk")
      {
        data_out.write_vtk (output);
      }
      else if(filetype == ".gpl")
      {
        data_out.write_gnuplot (output);
      }
      else
      {
        throw DOpEException("Don't know how to write filetype `" + filetype + "'!","InstatReducedProblem::WriteToFile");
      }
      data_out.clear();
#else
    std::ofstream output(outfile.c_str());
    Vector<double> off;
    off = v;
    for (unsigned int i = 0; i < off.size(); i++)
    {
      output << off(i) << std::endl;
    }
#endif
  }
  else
  {
    throw DOpEException("No such DoFHandler `" + dof_type + "'!",
                        "InstatReducedProblem::WriteToFile");
  }
}

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER, typename CONTROLINTEGRATOR,
    typename INTEGRATOR, typename PROBLEM, typename VECTOR, int dopedim,
    int dealdim>
void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
    PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(const ControlVector<VECTOR> &v,
                                                                     std::string name,
                                                                     std::string outfile,
                                                                     std::string dof_type,
                                                                     std::string filetype)
{
  WriteToFile(v.GetSpacialVector(), name, outfile, dof_type, filetype);
}

/******************************************************/

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::WriteToFile(
        const std::vector<double> &v, std::string outfile)
    {
      //TODO This should get timedofhandler later on.
      const std::vector<double>& t =
          this->GetProblem()->GetSpaceTimeHandler()->GetTimes();
      std::ofstream out(outfile.c_str());
      assert( t.size() == v.size());
      assert(out.is_open());

      out << "#Time\tvalue" << std::endl;
      for (unsigned int i = 0; i < v.size(); i++)
        {
          out << t[i] << "\t" << v[i] << std::endl;
        }
      out.close();
    }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
    typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
    typename VECTOR, int dopedim, int dealdim>
    template<typename PDE>
    void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
    CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
    ForwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname, bool eval_funcs)
  {
    VECTOR u_alt;

    unsigned int max_timestep =
      problem.GetSpaceTimeHandler()->GetMaxTimePoint();
    const std::vector<double> times =
      problem.GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval();
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(local_to_global[0], it);
      sol.SetTimeDoFNumber(local_to_global[0], it);
    }
    //u_alt auf initial_values setzen
    {
      //dazu erstmal gesamt-dof berechnen
      const std::vector<unsigned int>& dofs_per_block =
	this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
      {
	n_dofs += dofs_per_block[i];
      }
      //und dann auf den Helper zuerueckgreifen (wegen Templateisierung)
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_alt);
    }
    
    //Projection der Anfangsdaten
    this->GetOutputHandler()->SetIterationNumber(0, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
          4 + this->GetBasePriority());

      auto& initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      _build_state_matrix = this->GetNonlinearSolver("state").NonlinearSolve_Initial(
          initial_problem, u_alt, true, true);
      _build_state_matrix = true;
      
      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      
    }
    sol.GetSpacialVector() = u_alt;
    this->GetOutputHandler()->Write(u_alt, outname + this->GetPostIndex(),
          problem.GetDoFType());
    
    
    if(eval_funcs)
    {//Funktional Auswertung in t_0
      ComputeTimeFunctionals(0,
			     max_timestep);
          this->SetProblemType("state");
    }
    
    
    for (TimeIterator it =
	   problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval(); it
	   != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().after_last_interval(); ++it)
    {
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(local_to_global[0], it);
      sol.SetTimeDoFNumber(local_to_global[0], it);
      //TODO Eventuell waere ein Test mit nicht-gleichmaessigen Zeitschritten sinnvoll!
      
      //we start here at i=1 because we assume that the most
      //left DoF in the actual interval is already computed!
      for (unsigned int i = 1; i < n_dofs_per_interval; i++)
      {
	this->GetOutputHandler()->SetIterationNumber(local_to_global[i],
						     "Time");
	double time = times[local_to_global[i]];
	
	std::stringstream out;
	out << "\t\t Timestep: " << local_to_global[i] << " ("
	    << times[local_to_global[i - 1]] << " -> " << time
	    << ") using " << problem.GetName();
	problem.GetOutputHandler()->Write(out,
					  4 + this->GetBasePriority());
	
	sol.SetTimeDoFNumber(local_to_global[i], it);
	sol.GetSpacialVector() = 0;
	
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
	
	this->GetNonlinearSolver("state").NonlinearLastTimeEvals(problem,
								 u_alt, sol.GetSpacialVector());

	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	problem.SetTime(time, it);
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
	
	this->GetBuildStateMatrix()
	  = this->GetNonlinearSolver("state").NonlinearSolve(problem,
							     u_alt, sol.GetSpacialVector(), true,
							     this->GetBuildStateMatrix());

	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	//TODO do a transfer to the next grid for changing spatial meshes!
	u_alt = sol.GetSpacialVector();
	this->GetOutputHandler()->Write(sol.GetSpacialVector(),
					outname + this->GetPostIndex(), problem.GetDoFType());
	if(eval_funcs)
	{//Funktional Auswertung in t_n//if abfrage, welcher typ
	  ComputeTimeFunctionals(local_to_global[i], max_timestep);
	  this->SetProblemType("state");
	}
      }
    }
  }

/******************************************************/

template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
  typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
  typename VECTOR, int dopedim, int dealdim>
  template<typename PDE>
  void InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
  CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::
  BackwardTimeLoop(PDE& problem, StateVector<VECTOR>& sol, std::string outname)
  {
    VECTOR u_alt;

    unsigned int max_timestep =
      problem.GetSpaceTimeHandler()->GetMaxTimePoint();
    const std::vector<double> times =
      problem.GetSpaceTimeHandler()->GetTimes();
    const unsigned int
      n_dofs_per_interval =
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> local_to_global(n_dofs_per_interval);
    {
      TimeIterator it =
	problem.GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval();
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(local_to_global[local_to_global.size()-1], it);
      sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
    }
    //u_alt auf initial_values setzen
    {
      //dazu erstmal gesamt-dof berechnen
      const std::vector<unsigned int>& dofs_per_block =
	this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFsPerBlock();
      unsigned int n_dofs = 0;
      unsigned int n_blocks = dofs_per_block.size();
      for (unsigned int i = 0; i < n_blocks; i++)
      {
	n_dofs += dofs_per_block[i];
      }
      //und dann auf den Helper zuerueckgreifen (wegen Templateisierung)
      DOpEHelper::ReSizeVector(n_dofs, dofs_per_block, u_alt);
    }
    //Projection der Anfangsdaten
    this->GetOutputHandler()->SetIterationNumber(max_timestep, "Time");
    {
      this->GetOutputHandler()->Write("Computing Initial Values:",
          4 + this->GetBasePriority());

      auto& initial_problem = problem.GetInitialProblem();
      this->GetProblem()->AddAuxiliaryToIntegrator(this->GetIntegrator());

      //TODO: Possibly another solver for the initial value than for the pde...
      _build_state_matrix = this->GetNonlinearSolver("adjoint").NonlinearSolve_Initial(
          initial_problem, u_alt, true, true);
      _build_state_matrix = true;

      this->GetProblem()->DeleteAuxiliaryFromIntegrator(this->GetIntegrator());
      
    }
    sol.GetSpacialVector() = u_alt;
    this->GetOutputHandler()->Write(u_alt, outname + this->GetPostIndex(),
          problem.GetDoFType());
    
    //TODO: Maybe we should calculate the local gradient computations here
    
    for (TimeIterator it =
	   problem.GetSpaceTimeHandler()->GetTimeDoFHandler().last_interval(); it
	   != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().before_first_interval(); --it)
    {
      it.get_time_dof_indices(local_to_global);
      problem.SetTime(local_to_global[local_to_global.size()-1], it);
      sol.SetTimeDoFNumber(local_to_global[local_to_global.size()-1], it);
     //TODO Eventuell waere ein Test mit nicht-gleichmaessigen Zeitschritten sinnvoll!
      
      //we start here at i= 1 and transform i -> n_dofs_per_interval-1-i because we assume that the most
      //right DoF in the actual interval is already computed!
      for (unsigned int i = 1; i < n_dofs_per_interval; i++)
      {
	unsigned int j = n_dofs_per_interval-1-i;
	this->GetOutputHandler()->SetIterationNumber(local_to_global[j],
						     "Time");
	double time = times[local_to_global[j]];
	
	std::stringstream out;
	out << "\t\t Timestep: " << local_to_global[j] << " ("
	    << times[local_to_global[j + 1]] << " -> " << time
	    << ") using " << problem.GetName();
	problem.GetOutputHandler()->Write(out,
					  4 + this->GetBasePriority());
	
	sol.SetTimeDoFNumber(local_to_global[j], it);
	sol.GetSpacialVector() = 0;
	
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
		
	this->GetNonlinearSolver("adjoint").NonlinearLastTimeEvals(problem,
								   u_alt, sol.GetSpacialVector());
	
	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());

	problem.SetTime(time, it);
	this->GetProblem()->AddAuxiliaryToIntegrator(
	  this->GetIntegrator());
		
	this->GetBuildStateMatrix()
	  = this->GetNonlinearSolver("adjoint").NonlinearSolve(problem,
							     u_alt, sol.GetSpacialVector(), true,
							     this->GetBuildStateMatrix());
	
	this->GetProblem()->DeleteAuxiliaryFromIntegrator(
	  this->GetIntegrator());
	
	//TODO do a transfer to the next grid for changing spatial meshes!
	u_alt = sol.GetSpacialVector();
	this->GetOutputHandler()->Write(sol.GetSpacialVector(),
					outname + this->GetPostIndex(), problem.GetDoFType());

	//Maybe build local gradient here
      }
    }
  }
////////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
