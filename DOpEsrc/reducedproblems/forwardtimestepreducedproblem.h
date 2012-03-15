/*
 * forwardtimestepreducedproblem.h
 *
 *  Created on: 16.01.2012
 *      Author: cgoll
 */

#ifndef _FORWARDTIMESTEPREDUCEDPROBLEM_H_
#define _FORWARDTIMESTEPREDUCEDPROBLEM_H_

#include <instatreducedproblem.h>

namespace DOpE
{
  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    class ForwardTimestepReducedProblem : public InstatReducedProblem<
        CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR, INTEGRATOR,
        PROBLEM, VECTOR, dopedim, dealdim>
    {
      public:
        /**
         * Constructur for the InstatReducedProblem.
         *
         * @param OP                Problem is given to the time stepping scheme.
         * @param state_behavior    Indicates the behavior of the StateVector. Please see, documentation of
         StateVector in `statevector.h'.
         * @param param_reader      An object which has run time data.
         * @param quad_rule         Quadrature-Rule, which is given to the integrators.
         * @param face_quad_rule    FaceQuadrature-Rule, which is given to the integrators.
         */
      template<typename INTEGRATORDATACONT>
          ForwardTimestepReducedProblem(PROBLEM *OP,
              std::string state_behavior, ParameterReader &param_reader,
              INTEGRATORDATACONT& idc,
              int base_priority = 0) :
                InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                    CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim,
                    dealdim> (OP, state_behavior, param_reader, idc, base_priority)
          {
          }

        /**
         * Constructor for the InstatReducedProblem.
         *
         * @param OP                      Problem is given to the time stepping scheme.
         * @param state_behavior          Indicates the behavior of the StateVector. Please see, documentation of
         StateVector in `statevector.h'.
         * @param param_reader            An object which has run time data.
         * @param control_quad_rule       Quadrature-Rule, which is given to the control_integrator.
         * @param control_face_quad_rule  FaceQuadrature-Rule, which is given to the control_integrator.
         * @param state_quad_rule         Quadrature-Rule, which is given to the state_integrator.
         * @param state_face_quad_rule    FaceQuadrature-Rule, which is given to the state_integrator.
         */
      template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
        ForwardTimestepReducedProblem(PROBLEM *OP, std::string state_behavior,
            ParameterReader &param_reader,
            CONTROLINTEGRATORCONT& c_idc,
            STATEINTEGRATORDATACONT & s_idc,
            int base_priority = 0) :
              InstatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                  CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim,
                  dealdim> (OP, state_behavior, param_reader,
                  c_idc, s_idc, base_priority)
        {
        }

        virtual void
        TimeLoop(const ControlVector<VECTOR>& q);

        friend class SolutionExtractor<ForwardTimestepReducedProblem<
            CONTROLNONLINEARSOLVER, NONLINEARSOLVER, CONTROLINTEGRATOR,
            INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim> , VECTOR> ;
    };

  template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
      typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
      typename VECTOR, int dopedim, int dealdim>
    void
    ForwardTimestepReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, VECTOR, dopedim, dealdim>::TimeLoop(
        const ControlVector<VECTOR>& q)
    {
      auto& problem = this->GetProblem()->GetStateProblem();
      VECTOR u_alt;
      //u_alt auf initial_values setzen
      //FIXME Statevector[0] muss u_alt bekommen!
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
      if (problem.L2ProjectionInitialDataWithDeal())
        {
          //TODO Initial values should be included in the newton step and not here!

          //    if (this->GetProblem()->GetSpaceTimeHandler()->GetDoFHandlerType() == "classic")
          //    {
          VectorTools::project(
              this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler(),
              this->GetProblem()->GetSpaceTimeHandler()->GetStateHangingNodeConstraints(),
              this->GetIntegrator().GetIntegratorDataContainer().GetQuad(),
              problem.GetInitialValues(), u_alt);
          //    }
          //    else
          //    {
          //      //TODO: Projection fuer hp!
          //      dealii::VectorTools::interpolate(
          //                                       *(static_cast<const dealii::hp::DoFHandler<dealdim>*> (&this->GetProblem()->GetSpaceTimeHandler()->GetStateHpDoFHandler())),
          //                                       this->GetProblem()->GetInitialValues(), u_alt);
          //    }
        }
      else
        {
          throw DOpEException("Other Projection not implemented yet!",
              "ForwardTimestepReducedProblem::ComputeReducedState");
        }
      this->GetOutputHandler()->Write(u_alt, "State" + this->GetPostIndex(),
          problem.GetDoFType());
      unsigned int max_timestep =
          problem.GetSpaceTimeHandler()->GetMaxTimePoint();

        {//Funktional Auswertung in t_0
          problem.SetTime(
              problem.GetSpaceTimeHandler()->GetTime(0),
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval());
          ComputeTimeFunctionals(this->GetFunctionalValues(), q, 0,
              max_timestep);
          this->SetProblemType("state");
        }
      const std::vector<double> times =
          problem.GetSpaceTimeHandler()->GetTimes();
      const unsigned int
          n_dofs_per_interval =
              problem.GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
      std::vector<unsigned int> local_to_global(n_dofs_per_interval);

      for (TimeIterator it =
          problem.GetSpaceTimeHandler()->GetTimeDoFHandler().first_interval(); it
          != problem.GetSpaceTimeHandler()->GetTimeDoFHandler().after_last_interval(); ++it)
        {
          it.get_time_dof_indices(local_to_global);
          problem.SetTime(local_to_global[0], it);
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

              this->GetU().SetTimeDoFNumber(local_to_global[i], it);
              this->GetU().GetSpacialVector() = 0;

              this->GetProblem()->AddAuxiliaryToIntegrator(
                  this->GetIntegrator());

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
                      "ForwardTimestepReducedProblem::ComputeReducedState");
                }

              this->GetNonlinearSolver("state").NonlinearLastTimeEvals(problem,
                  u_alt, this->GetU().GetSpacialVector());

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
                      "ForwardTimestepReducedProblem::ComputeReducedState");
                }
              this->GetProblem()->DeleteAuxiliaryFromIntegrator(
                  this->GetIntegrator());

              q.SetTime(time, it);
              problem.SetTime(time, it);
              this->GetProblem()->AddAuxiliaryToIntegrator(
                  this->GetIntegrator());

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
                      "ForwardTimestepReducedProblem::ComputeReducedState");
                }

              this->GetBuildStateMatrix()
                  = this->GetNonlinearSolver("state").NonlinearSolve(problem,
                      u_alt, this->GetU().GetSpacialVector(), true,
                      this->GetBuildStateMatrix());

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
                      "ForwardTimestepReducedProblem::ComputeReducedState");
                }
              this->GetProblem()->DeleteAuxiliaryFromIntegrator(
                  this->GetIntegrator());

              //TODO do a transfer to the next grid for changing spatial meshes!
              u_alt = this->GetU().GetSpacialVector();
              this->GetOutputHandler()->Write(this->GetU().GetSpacialVector(),
                  "State" + this->GetPostIndex(), problem.GetDoFType());
                {//Funktional Auswertung in t_n//if abfrage, welcher typ
                  ComputeTimeFunctionals(this->GetFunctionalValues(), q,
                      local_to_global[i], max_timestep);
                  this->SetProblemType("state");
                }
            }
        }
    }
}//end of namespace

#endif /* FORWARDTIMESTEPREDUCEDPROBLEM_H_ */
