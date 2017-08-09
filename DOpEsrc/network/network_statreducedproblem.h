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

#ifndef NETWORK_STAT_REDUCED_PROBLEM_H_
#define NETWORK_STAT_REDUCED_PROBLEM_H_

#include <reducedproblems/statreducedproblem.h>
#include <network/mol_network_spacetimehandler.h>

namespace DOpE
{
  namespace Networks
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
     * @tparam <dopedim>                   The dimension for the control variable.
     * @tparam <dealdim>                   The dimension for the state variable.
     */
    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
             typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
             int dopedim, int dealdim>
    class Network_StatReducedProblem : public StatReducedProblem<CONTROLNONLINEARSOLVER,NONLINEARSOLVER,CONTROLINTEGRATOR,INTEGRATOR,PROBLEM,dealii::BlockVector<double>,dopedim,dealdim>
    {
    public:
      /**
       * Constructor for the Network_StatReducedProblem.
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
      Network_StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                                 ParameterReader &param_reader, INTEGRATORDATACONT &idc,
                                 int base_priority = 0);

      /**
       * Constructor for the Network_StatReducedProblem.
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
      Network_StatReducedProblem(PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                                 ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc,
                                 STATEINTEGRATORDATACONT &s_idc, int base_priority = 0);

      ~Network_StatReducedProblem();

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
      WriteToFile(const dealii::BlockVector<double> &v, std::string name, std::string outfile,
                  std::string dof_type, std::string filetype);

      /******************************************************/

      /**
       *  Here, the given ControlVector<VECTOR> v is printed to a file of *.vtk or *.gpl format.
       *  However, in later implementations other file formats will be available.
       *
       *  @param v           The ControlVector<VECTOR> to write to a file.
       *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
       *  @param dof_type    Has the DoF type: state or control.
       */
      void
      WriteToFile(const ControlVector<dealii::BlockVector<double>> &v, std::string name, std::string dof_type);

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
      /******************************************************/

    private:
      friend class SolutionExtractor<
        Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
        CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>,
        dealii::BlockVector<double>> ;

      typedef StatReducedProblem<CONTROLNONLINEARSOLVER,NONLINEARSOLVER,CONTROLINTEGRATOR,INTEGRATOR,PROBLEM,dealii::BlockVector<double>,dopedim,dealdim> BASE_;
      typedef MethodOfLines_Network_SpaceTimeHandler<FESystem,DoFHandler,BlockVector<double>,0,1> STH_;
    };

    /*************************************************************************/
    /*****************************IMPLEMENTATION******************************/
    /*************************************************************************/
    using namespace dealii;

    /******************************************************/
    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
             typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
             int dopedim, int dealdim>
    void
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::declare_params(
                                 ParameterReader &param_reader)
  {
      BASE_::declare_params(param_reader);
    }
    /******************************************************/

    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
    typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
    int dopedim, int dealdim>
    template<typename INTEGRATORDATACONT>
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::Network_StatReducedProblem(
                                 PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                                 ParameterReader &param_reader, INTEGRATORDATACONT &idc,
                                 int base_priority)
                                 : BASE_(OP,state_behavior,param_reader, idc, base_priority)
    {

    }

    /******************************************************/

    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
             typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
             int dopedim, int dealdim>
    template<typename STATEINTEGRATORDATACONT, typename CONTROLINTEGRATORCONT>
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::Network_StatReducedProblem(
                                 PROBLEM *OP, DOpEtypes::VectorStorageType state_behavior,
                                 ParameterReader &param_reader, CONTROLINTEGRATORCONT &c_idc,
                                 STATEINTEGRATORDATACONT &s_idc, int base_priority)
                                 : BASE_(OP,state_behavior,param_reader,c_idc,s_idc,base_priority)
    {
    }

    /******************************************************/

    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
             typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
             int dopedim, int dealdim>
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::~Network_StatReducedProblem()
    {
    }

    /******************************************************/

    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
    typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
    int dopedim, int dealdim>
    void
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::WriteToFile(
                                 const BlockVector<double> &v, std::string name, std::string outfile,
                                 std::string dof_type, std::string filetype)
    {
      if (dof_type == "state")
        {
          STH_* sth = dynamic_cast<STH_ *>(this->GetProblem()->GetSpaceTimeHandler());
          if (sth == NULL)
            {
              throw DOpEException("Using Networks::Network_StatReducedProblem with wrong SpaceTimeHandler","Networks::Network_StatReducedProblem::WriteToFile");
            }

          unsigned int n_pipes = sth->GetNPipes();
          unsigned int n_comp = sth->GetFESystem("state").n_components();

          for (unsigned int p = 0; p < n_pipes; p++)
            {
              sth->SelectPipe(p);
              std::string tmp = outfile;
              std::stringstream tmp2;
              tmp2 << "_Pipe_"<<std::setfill ('0')<<std::setw(5)<<p<<"_";
              std::string::size_type pos = tmp.rfind(name);
              assert(pos != std::string::npos);
              pos += name.length();
              tmp.insert(pos,tmp2.str());

              auto &data_out =
                this->GetProblem()->GetSpaceTimeHandler()->GetDataOut();
              data_out.attach_dof_handler(
                this->GetProblem()->GetSpaceTimeHandler()->GetStateDoFHandler());

              data_out.add_data_vector(v.block(p), name);
              data_out.build_patches();

              std::ofstream output(tmp.c_str());

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
                    "Network_StatReducedProblem::WriteToFile");
                }
              data_out.clear();
              output.close();
            }
          //Write remaining block into txt-file.
          sth->SelectPipe(n_pipes);
          std::string tmp = outfile;
          std::stringstream tmp2;
          tmp2 << "_Couplings";
          std::string::size_type pos = tmp.rfind(name);
          assert(pos != std::string::npos);
          pos += name.length();
          tmp.insert(pos,tmp2.str());

          std::ofstream out(tmp.c_str());
          assert(out.is_open());

          out << "#File Content is Coupling Block of Network."<<std::endl;
          out<<"# For Residuals the first "<<n_pipes<<" lines are the residual between the outflow of the "<<n_pipes<<" pipes compared to the outflow flux." << std::endl;
          out<<"#    the remaining "<<n_pipes<<" lines are the residuals of the node coupling conditions" << std::endl;
          out<<"# For all other vectors the first "<<n_pipes<<" lines are the fluxes at the 'left' of the pipes" << std::endl;
          out<<"#    the remaining "<<n_pipes<<" lines are the fluxes at the 'right'." << std::endl;
          assert(v.block(n_pipes).size() == 2*n_pipes * n_comp);
          for (unsigned int i = 0; i < 2*n_pipes; i++)
            {
              for (unsigned int c = 0; c < n_comp; c++)
                {
                  out <<  v.block(n_pipes)[i*n_pipes+c] << "\t";
                }
              out<<std::endl;
            }
          out.close();
        }
      else
        {
          BASE_::WriteToFile(v,name,outfile,dof_type,filetype);
        }
    }
    /******************************************************/

    template<typename CONTROLNONLINEARSOLVER, typename NONLINEARSOLVER,
    typename CONTROLINTEGRATOR, typename INTEGRATOR, typename PROBLEM,
    int dopedim, int dealdim>
    void
    Network_StatReducedProblem<CONTROLNONLINEARSOLVER, NONLINEARSOLVER,
                               CONTROLINTEGRATOR, INTEGRATOR, PROBLEM, dopedim, dealdim>::WriteToFile(
                                 const ControlVector<dealii::BlockVector<double>> &v, std::string name,
                                 std::string dof_type)
    {
      BASE_::WriteToFile(v,name,dof_type);
    }

////////////////////////////////ENDOF NAMESPACE Networks/////////////////////////////
  }
////////////////////////////////ENDOF NAMESPACE DOPE/////////////////////////////
}
#endif
