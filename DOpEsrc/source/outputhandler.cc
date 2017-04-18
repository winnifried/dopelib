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


#include <include/outputhandler.h>
#include <interfaces/reducedprobleminterface.h>
#include <include/version.h>

#include <cstdlib>
#include <assert.h>
#include <iomanip>

using namespace dealii;

namespace DOpE
{

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("output parameters");
    param_reader.declare_entry("printlevel", "-1",Patterns::Integer(-1),"Defines what strings should be printed, the higher the number the more output");
    param_reader.declare_entry("never_write_list","",Patterns::Anything(),"Do not write files whose name contains a substring given here by a list of `;` separated words");
    param_reader.declare_entry("ignore_iterations","",Patterns::Anything(),"Iteration Counters that should not reflect in the outputname, seperated by `;`");
    param_reader.declare_entry("results_dir","Results/",Patterns::Anything(),"Directory where the output goes to");
    param_reader.declare_entry("logfile","dope.log",Patterns::Anything(),"Name of the logfile");
    param_reader.declare_entry("file_format",".vtk",Patterns::Selection(".vtk|.gpl"),"File format for the output of solution variables");
    param_reader.declare_entry("control_file_format",".vtk",Patterns::Selection(".vtk|.txt"),"File format for the output of control variables");
    param_reader.declare_entry("debug","false",Patterns::Bool(),"Log Debug Information");
    param_reader.declare_entry("number_precision","5",Patterns::Integer(1),"Sets the precision of the output numbers in the newton schemes.");
    param_reader.declare_entry("functional_number_precision","6",Patterns::Integer(1),"Sets the precision of the output numbers for functionals.");
    param_reader.declare_entry("eps_machine_set_by_user","0.0",Patterns::Double(),"Correlation of the output and machine precision");

  }

  /*******************************************************/
  template <typename VECTOR>
  DOpEOutputHandler<VECTOR>::DOpEOutputHandler(ReducedProblemInterface_Base<VECTOR> *SI, ParameterReader &param_reader)
  {
    //assert(SI);
    if (!SI)
      {
        std::cerr<<"Attention: DOpEOutputHandler is configured without a ReducedProblem, hence no vectors can be written!"<<std::endl;
        Solver_ = NULL;
      }
    else
      {
        Solver_ = SI;
      }

    /* Note that smaller printlevel prints less*/
    /*******************************************
        priority 0 : Things that must be printed always and can't be turned of.
        priority 1 : Tell what Major Algorithms are running
        priority 2 : Things like number of Total Iterations
        priority 3 : Major subiterations
        priority 4 : Minor subiteration (e.g ComputeState)
        priority 5 : Subiteration infos
        priority 20: Debug Infos

         -1 Print everything
     *******************************************/

    param_reader.SetSubsection("output parameters");
    printlevel_        = param_reader.get_integer("printlevel");
    results_basedir_   = param_reader.get_string("results_dir");
    logfile_           = param_reader.get_string("logfile");
    ending_            = param_reader.get_string("file_format");
    control_ending_    = param_reader.get_string("control_file_format");
    debug_             = param_reader.get_bool("debug");
    number_precision_  = param_reader.get_integer("number_precision");
    functional_number_precision_ = param_reader.get_integer("functional_number_precision");
    user_eps_machine_  = param_reader.get_double("eps_machine_set_by_user");

    std::string tmp  = param_reader.get_string("never_write_list");
    ParseString(tmp,never_write_list);

    tmp  = param_reader.get_string("ignore_iterations");
    ParseString(tmp,ignore_iterations);

    std::string command = "mkdir -p " + results_basedir_;
    if (system(command.c_str())!=0)
      {
        throw DOpEException("The command " + command +"failed!",
                            "Outputhandler<VECTOR>::Outputhandler");
      }

    results_outdir_ = "";

    n_reinits_ = 0;

    std::string logfilename = results_basedir_+logfile_;
    log_.open(logfilename.c_str(), std::ios::out);
    disallow_all_ = false;
    stdout_backup_ = 0;

    PrintCopyrightNotice();
  }

  /*******************************************************/
  template <typename VECTOR>
  DOpEOutputHandler<VECTOR>::~DOpEOutputHandler()
  {
    if (log_.good())
      {
        log_.close();
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::ReInit()
  {
    std::stringstream tmp;
    tmp << "Mesh"<<n_reinits_<<"/";
    results_outdir_ =  tmp.str();
    std::string command = "mkdir -p " + results_basedir_ + results_outdir_;
    if (system(command.c_str())!=0)
      {
        throw DOpEException("The command " + command +"failed!",
                            "Outputhandler<VECTOR>::ReInit");
      }
    n_reinits_++;
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::SetIterationNumber(unsigned int iteration, std::string type)
  {
    if (AllowIteration(type))
      {
        std::map<std::string,unsigned int>::const_iterator pos = iteration_type_pos_.find(type);
        if (pos == iteration_type_pos_.end())
          {
            log_<<"LOG: Allowing IterationCounter `"<<type<<"' for filenames!"<<std::endl;
            pos = ReorderAndInsert(type);
          }
        iteration_number_[pos->second] = iteration;
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::ParseString(const std::string tmp, std::vector<std::string> &list)
  {
    //Parse the list for the substrings
    if (tmp.size() > 1)
      {
        size_t last = 0;
        size_t next  = 0;
        while ( last != std::string::npos)
          {
            next = tmp.find(";",last);
            if (tmp.substr(last,next-last) != "")
              list.push_back(tmp.substr(last,next-last));
            last = next;
            if (last != std::string::npos)
              last++;
          }
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  bool DOpEOutputHandler<VECTOR>::AllowWrite(std::string name)
  {
    if (disallow_all_)
      {
        if (debug_)
          {
            log_<<"DEBUG: Deny write of `"<<name<<"'! Since all output is supressed!"<<std::endl;
          }
        return false;
      }
    for (unsigned int i = 0; i < never_write_list.size(); i++)
      {
        if (name.find(never_write_list[i]) != std::string::npos)
          {
            if (debug_)
              {
                log_<<"DEBUG: Deny write of `"<<name<<"'! It containes the substring "<< never_write_list[i]<<std::endl;
              }
            return false;
          }
      }
    return true;
  }

  /*******************************************************/
  template <typename VECTOR>
  bool DOpEOutputHandler<VECTOR>::AllowIteration(std::string name)
  {
    for (unsigned int i = 0; i < ignore_iterations.size(); i++)
      {
        if (name.find(ignore_iterations[i]) != std::string::npos)
          {
            if (debug_)
              {
                log_<<"DEBUG: Deny Iteration counter `"<<name<<"'! It containes the substring "<< ignore_iterations[i]<<std::endl;
              }
            return false;
          }
      }
    return true;
  }

  /*******************************************************/
  template <typename VECTOR>
  std::map<std::string,unsigned int>::const_iterator DOpEOutputHandler<VECTOR>::ReorderAndInsert(std::string type)
  {
    //At the moment this does simple ordering using last_input is last field in Vector
    assert(iteration_number_.size() == iteration_type_pos_.size());
    iteration_type_pos_[type]  = iteration_number_.size();
    iteration_number_.push_back(0);
    return iteration_type_pos_.find(type);
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::WriteError(std::string msg)
  {
    if (disallow_all_)
      {
        if (debug_)
          {
            log_<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
          }
      }
    else
      {
        std::cerr<<msg<<std::endl;
        log_ <<" ERROR: "<<msg<<std::endl;
      }
  }
  /*******************************************************/
  template <typename VECTOR>
  void  DOpEOutputHandler<VECTOR>::WriteAux(std::string msg, std::string file, bool append)
  {
    std::stringstream ofile;
    ofile <<  results_basedir_ << results_outdir_ << file;
    std::ofstream outfile;
    if (append)
      {
        outfile.open(ofile.str().c_str(), std::ios::app|std::ios::out);
      }
    else
      {
        outfile.open(ofile.str().c_str(), std::ios::out);
      }
    outfile<<msg;
    outfile.close();
  }
  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(std::string msg,
                                        int priority,
                                        unsigned int pre_newlines,
                                        unsigned int post_newlines)
  {
    if (disallow_all_)
      {
        if (debug_)
          {
            log_<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
          }
      }
    else
      {
        if (priority < printlevel_ || printlevel_ < 0)
          {
            if (debug_)
              {
                log_<<"DEBUG: Write with priority "<<priority<<" allowed."<<std::endl;
              }
            for (unsigned int n=0; n < pre_newlines; n++)
              {
                log_<<std::endl;
                std::cout<<std::endl;
              }
            log_ << "\t"<< msg<<std::endl;
            std::cout<<msg<<std::endl;
            for (unsigned int n=0; n < post_newlines; n++)
              {
                log_<<std::endl;
                std::cout<<std::endl;
              }
          }
        else if (debug_)
          {
            log_<<"DEBUG: Write because priority "<<priority<<" is too small for printing at level "<<printlevel_<<std::endl;
            {
              log_ <<"D\t"<<msg<<std::endl;
            }
          }
      }
  }
  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(std::stringstream &msg, int priority,unsigned int pre_newlines,unsigned int post_newlines)
  {

    if (disallow_all_)
      {
        if (debug_)
          {
            log_<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
          }
      }
    else
      {
        if (priority < printlevel_ || printlevel_ < 0)
          {
            if (debug_)
              {
                log_<<"DEBUG: Write with priority "<<priority<<" allowed."<<std::endl;
              }
            for (unsigned int n=0; n < pre_newlines; n++)
              {
                log_<<std::endl;
                std::cout<<std::endl;
              }
            std::cout<<msg.str()<<std::endl;
            {
              //for logfile indentation
              std::string tmp = msg.str();
              size_t last = 0;
              size_t next  = 0;
              while (last != std::string::npos)
                {
                  next = tmp.find("\n",last);
                  if (next != std::string::npos)
                    {
                      tmp.insert(next," ");
                      tmp.replace(next,2,"\n\t");
                      next++;
                    }
                  last = next;
                }
              log_ <<"\t"<<tmp<<std::endl;
            }
            for (unsigned int n=0; n < post_newlines; n++)
              {
                log_<<std::endl;
                std::cout<<std::endl;
              }
          }
        else if (debug_)
          {
            log_<<"DEBUG: Write because priority "<<priority<<" is too small for printing at level "<<printlevel_<<std::endl;
            {
              //for logfile indentation
              std::string tmp = msg.str();
              size_t last = 0;
              size_t next  = 0;
              while (last != std::string::npos)
                {
                  next = tmp.find("\n",last);
                  if (next != std::string::npos)
                    {
                      tmp.insert(next,"  ");
                      tmp.replace(next,3,"\nD\t");
                      next++;
                    }
                  last = next;
                }
              log_ <<"D\t"<<tmp<<std::endl;
            }
          }
        msg.str("");
      }
  }
  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const VECTOR &q, std::string name, std::string dof_type)
  {
    if (AllowWrite(name))
      {
        //Construct Name
        std::stringstream outfile;
        outfile<<results_basedir_;
        outfile<<results_outdir_;
        outfile<<name;
        outfile<<GetPostIndex();
        if (dof_type == "control")
          outfile<<control_ending_;
        else if (dof_type == "state")
          outfile<<ending_;
        else
          abort();

        Write("Writing ["+outfile.str()+"]",4);

        if (dof_type == "control")
          GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,control_ending_);
        else if (dof_type == "state")
          GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,ending_);
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const ControlVector<VECTOR> &q, std::string name, std::string dof_type)
  {
    if (AllowWrite(name))
      {
        GetReducedProblem()->WriteToFile(q,name,dof_type);
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const std::vector<double> &q, std::string name, std::string dof_type)
  {
    if (AllowWrite(name))
      {
        //Construct Name
        std::stringstream outfile;
        outfile<<results_basedir_;
        outfile<<results_outdir_;
        outfile<<name;
        outfile<<GetPostIndex();
        if (dof_type == "time")
          outfile<<".gpl";
        else
          abort();
        Write("Writing ["+outfile.str()+"]",4);

        GetReducedProblem()->WriteToFile(q,outfile.str());
      }
  }

  /*******************************************************/
  template <typename VECTOR>
  std::string DOpEOutputHandler<VECTOR>::ZeroTolerance(double value, double reference_value)
  {
    std::stringstream ret;
    ret.precision(number_precision_);
    if (std::isnan(std::abs(value)))
      {
        std::string pre;
        pre.resize(number_precision_+5,' ');
        ret << pre << "NAN";
      }
    else if (std::isinf(std::abs(value)))
      {
        std::string pre;
        pre.resize(number_precision_+5,' ');
        ret << pre << "INF";
      }
    else if (std::abs(value) <= std::abs(reference_value) * user_eps_machine_)
      ret << "< " << std::setfill(' ') << std::setw(number_precision_+6) <<  std::scientific << user_eps_machine_;
    else
      ret << " " << std::setfill(' ') << std::setw(number_precision_+7) << std::scientific << value;

    return ret.str();
  }


  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::InitNewtonOut(std::stringstream &msg)
  {
    msg.precision(number_precision_);
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::InitOut(std::stringstream &msg)
  {
    msg.precision(functional_number_precision_);
  }


  /*******************************************************/

  template<typename VECTOR>
  void
  DOpEOutputHandler<VECTOR>::WriteElementwise(const Vector<double> &q,
                                              std::string name, std::string dof_type)
  {
    if (AllowWrite(name))
      {
        //Construct Name
        std::stringstream outfile;
        outfile << results_basedir_;
        outfile << results_outdir_;
        outfile << name;
        outfile << GetPostIndex();
        if (dof_type == "control")
          outfile << control_ending_;
        else if (dof_type == "state")
          outfile << ending_;
        else
          abort();

        Write("Writing [" + outfile.str() + "]", 4);

        if (dof_type == "control")
          GetReducedProblem()->WriteToFileElementwise(q, name, outfile.str(),
                                                      dof_type, control_ending_);
        else if (dof_type == "state")
          GetReducedProblem()->WriteToFileElementwise(q, name, outfile.str(),
                                                      dof_type, ending_);
      }
  }

  /*******************************************************/

  template<typename VECTOR>
  std::string
  DOpEOutputHandler<VECTOR>::ConstructOutputName(std::string name,
                                                 std::string dof_type)
  {
    std::string outfile;
    if (AllowWrite(name))
      {
        //Construct Name
        outfile += results_basedir_;
        outfile += results_outdir_;
        outfile += name;
        outfile += GetPostIndex();
        if (dof_type == "control")
          outfile += control_ending_;
        else if (dof_type == "state")
          outfile += ending_;
        else
          abort();

        Write("Writing [" + outfile + "]", 4);
      }
    return outfile;
  }


  /*******************************************************/
  template <typename VECTOR>
  std::string DOpEOutputHandler<VECTOR>::GetPostIndex()
  {
    std::stringstream ret;
    for (unsigned int n = 0; n < iteration_number_.size(); n++)
      {
        ret << "."<<std::setfill ('0')<<std::setw(5)<<iteration_number_[n];
      }
    return ret.str();
  }

  /*******************************************************/

  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::StartSaveCTypeOutputToLog()
  {
    if (log_.good())
      {
        log_.close();
      }
    assert(stdout_backup_ == 0);
    fgetpos(stdout, &std_out_pos_);
    stdout_backup_ = dup(fileno(stdout));
    std::string logfilename = results_basedir_+logfile_;
    if (freopen(logfilename.c_str(),"a+",stdout)==NULL)
      {
        throw DOpEException("Could not attach file to stdout",
                            "Outputhandler<VECTOR>::StartSaveCTypeOutputToLog");
      }
  }

  /*******************************************************/

  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::StopSaveCTypeOutputToLog()
  {
    fflush(stdout);
    dup2(stdout_backup_,fileno(stdout));
    close(stdout_backup_);
    stdout_backup_ = 0;
    clearerr(stdout);
    fsetpos(stdout,&std_out_pos_);

    std::string logfilename = results_basedir_+logfile_;
    log_.open(logfilename.c_str(), std::ios::app|std::ios::out);
  }
  /*******************************************************/

  template <typename VECTOR>
  std::string DOpEOutputHandler<VECTOR>::GetResultsDir() const
  {
    return results_basedir_+results_outdir_;
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::PrintCopyrightNotice()
  {
    std::stringstream out;

    out<<"DOpElib Copyright (C) 2012 - "<<DOpE::VERSION::year<<" DOpElib authors"<<std::endl;
    out<<"This program comes with ABSOLUTELY NO WARRANTY."<<std::endl;
    out<<"For License details read LICENSE.TXT distributed with this software!"<<std::endl;
    out<<std::endl;
    out<<"This is DOpElib Version: "<<DOpE::VERSION::major<<"."<<DOpE::VERSION::minor;
    out<<"."<<DOpE::VERSION::fix<<" "<<DOpE::VERSION::postfix<<std::endl;
    out<<"\tStatus as of: "<<std::setfill('0')<<std::setw(2)<<DOpE::VERSION::day;
    out<<"/"<<std::setfill('0')<<std::setw(2)<<DOpE::VERSION::month;
    out<<"/"<<DOpE::VERSION::year<<std::endl;
    std::cout<<out.str();
    std::cout.flush();
    log_<<out.str();
    log_.flush();
  }


}//Endof namespace
/******************************************************/
/******************************************************/
template class DOpE::DOpEOutputHandler<dealii::Vector<double> >;
template class DOpE::DOpEOutputHandler<dealii::BlockVector<double> >;



