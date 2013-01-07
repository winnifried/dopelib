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


#include "outputhandler.h"
#include "reducedprobleminterface.h"

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
  DOpEOutputHandler<VECTOR>::DOpEOutputHandler(ReducedProblemInterface_Base<VECTOR>* SI, ParameterReader &param_reader)
  {
    assert(SI);
    _Solver = SI;

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
    _printlevel        = param_reader.get_integer("printlevel");
    _results_basedir   = param_reader.get_string("results_dir");
    _logfile           = param_reader.get_string("logfile");
    _ending            = param_reader.get_string("file_format");
    _control_ending    = param_reader.get_string("control_file_format");
    _debug             = param_reader.get_bool("debug");
    _number_precision  = param_reader.get_integer("number_precision");
    _functional_number_precision = param_reader.get_integer("functional_number_precision");
    _user_eps_machine  = param_reader.get_double("eps_machine_set_by_user");

    std::string tmp  = param_reader.get_string("never_write_list");
    ParseString(tmp,never_write_list);

    tmp  = param_reader.get_string("ignore_iterations");
    ParseString(tmp,ignore_iterations);

    std::string command = "mkdir -p " + _results_basedir;
    if(system(command.c_str())!=0)
    {
      throw DOpEException("The command " + command +"failed!",
			  "Outputhandler<VECTOR>::Outputhandler");
    }

    _results_outdir = "";

    _n_reinits = 0;

    std::string logfilename = _results_basedir+_logfile;
    _log.open(logfilename.c_str(), std::ios::out);
    _disallow_all = false;
    _stdout_backup = 0;    

    PrintCopyrightNotice();
  }

/*******************************************************/
  template <typename VECTOR>
  DOpEOutputHandler<VECTOR>::~DOpEOutputHandler()
  {
    if(_log.good())
    {
      _log.close();
    }
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::ReInit()
  {
    std::stringstream tmp;
    tmp << "Mesh"<<_n_reinits<<"/";
    _results_outdir =  tmp.str();
    std::string command = "mkdir -p " + _results_basedir + _results_outdir;
    if(system(command.c_str())!=0)
    {
      throw DOpEException("The command " + command +"failed!",
			  "Outputhandler<VECTOR>::ReInit");
    }
    _n_reinits++;
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::SetIterationNumber(unsigned int iteration, std::string type)
  {
    if(AllowIteration(type))
    {
      std::map<std::string,unsigned int>::const_iterator pos = _iteration_type_pos.find(type);
      if(pos == _iteration_type_pos.end())
      {
	_log<<"LOG: Allowing IterationCounter `"<<type<<"' for filenames!"<<std::endl;
	pos = ReorderAndInsert(type);
      }
      _iteration_number[pos->second] = iteration;
    }
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::ParseString(const std::string tmp, std::vector<std::string>& list)
  {
    //Parse the list for the substrings
    if(tmp.size() > 1)
    {
      size_t last = 0;
      size_t next  = 0;
      while( last != std::string::npos)
      {
	next = tmp.find(";",last);
	if(tmp.substr(last,next-last) != "")
	  list.push_back(tmp.substr(last,next-last));
	last = next;
	if(last != std::string::npos)
	  last++;
      }
    }
  }

/*******************************************************/
  template <typename VECTOR>
  bool DOpEOutputHandler<VECTOR>::AllowWrite(std::string name)
  {
    if(_disallow_all)
    {
      if(_debug)
      {
	_log<<"DEBUG: Deny write of `"<<name<<"'! Since all output is supressed!"<<std::endl;
      }
      return false;
    }
    for(unsigned int i = 0; i < never_write_list.size(); i++)
    {
      if(name.find(never_write_list[i]) != std::string::npos)
      {
	if(_debug)
	{
	  _log<<"DEBUG: Deny write of `"<<name<<"'! It containes the substring "<< never_write_list[i]<<std::endl;
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
    for(unsigned int i = 0; i < ignore_iterations.size(); i++)
    {
      if(name.find(ignore_iterations[i]) != std::string::npos)
      {
	if(_debug)
	{
	  _log<<"DEBUG: Deny Iteration counter `"<<name<<"'! It containes the substring "<< ignore_iterations[i]<<std::endl;
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
    assert(_iteration_number.size() == _iteration_type_pos.size());
    _iteration_type_pos[type]  = _iteration_number.size();
    _iteration_number.push_back(0);
    return _iteration_type_pos.find(type);
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::WriteError(std::string msg)
  {
    if(_disallow_all)
    {
      if(_debug)
      {
	_log<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
      }
    }
    else
    {
      std::cerr<<msg<<std::endl;
      _log <<" ERROR: "<<msg<<std::endl;
    }
  }
/*******************************************************/
  template <typename VECTOR>
  void  DOpEOutputHandler<VECTOR>::WriteAux(std::string msg, std::string file, bool append)
  {
    std::stringstream ofile;
    ofile <<  _results_basedir << _results_outdir << file;
    std::ofstream outfile;
    if(append)
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
    if(_disallow_all)
    {
      if(_debug)
      {
	_log<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
      }
    }
    else
    {
      if(priority < _printlevel || _printlevel < 0)
      {
	if(_debug)
	{
	  _log<<"DEBUG: Write with priority "<<priority<<" allowed."<<std::endl;
	}
	for(unsigned int n=0; n < pre_newlines; n++)
	{
	  _log<<std::endl;
	  std::cout<<std::endl;
	}
	_log << "\t"<< msg<<std::endl;
	std::cout<<msg<<std::endl;
	for(unsigned int n=0; n < post_newlines; n++)
	{
	  _log<<std::endl;
	  std::cout<<std::endl;
	}
      }
      else if(_debug)
      {
	_log<<"DEBUG: Write because priority "<<priority<<" is too small for printing at level "<<_printlevel<<std::endl;
	{
	  _log <<"D\t"<<msg<<std::endl;
	}
      }
    }
  }
/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(std::stringstream& msg, int priority,unsigned int pre_newlines,unsigned int post_newlines)
  {

    if(_disallow_all)
    {
      if(_debug)
      {
	_log<<"DEBUG: Output of Error was suppresed since all output is supressed!"<<std::endl;
      }
    }
    else
    {
      if(priority < _printlevel || _printlevel < 0)
      {
	if(_debug)
	{
	  _log<<"DEBUG: Write with priority "<<priority<<" allowed."<<std::endl;
	}
	for(unsigned int n=0; n < pre_newlines; n++)
	{
	  _log<<std::endl;
	  std::cout<<std::endl;
	}
	std::cout<<msg.str()<<std::endl;
	{
	  //for logfile indentation
	  std::string tmp = msg.str();
	  size_t last = 0;
	  size_t next  = 0;
	  while(last != std::string::npos)
	  {
	    next = tmp.find("\n",last);
	    if(next != std::string::npos)
	    {
	      tmp.insert(next," ");
	      tmp.replace(next,2,"\n\t");
	      next++;
	    }
	    last = next;
	  }
	  _log <<"\t"<<tmp<<std::endl;
	}
	for(unsigned int n=0; n < post_newlines; n++)
	{
	  _log<<std::endl;
	  std::cout<<std::endl;
	}
      }
      else if(_debug)
      {
	_log<<"DEBUG: Write because priority "<<priority<<" is too small for printing at level "<<_printlevel<<std::endl;
	{
	  //for logfile indentation
	  std::string tmp = msg.str();
	  size_t last = 0;
	  size_t next  = 0;
	  while(last != std::string::npos)
	  {
	    next = tmp.find("\n",last);
	    if(next != std::string::npos)
	    {
	      tmp.insert(next,"  ");
	      tmp.replace(next,3,"\nD\t");
	      next++;
	    }
	    last = next;
	  }
	  _log <<"D\t"<<tmp<<std::endl;
	}
      }
      msg.str("");
    }
  }
/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const VECTOR&q, std::string name, std::string dof_type)
  {
    if(AllowWrite(name))
    {
      //Construct Name
      std::stringstream outfile;
      outfile<<_results_basedir;
      outfile<<_results_outdir;
      outfile<<name;
      outfile<<GetPostIndex();
      if(dof_type == "control")
	outfile<<_control_ending;
      else if(dof_type == "state")
	outfile<<_ending;
      else
	abort();

      Write("Writing ["+outfile.str()+"]",4);

      if(dof_type == "control")
	GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,_control_ending);
      else if(dof_type == "state")
	GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,_ending);
    }
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const ControlVector<VECTOR>&q, std::string name, std::string dof_type)
  {
    if(AllowWrite(name))
    {
      //Construct Name
      std::stringstream outfile;
      outfile<<_results_basedir;
      outfile<<_results_outdir;
      outfile<<name;
      outfile<<GetPostIndex();
      if(dof_type == "control")
	outfile<<_control_ending;
      else if(dof_type == "state")
	outfile<<_ending;
      else
	abort();
      Write("Writing ["+outfile.str()+"]",4);

      if(dof_type == "control")
	GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,_control_ending);
      else if(dof_type == "state")
	GetReducedProblem()->WriteToFile(q,name,outfile.str(),dof_type,_ending);
    }
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::Write(const std::vector<double>& q, std::string name, std::string dof_type)
  {
    if(AllowWrite(name))
    {
      //Construct Name
      std::stringstream outfile;
      outfile<<_results_basedir;
      outfile<<_results_outdir;
      outfile<<name;
      outfile<<GetPostIndex();
      if(dof_type == "time")
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
    ret.precision(_number_precision);
    if(std::isnan(std::abs(value)))
    {
      std::string pre;
      pre.resize(_number_precision+5,' ');
      ret << pre << "NAN";
    }
    else if(std::isinf(std::abs(value)))
    {
      std::string pre;
      pre.resize(_number_precision+5,' ');
      ret << pre << "INF";
    }   
    else
      if (std::abs(value) <= std::abs(reference_value) * _user_eps_machine)
      ret << "< " << std::setfill(' ') << std::setw(_number_precision+6) <<  std::scientific << _user_eps_machine;
    else 
      ret << " " << std::setfill(' ') << std::setw(_number_precision+7) << std::scientific << value; 

    return ret.str();
  }


  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::InitNewtonOut(std::stringstream &msg)
  {
    msg.precision(_number_precision);
  }

  /*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::InitOut(std::stringstream &msg)
  {
    msg.precision(_functional_number_precision);
  }


  /*******************************************************/

   template<typename VECTOR>
     void
     DOpEOutputHandler<VECTOR>::WriteCellwise(const Vector<double>&q,
         std::string name, std::string dof_type)
     {
       if (AllowWrite(name))
       {
         //Construct Name
         std::stringstream outfile;
         outfile << _results_basedir;
         outfile << _results_outdir;
         outfile << name;
         outfile << GetPostIndex();
         if (dof_type == "control")
           outfile << _control_ending;
         else if (dof_type == "state")
           outfile << _ending;
         else
           abort();

         Write("Writing [" + outfile.str() + "]", 4);

         if (dof_type == "control")
           GetReducedProblem()->WriteToFileCellwise(q, name, outfile.str(),
               dof_type, _control_ending);
         else if (dof_type == "state")
           GetReducedProblem()->WriteToFileCellwise(q, name, outfile.str(),
               dof_type, _ending);
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
        outfile += _results_basedir;
        outfile += _results_outdir;
        outfile += name;
        outfile += GetPostIndex();
        if (dof_type == "control")
          outfile += _control_ending;
        else if (dof_type == "state")
          outfile += _ending;
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
    for(unsigned int n = 0; n < _iteration_number.size(); n++)
    {
      ret << "."<<std::setfill ('0')<<std::setw(5)<<_iteration_number[n];
    }
    return ret.str();
  } 

/*******************************************************/

  template <typename VECTOR>
   void DOpEOutputHandler<VECTOR>::StartSaveCTypeOutputToLog()
  {
    if(_log.good())
    {
      _log.close();
    }
    assert(_stdout_backup == 0);
    fgetpos(stdout, &_std_out_pos);
    _stdout_backup = dup(fileno(stdout));
    std::string logfilename = _results_basedir+_logfile;
    if(freopen(logfilename.c_str(),"a+",stdout)==NULL)
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
    dup2(_stdout_backup,fileno(stdout));
    close(_stdout_backup);
    _stdout_backup = 0;
    clearerr(stdout);
    fsetpos(stdout,&_std_out_pos);

    std::string logfilename = _results_basedir+_logfile;
    _log.open(logfilename.c_str(), std::ios::app|std::ios::out);
  }

/*******************************************************/
  template <typename VECTOR>
  void DOpEOutputHandler<VECTOR>::PrintCopyrightNotice()
  {
    std::stringstream out;
    
    out<<"DOpElib Copyright (C) 2012  DOpElib authors"<<std::endl;
    out<<"This program comes with ABSOLUTELY NO WARRANTY."<<std::endl;
    out<<"For License details read LICENSE.TXT distributed with this software!"<<std::endl;
    std::cout<<out.str();
    std::cout.flush();
    _log<<out.str();
    _log.flush();
  }


}//Endof namespace
/******************************************************/
/******************************************************/
template class DOpE::DOpEOutputHandler<dealii::Vector<double> >;
template class DOpE::DOpEOutputHandler<dealii::BlockVector<double> >;



