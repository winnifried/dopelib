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

#ifndef DOPE_OUTPUT_HANDLER_H_
#define DOPE_OUTPUT_HANDLER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <deal.II/lac/vector.h>

#include <container/optproblemcontainer.h>
#include <include/controlvector.h>
#include <include/parameterreader.h>

namespace DOpE
{
//Predeclaration necessary
  template<typename VECTOR> class ReducedProblemInterface_Base;
/////////////////////////////

  /**
   * This class takes care of all output from the programm.
   * It includes special filter routines on what information
   * is to be displayed, and what vectors should be stored.
   * The filters and output opptions can be configured
   * using a parameter file.
   */
  template<typename VECTOR>
  class DOpEOutputHandler
  {
  public:

    DOpEOutputHandler(ReducedProblemInterface_Base<VECTOR> *SI, ParameterReader &param_reader);
    ~DOpEOutputHandler();

    /**
     * This method must be called prior to instanciation of the object to be able to read the required
     * parameters from the paramfile. Between the call of this function and the instanciation of the
     * OutputHandler param_reader.read_parameters must be called.
     */
    static void declare_params(ParameterReader &param_reader);

    /**
     * Call this function to write a string to an auxilliary file
     *
     * @param msg        The message that should be printed.
     * @param file       Filename
     * @param append     String appendend to file ?
     */
    void WriteAux(std::string msg, std::string file, bool append=true);

    /**
    * Call this function to write a string both to std::out and into the logfile.
    * If the debug mode is activated via the paramfile, the message is always printed into the log.
    *
    * @param msg        The message that should be printed.
    * @param priority   The priority of the message. This is used to decide whether the string is actually be
    *                   printed. By comparing this number with the printlevel given in the paramfile.
    *                   A message is printed only if priority < printlevel, or if printlevel < 0
    *                   As a general rule:
    *                   priority 0 : Things that must be printed always and can't be turned of.
    *                   priority 1 : Tell what Major Algorithms are running
    *                   priority 2 : Things like number of Total Iterations
    *                   priority 3 : Major subiterations
    *                   priority 4 : Minor subiteration (e.g ComputeState)
    *                   priority 5 : Subiteration infos
    *                   priority 20: Debug Infos
    *
    * @param pre_newlines Number of empty lines in front of the message.
    * @param post_newlines Number of empty lines after the message.
    */
    void Write(std::string msg, int priority = 20, unsigned int pre_newlines = 0,
               unsigned int post_newlines = 0);
    /**
     * Call this function to write a stringstream both to std::out and into the logfile.
     * If the debug mode is activated via the paramfile, the message is always printed into the log.
     *
     * @param msg        The message that should be printed.
     * @param priority   The priority of the message. This is used to decide whether the string is actually be
     *                   printed. By comparing this number with the printlevel given in the paramfile.
     *                   A message is printed only if priority < printlevel, or if printlevel < 0
     *                   As a general rule:
     *                   priority 0 : Things that must be printed always and can't be turned of.
     *                   priority 1 : Tell what Major Algorithms are running
     *                   priority 2 : Things like number of Total Iterations
     *                   priority 3 : Major subiterations
     *                   priority 4 : Minor subiteration (e.g ComputeState)
     *                   priority 5 : Subiteration infos
     *                   priority 20: Debug Infos
     *
     * @param pre_newlines Number of empty lines in front of the message.
     * @param post_newlines Number of empty lines after the message.
     */
    void Write(std::stringstream &msg, int priority = 20, unsigned int pre_newlines = 0,
               unsigned int post_newlines = 0);
    /**
     * Call this Function to write a BlockVector into a file. The method AllowWrite is called to check if
     * this actually happens. If successfull the filename of the output file will be written with priority 3.
     *
     * @param q          A Reference to the BlockVector to be written.
     * @param name       The Name of the output file (this is what is checked by AllowWrite).
     *                   The directory, iteration counters and ending are added automatically.
     * @param dof_type   A string indicating which dof_handler is associated to the BlockVector.
     *                   Valid options are 'control' and 'state'
     */
    void Write(const VECTOR &q, std::string name, std::string dof_type);

    /**
     * Call this Function to write a Vector containing element-related data into a file.
     * The method AllowWrite is called to check if
     * this actually happens. If successfull the filename of the output file will be written with priority 3.
     *
     * @param q          A Reference to the Vector to be written.
     * @param name       The Name of the output file (this is what is checked by AllowWrite).
     *                   The directory, iteration counters and ending are added automatically.
     * @param dof_type   A string indicating which dof_handler is associated to the BlockVector.
     *                   Valid options are 'control' and 'state'.
     */
    void
    WriteElementwise(const Vector<double> &q, std::string name,
                     std::string dof_type);

    /**
     * Call this Function to write a ControlVector into a file. The method AllowWrite is called to check if
     * this actually happens. If successfull the filename of the output file will be written with priority 3.
     *
     * @param q          A Reference to the ControlVector to be written.
     * @param name       The Name of the output file (this is what is checked by AllowWrite).
     *                   The directory, iteration counters and ending are added automatically.
     * @param dof_type   A string indicating which dof_handler is associated to the BlockVector.
     *                   Valid options are 'control' and 'state'
     */
    void Write(const ControlVector<VECTOR> &q, std::string name, std::string dof_type);
    //    /**
    //     * same as above for ControlVector<dealii::Vector<double> >
    //     *
    //     */
    //    void Write(const ControlVector<dealii::Vector<double> >& q, std::string name, std::string dof_type);
    /**
     * Call this Function to write a std::vector into a file. The method AllowWrite is called to check if
     * this actually happens. If successfull the filename of the output file will be written with priority 3.
     * The output is a gnuplot file.
     *
     * @param q          A Reference to the ControlVector to be written.
     * @param name       The Name of the output file (this is what is checked by AllowWrite).
     *                   The directory, iteration counters and ending are added automatically.
     * @param dof_type   A string indicating which dof_handler is associated to the BlockVector.
     *                   Valid options are 'time'.
     */
    void Write(const std::vector<double> &q, std::string name, std::string dof_type);
    /**
     * Writes an error message to std::cerr and into the Log. This message is printed always
     * regardless of priorities.
     *
     * @param msg   The message to be printed.
     */
    void WriteError(std::string msg);
    /**
     * Sets the iteration  number for the automatic file name extension.
     * Whether it is actually used is decided by the AllowIteration method.
     *
     * @param iteration    The number of the current iteration.
     * @param type         A name indicating which iteration the number corresponds to.
     */
    void SetIterationNumber(unsigned int iteration, std::string type);
    /**
     * If called this Method initializes a new subdirectory into which the solutions are written.
     * Usually this should be called once on each mesh.
     */
    void ReInit();
    /**
     * Decides if the exact output value is written out or
     * zero (in case if the exact output value is below a
     * certain tolerance.
     */
    std::string ZeroTolerance(double value, double reference_value);
    /**
     * This function sets the precision of the newton output values.
     */
    void InitNewtonOut(std::stringstream &msg);
    /**
     * This function sets the precision of the functional output values.
     */
    void InitOut(std::stringstream &msg);

    /**
     * This function can be called to disable all output.
     */
    void DisallowAllOutput()
    {
      disallow_all_ = true;
    };
    /**
     * This function is used to restore normal output behavior after a call
     * of DisallowAllOutput
     */
    void ResumeOutput()
    {
      disallow_all_ = false;
    }

    /**
     * This function constructs the correct output name given by name and dof_type.
     */
    std::string
    ConstructOutputName(std::string name, std::string dof_type);

    /**
     * Redirects all output to std::cout to the logfile.
     * This is usefull if third party software directly prints
     * output onto the screen.
     */
    void StartSaveCTypeOutputToLog();
    /**
     * Disables the redirect of std::cout
     */
    void StopSaveCTypeOutputToLog();

    /**
     * This function gives the current output directory
     */
    std::string GetResultsDir() const;

  protected:
    /**
     * For internal use. This function is used to insert a new iteration counter whose values
     * should be stored. If necessary the counters are reordered.
     *
     * @param type     The name of the counter to be stored.
     * @return         An iterator pointing onto the position of the newly inserted type.
     */
    std::map<std::string, unsigned int>::const_iterator ReorderAndInsert(std::string type);
    /**
     * This function computes the extension to the file name given by the stored iteration variables.
     *
     * @return The postfix to the filename.
     */
    std::string GetPostIndex();
    ReducedProblemInterface_Base<VECTOR> *GetReducedProblem()
    {
      if (NULL != Solver_)
        return Solver_;
      else
        {
          abort();
        }
    }

    /**
     * This function decides whether or not a Vector with the given name should be written.
     * It returns false if the name contains any substring that is given by the parameter file
     * option 'never_write_list', otherwise it returns true.
     * If the debug mode is activated via the paramfile, a declined name is noted in the log.
     *
     * @param name        The name to be checked
     * @return            A boolean that is true if writing this vector is ok, and false otherwise.
     */
    bool AllowWrite(std::string name);
    /**
     * This function decides whether or not to store an iteration counter for automatic filename extension.
     * It returns false if the name contains any substring that is given by the parameter file
     * option 'ignore_iterations', otherwise it returns true.
     * If the debug mode is activated via the paramfile, a declined name is noted in the log.
     *
     * @param name        The name to be checked
     * @return            A boolean that is true if this counter should be stored, and false otherwise.
     */
    bool AllowIteration(std::string name);
    /**
     * This function parses a string into words that are separated by a ';'
     *
     * @param tmp     The string to be parsed.
     * @param list    A reference to a vector of strings. All words in the string tmp
     *                are beeing stored as a separate string in this list.
     */
    void ParseString(const std::string tmp, std::vector<std::string> &list);

  private:
    std::map<std::string, unsigned int> iteration_type_pos_;
    std::vector<unsigned int> iteration_number_;
    int printlevel_;
    std::string results_basedir_, results_outdir_, ending_, control_ending_, logfile_;
    ReducedProblemInterface_Base<VECTOR> *Solver_;
    unsigned int n_reinits_;
    bool debug_;
    unsigned int number_precision_;
    unsigned int functional_number_precision_;
    double user_eps_machine_;
    bool disallow_all_;

    std::vector<std::string> never_write_list;
    std::vector<std::string> ignore_iterations;

    std::ofstream log_;

    fpos_t std_out_pos_;
    int    stdout_backup_;

    /**
      * Prints Copyright information
      */
    void PrintCopyrightNotice();

  };

}
#endif
