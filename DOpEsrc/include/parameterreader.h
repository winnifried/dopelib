#ifndef _PARAMETERREADER_h_
#define _PARAMETERREADER_h_

#include <base/parameter_handler.h>

using namespace dealii;
namespace DOpE
{
  /**
   * This class is designed to allow all components of the program to use the same 
   * parameter file. 
   */
class ParameterReader : public Subscriptor
 {
   public:
   inline ParameterReader();
   /**
    * In order to access a subsetion in a paramfile this Function must be called with
    * The apropriate label for the subsection.
    *
    * @param subsection        The name of the subsection to be used.
    */
   inline void SetSubsection(const std::string subsection);
   /**
    * This function is called in order to read all previously declared entries from a param file.
    *
    * @param parameter_file   The name of the parameter file from which the parameters are to be read.
    */
   inline void read_parameters(const std::string parameter_file);
   /**
    * This is a wrapper to the corresponding dealii::ParameterHandler routine.
    * But the previously subsection is set to the last value set by SetSubsection is used for the declaration.
    */
   inline void declare_entry (const std::string &entry, 
		       const std::string &default_value, 
		       const Patterns::PatternBase &pattern=Patterns::Anything(), 
		       const std::string &documentation=std::string());
   /**
    * This is a wrapper to the corresponding dealii::ParameterHandler routine.
    * But the previously subsection is set to the last value set by SetSubsection is used for the declaration.
    */
   inline double get_double (const std::string &entry_name);
   /**
    * This is a wrapper to the corresponding dealii::ParameterHandler routine.
    * But the previously subsection is set to the last value set by SetSubsection is used for the declaration.
    */
   inline int get_integer (const std::string &entry_name);
   /**
    * This is a wrapper to the corresponding dealii::ParameterHandler routine.
    * But the previously subsection is set to the last value set by SetSubsection is used for the declaration.
    */
   inline std::string get_string (const std::string &entry_name);
   /**
    * This is a wrapper to the corresponding dealii::ParameterHandler routine.
    * But the previously subsection is set to the last value set by SetSubsection is used for the declaration.
    */
   inline bool 	get_bool (const std::string &entry_name);

   private:
     ParameterHandler prm;
     std::string _subsection;
 };


  ParameterReader::ParameterReader()
  { 
    _subsection = "";
  }

void ParameterReader::SetSubsection(const std::string subsection)
  {
    _subsection = subsection;
  }

void ParameterReader::declare_entry(const std::string &entry, 
				    const std::string &default_value, 
				    const Patterns::PatternBase &pattern, 
				    const std::string &documentation)
{
  prm.enter_subsection (_subsection);
  {
    prm.declare_entry(entry,default_value,pattern,documentation);
  }
  prm.leave_subsection();
}

void ParameterReader::read_parameters (const std::string parameter_file)
 { 
   prm.read_input (parameter_file); 
 }

double ParameterReader::get_double (const std::string &entry_name) 
{
  double ret;
  prm.enter_subsection(_subsection);
  {
    ret = prm.get_double(entry_name);
  }
  prm.leave_subsection();
  return ret;
}

int ParameterReader::get_integer (const std::string &entry_name)
{
  int ret;
  prm.enter_subsection(_subsection);
  {
    ret = prm.get_integer(entry_name);
  }
  prm.leave_subsection();
  return ret;
}

 std::string ParameterReader::get_string (const std::string &entry_name)
{
  std::string ret;
  prm.enter_subsection(_subsection);
  {
    ret = prm.get(entry_name);
  }
  prm.leave_subsection();
  return ret;
}

bool ParameterReader::get_bool(const std::string &entry_name)
{
  bool ret;
  prm.enter_subsection(_subsection);
  {
    ret = prm.get_bool(entry_name);
  }
  prm.leave_subsection();
  return ret;
}

}
#endif
