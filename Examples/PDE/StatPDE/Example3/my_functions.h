#include "function_wrapper.h"

using namespace dealii;


/******************************************************/

class BoundaryParabel : public DOpEWrapper::Function<2> 
{
public:
  BoundaryParabel (ParameterReader &param_reader) : DOpEWrapper::Function<2>(5) 
  {
    param_reader.SetSubsection("My functions parameters");
    _mean_inflow_velocity = param_reader.get_double ("mean_inflow_velocity");
  }
  
  virtual double value (const Point<2>   &p,
			const unsigned int  component = 0) const;
  
  virtual void vector_value (const Point<2> &p, 
			     Vector<double>   &value) const;

  static void declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("My functions parameters");
    param_reader.declare_entry("mean_inflow_velocity", "0.0",
				 Patterns::Double(0));    
  }

  
private:
  double _mean_inflow_velocity;

};

/******************************************************/

double 
BoundaryParabel::value (const Point<2>  &p,
			    const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));
  
  if (component == 0)   
    {
      
      /*
      // Channel problem
      return   ( (p(0) == -6.0) && (p(1) <= 2.0)  ? - _mean_inflow_velocity * 
       (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)) : 0 );
      */
      
      
      // Benchmark with flag
      return ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * _mean_inflow_velocity * 
		     (4.0/0.1681) * 		     		    
		     (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );  
      

      /*
      // Benchmark
      return ( (p(0) == 0) && (p(1) <= 0.41) ? -_mean_inflow_velocity * 
	       (4.0/0.1681) * 		     		    
	       (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );  
      */
    }	 
  return 0;
}

/******************************************************/

void
BoundaryParabel::vector_value (const Point<2> &p,
				   Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = BoundaryParabel::value (p, c);
}

