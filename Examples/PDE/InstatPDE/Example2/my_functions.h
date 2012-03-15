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

  void SetTime(double t) const { mytime=t;}

private:
  double _mean_inflow_velocity;
  mutable double mytime;

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
      
      /*
      // Fluid Benchmark
      return ( (p(0) == 0) && (p(1) <= 0.41) ? -_mean_inflow_velocity * 
	       (4.0/0.1681) * 		     		    
	       (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );  
      
      */
      
      //	FSI Benchmark
      if (mytime < 2.0)
	{
	  return   ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * _mean_inflow_velocity * 
		     (1.0 - std::cos(M_PI/2.0 * mytime))/2.0 * 		     
		     (4.0/0.1681) * 		     		    
		     (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
	}
      else 
	{
	  return ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * _mean_inflow_velocity  * 			
		   (4.0/0.1681) * 		     		    
		   (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
	}  
      
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

/******************************************************/

class BoundaryParabelExact : public DOpEWrapper::Function<2> 
{
public:
  BoundaryParabelExact () : DOpEWrapper::Function<2>(5) {}
  
  virtual double value (const Point<2>   &p,
			const unsigned int  component = 0) const;
  
  virtual void vector_value (const Point<2> &p, 
			     Vector<double>   &value) const;
  
private:
  
};

/******************************************************/

double 
BoundaryParabelExact::value (const Point<2>  &p,
			    const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));

   double damping_inflow = 1.0; 
 
  if (component == 0)   
    {
      return (-damping_inflow * 
       (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)));
     
    
    }
  else if (component == 2)
    return -p(0) + 6.0;
  return 0;
}

/******************************************************/

void
BoundaryParabelExact::vector_value (const Point<2> &p,
				   Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = BoundaryParabelExact::value (p, c);
}
