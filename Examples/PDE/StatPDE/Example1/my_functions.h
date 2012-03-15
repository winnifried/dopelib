#include "function_wrapper.h"

using namespace dealii;


/******************************************************/

class BoundaryParabel : public DOpEWrapper::Function<2> 
{
public:
  BoundaryParabel () : DOpEWrapper::Function<2>(3) {}
  
  virtual double value (const Point<2>   &p,
			const unsigned int  component = 0) const;
  
  virtual void vector_value (const Point<2> &p, 
			     Vector<double>   &value) const;
  
private:
  
};

/******************************************************/

double 
BoundaryParabel::value (const Point<2>  &p,
			    const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));

   double damping_inflow = 1.0; 
 
  if (component == 0)   
    {
      return   ( (p(0) == -6.0) && (p(1) <= 2.0)  ? -damping_inflow * 
       (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)) : 0 );
     
    
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
