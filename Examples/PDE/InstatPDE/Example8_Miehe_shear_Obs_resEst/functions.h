#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <wrapper/function_wrapper.h>

namespace local
{
  double rhs(const Point<2> &/*p*/) 
  {
    //const double x = p[0];
    //const double y = p[1];
    
    //return sin(M_PI*x)*sin(2*M_PI*y);
    return 1.;
  }

  /*  class
    Obstacle : public Function<2>
  {
  public:
    Obstacle() : Function<2>(4){}

    double value(const Point<2> &p, const unsigned int component = 0 ) const
    {
      //return 0.5;
      if( component ==2 )
      {
	const double x = p[0];
	const double y = p[1];
	if( (x >= 0.25 && x <= 0.75) && (y >= 0.25 && y <= 0.75 ))
	  return 0.5;
	//return 1.-3*x*(1.-x)-3*y*(1.-y);
      }
      return 1.;
    }
    
    };*/
}

#endif
