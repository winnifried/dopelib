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
    
    return -5.;
  }

  class
    Obstacle : public Function<2>
  {
  public:
    Obstacle() : Function<2>(2){}

    double value(const Point<2> &p, const unsigned int component = 0 ) const
    {
      //return 0.5;
      if( component ==0 )
      {
	const double x = p[0];
	const double y = p[1];
	//Calculate dist to \partial \Omega
	const double dist1 = min(1-abs(x),1-abs(y));
	//Calculate dist to \Omega \setminus (-1/4,1/4)
	double dist2 = 0.;
	if(max(abs(x),abs(y))<0.25)
	{
	  dist2 = min(0.25-abs(x),0.25-abs(y));
	}
	return dist1 - 2.*dist2 - 1./5.;
      }
      return 1.;
    }
    
  };
}

#endif
