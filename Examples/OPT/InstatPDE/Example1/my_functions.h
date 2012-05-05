#ifndef _MY_FUNCTIONS_
#define _MY_FUNCTIONS_


#include "function_wrapper.h"

using namespace dealii;

/******************************************************/

class RightHandSideFunction: public DOpEWrapper::Function<2>
{
	public:
  RightHandSideFunction():DOpEWrapper::Function<2>()
  {

  }
  virtual double value(const Point<2> &p, const unsigned int component = 0) const;

  void SetTime(double t) const { mytime=t;}
			
        private:
  mutable double mytime;

};

/******************************************************/

double RightHandSideFunction::value(const Point<2> &p, const unsigned int/* component*/) const
{

  //    return ((std::cos(mytime) + 2*std::sin(mytime)) * std::sin(p[0]) * std::sin(p[1]));
  return ((3-2*mytime) * std::exp(mytime-mytime*mytime) * sin(p[0]) * sin(p[1]) + std::exp(mytime-mytime*mytime) * sin(p[0]) * sin(p[1]) * std::exp(mytime-mytime*mytime) * sin(p[0]) * sin(p[1]));
}

/******************************************************/


#endif
