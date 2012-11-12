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

#ifndef _MY_FUNCTIONS_
#define _MY_FUNCTIONS_


#include "function_wrapper.h"

using namespace dealii;

/******************************************************/


/******************************************************/

class InitialData: public DOpEWrapper::Function<1>
{
	public:
  InitialData() : DOpEWrapper::Function<1>()
  {

  }
			virtual double value(const Point<1> &p, const unsigned int component = 0) const;
			virtual void vector_value(const Point<1> &p, Vector<double> &value) const;
			
	private:


};

/******************************************************/

double InitialData::value(const Point<1> &p, const unsigned int /*component*/) const
{
  	double x = p[0];

  	return std::min(x,1.-x);

}

/******************************************************/

void InitialData::vector_value(const Point<1> &p, Vector<double> &values) const
{
	for (unsigned int c = 0; c < this->n_components; ++c)
		values(c) = InitialData::value(p, c);
}

/******************************************************/

/* class RightHandSideFunction: public DOpEWrapper::Function<2> */
/* { */
/* 	public: */
/*   RightHandSideFunction():DOpEWrapper::Function<2>() */
/*   { */

/*   } */
/*   virtual double value(const Point<2> &p, const unsigned int component = 0) const; */

/*   void SetTime(double t) const { mytime=t;} */
			
/*         private: */
/*   mutable double mytime; */

/* }; */

/* /\******************************************************\/ */

/* double RightHandSideFunction::value(const Point<2> &p, const unsigned int/\* component*\/) const */
/* { */

/*     return ((std::cos(mytime) + 2*std::sin(mytime)) * std::sin(p[0]) * std::sin(p[1])); */
/*   /\* return ((numbers::PI*std::cos(numbers::PI*mytime) + 2*numbers::PI*numbers::PI*std::sin(numbers::PI*mytime)) * std::sin(numbers::PI*p[0]) * std::sin(numbers::PI*p[1])); *\/ */
/*   /\* std::cout << mytime << std::endl; *\/ */
/*   /\* return 0; *\/ */
/* } */

/******************************************************/


#endif
