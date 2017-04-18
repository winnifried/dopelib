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

#ifndef MY_FUNCTIONS_H_
#define MY_FUNCTIONS_H_

#include <wrapper/function_wrapper.h>

class ExactSolution : public DOpEWrapper::Function<2>
{
public:

  double
  value(const dealii::Point<2> &p, const unsigned int /*component*/ = 0) const
  {
//      Assert(component < this->n_components,
//          ExcIndexRange (component, 0, this->n_components));

    return std::pow(p[0],2.) + std::pow(p[1],2.);
  }

  double
  laplacian(const dealii::Point<2> & /*p*/,
            const unsigned int /*component*/ = 0) const
  {
    return 4.;
  }

  void
  vector_value(const dealii::Point<2> &p, dealii::Vector<double> &value) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      value(c) = ExactSolution::value(p, c);
  }
};

#endif /* MY_FUNCTIONS_H_ */
