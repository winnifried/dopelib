/*
 * my_functions.h
 *
 *  Created on: Oct 10, 2012
 *      Author: cgoll
 */

#ifndef MY_FUNCTIONS_H_
#define MY_FUNCTIONS_H_

#include "function_wrapper.h"

class ExactSolution : public DOpEWrapper::Function<2>
{
  public:
    ExactSolution(unsigned int order)
        : _order(order)
    {
    }
    double
    value(const dealii::Point<2> &p, const unsigned int component = 0) const
    {
      Assert(component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

      return std::pow(p[0],2.) + std::pow(p[1],2.);
    }

    double
    laplacian(const dealii::Point<2> & p,
        const unsigned int component = 0) const
    {
      return 4.;
    }

    void
    vector_value(const dealii::Point<2> &p, dealii::Vector<double> &value) const
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        value(c) = ExactSolution::value(p, c);
    }
  private:
    const double _order;
};

#endif /* MY_FUNCTIONS_H_ */
