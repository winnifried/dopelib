#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <wrapper/function_wrapper.h>

namespace error_eval
{
  double rhs(const Point<2> &/*p*/)
  {
    return 1.;
  }
}

#endif
