#ifndef _MYFUNCTIONS_
#define _MYFUNCTIONS_

namespace MyFunctions{

  //Assumes boundary has been checked before hand, and that values has size 2!
  inline void Forces(std::vector<double>& values, double x, double y __attribute__((unused)))
  {
    values[0] = values [1] = 0.;
    if(fabs(x) <= 0.25)
    {
      values[1] = -1.;
    }
  }

}

#endif
