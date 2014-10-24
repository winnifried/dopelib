#ifndef VERSION_H_
#define VERSION_H_

namespace DOpE
{
  namespace VERSION
  {
    //Don't increase unless some major changes have happend
    //Then set minor and fix to zero
    const unsigned int major         = 2;
    //Update when you add new functionality
    //Then set fix to zero
    const unsigned int minor         = 0;
    //Update when you have fixed a bug
    const unsigned int fix           = 0;
    //If we want we can give additional information here 
    const std::string  postfix       = "";
    //When updating, please increase the time
    const unsigned int day           = 24;
    const unsigned int month         = 10;
    const unsigned int year          = 2014;
  }
}

#endif
