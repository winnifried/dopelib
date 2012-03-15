#ifndef __SNOPT_WRAPPER__
#define __SNOPT_WRAPPER__

#ifdef WITH_SNOPT
//BEGIN DONOT_TOUCH
//The following code is sensitive to ordering
//due to some bugs in f2c and snopt
//please do not touch this unless you are very 
//sure of what you are doing! And even then---don't.
//If you do something wrong here your code may compile
//and then fail if you include some other headers you 
//may additionally need for no apparent reason with no 
//reasonable compiler warning whatsoever.
//         You have been warned: HERE BE DRAGONS!
#include "snopt.hh"
#include "snfilewrapper.hh"
#include "snoptProblem.hh"
///NEEDED DUE TO CONFLICTING defines in f2c.h
#undef abs
#undef dabs
#undef min
#undef max
#undef dmin
#undef dmax
#undef bit_test
#undef bit_clear
#undef bit_set
//END OF f2c BUGFIX
//END DONOT_TOUCH

#include <boost/function.hpp>

namespace DOpEWrapper
{
  struct SNOPT_FUNC_DATA{
    integer    *Status; integer *n;    doublereal* x;
    integer    *needF;  integer *neF;  doublereal* F;
    integer    *needG;  integer *neG;  doublereal* G;
    char       *cu;     integer *lencu;
    integer    *iu;     integer *leniu;
    doublereal *ru;     integer *lenru;
  };
      
  boost::function1<int, SNOPT_FUNC_DATA&> SNOPT_A_userfunc_interface;
  int SNOPT_A_userfunc_(integer    *Status, integer *n,    doublereal x[],
			   integer    *needF,  integer *neF,  doublereal F[],
			   integer    *needG,  integer *neG,  doublereal G[],
			   char       *cu,     integer *lencu,
			   integer    iu[],    integer *leniu,
			   doublereal ru[],    integer *lenru )
  {
    SNOPT_FUNC_DATA data;
    data.Status = Status;
    data.n      = n;
    data.x      = x;
    data.needF  = needF;
    data.neF    = neF;
    data.F      = F;
    data.needG  = needG;
    data.neG    = neG;
    data.G      = G;
    data.cu     = cu;
    data.lencu  = lencu;
    data.iu     = iu;
    data.leniu  = leniu;
    data.ru     = ru;
    data.lenru  = lenru;

    if(SNOPT_A_userfunc_interface)
      return SNOPT_A_userfunc_interface(data);
    else
      throw DOpE::DOpEException("The boost::function SNOPT_userfunc_interface has not been declared","DOpEWrapper::SNOPT::dope_snopt_userfunc_");
  }

  class SNOPT_Problem : public snoptProblem
  {
  public:
    int GetReturnStatus() { return snoptProblem::inform; }
  };
}
#endif //Endof WITH_SNOPT
#endif //Endof File
