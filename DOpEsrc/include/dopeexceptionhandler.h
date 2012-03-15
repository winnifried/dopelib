#ifndef _DOPE_EXCEPTION_HANDLER_H_
#define _DOPE_EXCEPTION_HANDLER_H_

#include "dopeexception.h"
//#include "outputhandler.h"
#include <string>

namespace DOpE
{
//Predeclaration necessary
  template<typename VECTOR> class DOpEOutputHandler;
/////////////////////////////
template<typename VECTOR>
  class DOpEExceptionHandler
  {
  public:
    DOpEExceptionHandler(DOpEOutputHandler<VECTOR>* OutputHandler);
    ~DOpEExceptionHandler();

    /**
     * This function handles non critical exceptions, e.g.,  if an iterative
     * solver reached its maximal iteration count.
     * This function writes a warning message, but doesn't stop the computation.
     */
    void HandleException(DOpEException& e, std::string reporter = "undefined");
    /**
     * This function handles critical exceptions, e.g., the equation couln't be solved.
     * This function writes a warning message, and then terminates the program because
     * it is impossible to continue after a severe problem.
     */
    void HandleCriticalException(DOpEException& e, std::string reporter = "undefined");
  protected:
    DOpEOutputHandler<VECTOR>* GetOutputHandler() { return _OutputHandler; }
  private:
    DOpEOutputHandler<VECTOR>* _OutputHandler;
  };

}
#endif
