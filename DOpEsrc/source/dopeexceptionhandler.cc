#include "dopeexceptionhandler.h"
#include "outputhandler.h"

#include <iostream>
#include <string>
#include <cstdlib>


namespace DOpE
{

/*******************************************************/

template<typename VECTOR>
DOpEExceptionHandler<VECTOR>::DOpEExceptionHandler(DOpEOutputHandler<VECTOR>* OutputHandler)
{
  _OutputHandler = OutputHandler;
}

/*******************************************************/

template<typename VECTOR>
DOpEExceptionHandler<VECTOR>::~DOpEExceptionHandler()
{

}

/*******************************************************/

template<typename VECTOR>
void DOpEExceptionHandler<VECTOR>::HandleException(DOpEException& e,std::string reporter)
{
  GetOutputHandler()->WriteError("The following `" + e.GetName() + "` reports!");
  GetOutputHandler()->WriteError("Warning: During execution of `" + e.GetThrowingInstance()
      + "` the following Problem occurred!");
  GetOutputHandler()->WriteError(e.GetErrorMessage());
  GetOutputHandler()->WriteError("Reported by `"+reporter+"`");
}

/*******************************************************/

template<typename VECTOR>
void DOpEExceptionHandler<VECTOR>::HandleCriticalException(DOpEException& e,std::string reporter)
{
  GetOutputHandler()->WriteError("The following `" + e.GetName() + "` reports!");
  GetOutputHandler()->WriteError("Error: During execution of `" + e.GetThrowingInstance()
      + "` the following fatal Problem occurred!");
  GetOutputHandler()->WriteError(e.GetErrorMessage());
  GetOutputHandler()->WriteError("Reported by `"+reporter+"`");
  abort();
}

}//Endof namespace
/******************************************************/

/******************************************************/
/******************************************************/
template class DOpE::DOpEExceptionHandler<dealii::Vector<double> >;
template class DOpE::DOpEExceptionHandler<dealii::BlockVector<double> >;

