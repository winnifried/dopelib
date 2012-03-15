#ifndef _DOPE_EXCEPTION_H_
#define _DOPE_EXCEPTION_H_

#include <string>

namespace DOpE
{

  /**
   * This class is the base for all exceptions thrown by methods implemented in DOpE.
   */
  class DOpEException
  {
  public:
    /**
     * The Constructor for all exceptions.
     * 
     * @param message          An (hopefully) informative error message, e.g., what is wrong.
     * @param thrower          Information on where (wich method) the exception occured.
     */
    DOpEException(std::string message, std::string thrower="unspecified throwing instance") 
    { 
      _msg = message;
      _thrower = thrower;
    }
      
    ~DOpEException() {}
    
    /**
     * Access method for the stored error message.
     *
     * @return A string containing the message by which this object was initialized.
     */
    std::string GetErrorMessage() { return _msg; }
    /**
     * Access method for the stored throwing instance.
     *
     * @return A string containing the throwing instance by which this object was initialized.
     */
    std::string GetThrowingInstance() {return  _thrower; }
    virtual std::string GetName() {return "DOpEException";} 
  protected:

  private:
    std::string _msg;
    std::string _thrower;
  };

  /**
   * A specialized exception, to indicate that an error during an iterative method has occured,
   * e.g., too many iterations.
   */
  class DOpEIterationException : public DOpEException
  {
  public:
    DOpEIterationException(std::string message, std::string thrower="unspecified throwing instance")
      : DOpEException(message,thrower)
    {}
    virtual std::string GetName() {return "DOpEIterationException";} 
  }; 

  /**
   * A specialized exception that indicates that method a method, e.g., 
   * conjugate gradient for the reduced hessian, 
   * has terminated because of a negative curvature. 
   */
  class DOpENegativeCurvatureException : public DOpEException
  {
  public:
    DOpENegativeCurvatureException(std::string message, std::string thrower="unspecified throwing instance")
      : DOpEException(message,thrower)
    {}
    virtual std::string GetName() {return "DOpENegativeCurvatureException";} 
  };
}
#endif
