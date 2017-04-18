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

#ifndef DOPE_EXCEPTION_H_
#define DOPE_EXCEPTION_H_

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
      msg_ = message;
      thrower_ = thrower;
    }

    ~DOpEException() {}

    /**
     * Access method for the stored error message.
     *
     * @return A string containing the message by which this object was initialized.
     */
    std::string GetErrorMessage()
    {
      return msg_;
    }
    /**
     * Access method for the stored throwing instance.
     *
     * @return A string containing the throwing instance by which this object was initialized.
     */
    std::string GetThrowingInstance()
    {
      return  thrower_;
    }
    virtual std::string GetName()
    {
      return "DOpEException";
    }
  protected:

  private:
    std::string msg_;
    std::string thrower_;
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
    virtual std::string GetName()
    {
      return "DOpEIterationException";
    }
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
    virtual std::string GetName()
    {
      return "DOpENegativeCurvatureException";
    }
  };
}
#endif
