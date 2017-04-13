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

#ifndef PDEPROBLEM_INTERFACE_H_
#define PDEPROBLEM_INTERFACE_H_

#include <include/dopeexceptionhandler.h>
#include <include/outputhandler.h>
#include <include/controlvector.h>
#include <include/constraintvector.h>
#include <interfaces/reducedprobleminterface.h>
#include <container/dwrdatacontainer.h>

#include <assert.h>

#include <deal.II/lac/vector.h>

namespace DOpE
{
  //Predeclaration necessary
  template<typename VECTOR>
  class DOpEOutputHandler;
  template<typename VECTOR>
  class DOpEExceptionHandler;
  /////////////////////////////

  /**
   * This class is the basis for the reduced problems,
   * i.e., those in which the PDE has been elimiated for
   * the solution of stationary and nonstationary PDEs.
   */
  template<typename PROBLEM, typename VECTOR, int dealdim>
  class PDEProblemInterface : public ReducedProblemInterface_Base<VECTOR>
  {
  public:
    PDEProblemInterface(PROBLEM *OP, int base_priority = 0)
      : ReducedProblemInterface_Base<VECTOR>()
    {
      OP_ = OP;
      base_priority_ = base_priority;
      post_index_ = "_" + this->GetProblem()->GetName();
    }
    virtual
    ~PDEProblemInterface()
    {
    }

    /******************************************************/

    /**
    * Reinitialization when needed to adjust vector and matrix
    * sizes.
     *
     */
    virtual void
    ReInit()
    {
      this->GetProblem()->ReInit("reduced");
    }

    /******************************************************/

    /**
     * Evaluation of the functionals in the solution of the
     * PDE. This function needs to be specified separately for
    * stationary and non stationary problems since the
    * evaluation of the functionals differs.
     */
    virtual void
    ComputeReducedFunctionals()=0;

    /******************************************************/

    /**
     * Sets the type of the Problem OP_. This function secures the proper initialization of the
     * FEValues after the type has changed. See also the documentation of SetType in optproblemcontainer.h
     */
    void
    SetProblemType(std::string type, unsigned int num = 0)
    {
      this->GetProblem()->SetType(type, num);
    }

    /**
     * Initializes the HigherOrderDWRDataContainer
     * (we need GetStateNBlocks() and GetStateBlockComponent()!)
     */
    template<class DWRC>
    void
    InitializeDWRC(DWRC &dwrc)
    {
      dwrc.Initialize(GetProblem()->GetSpaceTimeHandler(),
                      GetProblem()->GetStateNBlocks(),
                      GetProblem()->GetStateBlockComponent());
    }

  protected:
    /**
     * Just calls the GetFunctioalPosition() method of the problem. See
     * there for further documentation of the method.
     */
    virtual const std::map<std::string, unsigned int> &
    GetFunctionalPosition() const
    {
      return GetProblem()->GetFunctionalPosition();
    }

    PROBLEM *
    GetProblem()
    {
      return OP_;
    }
    const PROBLEM *
    GetProblem() const
    {
      return OP_;
    }
    std::string
    GetPostIndex()
    {
      return post_index_;
    }
    int
    GetBasePriority()
    {
      return base_priority_;
    }
  private:
    PROBLEM *OP_;
    int base_priority_;
    std::string post_index_;
  };

}
#endif
