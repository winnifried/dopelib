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

#ifndef SOLVER_INTERFACE_H_
#define SOLVER_INTERFACE_H_

#include <include/dopeexceptionhandler.h>
#include <include/outputhandler.h>
#include <include/controlvector.h>
#include <include/constraintvector.h>
#include <basic/dopetypes.h>
#include <container/dwrdatacontainer.h>

#include <assert.h>
#include <deal.II/numerics/data_out.h>
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
   * The base class for all solvers.
   * Defines the non dimension dependent interface for the output handling
   */
  template<typename VECTOR>
  class ReducedProblemInterface_Base
  {
  public:
    ReducedProblemInterface_Base()
    {
      ExceptionHandler_ = NULL;
      OutputHandler_ = NULL;
    }

    virtual
    ~ReducedProblemInterface_Base()
    {
    }
    ;
    /**
     * Basic function to get information of the state size.
     *
     * @param out         The output stream.
     */
    virtual void
    StateSizeInfo(std::stringstream &out)=0;

    /******************************************************/

    /**
     * Basic function to write vectors in files.
     *
     *  @param v           The BlockVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     */
    virtual void
    WriteToFile(const VECTOR &v, std::string name, std::string outfile,
                std::string dof_type, std::string filetype)=0;

    /******************************************************/

    /**
     * Basic function to write vectors containing element-related data in files.
     *
     *  @param v           The BlockVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     */
    virtual void
    WriteToFileElementwise(const Vector<double> &/*v*/, std::string /*name*/,
                           std::string /*outfile*/, std::string /*dof_type*/,
                           std::string /*filetype*/)
    {
      throw DOpEException("NotImplemented", "WriteToFileElementwise");
    }

    /******************************************************/

    /**
     * Basic function to write vectors in files.
     *
     *  @param v           The ControlVector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
    *  @param dof_type    Has the DoF type: state or control.
     */
    virtual void
    WriteToFile(const ControlVector<VECTOR> &v, std::string name,
                std::string dof_type)=0;

    /******************************************************/

    /**
     * Basic function to write a std::vector to a file.
     *
     *  @param v           A std::vector to write to a file.
     *  @param outfile     The basic name for the output file to print.
     */
    virtual void
    WriteToFile(const std::vector<double> &v, std::string outfile) =0;

    void
    RegisterOutputHandler(DOpEOutputHandler<VECTOR> *OH)
    {
      OutputHandler_ = OH;
    }
    void
    RegisterExceptionHandler(DOpEExceptionHandler<VECTOR> *OH)
    {
      ExceptionHandler_ = OH;
    }

    DOpEExceptionHandler<VECTOR> *
    GetExceptionHandler()
    {
      assert(ExceptionHandler_);
      return ExceptionHandler_;
    }
    DOpEOutputHandler<VECTOR> *
    GetOutputHandler()
    {
      assert(OutputHandler_);
      return OutputHandler_;
    }

    /**
     * Grants access to the computed value of the functional named 'name'.
     * This method should be used in the stationary case, see GetTimeFunctionalValue
     * for the instationary one.
     *
     * @return                Value of the functional in question.
     * @param name            Name of the functional in question.
     */
    double
    GetFunctionalValue(std::string name) const
    {
      typename std::map<std::string, unsigned int>::const_iterator it =
        GetFunctionalPosition().find(name);
      if (it == GetFunctionalPosition().end())
        {
          throw DOpEException(
            "Did not find " + name + " in the list of functionals.",
            "ReducedProblemInterface_Base::GetFunctionalValue");
        }
      unsigned int pos = it->second;

      if (GetFunctionalValues()[pos].size() != 1)
        {
          if (GetFunctionalValues()[0].size() == 0)
            throw DOpEException(
              "Apparently the Functional in question was never evaluated!",
              "ReducedProblemInterface_Base::GetFunctionalValue");
          else
            throw DOpEException(
              "The Functional has been evaluated too many times! \n\tMaybe you should use GetTimeFunctionalValue.",
              "ReducedProblemInterface_Base::GetFunctionalValue");

        }
      else
        {
          return GetFunctionalValues()[pos][0];
        }
    }

    /**
     * Grants access to the computed value of the functional named 'name'.
     * This should be used in the instationary case.
     *
     * @return                Value of the functional in question.
     * @param name            Name of the functional in question.
     */
    const std::vector<double> &
    GetTimeFunctionalValue(std::string name) const
    {
      typename std::map<std::string, unsigned int>::const_iterator it =
        GetFunctionalPosition().find(name);
      if (it == GetFunctionalPosition().end())
        {
          throw DOpEException(
            "Did not find " + name + " in the list of functionals.",
            "ReducedProblemInterface_Base::GetFunctionalValue");
        }
      unsigned int pos = it->second;

      if (GetFunctionalValues()[pos].size != 1)
        {
          if (GetFunctionalValues()[0].size() == 0)
            throw DOpEException(
              "Apparently the Functional in question was never evaluated!",
              "ReducedProblemInterface_Base::GetTimeFunctionalValue");
          else
            throw DOpEException(
              "The Functional has been evaluated too many times! \n\tMaybe you should use GetTimeFunctionalValue.",
              "ReducedProblemInterface_Base::GetTimeFunctionalValue");
        }
      else
        {
          return GetFunctionalValues()[pos];
        }
    }


    /**
     * The user can add his own Domain Data (for example the coefficient
     * vector of a finite element function). The user has to make sure
     * that the vector new_data has the appropriate length. This data
     * is the accessible in the integrator routines.
     *
     * @param name      The unique identifier for the data-vector.
     * @param new_data  The vector one wishes to add.
     */
    void AddUserDomainData(std::string name,const VECTOR *new_data)
    {
      if (user_domain_data_.find(name) != user_domain_data_.end())
        {
          throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "ReducedProblemInterface::AddUserDomainData");
        }
      user_domain_data_.insert(
        std::pair<std::string, const VECTOR *>(name, new_data));
    }


    /**
     * This function allows to delete user-given domain data vectors,
     * see AddUserDomainData.
     */
    void DeleteUserDomainData(
      std::string name)
    {
      typename std::map<std::string, const VECTOR *>::iterator it =
        user_domain_data_.find(name);
      if (it == user_domain_data_.end())
        {
          throw DOpEException(
            "Deleting Data " + name + " is impossible! Data not found",
            "Integrator::DeleteDomainData");
        }
      user_domain_data_.erase(it);
    }
  protected:
    /**
     * Resets the functional_values_ to their proper size.
     *
     * @ param N            Number of functionals (aux + cost).
     */
    void
    InitializeFunctionalValues(unsigned int N)
    {
      //Initializing Functional Values
      GetFunctionalValues().resize(N);
      for (unsigned int i = 0; i < GetFunctionalValues().size(); i++)
        {
          GetFunctionalValues()[i].resize(0);
        }
    }
    std::vector<std::vector<double> > &
    GetFunctionalValues()
    {
      return functional_values_;
    }

    const std::vector<std::vector<double> > &
    GetFunctionalValues() const
    {
      return functional_values_;
    }

    const std::map<std::string, const VECTOR *> &
    GetUserDomainData() const
    {
      return user_domain_data_;
    }

    /**
     * This has to get implemented in the derived classes
     * like optproblem, pdeproblemcontainer etc.
     * It returns a map connecting the names of the
     * added functionals with their position in
     * functional_values_.
     * If a cost functional is present, its values are always
     * stored in functional_values_[0]. Auxiliary functionals are stored after
     * the cost functional (present or not!) in the order as they are added.
     */
    virtual const std::map<std::string, unsigned int> &
    GetFunctionalPosition() const
    {
      throw DOpEException("Method not implemented",
                          "ReducedProblemInterface_Base::GetFunctionalPosition");
    }
    ;

  private:
    DOpEExceptionHandler<VECTOR> *ExceptionHandler_;
    DOpEOutputHandler<VECTOR> *OutputHandler_;
    std::vector<std::vector<double> > functional_values_;
    std::map<std::string, const VECTOR *> user_domain_data_;
  };

  /**
   * A template for different solver types to be used for solving
   * PDE- as well as optimization problems.
   */
  template<typename PROBLEM, typename VECTOR>
  class ReducedProblemInterface : public ReducedProblemInterface_Base<VECTOR>
  {
  public:
    ReducedProblemInterface(PROBLEM *OP, int base_priority = 0)
      : ReducedProblemInterface_Base<VECTOR>()
    {
      OP_ = OP;
      base_priority_ = base_priority;
      post_index_ = "_" + this->GetProblem()->GetName();
    }
    ~ReducedProblemInterface()
    {
    }

    /******************************************************/

    /**
     * Basic function which is given to instatsolver.h and statsolver.h, respectively,
     * and reinitializes vectors, matrices, etc.
     *
     */
    virtual void
    ReInit()
    {
      this->GetProblem()->ReInit("reduced");
    }

    /******************************************************/

    /**
     * Basic function which is given to instatsolver.h and statsolver.h, respectively,
     * It computes the value of the constraint mapping and returns a boolean indicating
     * whether the point is feasible.
     *
     * @param q            The ControlVector is given to this function.
     * @param g            The ConstraintVector that contains the value of the constraint mapping after completion.
     *
     * @return             True if feasible, false otherwise.
     */
    virtual bool
    ComputeReducedConstraints(const ControlVector<VECTOR> &q,
                              ConstraintVector<VECTOR> &g) = 0;

    /******************************************************/

    /**
     * Basic function which is given to instatsolver.h and statsolver.h, respectively,
     * It fills the values of the lower and upper box constraints on the control variable in a vector
     *
     * @param lb           The ControlVector to store the lower bounds
     * @param ub           The ControlVector to store the upper bounds
     */
    virtual void
    GetControlBoxConstraints(ControlVector<VECTOR> &lb,
                             ControlVector<VECTOR> &ub)= 0;

    /******************************************************/

    /**
     * Basic function to compute the reduced gradient solution.
     * We assume that state u(q) is already computed.
     * However the adjoint is not assumed to be computed.
     *
     * @param q                    The ControlVector is given to this function.
     * @param gradient             The gradient vector.
     * @param gradient_transposed  The transposed version of the gradient vector.
     */
    virtual void
    ComputeReducedGradient(const ControlVector<VECTOR> &q,
                           ControlVector<VECTOR> &gradient,
                           ControlVector<VECTOR> &gradient_transposed)=0;

    /******************************************************/

    /**
     * Basic function to return the computed value of the reduced cost functional.
     *
     * @param q            The ControlVector is given to this function.
     */
    virtual double
    ComputeReducedCostFunctional(const ControlVector<VECTOR> &q)=0;

    /******************************************************/

    /**
     * Basic function to compute reduced functionals.
     * We assume that state u(q) is already computed.
     *
     * @param q            The ControlVector is given to this function.
     */
    virtual void
    ComputeReducedFunctionals(const ControlVector<VECTOR> &q)=0;

    /******************************************************/

    /**
     * Basic function to compute the reduced gradient solution.
     * We assume that adjoint state z(u(q)) is already computed.
     *
     * @param q                             The ControlVector is given to this function.
     * @param direction                     Documentation will follow later.
     * @param hessian_direction             Documentation will follow later.
     * @param hessian_direction_transposed  Documentation will follow later.
     */
    virtual void
    ComputeReducedHessianVector(const ControlVector<VECTOR> &q,
                                const ControlVector<VECTOR> &direction,
                                ControlVector<VECTOR> &hessian_direction,
                                ControlVector<VECTOR> &hessian_direction_transposed)=0;

    virtual void
    ComputeReducedHessianInverseVector(const ControlVector<VECTOR> & /*q*/,
                                       const ControlVector<VECTOR> & /*direction*/,
                                       ControlVector<VECTOR> &       /*hessian_direction*/)
    {
      throw DOpEException("Method not implemented",
                          "ReducedProblemInterface::ComputeReducedHessianInverseVector");
    }

    /**
     * We assume that the constraints g have been evaluated at the corresponding
     * point q. This comutes the reduced gradient of the global constraint num
     * with respect to the control variable.
     *
     * @param num                           Number of the global constraint to which we want to
     *                                      compute the gradient.
     * @param q                             The ControlVector<VECTOR> is given to this function.
     * @param g                             The ConstraintVector<VECTOR> which contains the
     *                                      value of the constraints at q.
     * @param gradient                      The vector where the gradient will be stored in.
     * @param gradient_transposed           The transposed version of the gradient vector.
     */
    virtual void
    ComputeReducedGradientOfGlobalConstraints(unsigned int /*num*/,
                                              const ControlVector<VECTOR> & /*q*/,
                                              const ConstraintVector<VECTOR> & /*g*/,
                                              ControlVector<VECTOR> & /*gradient*/,
                                              ControlVector<VECTOR> & /*gradient_transposed*/)

    {
      throw DOpEException("Method not implemented",
                          "ReducedProblemInterface::ComputeReducedGradientOfGlobalConstraints");
    }

    /*****************************************************/
    /**
     * Sets the type of the Problem OP_. This function secures the proper initialization of the
     * FEValues after the type has changed. See also the documentation of SetType in optproblemcontainer.h
     */
    void
    SetProblemType(std::string type, unsigned int num = 0)
    {
      this->GetProblem()->SetType(type, num);
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
//        /**
//         * Initializes the HigherOrderDWRDataContainer
//         */
//        template<class DWRC>
//          void
//          InitializeDWRC(DWRC& dwrc)
//          {
//            dwrc.Initialize(GetProblem()->GetSpaceTimeHandler(),
//                GetProblem()->GetControlNBlocks(),
//                GetProblem()->GetControlBlockComponent(),
//                GetProblem()->GetStateNBlocks(),
//                GetProblem()->GetStateBlockComponent());
//          }

  protected:
    virtual const std::map<std::string, unsigned int> &
    GetFunctionalPosition() const
    {
      return GetProblem()->GetFunctionalPosition();
    }
    std::string
    GetPostIndex() const
    {
      return post_index_;
    }
    int
    GetBasePriority() const
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
