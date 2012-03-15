#ifndef _CONSTRAINT_VECTOR_H_
#define _CONSTRAINT_VECTOR_H_

#include "spacetimehandler_base.h"

#include <lac/vector.h>
#include <lac/block_vector_base.h>
#include <lac/block_vector.h>

#include <vector>
#include <iostream>
#include <sstream>


namespace DOpE
{
  template<typename VECTOR>
  class ConstraintVector
  {
  public:
    //TODO: Currently we only consider one fixed control 
    //      for all timesteps, if more is desired one needs to augment the 
    //      Spacetimehandler to have a time discretization for the control,
    //      Then one can update this vector similar to the statevector 
    //      with different meshes for Vectors.
    //      Note that this requires to keep track of the interpolation
    //      between state and control time points...
    ConstraintVector(const ConstraintVector& ref);
    ConstraintVector(const SpaceTimeHandlerBase<VECTOR>* STH, std::string behavior);
    ~ConstraintVector();

    /**
     * Sets the time in the vector. This Function or SetTimeDoFNumber
     * must be called before calling GetSpacialVector
     * in order to load the required data.
     *
     * @param t            A double containing the time we are interested in. If t doesn't match the time given by
     *                     time_point, then an interpolation between the corresponding time_points is
     *			               computed.
     * @param interval      An TimeIterator. The interval containing t.
     *
     */
    void SetTime(double t, const TimeIterator& interval) const;

    /**
     * Sets the time in the vector. This Function or SetTime
     * must be called before calling GetSpacialVector
     * in order to load the required data.
     *
     * @param time_point   An unsigned integer. This gives the number of the point in the  time mesh.
     *
     */
    void SetTimeDoFNumber(unsigned int time_point) const;

    /**
     * Returns true if there is a constraint associated to the name.
     */
    bool HasType(std::string name) const;
   
    /**
     * Returns a reference to the spacial vector associated to the last time given by SetTime*
     * The Constrainttype must be indicated in the string name.
     * Feasible values are 'local' for local in time and space
     *                     'local_global' for local in space but global in time constraints.
     */
    VECTOR& GetSpacialVector(std::string name);

    /**
     * Returns a const reference to the spacial vector associated to the last time given by SetTime*
     * See also GetSpacialVector
     */
    const VECTOR& GetSpacialVector(std::string name) const;
    
    /**
     * Returns the vector containing information on global in space and time constraints
     */
    const dealii::Vector<double>& GetGlobalConstraints() const;
    /**
     * Returns the vector containing information on global in space and time constraints
     */
    dealii::Vector<double>& GetGlobalConstraints();
    /**
     * Sets all the vector to a constant value.
     *
     * @param value    The constant value to be assigned to the vector.
     */
    void operator=(double value);
    /**
     * Sets this vector to the values of an other given vector.
     * If required this vector is resized. This invalidates all prior SetTime* calls.
     *
     * @param dq    The other vector.
     */
    void operator=(const ConstraintVector& dq);
    /**
     * Upon completion each entry of this Vector contains the following
     * Result this = this + dq;
     * It is required that both this and dq have the same structure!
     *
     * @param dq    The increment.
     */
    void operator+=(const ConstraintVector& dq);
    /**
     * Multiplies the Vector with a constant.
     *
     * @param a    A double to be multiplied with the vector.
     */
    void operator*=(double a);
    /**
     * Computes the Euclidean scalar product of this vector with the argument.
     * Both Vectors must have the same struckture.
     *
     * @param dq    The argument for the computation of the scalarproduct.
     * @return      A double containing the scalar product.
     */
    double operator*(const ConstraintVector& dq) const;
    /**
     * Sets this vector adds a multiple of an other vector to this vector.
     * this = this + s * dq
     * It expects both vectors  to be of the same structure.
     *
     * @param s    A double, by which the other vector is scaled.
     * @param dq   The other vector.
     */
    void add(double s, const ConstraintVector& dq);
    /**
     * Sets this vector to the values of an other given vector.
     * The vector is not resized!
     *
     * @param dq    The other vector.
     */
    void equ(double s, const ConstraintVector& dq);

    /**
     * Prints Information on this vector into the given stream.
     *
     * @param out    The output stream.
     */
    void PrintInfos(std::stringstream& out);

    /**
     * This returns the behavior of the ConstraintVector
     * Currently implemented are the following posibilities
     * @par  fullmem         Means there is a spacial vector for each time point. The whole vector
     *                       is stored in main memory.
     *
     * @return               A string indicating the behavior.
     */
    std::string GetBehavior() const { return _behavior; }

    /**
     * @return               A const pointer to the SpaceTimeHandler associated with this vector.
     */
    const SpaceTimeHandlerBase<VECTOR>* GetSpaceTimeHandler() const { return _STH; }

    /**
     * Call if the SpaceTimeHandler has changed to reinitialize vector sizes.
     *
     */
    void ReInit();

    /**
     * Computes the norm given by name of the vector.
     * Feasible values are "infty", and "l1"
     * The string restriction defines if only certain values are
     * to be considered. Currently "all" and "positive" are feasible
     * Meaning that either all or only the positive entries are 
     * considered.
     */
    double Norm(std::string name,std::string restriction = "all") const;

  private:
    /**
     * This function resizes the spacial vector at a prior given time point.
     * Hence SetTimeDoFNumber should be called before this function.
     */
    void ReSizeLocalSpace(unsigned int ndofs, const std::vector<unsigned int>& dofs_per_block);

    void ReSizeGlobal(unsigned int ndofs);

    std::vector<VECTOR* > _local_control_constraint;
    mutable VECTOR _local_constraint__control;

    dealii::Vector<double> _global_constraint;

    mutable int _accessor;

    const SpaceTimeHandlerBase<VECTOR>* _STH;
    std::string _behavior;
    unsigned int _sfh_ticket;
  };


}
#endif
