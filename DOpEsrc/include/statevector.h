/**
*
* Copyright (C) 2012 by the DOpElib authors
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

#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include "spacetimehandler_base.h"
#include "parameterreader.h"
#include "dopetypes.h"

#include <base/utilities.h>
#include <lac/vector.h>
#include <lac/block_vector_base.h>
#include <lac/block_vector.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace DOpE
{
/**
 * This class represents the Statevector.
 *
 * @tparam <VECTOR>     Class in which we want to store the spatial vector
 *                      (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
 */
template<typename VECTOR>
class StateVector
{
	public:
    //FIXME this is not a real copyconstructor, it just
    //uses the information of ref about size and so on. Is this correct?
		StateVector(const StateVector<VECTOR>& ref);
		StateVector(const SpaceTimeHandlerBase<VECTOR>* STH, 
			    DOpEtypes::VectorStorageType behavior,
			    ParameterReader &param_reader);
		~StateVector();

    /**
     * Sets the time in the vector. This Function or SetTimeDoFNumber or SetTime
     * must be called before calling GetSpacialVector
     * in order to load the required data.
     *
     * @param dof_number          An unsigned int containing the global time DoF number
     *                            of the given  interval we are interested in.
     * @param interval            A TimeIterator. The interval we are currently looking on.
     *
     */
		void SetTimeDoFNumber(unsigned int dof_number,
		    const TimeIterator& interval) const;

		/**
		 * Sets the time in the vector for interpolation. This Function or SetTimeDoFNumber
		 * or SetTimeDoFNumber must be called before calling GetSpacialVector
		 * in order to load the required data.
		 *
		 * @param t            A double containing the time we are interested in.
     *
		 * @param interval      An TimeIterator. The interval containing t.
		 *
		 */
		void SetTime(double t, const TimeIterator& interval) const;
		/**
		 * Sets the time in the vector. This Function or SetTime or SetTimeDoFNumber
		 * must be called before calling GetSpacialVector
		 * in order to load the required data.
		 *
		 * @param time_point   An unsigned integer. This gives the number of the point in the  time mesh.
		 *
		 */
		void SetTimeDoFNumber(unsigned int time_point) const;
		/**
		 * Returns a reference to the spatial vector associated to the last time given by SetTime*
		 */
		VECTOR& GetSpacialVector();
		/**
		 * Returns a const reference to the spatial vector associated to the last time given by SetTime* or SetTimeDoFNumber
		 */
		const VECTOR& GetSpacialVector() const;
		/**
		 * Returns a const reference to the spatial vector associated to the last time given by SetTime*
		 * This makes a copy of the real vector  in order to change the vector type.
		 * To assert data integrity this Only one Copy may  be obtained at any time.
		 * Hence prior to calling this Function again UnLockCopy must be called.
		 */
		const dealii::Vector<double>& GetSpacialVectorCopy() const;
		/**
		 * Sets all the vector to a constant value. This function calls SetTime(0).
		 *
		 * @param value    The constant value to be assigned to the vector.
		 */
		void operator=(double value);
		/**
		 * Sets this vector to the values of an other given vector.
		 * If required this vector is resized. This function calls SetTime(0).
		 *
		 * @param dq    The other vector.
		 */
		void operator=(const StateVector<VECTOR>& dq);
		/**
		 * Upon completion each entry of this Vector contains the following
		 * Result this = this + dq;
		 * It is required that both this and dq have the same structure!
		 * This function calls SetTime(0).
		 *
		 * @param dq    The increment.
		 */
		void operator+=(const StateVector<VECTOR>& dq);
		/**
		 * Multiplies the Vector with a constant.  It expects both vectors  to be of
		 * the same structure. This function calls SetTime(0).
		 *
		 * @param a    A double to be multiplied with the vector.
		 */
		void operator*=(double a);
		/**
		 * Computes the Euclidean scalar product of this vector with the argument.
		 * Both Vectors must have the same structure.
		 *
		 * @param dq    The argument for the computation of the scalarproduct.
		 * @return      A double containing the scalar product.
		 */
		double operator*(const StateVector<VECTOR>& dq) const;
		/**
		 * Sets this vector adds a multiple of an other vector to this vector.
		 * this = this + s * dq
		 * It expects both vectors  to be of the same structure.
		 * This function calls SetTime(0).
		 *
		 * @param s    A double, by which the other vector is scaled.
		 * @param dq   The other vector.
		 */
		void add(double s, const StateVector<VECTOR>& dq);
		/**
		 * Sets this vector to the values of an other given vector.
		 * The vector is not resized! It expects both vectors  to be of
		 * the same structure. This function calls SetTime(0).
		 *
		 * @param dq    The other vector.
		 */
		void equ(double s, const StateVector<VECTOR>& dq);

		/**
		 * Prints Information on this vector into the given stream.
		 *
		 * @param out    The output stream.
		 */
		void PrintInfos(std::stringstream& out);
		/**
		 * This unlocks the function GetSpacialVectorCopy
		 */
		void UnLockCopy() const
		{
			_lock = false;
		}
		/**
		 * This returns the behavior of the StateVector
		 * Currently implemented are the following possibilities
		 * @par  fullmem         	Means there is a spatial vector for each time point. The whole vector
		 *                       	is stored in main memory.
		 *
		 * @par  store_on_disc   	Means there are only three spatial vectors (for the actual timepoint
		 * 												and his two neighbors) stored in the main memory whereas the rest of
		 * 												the statevector is stored on the hard disc.
		 *
		 * @return               A string indicating the behavior.
		 */
		DOpEtypes::VectorStorageType GetBehavior() const
		{
			return _behavior;
		}
		/**
		 * @return               A const pointer to the SpaceTimeHandler associated with this vector.
		 */
		const SpaceTimeHandlerBase<VECTOR>* GetSpaceTimeHandler() const
		{
			return _STH;
		}
		/**
		 * Call if the SpaceTimeHandler has changed to reinitialize vector sizes.
		 *
		 */
		void ReInit();

	private:
		struct SpatialVectorInfos
        {
            int _size;
            bool _on_disc;

            SpatialVectorInfos(int size = -1, bool on_disc = false)
            {
              _size = size;
              _on_disc = on_disc;
            }
            ;
        };
		/**
		 * This function resizes the spatial vector at a prior given time point.
		 * Hence SetTimeDoFNumber must be called before this function.
		 */
		void ReSizeSpace(unsigned int ndofs, const std::vector<unsigned int>& dofs_per_block);

		/**
		 * Sets the membervariable '_filename' to the name of the file (e.g. the whole path!) corresponding to 'time_point'.
		 *
		 * @ param time_point			The timepoint we are actually interested in.
		 */
		void MakeName(unsigned int time_point) const;
		/**
		 * Stores the BlockVector stored in *_state[1] on the Disc. The name of the file will be
		 * createt by the function call 'MakeName(_local_state.at(1))'.
		 *
		 */
		void StoreOnDisc() const;
		/**
		 * This function reads the BlockVector<double> stored in the file with the name
		 * 'MakeName(time_point)' and stores him in vector.
		 *
		 * This function is set const due to compatibility reasons.
		 *
		 * @ param time_point		The timepoint we are actually interested in.
		 * @ param vector				A BlockVector in which the Data read out from Disc will be stored.
		 */
		void FetchFromDisc(unsigned int time_point, VECTOR & vector) const;
		/**
		 * This function checks if a file named '_filename' exists in _tmp_dir.
		 *
		 * @param time_point			The timepoint we are actually interested in.
		 *
		 * @return								A bool indicating whether the file exists or not.
		 */
		bool FileExists(unsigned int time_point) const;
		/**
		 * The Function swaps the Pointers a and b.
		 *
		 * @param a, b			References to two BlockVector<double>-Pointers which will be swapped.
		 */
		void SwapPtr(VECTOR* & a,VECTOR* & b ) const;

		/**
		 * Writes the vectors corresponding to the current interval
		 * into _local_vectors, and adjusts _global_to_local;
		 */
		void ComputeLocalVectors(const TimeIterator& interval) const;

		/**
		 * Helper function, resizes _local_vectors to the size given by size.
		 */
		void ResizeLocalVectors(unsigned int size) const;

		mutable std::vector<VECTOR*> _state;
		mutable std::vector<SpatialVectorInfos> _state_information;

		mutable VECTOR _local_state;
		mutable dealii::Vector<double> _copy_state;//FIXME why is this not of type VECTOR??
		mutable int _accessor;

		mutable bool _lock;

		//Needed in the store_on_disc case for read/write operations on the hard disc
		mutable std::string _filename;
		mutable std::fstream _filestream;

		//pointer to the dofs in the actual interval. Is only used if the interval is set!
		mutable std::vector<VECTOR*> _local_vectors;


		//Map: global time dof index - local time DoF index
		mutable std::map<unsigned int, unsigned int> _global_to_local;
		//the index of the interval, to which the vectors stored in local_vectors belong
		mutable int _accessor_index;

		DOpEtypes::VectorStorageType _behavior;
		std::string _tmp_dir;
		unsigned int _sfh_ticket;

		const SpaceTimeHandlerBase<VECTOR>* _STH;
		const unsigned int _unique_id;

		static unsigned int _id_counter;
		static unsigned int _num_active;
};

}
#endif
