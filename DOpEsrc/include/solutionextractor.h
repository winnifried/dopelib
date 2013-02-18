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

#ifndef SOLUTIONEXTRACTOR_H_
#define SOLUTIONEXTRACTOR_H_

#include "statevector.h"

/**
 * This class is used to extract the computed solution u out of the template
 * Parameter SOLVERCLASS, which should have a memberfunction GetU() as well as
 * GetZforEE() with the return type StateVector. This class is necessary due
 * to some issues  connected with the resolution of overloaded functions.
 */


namespace DOpE
{
template<class SOLVERCLASS, class VECTOR>
class SolutionExtractor
{
	public:
		SolutionExtractor(const SOLVERCLASS &solver) :
			_solverpointer(&solver) {};
		const StateVector<VECTOR> & GetU() const
		{
			return _solverpointer->GetU();
		}
    const StateVector<VECTOR> & GetZForEE() const
    {
      return _solverpointer->GetZForEE();
    }
	private:
		const SOLVERCLASS* _solverpointer;
};
}

#endif /* SOLUTIONEXTRACTOR_H_ */
