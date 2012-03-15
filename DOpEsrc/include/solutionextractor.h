#ifndef SOLUTIONEXTRACTOR_H_
#define SOLUTIONEXTRACTOR_H_

#include "statevector.h"

/**
 * This class is used to extract the computed solution u out of the template Parameter SOLVERCLASS,
 *  which should have a memberfunction GetU() with the return type StateVector.
 *   This class is necessary due to some issues  connected with the resolution of overloaded functions.
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
	private:
		const SOLVERCLASS* _solverpointer;
};
}

#endif /* SOLUTIONEXTRACTOR_H_ */
