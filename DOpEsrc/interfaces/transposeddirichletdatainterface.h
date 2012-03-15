#ifndef _TRANSPOSED_DIRICHLET_INTERFAC_H_
#define _TRANSPOSED_DIRICHLET_INTERFAC_H_

#include "function_wrapper.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"

namespace DOpE
{
  /**
   * Interface for TransposedDirichletData to compute reduced Hessian and Gradient from the Adjoint.
   */
  template<int dopedim, int dealdim>
    class TransposedDirichletDataInterface 
  {
  public:
    virtual ~TransposedDirichletDataInterface() {}

    virtual void value (const dealii::Point<dealdim>   &p,
			const unsigned int  component,
			const unsigned int  dof_number, 
			dealii::Vector<double>& local_vector) const=0;
  }; 
}
#endif
