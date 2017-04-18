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

#ifndef TRANSPOSED_DIRICHLET_INTERFAC_H_
#define TRANSPOSED_DIRICHLET_INTERFAC_H_

namespace DOpE
{
  /**
   * Interface for TransposedDirichletData to compute reduced Hessian and Gradient from the Adjoint.
   *
   * For details see the description of TransposedGradientDirichletData and
   * TransposedHessianDirichletData
   *
   */
  template<int dealdim>
  class TransposedDirichletDataInterface
  {
  public:
    virtual ~TransposedDirichletDataInterface() {}

    virtual void value (const dealii::Point<dealdim>   &p,
                        const unsigned int  component,
                        const unsigned int  dof_number,
                        dealii::Vector<double> &local_vector) const=0;
  };
}
#endif
