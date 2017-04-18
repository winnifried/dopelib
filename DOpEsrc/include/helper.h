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

#ifndef DOpEHelper_H_
#define DOpEHelper_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_indices.h>

namespace DOpEHelper
{
  /**
   * Helper class for the VECTOR-templates. Resizes the given BlockVector vector.
   */
  void ReSizeVector(unsigned int ndofs,
                    const std::vector<unsigned int> &dofs_per_block,
                    dealii::BlockVector<double> &vector);

  /**
   * Same as above with different input parameters.
   */
  void ReSizeVector(const dealii::BlockIndices &, dealii::BlockVector<double> &vector);

  /*************************************************************************************/
  /**
   * Helper class for the VECTOR-templates. Resizes the given Vector vector.
   */
  void ReSizeVector(unsigned int ndofs,
                    const std::vector<unsigned int> & /*dofs_per_block*/,
                    dealii::Vector<double> &vector);
  /**
   * Same as above with different input parameters.
   */
  void ReSizeVector(const dealii::BlockIndices &, dealii::Vector<double> &vector);


}//end of namespace
#endif /* DOpEHelper_H_ */
