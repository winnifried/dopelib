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

#include <include/helper.h>
namespace DOpEHelper
{
  /**
   * Helper class for the VECTOR-templates. Resizes the given BlockVector vector.
   */
  void ReSizeVector(unsigned int ndofs,
                    const std::vector<unsigned int> &dofs_per_block,
                    dealii::BlockVector<double> &vector)
  {
    unsigned int nblocks = dofs_per_block.size();
    if (vector.size() != ndofs)
      {
        vector.reinit(nblocks);
        for (unsigned int i = 0; i < nblocks; i++)
          {
            vector.block(i).reinit(dofs_per_block[i]);
          }
        vector.collect_sizes();
      }
  }

  /*************************************************************************************/
  void ReSizeVector(const dealii::BlockIndices &bi, dealii::BlockVector<double> &vector)
  {
    unsigned int ndofs = bi.total_size();
    std::vector<unsigned int>  dofs_per_block(bi.size(), 0);
    for (unsigned int i = 0; i < bi.size(); i++)
      {
        dofs_per_block.at(i) = bi.block_size(i);
      }
    DOpEHelper::ReSizeVector(ndofs, dofs_per_block, vector);
  }

  /*************************************************************************************/
  /**
   * Helper class for the VECTOR-templates. Resizes the given Vector vector.
   */
  void ReSizeVector(unsigned int ndofs,
                    const std::vector<unsigned int> & /*dofs_per_block*/,
                    dealii::Vector<double> &vector)
  {
    if (vector.size() != ndofs)
      {
        vector.reinit(ndofs);
      }
  }
  /*************************************************************************************/
  void ReSizeVector(const dealii::BlockIndices &bi, dealii::Vector<double> &vector)
  {
    unsigned int ndofs = bi.total_size();
    std::vector<unsigned int>  dofs_per_block(1, 0);
    DOpEHelper::ReSizeVector(ndofs, dofs_per_block, vector);
  }

}//end of namespace
