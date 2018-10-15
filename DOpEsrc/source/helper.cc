/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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
//  void ReSizeVector(unsigned int ndofs,
//                    const std::vector<unsigned int> &dofs_per_block,
//                    dealii::BlockVector<double> &vector)
//  {
//    unsigned int nblocks = dofs_per_block.size();
//    if (vector.size() != ndofs)
//      {
//        vector.reinit(nblocks);
//        for (unsigned int i = 0; i < nblocks; i++)
//          {
//            vector.block(i).reinit(dofs_per_block[i]);
//          }
//        vector.collect_sizes();
//      }
//  }
//
//  /*************************************************************************************/
//  void ReSizeVector(const dealii::BlockIndices &bi, dealii::BlockVector<double> &vector)
//  {
//    unsigned int ndofs = bi.total_size();
//    std::vector<unsigned int>  dofs_per_block(bi.size(), 0);
//    for (unsigned int i = 0; i < bi.size(); i++)
//      {
//        dofs_per_block.at(i) = bi.block_size(i);
//      }
//    DOpEHelper::ReSizeVector(ndofs, dofs_per_block, vector);
//  }
//
//  /*************************************************************************************/
//  /**
//   * Helper class for the VECTOR-templates. Resizes the given Vector vector.
//   */
//  void ReSizeVector(unsigned int ndofs,
//                    const std::vector<unsigned int> & /*dofs_per_block*/,
//                    dealii::Vector<double> &vector)
//  {
//    if (vector.size() != ndofs)
//      {
//        vector.reinit(ndofs);
//      }
//  }
//  /*************************************************************************************/
//  void ReSizeVector(const dealii::BlockIndices &bi, dealii::Vector<double> &vector)
//  {
//    unsigned int ndofs = bi.total_size();
//    std::vector<unsigned int>  dofs_per_block(1, 0);
//    DOpEHelper::ReSizeVector(ndofs, dofs_per_block, vector);
//  }
//
//
//// !!! Daniel !!!
//
//  /**
//   * Helper class for the VECTOR-templates. Resizes the given BlockVector vector.
//   */
//  void ReSizeVector(unsigned int ndofs,
//                    const std::vector<unsigned int> &dofs_per_block,
//                    dealii::TrilinosWrappers::MPI::BlockVector &vector)
//  {
//    assert(false);
//    (void)ndofs;
//    (void)dofs_per_block;
//    (void)vector;
//  }
//
//  /**
//   * Same as above with different input parameters.
//   */
//  void ReSizeVector(const dealii::BlockIndices &, dealii::TrilinosWrappers::MPI::BlockVector &vector)
//  {
//    assert(false);
//    (void)vector;
//  }
//
//  /*************************************************************************************/
//  /**
//   * Helper class for the VECTOR-templates. Resizes the given Vector vector.
//   */
//  void ReSizeVector(unsigned int ndofs,
//                    const std::vector<unsigned int> &dofs_per_block,
//                    dealii::TrilinosWrappers::MPI::Vector &vector)
//  {
//    assert(false);
//    (void)ndofs;
//    (void)dofs_per_block;
//    (void)vector;
//  }
//  /**
//   * Same as above with different input parameters.
//   */
//  void ReSizeVector(const dealii::BlockIndices &, dealii::TrilinosWrappers::MPI::Vector &vector)
//  {
//    assert(false);
//    (void)vector;
//  }
  /**
   * Splits an index set source into different blocks, block_counts[i] = n_dofs within block i
   * Application: split locally_owned for block vectors
   */
  std::vector<dealii::IndexSet>
  split_blockwise (const dealii::IndexSet &source,
                   const std::vector<unsigned int> &block_counts)
  {
    std::vector<dealii::IndexSet> result;
    unsigned int start = 0;
    for (unsigned int i = 0; i < block_counts.size (); i++)
      {
        result.push_back (source.get_view (start, start + block_counts[i]));
        start += block_counts[i];
      }
    return result;
  }

}//end of namespace
