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
#include <deal.II/base/index_set.h>

#include <include/parallel_vectors.h>

using namespace dealii;

namespace DOpEHelper
{

  /**
   * Writes a given vector to a stream.
   * Wraps around Vector<double>::block_write, since this function does not exist for trilinos-based vectors.
   * Specializations exist for other vector types.
   */
  template <typename VECTOR>
  void
  write (const VECTOR &v,
         std::ostream &stream)
  {
    v.block_write (stream);
  }

  /**
   * Reads a given vector from a stream.
   * Wraps around Vector<double>::block_read, since this function does not exist for trilinos-based vectors.
   * Specializations exist for other vector types.
   */
  template <typename VECTOR>
  void
  read (VECTOR &v,
        std::istream &stream)
  {
    v.block_read (stream);
  }

#ifdef DOPELIB_WITH_TRILINOS
  template <>
  inline void
  write<dealii::TrilinosWrappers::MPI::Vector> (const dealii::TrilinosWrappers::MPI::Vector &v,
                                                std::ostream &stream)
  {
    (void) v;
    (void) stream;
    throw ExcNotImplemented ();
  }

  template <>
  inline void
  write<dealii::TrilinosWrappers::MPI::BlockVector> (const dealii::TrilinosWrappers::MPI::BlockVector &v,
                                                     std::ostream &stream)
  {
    (void) v;
    (void) stream;
    throw ExcNotImplemented ();
  }

  template <>
  inline void
  read<dealii::TrilinosWrappers::MPI::Vector> (dealii::TrilinosWrappers::MPI::Vector &v,
                                               std::istream &stream)
  {
    (void) v;
    (void) stream;
    throw ExcNotImplemented ();
  }

  template <>
  inline void
  read<dealii::TrilinosWrappers::MPI::BlockVector> (dealii::TrilinosWrappers::MPI::BlockVector &v,
                                                    std::istream &stream)
  {
    (void) v;
    (void) stream;
    throw ExcNotImplemented ();
  }
#endif

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

#ifdef DOPELIB_WITH_TRILINOS

// TODO these are just for temporary compatibility, to be removed !!!
  inline void ReSizeVector(const dealii::BlockIndices &, dealii::TrilinosWrappers::MPI::BlockVector &)
  {
  }

  inline void ReSizeVector(const dealii::BlockIndices &, dealii::TrilinosWrappers::MPI::Vector &)
  {
  }

  inline void ReSizeVector(unsigned int ,
                           const std::vector<unsigned int> &,
                           dealii::TrilinosWrappers::MPI::Vector &)
  {
  }
  inline void ReSizeVector(unsigned int ,
                           const std::vector<unsigned int> &,
                           dealii::TrilinosWrappers::MPI::BlockVector &)
  {
  }

#endif
}//end of namespace

#endif /* DOpEHelper_H_ */
