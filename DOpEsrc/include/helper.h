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

#ifndef DOpEHelper_H_
#define DOpEHelper_H_

#include <deal.II/base/index_set.h>

#include <deal.II/lac/block_indices.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

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
  write(const VECTOR &v, std::ostream &stream)
  {
    v.block_write(stream);
  }

  /**
   * Reads a given vector from a stream.
   * Wraps around Vector<double>::block_read, since this function does not exist for trilinos-based vectors.
   * Specializations exist for other vector types.
   */
  template <typename VECTOR>
  void
  read(VECTOR &v, std::istream &stream)
  {
    v.block_read(stream);
  }

#ifdef DOPELIB_WITH_TRILINOS
  template <>
  inline void
  write<dealii::TrilinosWrappers::MPI::Vector>(const dealii::TrilinosWrappers::MPI::Vector &v, std::ostream &stream)
  {
    (void)v;
    (void)stream;
    throw ExcNotImplemented();
  }

  template <>
  inline void
  write<dealii::TrilinosWrappers::MPI::BlockVector>(const dealii::TrilinosWrappers::MPI::BlockVector &v,
                                                    std::ostream                                     &stream)
  {
    (void)v;
    (void)stream;
    throw ExcNotImplemented();
  }

  template <>
  inline void
  read<dealii::TrilinosWrappers::MPI::Vector>(dealii::TrilinosWrappers::MPI::Vector &v, std::istream &stream)
  {
    (void)v;
    (void)stream;
    throw ExcNotImplemented();
  }

  template <>
  inline void
  read<dealii::TrilinosWrappers::MPI::BlockVector>(dealii::TrilinosWrappers::MPI::BlockVector &v, std::istream &stream)
  {
    (void)v;
    (void)stream;
    throw ExcNotImplemented();
  }
#endif

  /**
   * Splits an index set source into different blocks according to block_counts.
   * Application: split locally_owned for block vectors
   */
  inline std::vector<dealii::IndexSet>
  split_blockwise(const dealii::IndexSet &source, const std::vector<unsigned int> &block_counts)
  {
    std::vector<dealii::IndexSet> result;
    unsigned int                  start = 0;
    for (unsigned int i = 0; i < block_counts.size(); i++)
      {
        result.push_back(source.get_view(start, start + block_counts[i]));
        start += block_counts[i];
      }
    return result;
  }

  // Distributed: vmult, solve, +, -, constraints, assemble into
  // Ghosted: linearization point, output, anything that evaluates

  // TODO document
  template <typename VECTOR>
  void
  make_distributed(VECTOR &source)
  {
    (void)source;
  }

#ifdef DOPELIB_WITH_TRILINOS
  inline void
  make_distributed(TrilinosWrappers::MPI::Vector &source)
  {
    if (source.has_ghost_elements())
      {
        TrilinosWrappers::MPI::Vector res(source);
        source.reinit(source.locally_owned_elements(), source.get_mpi_communicator(), true);
        source = res;
      }
  }

  inline void
  make_distributed(TrilinosWrappers::MPI::BlockVector &source)
  {
    if (source.has_ghost_elements())
      {
        std::vector<unsigned int> counts;
        for (unsigned int b = 0; b < source.n_blocks(); b++)
          counts.push_back(source.block(b).size());

        const auto block_owned = split_blockwise(source.locally_owned_elements(), counts);

        TrilinosWrappers::MPI::BlockVector res(source);
        source.reinit(block_owned, source.block(0).get_mpi_communicator(), true);
        source = res;
      }
  }
#endif

} // namespace DOpEHelper

#endif /* DOpEHelper_H_ */
