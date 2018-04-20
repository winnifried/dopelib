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

// TODO if Trilinos, ... + documentation
// !!! Daniel !!!
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>

#include <deal.II/base/index_set.h>

using namespace dealii;

// !!! Daniel !!!
template <typename VectorType>
struct IsParallelVector
{
private:
    struct yes_type
    {
        char c[1];
    };
    struct no_type
    {
        char c[2];
    };

    /**
     * Overload returning true if the class is MPI vector.
     */
    // TODO move to dealii library
    // TODO more parallel vectors
    static yes_type check_for_parallel_vector (const TrilinosWrappers::MPI::Vector *);
    static yes_type check_for_parallel_vector (const TrilinosWrappers::MPI::BlockVector *);
    template<typename T>
    static yes_type check_for_parallel_vector (const LinearAlgebra::distributed::Vector<T> *);
    template<typename T>
    static yes_type check_for_parallel_vector (const LinearAlgebra::distributed::BlockVector<T> *);


    /**
     * Catch all for all other potential vector types that are not MPI vectors
     */
    static no_type check_for_parallel_vector (...);

public:
    /**
     * A statically computable value that indicates whether the template
     * argument to this class is a block vector (in fact whether the type is
     * derived from BlockVectorBase<T>).
     */
    static const bool value = (sizeof(check_for_parallel_vector
                                      ((VectorType *)0))
                               ==
                               sizeof(yes_type));
};


// instantiation of the static member
template <typename VectorType>
const bool IsParallelVector<VectorType>::value;

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

// !!! Daniel !!!
/**
 * Helper class for the VECTOR-templates. Resizes the given BlockVector vector.
 */
void ReSizeVector(unsigned int ndofs,
                  const std::vector<unsigned int>& dofs_per_block,
                  dealii::TrilinosWrappers::MPI::BlockVector & vector);

/**
 * Same as above with different input parameters.
 */
void ReSizeVector(const dealii::BlockIndices&, dealii::TrilinosWrappers::MPI::BlockVector & vector);

/*************************************************************************************/
/**
 * Helper class for the VECTOR-templates. Resizes the given Vector vector.
 */
void ReSizeVector(unsigned int ndofs,
                  const std::vector<unsigned int>& /*dofs_per_block*/,
                  dealii::TrilinosWrappers::MPI::Vector & vector);
/**
 * Same as above with different input parameters.
 */
void ReSizeVector(const dealii::BlockIndices&, dealii::TrilinosWrappers::MPI::Vector & vector);

/**
 * Splits an index set source into different blocks, block_counts[i] = n_dofs within block i
 * Application: split locally_owned for block vectors
 */
std::vector<dealii::IndexSet> split_blockwise(const dealii::IndexSet & source, const std::vector<unsigned int> & block_counts);


// TODO document, extend for other types, move to cc
// TODO template + IsBlock, IsMPI, ...
inline TrilinosWrappers::MPI::Vector make_distributed(const TrilinosWrappers::MPI::Vector & source, const bool copy_values = true)
{
    TrilinosWrappers::MPI::Vector res;
    res.reinit(source.locally_owned_elements(), MPI_COMM_WORLD, true);
    if (copy_values) res = source;
    return res;
}

inline TrilinosWrappers::MPI::BlockVector make_distributed(const TrilinosWrappers::MPI::BlockVector & source, const bool copy_values = true)
{
    std::vector<unsigned int> counts;
    for (unsigned int b = 0; b < source.n_blocks(); b++)
        counts.push_back(source.block(b).size());

    const auto block_owned = DOpEHelper::split_blockwise(source.locally_owned_elements(), counts);

    TrilinosWrappers::MPI::BlockVector res;
    res.reinit(block_owned, MPI_COMM_WORLD, true);
    if (copy_values) res = source;
    return res;
}

inline Vector<double> make_distributed(const Vector<double> & source, const bool copy_values = true)
{
    return source;
}

inline BlockVector<double> make_distributed(const BlockVector<double> & source, const bool copy_values = true)
{
    return source;
}
}//end of namespace
#endif /* DOpEHelper_H_ */
