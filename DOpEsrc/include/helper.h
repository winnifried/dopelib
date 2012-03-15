#ifndef _DOpEHelper_H_
#define _DOpEHelper_H_

#include <lac/block_vector.h>
#include <lac/vector.h>
#include <lac/block_indices.h>

namespace DOpEHelper
{
/**
 * Helper class for the VECTOR-templates. Resizes the given BlockVector vector.
 */
void ReSizeVector(unsigned int ndofs,
                  const std::vector<unsigned int>& dofs_per_block,
                  dealii::BlockVector<double>& vector);

/**
 * Same as above with different input parameters.
 */
void ReSizeVector(const dealii::BlockIndices&, dealii::BlockVector<double>& vector);

/*************************************************************************************/
/**
 * Helper class for the VECTOR-templates. Resizes the given Vector vector.
 */
void ReSizeVector(unsigned int ndofs,
                  const std::vector<unsigned int>& /*dofs_per_block*/,
                  dealii::Vector<double>& vector);
/**
 * Same as above with different input parameters.
 */
void ReSizeVector(const dealii::BlockIndices&, dealii::Vector<double>& vector);

}//end of namespace
#endif /* _DOpEHelper_H_ */
