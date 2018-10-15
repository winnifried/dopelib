#ifndef DOPE_PARALLEL_VECTORS_H_
#define DOPE_PARALLEL_VECTORS_H_

#ifdef DOPELIB_WITH_TRILINOS
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#endif

// These work without MPI as well
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>

#endif
