#ifndef DOPE_PARALLEL_VECTORS_H_
#define DOPE_PARALLEL_VECTORS_H_

#ifdef DOPELIB_WITH_TRILINOS
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#endif

// These work without MPI as well
#if DEAL_II_VERSION_GTE(9,1,1)
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#else
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
#endif

#endif
