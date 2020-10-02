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

#include <interfaces/pdeinterface.h>
#include <include/dopeexception.h>

#include <iostream>

//FIXME: For developement of MG-support, please uncomment.
//#include "../../Examples/Experimental/Example12/mgelementdatacontainer.h"

using namespace dealii;

namespace DOpE
{

 
  /********************************************/

} //Endof namespace
/********************************************/
/********************************************/

template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;

#ifdef DOPELIB_WITH_TRILINOS
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
                                  DOpE::FaceDataContainer, dealii::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::BlockVector, deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::ElementDataContainer,
                                  DOpE::FaceDataContainer, dealii::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::Vector, deal_II_dimension>;
#endif

/********************************************/

template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::BlockVector<double>, deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::Vector<double>, deal_II_dimension>;

#ifdef DOPELIB_WITH_TRILINOS
template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
                                  DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::BlockVector, deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
                                  DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::Vector, deal_II_dimension>;
#endif

/********************************************/
#if DEAL_II_VERSION_GTE(9,3,0)
#else
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler,
         dealii::BlockVector<double>, deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;

#ifdef DOPELIB_WITH_TRILINOS
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
                                  DOpE::FaceDataContainer, dealii::hp::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::BlockVector, deal_II_dimension>;

template class DOpE::PDEInterface<DOpE::ElementDataContainer,
                                  DOpE::FaceDataContainer, dealii::hp::DoFHandler,
                                  dealii::TrilinosWrappers::MPI::Vector, deal_II_dimension>;
#endif
#endif//Endof deal older than 9.3.0
/********************************************/

template class DOpE::PDEInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;

