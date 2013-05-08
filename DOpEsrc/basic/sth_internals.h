/**
*
* Copyright (C) 2012 by the DOpElib authors
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

/*
 * sth_internals.h
 *
 *  Created on: 23.01.2012
 *      Author: cgoll
 */

#ifndef _STH_INTERNALS_H_
#define _STH_INTERNALS_H_

#include "mapping_wrapper.h"

#include <dofs/dof_tools.h>
#include <fe/mapping_q1.h>
#include <hp/mapping_collection.h>

#include "dofhandler_wrapper.h"

using namespace dealii;

namespace DOpE
{
  namespace STHInternals
  {
    template<typename VECTOR, int dealdim>
      void
      MapDoFsToSupportPoints(
          const DOpEWrapper::Mapping<dealdim, dealii::DoFHandler >& mapping,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler >& dh,
          VECTOR& support_points)
      {

//        MappingQ1 < dealdim > mapping;

        DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
      }

    template<typename VECTOR, int dealdim>
      void
      MapDoFsToSupportPoints(
          const DOpEWrapper::Mapping<dealdim, dealii::hp::DoFHandler >& mapping,
          const DOpEWrapper::DoFHandler<dealdim, dealii::hp::DoFHandler >& dh,
          VECTOR& support_points)
      {

#if DEAL_II_MAJOR_VERSION >= 7
#if DEAL_II_MINOR_VERSION >= 2
//        MappingQ1<dealdim> mapping;
//        hp::MappingCollection<dealdim> map_col(mapping);

        DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
//        DoFTools::map_dofs_to_support_points(map_col, dh, support_points);
#else
        throw DOpEException(
            "Your deal.ii version is too old. We need DoFTools::map_dofs_to_support_points for hp::DoFhandler"
            " (Implemented since 7.2, revision 24975)!",
            "MapDoFsToSupportPoints");
#endif
#else
        throw DOpEException(
            "Your deal.ii version is too old. We need DoFTools::map_dofs_to_support_points for hp::DoFhandler"
                " (Implemented since 7.2, revision 24975)!",
            "MapDoFsToSupportPoints");
#endif
      }
  }
}

#endif /* STH_INTERNALS_H_ */
