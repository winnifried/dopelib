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
#ifndef INDEXSETTER_H_
#define INDEXSETTER_H_

#include <interfaces/active_fe_index_setter_interface.h>

using namespace DOpE;

template<int dealdim>
class ActiveFEIndexSetter : public ActiveFEIndexSetterInterface<dealdim>
{
public:
  ActiveFEIndexSetter(ParameterReader &/*param_reader*/)
  {
  }

  static void
  declare_params(ParameterReader &/*param_reader*/)
  {
  }

  /*
   * Gets an iterator to a element and sets an active FE index
   * on this element for the state variable. This function is
   * used after the first grid generation.
   *
   */
  virtual void
  SetActiveFEIndexState(
    typename dealii::hp::DoFHandler<dealdim>::active_cell_iterator &element) const
  {

    if (element->center()[0] < 0)
      element->set_active_fe_index(0);
    else
      element->set_active_fe_index(1);
  }

};

#endif /* INDEXSETT
ER_H_ */
