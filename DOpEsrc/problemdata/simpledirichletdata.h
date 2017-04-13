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

#ifndef SIMPLE_DIRICHLET_H_
#define SIMPLE_DIRICHLET_H_

#include <interfaces/dirichletdatainterface.h>
#include <wrapper/function_wrapper.h>

namespace DOpE
{

  /**
   * A Simple Interface Class, that sets DirichletData given by a DOpEWrapper::Function.
   * This means they don't depend on control or state values
   */
  template<typename VECTOR, int dealdim>
  class SimpleDirichletData : public DirichletDataInterface<VECTOR, dealdim>
  {
  public:
    SimpleDirichletData(const DOpEWrapper::Function<dealdim> &data) : DirichletDataInterface<VECTOR,dealdim>(), data_(data)
    {}

    double Data(const std::map<std::string, const dealii::Vector<double>* > */*param_values*/,
                const std::map<std::string, const VECTOR * > */*domain_values*/,
                unsigned int /*color*/,
                const dealii::Point<dealdim> &point,
                const unsigned int component) const
    {
      return data_.value(point,component);
    }

    double Data_Q(const std::map<std::string, const dealii::Vector<double>* > */*param_values*/,
                  const std::map<std::string, const VECTOR * > */*domain_values*/,
                  unsigned int /*color*/,
                  const dealii::Point<dealdim> & /*point*/,
                  const unsigned int /*component*/) const
    {
      return 0.;
    }

    void SetTime(double time) const
    {
      data_.SetTime(time);
    }

    unsigned int n_components() const
    {
      return data_.n_components;
    }

    double InitialTime() const
    {
      return data_.InitialTime();
    }

  private:
    const DOpEWrapper::Function<dealdim> &data_;
  };

}


#endif
