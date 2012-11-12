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

#ifndef _SIMPLE_DIRICHLET_H_
#define _SIMPLE_DIRICHLET_H_

#include "dirichletdatainterface.h"
#include "function_wrapper.h"

namespace DOpE
{

  /**
   * A Simple Interface Class, that sets DirichletData given by a DOpEWrapper::Function. This means they don't depend on control or state values
   */
  template<typename VECTOR, int dopedim, int dealdim=dopedim>
    class SimpleDirichletData : public DirichletDataInterface<VECTOR, dopedim,dealdim>
  {
  public:
  SimpleDirichletData(const DOpEWrapper::Function<dealdim>& data) : DirichletDataInterface<VECTOR, dopedim,dealdim>(), _data(data)
    {}

  double Data(
//              const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//	      const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
	      const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
	      const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
	      unsigned int color __attribute__((unused)),
	      const dealii::Point<dealdim>& point,
	      const unsigned int component) const
  {
    return _data.value(point,component);
  }

  double Data_Q(
//                const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
		const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
		unsigned int color __attribute__((unused)),
		const dealii::Point<dealdim>& point __attribute__((unused)),
		const unsigned int component __attribute__((unused))) const
  {
    return 0.;
  }

  void SetTime(double time) const
  {
    _data.SetTime(time);
  }

  unsigned int n_components() const
  {
    return _data.n_components;
  }

  double InitialTime() const { return _data.InitialTime();}

  private:
    const DOpEWrapper::Function<dealdim>& _data;
  };

}


#endif
