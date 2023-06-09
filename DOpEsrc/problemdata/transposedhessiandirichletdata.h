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

#ifndef TRANSPOSED_HESSIAN_DIRICHLET_DATA_H_
#define TRANSPOSED_HESSIAN_DIRICHLET_DATA_H_

#include <interfaces/transposeddirichletdatainterface.h>

namespace DOpE
{

  /**
   * This class is used to compute the reduced hessian in the case of dirichlet control
   *
   * @tparam  DD              The Dirichlet Data Object under consideration
   * @tparam  VECTOR          The Vector type
   * @tparam  dealdim         The dimension of the domain
   *
   */
  template<typename DD, typename VECTOR, int dealdim>
  class TransposedHessianDirichletData : public TransposedDirichletDataInterface<dealdim>
  {
  public:
    TransposedHessianDirichletData(const DD &data) : TransposedDirichletDataInterface<dealdim>(), dirichlet_data_(data)
    {
      param_values_ = NULL;
      domain_values_ = NULL;
      color_ = 0;
    }

    /**
     * Initializes the private data, should be called prior to any value call!
     */
    void ReInit(
      const std::map<std::string, const dealii::Vector<double>* > &param_values,
      const std::map<std::string, const VECTOR * > &domain_values,
      unsigned int color)
    {
      param_values_ = &param_values;
      domain_values_ = &domain_values;
      color_ = color;
    }

    /**
     * Accesses the values of the dirichlet data transposed of the second
     * derivative of the control-to-dirichlet-value map
     *
     * For details see TransposedGradientDirichletData
     */
    void value (const dealii::Point<dealdim>   &p,
                const unsigned int  component,
                const unsigned int  dof_number,
                dealii::Vector<double> &local_vector) const override
    {
      dirichlet_data_.Data_QT(
        param_values_,
        domain_values_,
        color_,
        p,
        component,
        dof_number,
        local_vector);
      dirichlet_data_.Data_QQT(
        param_values_,
        domain_values_,
        color_,
        p,
        component,
        dof_number,
        local_vector);
    }

    /**
     * This Function is used to transfer the current time to the dirichlet data if needed this should be stored.
     *
     * @param time      The current time
     */
    void SetTime(double time) const
    {
      dirichlet_data_.SetTime(time);
    }
  private:
    const DD &dirichlet_data_;
    const std::map<std::string, const dealii::Vector<double>* > *param_values_;
    const std::map<std::string, const VECTOR * > *domain_values_;
    unsigned int color_;
  };

}
#endif
