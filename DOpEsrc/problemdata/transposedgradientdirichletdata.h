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

#ifndef TRANSPOSED_GRADIENT_DIRICHLET_DATA_H_
#define TRANSPOSED_GRADIENT_DIRICHLET_DATA_H_

#include <interfaces/transposeddirichletdatainterface.h>

namespace DOpE
{

  /**
   * This class is used to compute the reduced gradient in the case of dirichlet control
   *
   * @tparam  DD              The Dirichlet Data Object under consideration
   * @tparam  VECTOR          The Vector type
   * @tparam  dealdim         The dimension of the domain
   *
   */
  template<typename DD, typename VECTOR, int dealdim>
  class TransposedGradientDirichletData : public TransposedDirichletDataInterface<dealdim>
  {
  public:
    TransposedGradientDirichletData(const DD &data) : TransposedDirichletDataInterface<dealdim>(), dirichlet_data_(data)
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
     * Accesses the values of the transposed of the first derivative of the
     * control-to-dirichlet-values map. I.e. if DD'(q): \R^n \rightarrow L^2(\partial\Omega;R^m)
     * then here we calculate DD'(q)^*: L^2(\partial\Omega;R^m) \rightarrow \R^n
     *
     * @param p             The point (on the boundary of the domain) where the Dirichlet
     *                      values are evaluated
     * @param component     The component of the Dirichlet data (in \R^m)
     * @param dof_number    The number of the dof for the domain data in V_h(\Omega;R^m)
     *                      from which the influence is calculated.
     * @param local_vector  The vector for the result in \R^n
     */
    void value (const dealii::Point<dealdim>   &p,
                const unsigned int  component,
                const unsigned int  dof_number,
                dealii::Vector<double> &local_vector) const
    {
      dirichlet_data_.Data_QT(
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
