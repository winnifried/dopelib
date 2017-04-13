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

#ifndef LOCAL_DIRICHLET_INTERFAC_H_
#define LOCAL_DIRICHLET_INTERFAC_H_

#include <interfaces/dirichletdatainterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dealdim>
class LocalDirichletData : public DirichletDataInterface<VECTOR, dealdim>
{
public:

  double
  Data(
    const std::map<std::string, const dealii::Vector<double>*> *param_values,
    const std::map<std::string, const VECTOR *> */*domain_values*/,
    unsigned int color, const dealii::Point<dealdim> & /*point*/,
    unsigned int component) const
  {
    qvalues_.reinit(5);
    GetParams(*param_values, "control", qvalues_);

    if (component == 0)
      {
        if (color == 0)
          return qvalues_(0);
        else if (color == 1)
          return qvalues_(1);
        else if (color == 2)
          return qvalues_(2);
        else if (color == 3)
          return qvalues_(3);
      }
    return qvalues_(4) * qvalues_(4) * qvalues_(4);
  }

  double
  Data_Q(
    const std::map<std::string, const dealii::Vector<double>*> *param_values,
    const std::map<std::string, const VECTOR *> */*domain_values*/,
    unsigned int color, const dealii::Point<dealdim> & /*point*/,
    unsigned int component) const
  {
    qvalues_.reinit(5);
    GetParams(*param_values, "control", qvalues_);
    dqvalues_.reinit(5);
    GetParams(*param_values, "dq", dqvalues_);

    if (component == 0)
      {
        if (color == 0)
          return dqvalues_(0);
        else if (color == 1)
          return dqvalues_(1);
        else if (color == 2)
          return dqvalues_(2);
        else if (color == 3)
          return dqvalues_(3);
      }
    return 3 * qvalues_(4) * qvalues_(4) * dqvalues_(4);
  }

  void
  Data_QT(
    const std::map<std::string, const dealii::Vector<double>*> *param_values,
    const std::map<std::string, const VECTOR *> *domain_values,
    unsigned int color, const dealii::Point<dealdim> & /*point*/,
    unsigned int component, unsigned int dof_number,
    dealii::Vector<double> &local_vector) const
  {
    qvalues_.reinit(5);
    GetParams(*param_values, "control", qvalues_);
    GetValues(*domain_values, "adjoint_residual", resvalues_, dof_number);
    if (component == 0)
      {
        if (color == 0)
          local_vector(0) += resvalues_;
        if (color == 1)
          local_vector(1) += resvalues_;
        else if (color == 2)
          local_vector(2) += resvalues_;
        else if (color == 3)
          local_vector(3) += resvalues_;
      }
    if (component == 1)
      local_vector(4) += 3 * qvalues_(4) * qvalues_(4) * resvalues_;
  }

  void
  Data_QQT(
    const std::map<std::string, const dealii::Vector<double>*> *param_values,
    const std::map<std::string, const VECTOR *> *domain_values,
    unsigned int /*color*/, const dealii::Point<dealdim> & /*point*/,
    unsigned int component, unsigned int dof_number,
    dealii::Vector<double> &local_vector) const
  {
    qvalues_.reinit(5);
    GetParams(*param_values, "control", qvalues_);
    dqvalues_.reinit(5);
    GetParams(*param_values, "dq", dqvalues_);
    GetValues(*domain_values, "hessian_residual", resvalues_, dof_number);
    if (component == 1)
      local_vector(4) += 6 * qvalues_(4) * dqvalues_(4) * resvalues_;
  }

  unsigned int
  n_components() const
  {
    return 2;
  }
  bool
  NeedsControl() const
  {
    return true;
  }
  /*****************************************************************************/
protected:
  inline void
  GetValues(const map<string, const VECTOR *> &domain_values, string name,
            double &values, unsigned int dof_number) const
  {
    typename map<string, const BlockVector<double>*>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    values = (*(it->second))(dof_number);
  }
  inline void
  GetParams(const map<string, const Vector<double>*> &param_values,
            string name, Vector<double> &values) const
  {
    typename map<string, const Vector<double>*>::const_iterator it =
      param_values.find(name);
    if (it == param_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    values = *(it->second);
  }
private:
  mutable Vector<double> qvalues_;
  mutable Vector<double> dqvalues_;
  mutable double resvalues_;
};
#endif
