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

#ifndef LOCALFunctional_
#define LOCALFunctional_

//#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional()
  {
  }

  bool
  NeedTime() const
  {
    return true;
  }
  double AlgebraicValue(const std::map<std::string, const dealii::Vector<double>*> &param_values,
                        const std::map<std::string, const VECTOR *> &/*domain_values*/)
  {
    assert(this->GetProblemType() == "cost_functional");
    //Search the vector with the precomputed functional values
    std::map<std::string, const dealii::Vector<double>*>::const_iterator vals = param_values.find("cost_functional_pre");
    //Return the postprocessed value
    return 0.5*(vals->second->operator[](0)*vals->second->operator[](0));
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    if ( this->GetProblemType() == "cost_functional_pre" ||
         this->GetProblemType() == "cost_functional_pre_tangent" )
      {
        //Precalculation
        unsigned int n_q_points = edc.GetNQPoints();
        double ret = 0.;

        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          edc.GetFEValuesState();
        uvalues_.resize(n_q_points);

        if (this->GetProblemType() == "cost_functional_pre" )
          {
            edc.GetValuesState("state", uvalues_);
          }
        else if (this->GetProblemType() == "cost_functional_pre_tangent" )
          {
            edc.GetValuesState("tangent", uvalues_);
          }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            ret += uvalues_[q_point]
                   * state_fe_values.JxW(q_point);
          }
        return ret;
      }
    else
      {
        assert( this->GetProblemType() == "cost_functional" );
        if (fabs(this->GetTime()) < 1.e-13)
          {
            unsigned int n_q_points = edc.GetNQPoints();
            double ret = 0.;
            //Factor to reverse the integration of this
            //integral over time. It is 0.5 * k
            double time_scale = 2./this->GetTimeStepSize();
            const DOpEWrapper::FEValues<dealdim> &state_fe_values =
              edc.GetFEValuesControl();
            //initialvalue
            fvalues_.resize(n_q_points);
            qvalues_.resize(n_q_points);
            edc.GetValuesControl("control", qvalues_);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                fvalues_[q_point] = sin(
                                      state_fe_values.quadrature_point(q_point)(0))
                                    * sin(state_fe_values.quadrature_point(q_point)(1));

                ret += 0.5 * time_scale
                       * (qvalues_[q_point] - fvalues_[q_point])
                       * (qvalues_[q_point] - fvalues_[q_point])
                       * state_fe_values.JxW(q_point);
              }
            return ret;
          }
        return 0.;
      }
    throw DOpEException("This should not be evaluated here!",
                        "LocalFunctional::Value");
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    dealii::Vector<double> pre_values(1);
    //Search the vector with the precomputed functional values
    edc.GetParamValues("cost_functional_pre", pre_values);
    // The derivative of the outer function
    double g_prime = pre_values[0];
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * g_prime
                               * state_fe_values.shape_value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    //Factor to reverse the integration of this
    //integral over time. Note that for the derivative the
    // factor is different  than for the ElementValue
    //This is due to the fact, that the evaluation of the
    //derivative is done only at the initial time without the
    //factor 0.5
    double time_scale = 1./this->GetTimeStepSize();

    if (fabs(this->GetTime()) < 1.e-13)
      {
        //endtimevalue
        fvalues_.resize(n_q_points);
        qvalues_.resize(n_q_points);

        edc.GetValuesControl("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            fvalues_[q_point] = sin(
                                  state_fe_values.quadrature_point(q_point)(0))
                                * sin(state_fe_values.quadrature_point(q_point)(1));
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale * time_scale
                                   * (qvalues_[q_point] - fvalues_[q_point])
                                   * state_fe_values.shape_value(i, q_point)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    dealii::Vector<double> pre_values_tangent(1);
    //Search the vector with the precomputed functional values
    //Values of \tilde{f} not needed since g'' is constant
    edc.GetParamValues("cost_functional_pre_tangent", pre_values_tangent); //\tilde{f}_u
    // The derivative of the outer function
    double g_primeprime = 1.;
    double f_u = pre_values_tangent[0];

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * g_primeprime * f_u
                               * state_fe_values.shape_value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    double time_scale = 1./this->GetTimeStepSize();

    if (fabs(this->GetTime()) < 1.e-13)
      {
        //endtimevalue
        dqvalues_.resize(n_q_points);

        edc.GetValuesControl("dq", dqvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                local_vector(i) += scale * time_scale * dqvalues_[q_point]
                                   * state_fe_values.shape_value(i, q_point)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const
  {
    //More complicated selection to avoid implementing
    //unneeded derivatives
    if ( this->GetProblemType() == "cost_functional_pre" //Only calculates drag
         ||this->GetProblemType() == "cost_functional_pre_tangent" //Only calculates drag of du
         || this->GetProblemType() == "adjoint" //J'_u is calculated as a boundary integral!
         || this->GetProblemType() == "gradient" //J'_q is calculated as a boundary integral!
         || this->GetProblemType() == "adjoint_hessian" //J'_{uu} is calculated as a boundary integral!
         || this->GetProblemType() == "hessian" //J'_{qq} is calculated as a boundary integral!
       )
      {
        return "domain timedistributed";
      }
    else
      {
        if ( this->GetProblemType() == "cost_functional" )
          {
            return "domain algebraic timedistributed";
          }
        else
          {
            std::cout<<"Unknown type ,,"<<this->GetProblemType()<<"'' in LocalFunctional::GetType"<<std::endl;
            abort();
          }
      }
  }

  std::string
  GetName() const
  {
    return "Cost-functional";
  }

  unsigned int NeedPrecomputations() const
  {
    return 1;
  }

private:
  vector<double> qvalues_;
  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<double> duvalues_;
  vector<double> dqvalues_;

};
#endif
