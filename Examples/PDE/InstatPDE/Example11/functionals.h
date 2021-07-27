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

#ifndef LOCALFunctionalS_
#define LOCALFunctionalS_

#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;


/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalTCV : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
	LocalFunctionalTCV()
	{
	}

	static void
	  declare_params(ParameterReader &/*param_reader*/)
	{
	}

	LocalFunctionalTCV(ParameterReader &/*param_reader*/)
	{

	}

	bool
	NeedTime() const
	{
		return true;
	}

	double
	ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
	{
		const auto &state_fe_values = edc.GetFEValuesState();
		unsigned int n_q_points = edc.GetNQPoints();
		double ret = 0.;

		vector<Vector<double> > uvalues_;
		vector<vector<Tensor<1, dealdim> > > ugrads_;

		uvalues_.resize(n_q_points, Vector<double>(4));
		ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

		edc.GetValuesState("state", uvalues_);
		edc.GetGradsState("state", ugrads_);

		if(fabs(edc.GetCenter()[0]) < 10 && fabs(edc.GetCenter()[1]) < 10)
		{
		  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
		  {
		    // displacement gradient
		    Tensor<1, 2> grad_phi;
		    grad_phi.clear();
		    grad_phi[0] = ugrads_[q_point][2][0];
		    grad_phi[1] = ugrads_[q_point][2][1];
		    
		    Tensor<1,2> u_val;
		    u_val.clear();
		    u_val[0] = uvalues_[q_point][0];
		    u_val[1] = uvalues_[q_point][1];
		    ret += u_val*grad_phi * state_fe_values.JxW(q_point);
		  }
		}
		return ret;
	}

	UpdateFlags
	GetUpdateFlags() const
	{
		return update_values | update_quadrature_points | update_gradients;
	}

	string
	GetType() const
	{
		return "domain timelocal";
	}

	bool HasFaces() const
	{
		return false;
	}

	string
	GetName() const
	{
		return "TCV";
	}

};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalBulk : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_;

public:
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    // param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    // param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));

  }

  LocalFunctionalBulk(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");

  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_bulk_energy = 0;

    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    if(fabs(edc.GetCenter()[0]) < 10 && fabs(edc.GetCenter()[1]) < 10)
    {
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	// displacement gradient
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];
	
        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);
        double pf = uvalues_[q_point](2);
        const double tr_e_2 = trace(E*E);
	const double psi_e = 0.5 * lame_coefficient_lambda_ * tr_E*tr_E + lame_coefficient_mu_ * tr_e_2;
	
        local_bulk_energy += (( 1 + constant_k_ ) * pf * pf + constant_k_) * 
	  psi_e * state_fe_values.JxW(q_point);
      }
    }
    return local_bulk_energy;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    return "BulkEnergy";
  }

};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalCrack : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double G_c_, alpha_eps_;

public: 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    // param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
  }

  LocalFunctionalCrack(ParameterReader &param_reader,double eps)
  {
    param_reader.SetSubsection("Local PDE parameters");
    G_c_ = param_reader.get_double("G_c");
    alpha_eps_ = eps;
    // alpha_eps_ = param_reader.get_double("alpha_eps");
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }
  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_crack_energy = 0;

    
    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    if(fabs(edc.GetCenter()[0]) < 10 && fabs(edc.GetCenter()[1]) < 10)
    {
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double pf = uvalues_[q_point](2);
	
	//phase field gradient
	Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];
	
        local_crack_energy += G_c_/2.0 * ((pf-1) * (pf-1)/alpha_eps_ + alpha_eps_ * 
					  (grad_pf[0] * grad_pf[0] + grad_pf[1] * grad_pf[1])) * state_fe_values.JxW(q_point);
      }
    }
    return local_crack_energy;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    return "CrackEnergy";
  }

};
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispXLeft : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(-1., 0.);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(0);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispXLeft";
  }
};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispXRight : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(1., 0.);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(0);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispXRight";
  }
};
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispXTop : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:


  PointFunctionalDispXTop(double meshsize)
  {
    d_ = meshsize;
  }
  void SetParams(double meshsize)
  {
    d_ = meshsize;
  }
  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(0., d_);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(0);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispXTop";
  }

private:
double d_;
};
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispXBottom : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:


  PointFunctionalDispXBottom(double meshsize)
  {
    d_ = meshsize;
  }
  void SetParams(double meshsize)
  {
    d_ = meshsize;
  }
  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(0., -1.*d_);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(0);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispXBottom";
  }

private:
double d_;
};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispYLeft : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(-1., 0.);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(1);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispYLeft";
  }
};

/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispYRight : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(1., 0.);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(1);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispYRight";
  }
};
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispYTop : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:


  PointFunctionalDispYTop(double meshsize)
  {
    d_ = meshsize;
  }
  void SetParams(double meshsize)
  {
    d_ = meshsize;
  }
  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(0., d_);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(1);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispYTop";
  }

private:
double d_;
};
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class PointFunctionalDispYBottom : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:


  PointFunctionalDispYBottom(double meshsize)
  {
    d_ = meshsize;
  }
  void SetParams(double meshsize)
  {
    d_ = meshsize;
  }
  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p(0., -1.*d_);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(4);

    VectorTools::point_value(state_dof_handler, *(it->second), p,
                             tmp_vector);
    
    return tmp_vector(1);
  }

  bool
  NeedTime() const
  {
    return true;
  }
  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "DispYBottom";
  }

private:
double d_;
};

/****************************************************************************************/
/** Functional for the error in u in its energy norm, i.e. the norm for elasticity weighted with g(phi) **/
/**********************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalEnergyNormU : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_, alpha_eps_, G_c_;
  std::string ref_string_;
    
public:
  static void
    declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    //param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalEnergyNormU(ParameterReader &param_reader,double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
    
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    //alpha_eps_ = param_reader.get_double("alpha_eps");

    alpha_eps_ = eps;
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
  
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }
  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_Uenergy_norm = 0;

    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    //state is the current solution
    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    //while ref is the solution on the coarser (adaptively) refined meshes
    vector<Vector<double> > refvalues_;
    vector<vector<Tensor<1, dealdim> > > refgrads_;

    refvalues_.resize(n_q_points, Vector<double>(4));
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState(ref_string_, refvalues_);
    edc.GetGradsState(ref_string_, refgrads_);

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
     for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

	// phase fiel variable to weight
	double refpf = refvalues_[q_point](2);

	// eigentlich ist das nicht ganz richtig, da wir in der time lagged version
	// den Wert von phi zum alten Zeitschritt nehmen muessten
	// wir haben aber nur die aktuellen Werte uebergeben
	double g_pf;				
	g_pf = (1-constant_k_)*refpf*refpf + constant_k_;

	// displacement gradient  
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];

	// displacement gradient on the coarser mesh   
        Tensor<2, 2> refgrad_u;
        refgrad_u.clear();
        refgrad_u[0][0] = refgrads_[q_point][0][0];
        refgrad_u[0][1] = refgrads_[q_point][0][1];
        refgrad_u[1][0] = refgrads_[q_point][1][0];
        refgrad_u[1][1] = refgrads_[q_point][1][1];

	  
        const Tensor<2,2> E = 0.5 * ((grad_u-refgrad_u) + transpose(grad_u-refgrad_u));
        const double tr_E = trace(E);

	Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;

    	double Energy;
	Energy = scalar_product(stress_term, E);
	
	local_Uenergy_norm += g_pf*Energy*state_fe_values.JxW(q_point);
	
	}
    
   return local_Uenergy_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"UEnergyNorm-"<<ref_string_;
    return out.str();
   }

};



/** Functional for the error in u in its elasticity energy norm **/
/**********************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalElasticityNorm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_, alpha_eps_, G_c_;
  std::string ref_string_;
    
public:
  static void
    declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    //param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalElasticityNorm(ParameterReader &param_reader,double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
   
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    //alpha_eps_ = param_reader.get_double("alpha_eps");
    alpha_eps_ = eps;
    
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
  
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_elasticity_norm = 0;

    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    //state is the current solution
    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    //while ref is the solution on the coarser (adaptively) refined meshes
    vector<Vector<double> > refvalues_;
    vector<vector<Tensor<1, dealdim> > > refgrads_;

    refvalues_.resize(n_q_points, Vector<double>(4));
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState(ref_string_, refvalues_);
    edc.GetGradsState(ref_string_, refgrads_);

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
     for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

	// displacement gradient  
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];

	// displacement gradient on the coarser mesh   
        Tensor<2, 2> refgrad_u;
        refgrad_u.clear();
        refgrad_u[0][0] = refgrads_[q_point][0][0];
        refgrad_u[0][1] = refgrads_[q_point][0][1];
        refgrad_u[1][0] = refgrads_[q_point][1][0];
        refgrad_u[1][1] = refgrads_[q_point][1][1];

	  
        const Tensor<2,2> E = 0.5 * ((grad_u-refgrad_u) + transpose(grad_u-refgrad_u));
        const double tr_E = trace(E);

	Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;

    	double Energy;
	Energy = scalar_product(stress_term, E);
	
	local_elasticity_norm += Energy*state_fe_values.JxW(q_point);
	
	}
    
   return local_elasticity_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"ElasticityNorm-"<<ref_string_;
    return out.str();
   }

};



/** Functional for the error in phi in the energy norm **/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalEnergyNorm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_, alpha_eps_, G_c_;
  std::string ref_string_;
  
public:
 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    //param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalEnergyNorm(ParameterReader &param_reader,double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
    
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    //alpha_eps_ = param_reader.get_double("alpha_eps");
    alpha_eps_ = eps;
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
    
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_energy_norm = 0;

    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    //state is the current solution on the finest grid 
    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    //while ref is the solution on the coarser (adaptively) refined meshes
    vector<Vector<double> > refvalues_;
    vector<vector<Tensor<1, dealdim> > > refgrads_;

    refvalues_.resize(n_q_points, Vector<double>(4));
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState(ref_string_, refvalues_);
    edc.GetGradsState(ref_string_, refgrads_);

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	//phase field variable for both solutions
	double pf = uvalues_[q_point](2);
	double refpf = refvalues_[q_point](2);
	
	//phase field gradient
	Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

	//phase field gradient to compare with
	Tensor<1,2> refgrad_pf;
        refgrad_pf.clear();
        refgrad_pf[0] = refgrads_[q_point][2][0];
        refgrad_pf[1] = refgrads_[q_point][2][1];
        
	// displacement gradient on the coarser mesh because this enters in the error measure  
        Tensor<2, 2> refgrad_u;
        refgrad_u.clear();
        refgrad_u[0][0] = refgrads_[q_point][0][0];
        refgrad_u[0][1] = refgrads_[q_point][0][1];
        refgrad_u[1][0] = refgrads_[q_point][1][0];
        refgrad_u[1][1] = refgrads_[q_point][1][1];

        const Tensor<2,2> E = 0.5 * (refgrad_u + transpose(refgrad_u));
        const double tr_E = trace(E);

	Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term, E);

	// it is the square of the enery norm
	local_energy_norm += (G_c_*alpha_eps_*((grad_pf[0]-refgrad_pf[0])*(grad_pf[0]-refgrad_pf[0]) + (grad_pf[1]-refgrad_pf[1])*(grad_pf[1]-refgrad_pf[1]))
			      + ((G_c_/alpha_eps_) + (1-constant_k_)*weightEnergy)*(pf-refpf)*(pf-refpf))* state_fe_values.JxW(q_point);

	
      }
    return local_energy_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
     stringstream out;
    out<<"EnergyNorm-"<<ref_string_;
    return out.str();
  }

};


/** Functional for the error in phi in the energy norm with u on the finest level**/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalEnergyVarNorm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_, alpha_eps_, G_c_;
  std::string ref_string_;
  
public:
 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    //param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalEnergyVarNorm(ParameterReader &param_reader,double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
    
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    //alpha_eps_ = param_reader.get_double("alpha_eps");
    alpha_eps_ = eps;
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
    
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_energyVar_norm = 0;

    vector<Vector<double> > uvalues_;
    vector<vector<Tensor<1, dealdim> > > ugrads_;

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    //state is the current solution on the finest grid 
    edc.GetValuesState("state", uvalues_);
    edc.GetGradsState("state", ugrads_);

    //while ref is the solution on the coarser (adaptively) refined meshes
    vector<Vector<double> > refvalues_;
    vector<vector<Tensor<1, dealdim> > > refgrads_;

    refvalues_.resize(n_q_points, Vector<double>(4));
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState(ref_string_, refvalues_);
    edc.GetGradsState(ref_string_, refgrads_);

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	//phase field variable for both solutions
	double pf = uvalues_[q_point](2);
	double refpf = refvalues_[q_point](2);
	
	//phase field gradient
	Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

	//phase field gradient to compare with
	Tensor<1,2> refgrad_pf;
        refgrad_pf.clear();
        refgrad_pf[0] = refgrads_[q_point][2][0];
        refgrad_pf[1] = refgrads_[q_point][2][1];
        
	// displacement gradient on the coarser mesh because this enters in the error measure  
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];

        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);

	Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term, E);

	// it is the square of the enery norm
	local_energyVar_norm += (G_c_*alpha_eps_*((grad_pf[0]-refgrad_pf[0])*(grad_pf[0]-refgrad_pf[0]) + (grad_pf[1]-refgrad_pf[1])*(grad_pf[1]-refgrad_pf[1]))
			      + ((G_c_/alpha_eps_) + (1-constant_k_)*weightEnergy)*(pf-refpf)*(pf-refpf))* state_fe_values.JxW(q_point);

	
      }
    return local_energyVar_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
     stringstream out;
    out<<"EnergyVarNorm-"<<ref_string_;
    return out.str();
  }

};



/** Functional for the error in phi in the L^2 part of the energy norm **/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalWeightedL2Norm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double constant_k_, lame_coefficient_mu_, lame_coefficient_lambda_, alpha_eps_, G_c_;
  std::string ref_string_;
  
public:
 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("Young_modulus", "1.0", Patterns::Double(0));
    param_reader.declare_entry("Poisson_ratio", "0.2", Patterns::Double(0));	
    //param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalWeightedL2Norm(ParameterReader &param_reader,double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    double E = param_reader.get_double("Young_modulus");
    double nu  = param_reader.get_double("Poisson_ratio");
    if(nu == 0.5)
      {
	std::cout<<"not yet implemented for \nu = 0.5"<<std::endl;
	abort();
      }
    lame_coefficient_mu_ = E/(2.*(1+nu));
    lame_coefficient_lambda_ = nu*E/((1+nu)*(1-2*nu));
    
    //lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    //lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    //alpha_eps_ = param_reader.get_double("alpha_eps");
    alpha_eps_ = eps;
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
    
  }
  void SetParams(double eps)
  {
    alpha_eps_ = eps;
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_weightedL2_norm = 0;

    vector<Vector<double> > uvalues_;
    
    uvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("state", uvalues_);
    
    vector<Vector<double> > refvalues_;
    vector<vector<Tensor<1, dealdim> > > refgrads_;

    refvalues_.resize(n_q_points, Vector<double>(4));
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetValuesState(ref_string_, refvalues_);
    edc.GetGradsState(ref_string_, refgrads_);

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	//phase field variable for both solutions
	double pf = uvalues_[q_point](2);
	double refpf = refvalues_[q_point](2);
	

	// displacement gradient which we need for reference because this enters in the error measure  
        Tensor<2, 2> refgrad_u;
        refgrad_u.clear();
        refgrad_u[0][0] = refgrads_[q_point][0][0];
        refgrad_u[0][1] = refgrads_[q_point][0][1];
        refgrad_u[1][0] = refgrads_[q_point][1][0];
        refgrad_u[1][1] = refgrads_[q_point][1][1];

        const Tensor<2,2> E = 0.5 * (refgrad_u + transpose(refgrad_u));
        const double tr_E = trace(E);

	Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term, E);
	
	local_weightedL2_norm += (((G_c_/alpha_eps_) + (1-constant_k_)*weightEnergy)*(pf-refpf)*(pf-refpf))* state_fe_values.JxW(q_point);
	
      }
    return local_weightedL2_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"WeightedL2Norm-"<<ref_string_;
    return out.str();
  }

};


/** Functional for the error in phi in the L^2 norm  **/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalL2Norm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
 
  std::string ref_string_;
  
public:
  
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    
  }

  LocalFunctionalL2Norm(ParameterReader &param_reader, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");

    ref_string_=ref_string;
    
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_L2_norm = 0;

    vector<Vector<double> > uvalues_;
    uvalues_.resize(n_q_points, Vector<double>(4));  
    edc.GetValuesState("state", uvalues_);
   
    vector<Vector<double> > refvalues_;
    refvalues_.resize(n_q_points, Vector<double>(4));

    edc.GetValuesState(ref_string_, refvalues_);
      
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	//phase field variable for both solutions
	double pf = uvalues_[q_point](2);
	double refpf = refvalues_[q_point](2);
		
	local_L2_norm += ((pf-refpf)*(pf-refpf))* state_fe_values.JxW(q_point);
	
      }
    return local_L2_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"L2Norm-"<<ref_string_;
    return out.str();
  }

};

/** Functional for the error in phi in the H^1 part of the energy norm  **/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalWeightedH1Norm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  double alpha_eps_, G_c_;
  std::string ref_string_;
  
public:
 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    //param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    
  }

  LocalFunctionalWeightedH1Norm(ParameterReader &param_reader, double eps, std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");
    alpha_eps_ = eps;
    //alpha_eps_ = param_reader.get_double("alpha_eps");
    G_c_ = param_reader.get_double("G_c");

    ref_string_=ref_string;
    
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_weightedH1_norm = 0;

    vector<vector<Tensor<1, dealdim> > > ugrads_;
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
    edc.GetGradsState("state", ugrads_);


    vector<vector<Tensor<1, dealdim> > > refgrads_;
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetGradsState(ref_string_, refgrads_);

    
    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

      
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

	//phase field gradient
	Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

	//phase field gradient to compare with
	Tensor<1,2> refgrad_pf;
        refgrad_pf.clear();
        refgrad_pf[0] = refgrads_[q_point][2][0];
        refgrad_pf[1] = refgrads_[q_point][2][1];
        
	
	local_weightedH1_norm += (G_c_*alpha_eps_*((grad_pf[0]-refgrad_pf[0])*(grad_pf[0]-refgrad_pf[0]) + (grad_pf[1]-refgrad_pf[1])*(grad_pf[1]-refgrad_pf[1])))
			      * state_fe_values.JxW(q_point);

	
      }
    return local_weightedH1_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"WeightedH1Norm-"<<ref_string_;
    return out.str();
  }

};

/** Functional for the error in the H^1 norm  **/
/****************************************************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
#endif
class LocalFunctionalH1Norm : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
  std::string ref_string_;
 
public:
 
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    
  }

  LocalFunctionalH1Norm(ParameterReader &param_reader,std::string ref_string)
  {
    param_reader.SetSubsection("Local PDE parameters");

    ref_string_=ref_string;
    
  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(const EDC<DH,VECTOR,dealdim> &edc)
  {
    const auto &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    double local_H1_norm = 0;

    vector<vector<Tensor<1, dealdim> > > ugrads_;
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
    edc.GetGradsState("state", ugrads_);

    vector<vector<Tensor<1, dealdim> > > refgrads_;
    refgrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));

    edc.GetGradsState(ref_string_, refgrads_);
   
          
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	//phase field gradient
	Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

	//phase field gradient to compare with
	Tensor<1,2> refgrad_pf;
        refgrad_pf.clear();
        refgrad_pf[0] = refgrads_[q_point][2][0];
        refgrad_pf[1] = refgrads_[q_point][2][1];
        
	
	local_H1_norm += ((grad_pf[0]-refgrad_pf[0])*(grad_pf[0]-refgrad_pf[0]) + (grad_pf[1]-refgrad_pf[1])*(grad_pf[1]-refgrad_pf[1]))
			      * state_fe_values.JxW(q_point);

	
      }
    return local_H1_norm;
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients;
  }

  string
  GetType() const
  {
    return "domain timelocal";
  }

  bool HasFaces() const
  {
    return false;
  }

  string
  GetName() const
  {
    stringstream out;
    out<<"H1Norm-"<<ref_string_;
    return out.str();
  }

};

#endif
