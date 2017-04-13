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

/* Thomas Wick
   University of Heidelberg
   Date: Wed, Mar 18, 2009,
         Fri, Mar 27, 2009

   Contents:
   1) namespaces: ALE transformations
   2)             Stokes Terms in ALE
   3)             NSE Terms in ALE
   4)             NSE Terms without ALE

*/


#ifndef ALE_TRANSFORMATIONS_H_
#define ALE_TRANSFORMATIONS_H_

namespace ALE_Transformations
{
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_pI (unsigned int q,
          std::vector<Vector<double> > &old_solution_values)
  {
    Tensor<2,dealdim> tmp;
    tmp[0][0] =  old_solution_values[q](dealdim+dealdim);
    tmp[1][1] =  old_solution_values[q](dealdim+dealdim);

    return tmp;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_pI_LinP (const double &phi_i_p)
  {
    Tensor<2,dealdim> tmp;
    tmp.clear();
    tmp[0][0] = phi_i_p;
    tmp[1][1] = phi_i_p;

    return tmp;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_grad_p (unsigned int q,
              std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<1,dealdim> grad_p;
    grad_p[0] =  old_solution_grads[q][dealdim+dealdim][0];
    grad_p[1] =  old_solution_grads[q][dealdim+dealdim][1];

    return grad_p;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_grad_p_LinP (const Tensor<1,dealdim> &phi_i_grad_p)
  {
    Tensor<1,dealdim> grad_p;
    grad_p[0] =  phi_i_grad_p[0];
    grad_p[1] =  phi_i_grad_p[1];

    return grad_p;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_grad_u (unsigned int q,
              std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<2,dealdim> structure_continuation;
    structure_continuation[0][0] = old_solution_grads[q][dealdim][0];
    structure_continuation[0][1] = old_solution_grads[q][dealdim][1];
    structure_continuation[1][0] = old_solution_grads[q][dealdim+1][0];
    structure_continuation[1][1] = old_solution_grads[q][dealdim+1][1];

    return structure_continuation;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_grad_v (unsigned int q,
              std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<2,dealdim> grad_v;
    grad_v[0][0] =  old_solution_grads[q][0][0];
    grad_v[0][1] =  old_solution_grads[q][0][1];
    grad_v[1][0] =  old_solution_grads[q][1][0];
    grad_v[1][1] =  old_solution_grads[q][1][1];

    return grad_v;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_grad_v_T (const Tensor<2,dealdim> &tensor_grad_v)
  {
    Tensor<2,dealdim> grad_v_T;
    grad_v_T = transpose (tensor_grad_v);

    return grad_v_T;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_grad_v_LinV (const Tensor<2,dealdim> &phi_i_grads_v)
  {
    Tensor<2,dealdim> tmp;
    tmp[0][0] = phi_i_grads_v[0][0];
    tmp[0][1] = phi_i_grads_v[0][1];
    tmp[1][0] = phi_i_grads_v[1][0];
    tmp[1][1] = phi_i_grads_v[1][1];

    return tmp;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_Identity ()
  {
    Tensor<2,dealdim> identity;
    identity[0][0] = 1.0;
    identity[0][1] = 0.0;
    identity[1][0] = 0.0;
    identity[1][1] = 1.0;

    return identity;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F (unsigned int q,
         std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<2,dealdim> F;
    F[0][0] = 1.0 +  old_solution_grads[q][dealdim][0];
    F[0][1] = old_solution_grads[q][dealdim][1];
    F[1][0] = old_solution_grads[q][dealdim+1][0];
    F[1][1] = 1.0 + old_solution_grads[q][dealdim+1][1];
    return F;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F_T (const Tensor<2,dealdim> &F)
  {
    return  transpose (F);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F_Inverse (const Tensor<2,dealdim> &F)
  {
    return invert (F);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F_Inverse_T (const Tensor<2,dealdim> &F_Inverse)
  {
    return transpose (F_Inverse);
  }

  template <int dealdim>
  inline
  double
  get_J (const Tensor<2,dealdim> &tensor_F)
  {
    return determinant (tensor_F);
  }


  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_v (unsigned int q,
         std::vector<Vector<double> > &old_solution_values)
  {
    Tensor<1,dealdim> v;
    v[0] = old_solution_values[q](0);
    v[1] = old_solution_values[q](1);

    return v;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_v_LinV (const Tensor<1,dealdim> &phi_i_v)
  {
    Tensor<1,dealdim> tmp;
    tmp[0] = phi_i_v[0];
    tmp[1] = phi_i_v[1];

    return tmp;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_u (unsigned int q,
         std::vector<Vector<double> > &old_solution_values)
  {
    Tensor<1,dealdim> u;
    u[0] = old_solution_values[q](dealdim);
    u[1] = old_solution_values[q](dealdim+1);

    return u;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_u_LinU (const Tensor<1,dealdim> &phi_i_u)
  {
    Tensor<1,dealdim> tmp;
    tmp[0] = phi_i_u[0];
    tmp[1] = phi_i_u[1];

    return tmp;
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_w (unsigned int q,
         std::vector<Vector<double> > &old_solution_values)
  {
    Tensor<1,dealdim> w;
    w[0] = old_solution_values[q](dealdim+dealdim+1);
    w[1] = old_solution_values[q](dealdim+dealdim+2);

    return w;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_grad_w (unsigned int q,
              std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<2,dealdim>
    tmp;
    tmp[0][0] = old_solution_grads[q][dealdim+dealdim+1][0];
    tmp[0][1] = old_solution_grads[q][dealdim+dealdim+1][1];
    tmp[1][0] = old_solution_grads[q][dealdim+dealdim+2][0];
    tmp[1][1] = old_solution_grads[q][dealdim+dealdim+2][1];

    return tmp;
  }



  template <int dealdim>
  inline
  double
  get_J_LinU (unsigned int &q,
              const std::vector<std::vector<Tensor<1,dealdim> > >
              &old_solution_grads,
              const Tensor<2,dealdim> &phi_i_grads_u)
  {
    return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dealdim+1][1]) +
            (1 + old_solution_grads[q][dealdim][0]) * phi_i_grads_u[1][1] -
            phi_i_grads_u[0][1] * old_solution_grads[q][dealdim+1][0] -
            old_solution_grads[q][dealdim][1] * phi_i_grads_u[1][0]);
  }

  template <int dealdim>
  inline
  double
  get_J_Inverse_LinU (const double &J,
                      const double &J_LinU)
  {
    return (-1.0/std::pow(J,2) * J_LinU);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F_LinU (const Tensor<2,dealdim> &phi_i_grads_u)
  {
    Tensor<2,dealdim> systemMatrix_A401_LinU1;
    systemMatrix_A401_LinU1[0][0] = phi_i_grads_u[0][0];
    systemMatrix_A401_LinU1[0][1] = phi_i_grads_u[0][1];
    systemMatrix_A401_LinU1[1][0] = phi_i_grads_u[1][0];
    systemMatrix_A401_LinU1[1][1] = phi_i_grads_u[1][1];

    return systemMatrix_A401_LinU1;
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_F_Inverse_LinU (const Tensor<2,dealdim> &phi_i_grads_u,
                      const double J,
                      const double J_LinU,
                      unsigned int q,
                      std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads
                     )
  {
    Tensor<2,dealdim> F_tilde;
    F_tilde[0][0] = 1.0 + old_solution_grads[q][dealdim+1][1];
    F_tilde[0][1] = -old_solution_grads[q][dealdim][1];
    F_tilde[1][0] = -old_solution_grads[q][dealdim+1][0];
    F_tilde[1][1] = 1.0 + old_solution_grads[q][dealdim][0];

    Tensor<2,dealdim> F_tilde_LinU;
    F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
    F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
    F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
    F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

    return (-1.0/std::pow(J,2) * J_LinU * F_tilde +
            1.0/J * F_tilde_LinU);

  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_J_F_Inverse_T_LinU (const Tensor<2,dealdim> &phi_i_grads_u)
  {
    Tensor<2,dealdim> systemMatrix_A41_LinU1;
    systemMatrix_A41_LinU1[0][0] = phi_i_grads_u[1][1];
    systemMatrix_A41_LinU1[0][1] = -phi_i_grads_u[1][0];
    systemMatrix_A41_LinU1[1][0] = -phi_i_grads_u[0][1];
    systemMatrix_A41_LinU1[1][1] = phi_i_grads_u[0][0];

    return  systemMatrix_A41_LinU1;
  }


  template <int dealdim>
  inline
  double
  get_tr_C_LinU (unsigned int &q,
                 const std::vector<std::vector<Tensor<1,dealdim> > >
                 &old_solution_grads,
                 const Tensor<2,dealdim> &phi_i_grads_u)
  {
    return ((1 + old_solution_grads[q][dealdim][0]) *
            phi_i_grads_u[0][0] +
            old_solution_grads[q][dealdim][1] *
            phi_i_grads_u[0][1] +
            (1 + old_solution_grads[q][dealdim+1][1]) *
            phi_i_grads_u[1][1] +
            old_solution_grads[q][dealdim+1][0] *
            phi_i_grads_u[1][0]);
  }


}

///////////////////////////////////////////////////////////////////
namespace NSE_in_ALE
{
// get_sigma_ALE
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_ALE (const double &density,
                        const double &viscosity,
                        const Tensor<2,dealdim> &pI,
                        const Tensor<2,dealdim> &grad_v,
                        const Tensor<2,dealdim> &grad_v_T,
                        const Tensor<2,dealdim> &F_Inverse,
                        const Tensor<2,dealdim> &F_Inverse_T)
  {
    Tensor<2,dealdim> tmp;
    tmp = (-pI + density * viscosity *
           (grad_v * F_Inverse +
            F_Inverse_T * grad_v_T ));

    return tmp;

  }


// get_sigma_ALE
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_except_pressure_ALE (const double &density,
                                        const double &viscosity,
                                        const Tensor<2,dealdim> &grad_v,
                                        const Tensor<2,dealdim> &grad_v_T,
                                        const Tensor<2,dealdim> &F_Inverse,
                                        const Tensor<2,dealdim> &F_Inverse_T)
  {
    Tensor<2,dealdim> tmp;
    tmp = (density * viscosity *
           (grad_v * F_Inverse +
            F_Inverse_T * grad_v_T ));

    return tmp;

  }

// get_J_pI_F_InverseT_LinAll
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_ALE_1st_term_LinAll_short (const Tensor<2,dealdim> &pI,
                                              const Tensor<2,dealdim> &F_Inverse_T,
                                              const Tensor<2,dealdim> &J_F_Inverse_T_LinU,
                                              const Tensor<2,dealdim> &pI_LinP,
                                              const double &J)
  {
    //   (J * sigma_ALE * F_Inverse_T , phi_i_grads_v )
    return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_ALE_2nd_term_LinAll_short (const Tensor<2,dealdim> J_F_Inverse_T_LinU,
                                              const Tensor<2,dealdim> stress_fluid_ALE,
                                              const Tensor<2,dealdim> grad_v,
                                              const Tensor<2,dealdim> grad_v_LinV,
                                              const Tensor<2,dealdim> F_Inverse,
                                              const Tensor<2,dealdim> F_Inverse_LinU,
                                              const double J,
                                              const double &viscosity,
                                              const double &density
                                             )
  {
    Tensor<2,dealdim> sigma_LinV;
    Tensor<2,dealdim> sigma_LinU;

    sigma_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    sigma_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);

    return (density * viscosity *
            (sigma_LinV + sigma_LinU) * J * transpose(F_Inverse) +
            stress_fluid_ALE * J_F_Inverse_T_LinU);


  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_ALE_3rd_term_LinAll_short (const Tensor<2,dealdim> F_Inverse,
                                              const Tensor<2,dealdim> F_Inverse_LinU,
                                              const Tensor<2,dealdim> grad_v,
                                              const Tensor<2,dealdim> grad_v_LinV,
                                              const double &viscosity,
                                              const double &density,
                                              const double J,
                                              const Tensor<2,dealdim> J_F_Inverse_T_LinU)
  {
    return density * viscosity *
           ( J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
             J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
             J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU));

  }

// for p-Stokes
  template <int dealdim>
  inline
  double
  get_shear_rate_ALE (const Tensor<2,dealdim> F_Inverse,
                      const Tensor<2,dealdim> grad_v
                     )
  {
    return (1.0/4.0 * scalar_product(grad_v * F_Inverse +
                                     transpose(F_Inverse) * transpose(grad_v),
                                     grad_v * F_Inverse +
                                     transpose(F_Inverse) * transpose(grad_v)));
  }


// p-Stokes
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_PowerLawP_ALE (const double &density,
                                  const double &viscosity,
                                  const Tensor<2,dealdim> &pI,
                                  const Tensor<2,dealdim> &grad_v,
                                  const Tensor<2,dealdim> &grad_v_T,
                                  const Tensor<2,dealdim> &F_Inverse,
                                  const Tensor<2,dealdim> &F_Inverse_T,
                                  const double powerLaw_p,
                                  const double shear_rate_ALE,
                                  const double powerLaw_eps)
  {
    return (-pI + density * viscosity * std::pow(std::pow(powerLaw_eps, 2) + shear_rate_ALE, (powerLaw_p - 2.0)/2.0) *
            (grad_v * F_Inverse +  F_Inverse_T * grad_v_T ));
  }


// p-Stokes
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_PowerLawP_except_pressure_ALE (const double density,
                                                  const double viscosity,
                                                  const Tensor<2,dealdim> &grad_v,
                                                  const Tensor<2,dealdim> &grad_v_T,
                                                  const Tensor<2,dealdim> &F_Inverse,
                                                  const Tensor<2,dealdim> &F_Inverse_T,
                                                  const double powerLaw_p,
                                                  const double shear_rate_ALE,
                                                  const double powerLaw_eps)
  {
    return (density * viscosity * std::pow(powerLaw_eps * powerLaw_eps + shear_rate_ALE, (powerLaw_p - 2.0)/2.0) *
            (grad_v * F_Inverse + F_Inverse_T * grad_v_T ));

  }

// p-Stokes
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_PowerLawP_ALE_2nd_term_LinAll (const Tensor<2,dealdim> J_F_Inverse_T_LinU,
                                                  const Tensor<2,dealdim> stress_fluid_PowerLaw_ALE,
                                                  const Tensor<2,dealdim> grad_v,
                                                  const Tensor<2,dealdim> grad_v_LinV,
                                                  const Tensor<2,dealdim> F_Inverse,
                                                  const Tensor<2,dealdim> F_Inverse_LinU,
                                                  const double J,
                                                  const double viscosity,
                                                  const double density,
                                                  const double powerLaw_p,
                                                  const double shear_rate_ALE,
                                                  const double powerLaw_eps
                                                 )
  {
    Tensor<2,dealdim> strain_rate_tensor_LinV;
    Tensor<2,dealdim> strain_rate_tensor_LinU;

    Tensor<2,dealdim> strain_rate_tensor;

    strain_rate_tensor = (grad_v * F_Inverse + transpose(F_Inverse) * transpose(grad_v));

    strain_rate_tensor_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    strain_rate_tensor_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);

    double powerLawP_LinAll = (density * viscosity * (powerLaw_p - 2.0)/2.0 *
                               std::pow(powerLaw_eps * powerLaw_eps + shear_rate_ALE, (powerLaw_p - 4.0)/2.0) *
                               1.0/2.0 * scalar_product(strain_rate_tensor_LinV + strain_rate_tensor_LinU, strain_rate_tensor));


    //powerLawP_LinAll
    return (powerLawP_LinAll * strain_rate_tensor * J * transpose(F_Inverse) +
            density * viscosity * std::pow(powerLaw_eps * powerLaw_eps + shear_rate_ALE, (powerLaw_p - 2.0)/2.0) *
            ((strain_rate_tensor_LinV + strain_rate_tensor_LinU) * J * transpose(F_Inverse) +
             strain_rate_tensor * J_F_Inverse_T_LinU));

  }



// p-Stokes
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_fluid_PowerLawP_ALE_3rd_term_LinAll_short (const Tensor<2,dealdim> F_Inverse,
                                                        const Tensor<2,dealdim> F_Inverse_LinU,
                                                        const Tensor<2,dealdim> grad_v,
                                                        const Tensor<2,dealdim> grad_v_LinV,
                                                        const double viscosity,
                                                        const double density,
                                                        const double J,
                                                        const Tensor<2,dealdim> J_F_Inverse_T_LinU,
                                                        const double powerLaw_p,
                                                        const double shear_rate_ALE,
                                                        const double powerLaw_eps
                                                       )
  {
    Tensor<2,dealdim> strain_rate_tensor_LinV;
    Tensor<2,dealdim> strain_rate_tensor_LinU;

    Tensor<2,dealdim> strain_rate_tensor;

    strain_rate_tensor = (grad_v * F_Inverse + transpose(F_Inverse) * transpose(grad_v));

    strain_rate_tensor_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    strain_rate_tensor_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);

    double powerLawP_LinAll = (density * viscosity * (powerLaw_p - 2.0)/2.0 *
                               std::pow(powerLaw_eps * powerLaw_eps + shear_rate_ALE, (powerLaw_p - 4.0)/2.0) *
                               1.0/2.0 * scalar_product(strain_rate_tensor_LinV + strain_rate_tensor_LinU, strain_rate_tensor));


    return (density * viscosity *
            (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
             J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
             J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU)) *
            std::pow(powerLaw_eps * powerLaw_eps + shear_rate_ALE, (powerLaw_p - 2.0)/2.0) +
            powerLawP_LinAll * transpose(grad_v) * transpose(F_Inverse) * J * transpose(F_Inverse));

  }



  template <int dealdim>
  inline
  double
  get_Incompressibility_ALE (unsigned int q,
                             std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    return (old_solution_grads[q][0][0] +
            old_solution_grads[q][dealdim+1][1] * old_solution_grads[q][0][0] -
            old_solution_grads[q][dealdim][1] * old_solution_grads[q][1][0] -
            old_solution_grads[q][dealdim+1][0] * old_solution_grads[q][0][1] +
            old_solution_grads[q][1][1] +
            old_solution_grads[q][dealdim][0] * old_solution_grads[q][1][1]);

  }

  template <int dealdim>
  inline
  double
  get_Incompressibility_ALE_LinAll ( const Tensor<2,dealdim> phi_i_grads_v,
                                     const Tensor<2,dealdim> &phi_i_grads_u,
                                     unsigned int &q,
                                     const std::vector<std::vector<Tensor<1,dealdim> > >
                                     &old_solution_grads)

  {
    return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] +
            phi_i_grads_u[1][1] * old_solution_grads[q][0][0] -
            phi_i_grads_u[0][1] * old_solution_grads[q][1][0] -
            phi_i_grads_u[1][0] * old_solution_grads[q][0][1] +
            phi_i_grads_u[0][0] * old_solution_grads[q][1][1]);
  }


  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_Convection_LinAll_short (const Tensor<2,dealdim> &phi_i_grads_v,
                               const Tensor<1,dealdim> &phi_i_v,
                               const double &J,
                               const double &J_LinU,
                               const Tensor<2,dealdim> &F_Inverse,
                               const Tensor<2,dealdim> &F_Inverse_LinU,
                               const Tensor<1,dealdim> &v,
                               const Tensor<2,dealdim> &grad_v,
                               const double &density
                              )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v

    Tensor<1,dealdim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * v +
                       J * grad_v * F_Inverse_LinU * v);

    Tensor<1,dealdim> convection_LinV;
    convection_LinV = (J * (phi_i_grads_v * F_Inverse * v +
                            grad_v * F_Inverse * phi_i_v));

    return density * (convection_LinU + convection_LinV);
  }


  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_Convection_u_LinAll_short (const Tensor<2,dealdim> &phi_i_grads_v,
                                 const Tensor<1,dealdim> &phi_i_u,
                                 const double &J,
                                 const double &J_LinU,
                                 const Tensor<2,dealdim> &F_Inverse,
                                 const Tensor<2,dealdim> &F_Inverse_LinU,
                                 const Tensor<1,dealdim> &u,
                                 const Tensor<2,dealdim> &grad_v,
                                 const double &density
                                )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1,dealdim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * u +
                       J * grad_v * F_Inverse_LinU * u +
                       J * grad_v * F_Inverse * phi_i_u);

    Tensor<1,dealdim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * u);


    return density * (convection_LinU + convection_LinV);
  }



  template <int dealdim>
  inline
  Tensor<1,dealdim>
  get_Convection_u_old_LinAll_short (const Tensor<2,dealdim> &phi_i_grads_v,
                                     const double &J,
                                     const double &J_LinU,
                                     const Tensor<2,dealdim> &F_Inverse,
                                     const Tensor<2,dealdim> &F_Inverse_LinU,
                                     const Tensor<1,dealdim> &old_timestep_solution_displacement,
                                     const Tensor<2,dealdim> &grad_v,
                                     const double &density
                                    )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1,dealdim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
                       J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);

    Tensor<1,dealdim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement);


    return density * (convection_LinU  + convection_LinV);
  }

  template <int dealdim>
  inline
  Tensor<1,dealdim> get_accelaration_term_LinAll (const Tensor<1,dealdim> &phi_i_v,
                                                  const Tensor<1,dealdim> v,
                                                  const Tensor<1,dealdim> old_timestep_v,
                                                  const double &J_LinU,
                                                  const double &J,
                                                  const double old_timestep_J,
                                                  const double &density_fluid)
  {
    //FSI
    return density_fluid/2.0 * (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);

  }


}

/// <>





namespace Structure_Terms_in_ALE
{
  // Green-Lagrange strain tensor
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_E (const Tensor<2,dealdim> &F_T,
         const Tensor<2,dealdim> &F,
         const Tensor<2,dealdim> &Identity)
  {

    return 0.5 * (F_T * F - Identity);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_linear_E (unsigned int q,
                std::vector<std::vector<Tensor<1,dealdim> > > &old_solution_grads)
  {
    Tensor<2,dealdim> F;
    F[0][0] = old_solution_grads[q][dealdim][0];
    F[0][1] = old_solution_grads[q][dealdim][1];
    F[1][0] = old_solution_grads[q][dealdim+1][0];
    F[1][1] = old_solution_grads[q][dealdim+1][1];
    return 0.5*(F + transpose(F));

  }


  template <int dealdim>
  inline
  double
  get_tr_E (const Tensor<2,dealdim> &E)
  {
    return trace (E);
  }

  template <int dealdim>
  inline
  double
  get_tr_E_LinU (unsigned int &q,
                 const std::vector<std::vector<Tensor<1,dealdim> > >
                 &old_solution_grads,
                 const Tensor<2,dealdim> &phi_i_grads_u)
  {
    return ((1 + old_solution_grads[q][dealdim][0]) *
            phi_i_grads_u[0][0] +
            old_solution_grads[q][dealdim][1] *
            phi_i_grads_u[0][1] +
            (1 + old_solution_grads[q][dealdim+1][1]) *
            phi_i_grads_u[1][1] +
            old_solution_grads[q][dealdim+1][0] *
            phi_i_grads_u[1][0]);
  }

  /// neo-Hookean material , INH


  // get_systemMatrix_A40_and_A41_LinAll
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_INH_ALE_2nd_3rd_term_LinU_TD (const Tensor<2,dealdim> F,
                                           const double J,
                                           const double lame_coefficient_mu,
                                           const double  J_LinU,
                                           const Tensor<2,dealdim> &F_LinU,
                                           const Tensor<2,dealdim> &J_F_Inverse_T_LinU)
  {
    return lame_coefficient_mu * (J_LinU * F + J * F_LinU - J_F_Inverse_T_LinU);
  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_INH_ALE_2nd_3rd_term_LinU (const Tensor<2,dealdim> F,
                                        const double J,
                                        const double lame_coefficient_mu,
                                        const double  J_LinU,
                                        const Tensor<2,dealdim> &F_LinU)
  {
    return lame_coefficient_mu * (J_LinU * F + J * F_LinU);
  }

  /// Mooney-Rivlin , IMR
  // get_systemMatrix_A40_and_A41_LinAll
  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_MooneyRivlin_ALE_2nd_3rd_term_LinU (const double &J,
                                                 const double &J_LinU1,
                                                 const double &J_LinU2,
                                                 const double &lame_coefficient_mu,
                                                 const double &mu_2,
                                                 const Tensor<2,dealdim> &F,
                                                 const Tensor<2,dealdim> &F_LinU1,
                                                 const Tensor<2,dealdim> &F_LinU2,
                                                 const Tensor<2,dealdim> &F_Inverse,
                                                 const Tensor<2,dealdim> &F_Inverse_T,
                                                 const Tensor<2,dealdim> &F_Inverse_LinU1,
                                                 const Tensor<2,dealdim> &F_Inverse_LinU2,
                                                 const Tensor<2,dealdim> &F_Inverse_T_LinU1,
                                                 const Tensor<2,dealdim> &F_Inverse_T_LinU2,
                                                 const Tensor<2,dealdim> &J_F_Inverse_T_LinU1,
                                                 const Tensor<2,dealdim> &J_F_Inverse_T_LinU2
                                                )
  {

    // 2nd term
    Tensor<2,dealdim> systemMatrix_A40_LinU1;
    systemMatrix_A40_LinU1 = (J_LinU1 * F + J * F_LinU1);

    Tensor<2,dealdim> systemMatrix_A40_LinU2;
    systemMatrix_A40_LinU2 = (J_LinU2 * F + J * F_LinU2);




    Tensor<2,dealdim> tmp = (lame_coefficient_mu *
                             (systemMatrix_A40_LinU1 +
                              systemMatrix_A40_LinU2) +
                             mu_2 *
                             (J_F_Inverse_T_LinU1 * F_Inverse * F_Inverse_T +
                              J * F_Inverse_T * F_Inverse_LinU1 * F_Inverse_T +
                              J * F_Inverse_T * F_Inverse * F_Inverse_T_LinU1 +
                              J_F_Inverse_T_LinU2 * F_Inverse * F_Inverse_T +
                              J * F_Inverse_T * F_Inverse_LinU2 * F_Inverse_T +
                              J * F_Inverse_T * F_Inverse * F_Inverse_T_LinU2));


    return tmp;

  }

  template <int dealdim>
  inline
  Tensor<2,dealdim>
  get_stress_Delfino_ALE_2nd_3rd_term_LinU (const double &tr_C,
                                            const double &tr_C_LinU1,
                                            const double &tr_C_LinU2,
                                            const double &delfino_a,
                                            const double &delfino_b,
                                            const Tensor<2,dealdim> &F,
                                            const Tensor<2,dealdim> &F_LinU1,
                                            const Tensor<2,dealdim> &F_LinU2
                                           )
  {
    Tensor<2,dealdim> systemMatrix_A40_LinU1;

    return 2.0 * (delfino_a/2.0 * std::exp(delfino_b/2.0 * (tr_C - 3.0)) *
                  delfino_b/2.0 * (tr_C_LinU1 + tr_C_LinU2)) * F +
           2.0 * delfino_a/2.0 * std::exp(delfino_b/2.0 * (tr_C - 3.0)) * (F_LinU1 + F_LinU2);

  }

}


#endif




