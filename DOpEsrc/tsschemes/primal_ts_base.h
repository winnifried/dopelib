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

#ifndef PRIMAL_TS_BASE_H_
#define PRIMAL_TS_BASE_H_

#include <tsschemes/ts_base.h>
namespace DOpE
{

  /**
   * This class contains the methods which all primal time stepping schemes share.
   *
   * @tparam <OPTPROBLEM>       The problem to deal with.
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state.
   * @tparam <VECTOR>           The vector type for control & state
   *                            (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <dealdim>          The dimension of the state variable.
   * @tparam <FE>               The type of finite elements in use, must be compatible with the DH.
   * @tparam <DH>               The type of the DoFHandler in use
   *                            (to be more precise: The type of the dealii-DoFhandler which forms
   *                            the base class of the DOpEWrapper::DoFHandler in use.)
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
           int dealdim,
           template <int, int> class FE = dealii::FESystem,
           template <int, int> class DH = dealii::DoFHandler>
  class PrimalTSBase : public TSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
    dealdim, FE, DH>
  {
  public:
    PrimalTSBase(OPTPROBLEM &OP) :
      TSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dealdim, FE,
      DH>(OP)
    {
    }

    ~PrimalTSBase()
    {
    }

    /******************************************************/
    /****For the initial values ***************/
    /**
     * Computes the value of the element contributions to the
    * equation for the calculation of the initial values
     *
    * @tparam <EDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., ElementDataContainer
     *
    * @param edc                      The EDC object.
     * @param local_vector        This vector contains the locally computed values
     *                                 of the element equation. For more information
     *                                 on dealii::Vector, please visit, the deal.ii manual pages.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    * @param scale_ico                Given for compatibility reasons with the ElementEquation
    *                                 in PDEInterface. Should not be used here!
     */
    template<typename EDC>
    void
    Init_ElementEquation(const EDC &edc,
                         dealii::Vector<double> &local_vector, double scale,
                         double scale_ico)
    {
      this->GetProblem().Init_ElementEquation(edc, local_vector, scale,
                                              scale_ico);
    }

    /**
     * Computes the value of the element contributions to the
    * RHS for the calculation of the initial values
     *
    * @tparam <EDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., ElementDataContainer
     *
    * @param edc                      The EDC object.
     * @param local_vector        This vector contains the locally computed values
     *                                 of the elementrhs.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    */
    template<typename EDC>
    void
    Init_ElementRhs(const EDC &edc,
                    dealii::Vector<double> &local_vector, double scale)
    {
      this->GetProblem().Init_ElementRhs(edc, local_vector, scale);
    }

    /**
     * Computes the value of the element contributions to the
    * Matrix for the calculation of the initial values
     *
    * @tparam <EDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., ElementDataContainer
     *
    * @param edc                      The EDC object.
     * @param local_matrix       This vector contains the locally computed values
     *                                 of the elementmatrix.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    * @param scale_ico                Given for compatibility reasons with the ElementEquation
    *                                 in PDEInterface. Should not be used here!
    */
    template<typename EDC>
    void
    Init_ElementMatrix(const EDC &edc,
                       dealii::FullMatrix<double> &local_matrix, double scale, double scale_ico)
    {
      this->GetProblem().Init_ElementMatrix(edc, local_matrix, scale,
                                            scale_ico);
    }

    /**
    * Computes the value of the point contributions to the
    * Rhs for the calculation of the initial values
    *
     * @param param_values             A std::map containing parameter data
    *                                 (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map
    *                                 at the position "control".
     * @param domain_values            A std::map containing domain data
    *                                 (i.e., nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the
    *                                 position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param local_vector        This vector contains the locally computed values
    *                                 of the PointRhs. For more information
     *                                 on dealii::Vector, please visit, the deal.ii manual pages.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
     */
    void
    Init_PointRhs(
      const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
      const std::map<std::string, const VECTOR *> &/*domain_values*/,
      VECTOR & /*rhs_vector*/, double /*scale*/)
    {
    }

    /**
       * Same functionality as for the Init_ElementEquation, but on Faces.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
       *
    * @param fdc                      The FDC object.
       * @param local_vector        This vector contains the locally computed values
       *                                 of the Facequation.
       * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    * @param scale_ico                Given for compatibility reasons with the ElementEquation
    *                                 in PDEInterface. Should not be used here!
        */
    template<typename FDC>
    void
    Init_FaceEquation(const FDC & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /**
      * Same functionality as for the Init_ElementEquation, but on Interfaces, i.e. the same as
    * FaceEquation but with access to the FEValues on both sides.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
      *
    * @param fdc                      The FDC object.
      * @param local_vector        This vector contains the locally computed values
      *                                 of the InterfaceEquation.
      * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    * @param scale_ico                Given for compatibility reasons with the ElementEquation
    *                                 in PDEInterface. Should not be used here!
    */
    template<typename FDC>
    void
    Init_InterfaceEquation(const FDC & /*fdc*/,
                           dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /**
     * Same functionality as for the ElementEquation, but on Boundaries.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
     *
    * @param fdc                      The FDC object.
     * @param local_vector        This vector contains the locally computed values
     *                                 of the Facequation.
     * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine
    *                                 to compute.
    * @param scale_ico                Given for compatibility reasons with the ElementEquation
    *                                 in PDEInterface. Should not be used here!
      */
    template<typename FDC>
    void
    Init_BoundaryEquation(const FDC & /*fdc*/,
                          dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /**
    * Same functionality as for the ElementMatrix, but on Faces.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
    *
    * @param fdc                      The FDC object.
    * @param local_matrix       This matrix contains the locally computed values
    *                                 of the FaceMatrix.
    */
    template<typename FDC>
    void
    Init_FaceMatrix(const FDC & /*fdc*/,
                    FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /**
       * Same functionality as for the ElementMatrix, but on Interfaces.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
       *
    * @param fdc                      The FDC object.
       * @param local_matrix       This matrix contains the locally computed values
       *                                 of the InterfaceMatrix.
    */
    template<typename FDC>
    void
    Init_InterfaceMatrix(const FDC & /*fdc*/,
                         FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /**
     * Same functionality as for the ElementMatrix, but on Boundaries.
    * Note that no time derivatives may occure on faces of the domain at present!
    * @tparam <FDC>                   A container that contains all relevant data
    *                                 needed on the element, e.g., element size, finite element values;
    *                                 see, e.g., FaceDataContainer
     *
    * @param fdc                      The FDC object.
     * @param local_matrix       This matrix contains the locally computed values
     *                                 of the FaceMatrix.
    */
    template<typename FDC>
    void
    Init_BoundaryMatrix(const FDC & /*fdc*/,
                        FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
    {
    }

    /****End the initial values ***************/
  };
}

#endif /* TS_BASE_PRIMAL_H_ */
