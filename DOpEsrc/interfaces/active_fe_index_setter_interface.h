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

#ifndef ACTIVE_FE_INDEX_INTERFACE_H_
#define ACTIVE_FE_INDEX_INTERFACE_H_

#include <wrapper/dofhandler_wrapper.h>

namespace DOpE
{
  /**
   * Implements different methods to set the ActiveFEIndex on the elements in the
   * case of an hp FEMethod. This class defines the interface needed
   * by the HP::DoFHandler. The user needs to reimplement the methods
   * according to the specific rules used for the element selection.
   */

  template<int dopedim, int dealdim=dopedim>
  class ActiveFEIndexSetterInterface
  {
  public:
    ActiveFEIndexSetterInterface() {};

    /**
     * Gets an iterator to a element and sets an active FE index
     * on this element for the state variable. This function is
     * used after the first grid generation.
     *
     */
    virtual void
    SetActiveFEIndexState(
      typename dealii::hp::DoFHandler<dealdim>::active_cell_iterator &) const
    {
    }
    ;
    /**
     * Just for compatibility issues.
     */
    void
    SetActiveFEIndexState(
      typename dealii::DoFHandler<dealdim>::active_cell_iterator &) const
    {
    }
    ;
    /**
     * Just for compatibility issues.
     */
//    virtual void
//    SetActiveFEIndexState(
//        typename dealii::MGDoFHandler<dealdim>::active_cell_iterator&) const
//    {
//    }
//    ;
    /**
     * Gets an iterator to a element and sets an active FE index
     * on this element for the control variable. This function is
     * used after the first grid generation.
     *
     */
    virtual void
    SetActiveFEIndexControl(
      typename dealii::hp::DoFHandler<dopedim>::active_cell_iterator &) const
    {
    }
    ;
    /**
     * Just for compatibility issues.
     */
    void
    SetActiveFEIndexControl(
      typename dealii::DoFHandler<dopedim>::active_cell_iterator &) const
    {
    }
    ;
    /**
     * Just for compatibility issues.
     */
//    virtual void
//    SetActiveFEIndexControl(
//        typename dealii::MGDoFHandler<dopedim>::active_cell_iterator&) const
//    {
//    }
//    ;


  protected:
  };

  template<int dealdim>
  class ActiveFEIndexSetterInterface<0, dealdim>
  {
  public:
    ActiveFEIndexSetterInterface() {};
    /*
     * Gets an iterator to a element and sets an active FE index
     * on this element for the state variable. This function is
     * used after the first grid generation.
     *
     */
    virtual void SetActiveFEIndexState(typename dealii::hp::DoFHandler<dealdim>::active_cell_iterator) const
    {
    }
    ;
    /**
      * Just for compatibility issues.
      */
    virtual void SetActiveFEIndexState(typename dealii::DoFHandler<dealdim>::active_cell_iterator) const
    {
    }
    ;
    /**
      * Just for compatibility issues.
      */
//    virtual void SetActiveFEIndexState(typename dealii::MGDoFHandler<dealdim>::active_cell_iterator) const
//    {
//    }
//    ;

  protected:
  };

}//end of namespace

#endif /* ACTIVE_FE_INDEX_INTERFACE_H_ */
