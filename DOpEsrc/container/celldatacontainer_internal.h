/*
 * CellDataContainer_internal.h
 *
 *  Created on: Apr 3, 2012
 *      Author: cgoll
 */

#ifndef _CELLDATACONTAINER_INTERNAL_H_
#define _CELLDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include "fevalues_wrapper.h"

namespace DOpE
{
  namespace cdcinternal
  {
    /**
     * This class houses all the functionality which is shared between
     * the CellDataContainer for normal and hp::DoFHandlers.
     *
     * @template VECTOR     Type of the vector we use in our computations
     *                      (i.e. Vector<double> or BlockVector<double>)
     * @template dim        The dimension of the integral we are actually
     *                      interested in.
     */
    template<typename VECTOR, int dim>
      class CellDataContainerInternal
      {
        public:
          CellDataContainerInternal(
              const std::map<std::string, const dealii::Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values);

          virtual
          ~CellDataContainerInternal()
          {
          }
          ;

          /**
           * Looks up the given name in _parameter_data and returns the
           * corresponding value through 'value'.
           */
          void
          GetParamValues(std::string name, dealii::Vector<double>& value) const;

          /**
           * Returns the domain values.
           */
          const std::map<std::string, const VECTOR*> &
          GetDomainValues() const
          {
            return _domain_values;
          }
          ;

          virtual const DOpEWrapper::FEValues<dim>&
          GetFEValuesState() const = 0;

          virtual const DOpEWrapper::FEValues<dim>&
          GetFEValuesControl() const = 0;

          /********************************************************************/
          /**
           * Functions to extract values and gradients out of the FEValues
           */

          /**
           * Writes the values of the state variable at the quadrature points into values.
           */
          inline void
          GetValuesState(std::string name, std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          inline void
          GetValuesState(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control variable at the quadrature points into values
           */
          inline void
          GetValuesControl(std::string name, std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          inline void
          GetValuesControl(std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /*********************************************/
          /*
           * Writes the values of the state gradient at the quadrature points into values.
           */

          template<int targetdim>
            inline void
            GetGradsState(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            inline void
            GetGradsState(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control gradient at the quadrature points into values.
           */
          template<int targetdim>
            inline void
            GetGradsControl(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            inline void
            GetGradsControl(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/
          /*
           * Writes the values of the state laplacian
           * at the quadrature points into values.
           */

          inline void
          GetLaplaciansState(std::string name,
              std::vector<double> & values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */

          inline void
          GetLaplaciansState(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control laplacian
           * at the quadrature points into values.
           */

          inline void
          GetLaplaciansControl(std::string name,
              std::vector<double> & values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */

          inline void
          GetLaplaciansControl(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

        private:
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          inline void
          GetValues(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name, std::vector<double>& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          inline void
          GetValues(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /***********************************************************/
          /**
           * Helper Function.
           */
          template<int targetdim>
            inline void
            GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
                std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          template<int targetdim>
            inline void
            GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
                std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;
          /***********************************************************/
          /**
           * Helper Function.
           */
          inline void
          GetLaplacians(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name, std::vector<double>& values) const;

          /***********************************************************/
          /**
           * Helper Function.
           */
          inline void
          GetLaplacians(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          const std::map<std::string, const dealii::Vector<double>*> &_param_values;
          const std::map<std::string, const VECTOR*> &_domain_values;
      };
  } //end of namespace cdcinternal
}

#endif /* CellDataContainer_INTERNAL_H_ */
