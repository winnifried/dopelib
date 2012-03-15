#ifndef _TRANSPOSED_GRADIENT_DIRICHLET_DATA_H_
#define _TRANSPOSED_GRADIENT_DIRICHLET_DATA_H_

#include "function_wrapper.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"
#include "transposeddirichletdatainterface.h"

namespace DOpE
{

  /**
   * This class is used to compute the reduced gradient in the case of dirichlet control
   */
  template<typename DD, typename VECTOR, int dopedim, int dealdim>
    class TransposedGradientDirichletData : public TransposedDirichletDataInterface<dopedim,dealdim>
  {
  public:
  TransposedGradientDirichletData(const DD& data) : TransposedDirichletDataInterface<dopedim,dealdim>(), _dirichlet_data(data)
    {
//      _control_dof_handler = NULL;
//      _state_dof_handler = NULL;
      _param_values = NULL;
      _domain_values = NULL;
      _color = 0;
    }

    /**
     * Initializes the private data, should be called prior to any value call!
     */
    void ReInit(
//                const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler,
//		const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
		const std::map<std::string, const dealii::Vector<double>* > &param_values,
		const std::map<std::string, const VECTOR* > &domain_values,
		unsigned int color)
    {
//      _control_dof_handler = &control_dof_handler;
//      _state_dof_handler = &state_dof_handler;
      _param_values = &param_values;
      _domain_values = &domain_values;
      _color = color;
    }


    /**
     * Accesses the values of the dirichlet data
     */
    void value (const dealii::Point<dealdim>   &p,
		const unsigned int  component,
		const unsigned int  dof_number,
		dealii::Vector<double>& local_vector) const
    {
      _dirichlet_data.Data_QT(
//                              _control_dof_handler,
//			      _state_dof_handler,
			      _param_values,
			      _domain_values,
			      _color,
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
      _dirichlet_data.SetTime(time);
    }
  private:
    const DD& _dirichlet_data;
//    const DOpEWrapper::DoFHandler<dopedim>*  _control_dof_handler;
//    const DOpEWrapper::DoFHandler<dealdim>* _state_dof_handler;
    const std::map<std::string, const dealii::Vector<double>* >* _param_values;
    const std::map<std::string, const VECTOR* >* _domain_values;
    unsigned int _color;
  };

}
#endif
