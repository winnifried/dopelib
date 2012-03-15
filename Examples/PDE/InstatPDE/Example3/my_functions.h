#include "function_wrapper.h"

using namespace dealii;

/******************************************************/

//class BoundaryFunction : public DOpEWrapper::Function<2>
//{
//public:
//  BoundaryFunction (ParameterReader &param_reader) : DOpEWrapper::Function<2>()
//  {

//  }
//
//  virtual double value (const Point<2>   &p,
//			const unsigned int  component = 0) const;
//
//  virtual void vector_value (const Point<2> &p,
//			     Vector<double>   &value) const;
//
//  static void declare_params(ParameterReader &param_reader)
//  {
//    param_reader.SetSubsection("My functions parameters");
//    param_reader.declare_entry("mean_inflow_velocity", "0.0",
//			       Patterns::Double(0));
//  }
//
// void SetTime(double t) const { mytime=t;}
//
//private:
//  double _mean_inflow_velocity;
//  mutable double mytime;
//};
//
///******************************************************/
//
//double
//BoundaryFunction::value (const Point<2>  &p,
//			    const unsigned int component) const
//{
//  Assert (component < this->n_components,
//	  ExcIndexRange (component, 0, this->n_components));
//
//  //double _mean_inflow_velocity = 1.5;
//
//  if (component == 0)
//    {
//
//      /*
//      // Benchmark: BFAC 2D-1, 2D-2
//      return ( (p(0) == 0) && (p(1) <= 0.41) ? -_mean_inflow_velocity *
//	       (4.0/0.1681) *
//	       (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
//      */
//      // Benchmark: BFAC 2D-3
//      return ( (p(0) == 0) && (p(1) <= 0.41) ? -_mean_inflow_velocity *
//	       (4.0/0.1681) *
//	       std::sin(M_PI * mytime/8) *
//	       (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
//
//      /*
//	// Channel problem
//      return   ( (p(0) == -6.0) && (p(1) <= 2.0)  ? - _mean_inflow_velocity*
//       (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)) : 0 );
//      */
//
//    }
//  return 0;
//}
//
///******************************************************/
//
//void
//BoundaryFunction::vector_value (const Point<2> &p,
//				   Vector<double>   &values) const
//{
//  for (unsigned int c=0; c<this->n_components; ++c)
//    values (c) = BoundaryFunction::value (p, c);
//}


/******************************************************/

class InitialData: public DOpEWrapper::Function<2>
{
	public:
		InitialData(ParameterReader &param_reader) :
			DOpEWrapper::Function<2>()
			{
				param_reader.SetSubsection("Local PDE parameters");
				_strike= param_reader.get_double("strike price");
			}
			virtual double value(const Point<2> &p, const unsigned int component = 0) const;
			virtual void vector_value(const Point<2> &p, Vector<double> &value) const;
			static void declare_params(ParameterReader &param_reader);
	private:
		double _strike;

};

/******************************************************/

double InitialData::value(const Point<2> &p, const unsigned int /*component*/) const
{
	double x = p[0];
	double y = p[1];

	return std::max(_strike - 0.5 * x - 0.5 * y, 0.);
}

/******************************************************/

void InitialData::vector_value(const Point<2> &p, Vector<double> &values) const
{
	for (unsigned int c = 0; c < this->n_components; ++c)
		values(c) = InitialData::value(p, c);
}

/******************************************************/

void InitialData::declare_params(ParameterReader &param_reader)
{
	param_reader.SetSubsection("Local PDE parameters");
	param_reader.declare_entry("strike price", "0.0", Patterns::Double(0));
}
