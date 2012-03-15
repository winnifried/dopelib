#ifndef _DOPE_FINITEELEMENT_H_
#define _DOPE_FINITEELEMENT_H_

#include <fe/fe_system.h>
#include <hp/fe_collection.h>
#include <dopeexception.h>

namespace DOpEWrapper
{

template<int dim>
class FiniteElement: public dealii::FESystem<dim>
{
  public:
    FiniteElement(const dealii::FESystem<dim> &fe) :
      dealii::FESystem<dim>(fe)
    {
    }
    FiniteElement(const dealii::FiniteElement<dim> &fe, const unsigned int n_elements) :
      dealii::FESystem<dim>(fe, n_elements)
    {
    }

    FiniteElement(const dealii::FiniteElement<dim> &fe1, const unsigned int n1,
                  const dealii::FiniteElement<dim> &fe2, const unsigned int n2) :
      dealii::FESystem<dim>(fe1, n1, fe2, n2)
    {
    }

    FiniteElement(const dealii::FiniteElement<dim> &fe1, const unsigned int n1,
                  const dealii::FiniteElement<dim> &fe2, const unsigned int n2,
                  const dealii::FiniteElement<dim> &fe3, const unsigned int n3) :
      dealii::FESystem<dim>(fe1, n1, fe2, n2, fe3, n3)
    {
    }

    FiniteElement(const dealii::FiniteElement<dim> &fe1, const unsigned int n1,
                  const dealii::FiniteElement<dim> &fe2, const unsigned int n2,
                  const dealii::FiniteElement<dim> &fe3, const unsigned int n3,
                  const dealii::FiniteElement<dim> &fe4, const unsigned int n4) :
      dealii::FESystem<dim>(fe1, n1, fe2, n2, fe3, n3, fe4, n4)
    {
    }

    FiniteElement(const dealii::FiniteElement<dim> &fe) :
      dealii::FESystem<dim>(fe, 1)
    {
    }
};

//template<>
//class FiniteElement<0> : public dealii::Subscriptor
//{
//  private:
//    unsigned int _comps;
//  public:
//    FiniteElement(unsigned int n)
//    {
//      _comps = n;
//    }
//    unsigned int n_base_elements() const
//    {
//      return _comps;
//    }
//};

template<int dim>
class FECollection: public dealii::hp::FECollection<dim>
{
  public:
    FECollection();
    FECollection(const FiniteElement<dim> &fe) :
      dealii::hp::FECollection<dim>(fe)
    {
    }
    FECollection(const FECollection<dim> &fe_collection) :
      dealii::hp::FECollection<dim>(fe_collection)
    {
    }
};

////TODO Implement FECollection<0>. At the moment, FECollection<0> == FiniteElement<0>
////FECollection sinnlos in 0d
//template<>
//class FECollection<0> : public dealii::Subscriptor
//{
//  private:
//    unsigned int _comps;
//  public:
//    FECollection(unsigned int n)
//    {
//      _comps = n;
//    }
//    unsigned int n_base_elements() const
//    {
//      return _comps;
//    }
//};

}

#endif
