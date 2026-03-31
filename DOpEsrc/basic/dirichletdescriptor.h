#ifndef DIRICHLET_DESCRIPTOR_
#define DIRICHLET_DESCRIPTOR_
namespace DOpE
{

  class DirichletDescriptor
  {
  public:
    DirichletDescriptor(const std::vector<unsigned int> &dirichlet_colors,
#if DEAL_II_VERSION_GTE(9,7,0)
			const std::vector<dealii::ComponentMask > &dirichlet_comps
#else
                        const std::vector<std::vector<bool> > &dirichlet_comps
#endif
                       )
      : dirichlet_colors_(dirichlet_colors), dirichlet_comps_(dirichlet_comps)
    {

    }

#if DEAL_II_VERSION_GTE(9,7,0)
    const dealii::ComponentMask &
#else
    const std::vector<bool> &
#endif
    GetDirichletCompMask(unsigned int color) const
    {
      unsigned int comp = dirichlet_colors_.size();
      for (unsigned int i = 0; i < dirichlet_colors_.size(); ++i)
        {
          if (dirichlet_colors_[i] == color)
            {
              comp = i;
              break;
            }
        }
      if (comp == dirichlet_colors_.size())
        {
          std::stringstream s;
          s << "ControlDirichletColor" << color << " has not been found !";
          throw DOpEException(s.str(),
                              "DirichletDescriptor::GetDirichletCompMask");
        }
      return dirichlet_comps_[comp];
    }

    const std::vector<unsigned int> &GetDirichletColors() const
    {
      return dirichlet_colors_;
    }

  private:
    const std::vector<unsigned int> &dirichlet_colors_;
#if DEAL_II_VERSION_GTE(9,7,0)
    const std::vector<dealii::ComponentMask> &dirichlet_comps_;
#else
    const std::vector<std::vector<bool> > &dirichlet_comps_;
#endif
    
  };

}

#endif
