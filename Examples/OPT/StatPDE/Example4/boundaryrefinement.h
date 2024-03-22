#ifndef BOUNDARYREFINEMENT_H_
#define BOUNDARYREFINEMENT_H_

template<int dim,int spacedim=dim>
class BoundaryRefinement : public RefineByGeometry<dim,spacedim>
{
public:

  virtual void MarkElements(Triangulation<dim,spacedim> &tria) const override
  {
    auto cell=tria.begin_active();
    auto endc=tria.end();
    for (; cell != endc; ++cell)
      {
        if (cell->at_boundary())
          {
            cell->set_refine_flag();
          }
      }

  }
};

#endif
