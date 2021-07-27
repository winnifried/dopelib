#ifndef SNEDDONGEOMETRICREFINEMENT_H_
#define SNEDDONGEOMETRICREFINEMENT_H_

template<int dim,int spacedim=dim>
  class SneddonGeometricVolumeRefinement : public RefineByGeometry<dim,spacedim>
    {
    public:
    
    virtual void MarkElements(Triangulation<dim,spacedim> & tria) const
    {
      auto cell=tria.begin_active();
      auto endc=tria.end();
      for(;cell != endc; ++cell)
      {

	for (unsigned int vertex=0;vertex < GeometryInfo<dim>::vertices_per_cell;++vertex)
         {
            Tensor<1,dim> cell_vertex = (cell->vertex(vertex));
	    if (cell_vertex[0] <= 2.5 && cell_vertex[0] >= -2.5
                    && cell_vertex[1] <= 1.25 && cell_vertex[1] >= -1.25)
            {
              cell->set_refine_flag();
              break;
            }
         }
	
      }
      
    }
    };


#endif
