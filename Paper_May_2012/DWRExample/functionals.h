/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalPointFunctionalPressure : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    public:
      bool
      NeedTime() const
      {
        return false;
      }

      double
      PointValue(
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & /*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &domain_values)
      {

        Point < 2 > p1(0.15, 0.2);
        Point < 2 > p2(0.25, 0.2);

        typename map<string, const VECTOR*>::const_iterator it =
            domain_values.find("state");
        Vector<double> tmp_vector(3);

        VectorTools::point_value(state_dof_handler, *(it->second), p1,
            tmp_vector);
        double p1_value = tmp_vector(2);
        tmp_vector = 0;
        VectorTools::point_value(state_dof_handler, *(it->second), p2,
            tmp_vector);
        double p2_value = tmp_vector(2);

        // pressure analysis
        return (p1_value - p2_value);

      }

      string
      GetType() const
      {
        return "point";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Pressure_difference";
      }

  };

/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalBoundaryFunctionalDrag : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    private:
      double _density_fluid, _viscosity;
      double _drag_lift_constant;

    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("density_fluid", "1.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
        param_reader.declare_entry("drag_lift_constant", "1.0",
            Patterns::Double(0));
      }

      LocalBoundaryFunctionalDrag(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      bool
      NeedTime() const
      {
        return false;
      }

      double
      BoundaryValue(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc)
      {
        unsigned int color = fdc.GetBoundaryIndicator();
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();

        if (color == 80)
        {
          Tensor < 1, 2 > drag_lift_value;
          //double drag_lift_constant = 20; // 2D-1: 500 , 2D-2: 20

          vector < Vector<double> > _ufacevalues;
          vector < vector<Tensor<1, dealdim> > > _ufacegrads;

          _ufacevalues.resize(n_q_points, Vector<double>(3));
          _ufacegrads.resize(n_q_points, vector < Tensor<1, 2> > (3));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor < 2, 2 > fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_ufacevalues[q_point](2);
            fluid_pressure[1][1] = -_ufacevalues[q_point](2);

            Tensor < 2, 2 > vgrads;
            vgrads.clear();
            vgrads[0][0] = _ufacegrads[q_point][0][0];
            vgrads[0][1] = _ufacegrads[q_point][0][1];
            vgrads[1][0] = _ufacegrads[q_point][1][0];
            vgrads[1][1] = _ufacegrads[q_point][1][1];

            drag_lift_value -= (fluid_pressure
                + _density_fluid * _viscosity * (vgrads + transpose(vgrads)))
                * state_fe_face_values.normal_vector(q_point)
                * state_fe_face_values.JxW(q_point);
          }

          drag_lift_value *= _drag_lift_constant;

          return drag_lift_value[0];
        }
        return 0.;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_normal_vectors;
      }

      string
      GetType() const
      {
        return "boundary";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Drag";
      }
  };

/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalBoundaryFunctionalLift : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    private:
      mutable double time;
      double _density_fluid, _viscosity;
      double _drag_lift_constant;

    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("density_fluid", "1.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
        param_reader.declare_entry("drag_lift_constant", "1.0",
            Patterns::Double(0));
      }

      LocalBoundaryFunctionalLift(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      bool
      NeedTime() const
      {
        return false;
      }

      double
      BoundaryValue(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc)
      {
        unsigned int color = fdc.GetBoundaryIndicator();
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();

        if (color == 80)
        {
          Tensor < 1, 2 > drag_lift_value;
          //  double drag_lift_constant = 20;// 2D-1: 500 , 2D-2: 20

          vector < Vector<double> > _ufacevalues;
          vector < vector<Tensor<1, dealdim> > > _ufacegrads;

          _ufacevalues.resize(n_q_points, Vector<double>(3));
          _ufacegrads.resize(n_q_points, vector < Tensor<1, 2> > (3));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor < 2, 2 > fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_ufacevalues[q_point](2);
            fluid_pressure[1][1] = -_ufacevalues[q_point](2);

            Tensor < 2, 2 > vgrads;
            vgrads.clear();
            vgrads[0][0] = _ufacegrads[q_point][0][0];
            vgrads[0][1] = _ufacegrads[q_point][0][1];
            vgrads[1][0] = _ufacegrads[q_point][1][0];
            vgrads[1][1] = _ufacegrads[q_point][1][1];

            drag_lift_value -= (fluid_pressure
                + _density_fluid * _viscosity * (vgrads + transpose(vgrads)))
                * state_fe_face_values.normal_vector(q_point)
                * state_fe_face_values.JxW(q_point);
          }

          drag_lift_value *= _drag_lift_constant;

          return drag_lift_value[1];
        }
        return 0.;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_normal_vectors;
      }

      string
      GetType() const
      {
        return "boundary";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Lift";
      }
  };
