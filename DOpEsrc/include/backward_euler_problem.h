#ifndef _BackwardEulerProblem_H_
#define _BackwardEulerProblem_H_

namespace DOpE
{
  /**
   * Class to compute time dependent problems with the backward Euler
   * time stepping scheme.
   *
   * @tparam <OPTPROBLEM>       The problem to deal with.
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam <FE>               The type of finite elements in use, must be compatible with the DOFHANDLER.
   * @tparam <DOFHANDLER>       The type of the DoFHandler in use (to be more precise: The type of the dealii-DoFhandler which forms
   *                            the base class of the DOpEWrapper::DoFHandler in use.)
   * @tparam <dopedim>          The dimension for the control variable.
   * @tparam <dealdim>          The dimension of the state variable.
   *
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim, typename FE = DOpEWrapper::FiniteElement<
          dealdim>, typename DOFHANDLER = dealii::DoFHandler<dealdim> >
    class BackwardEulerProblem
    {
    public:
      BackwardEulerProblem (OPTPROBLEM& OP) :
        _OP(OP)
      {
      }
      ~BackwardEulerProblem()
      {
      }

      /******************************************************/

      /**
       * Returns the name of the time stepping scheme.
       *
       * @return A string containing the name of the time stepping scheme.
       */
      std::string
      GetName()
      {
        return "backward Euler";
      }
      /******************************************************/
      BackwardEulerProblem<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim,
          dealdim, FE, DOFHANDLER> &
      GetBaseProblem()
      {
        return *this;
      }
      /******************************************************/

      /**
       * Sets the step part which should actually computed, e.g.,
       * previous solution within the NewtonStepSolver or
       * last time step solutions.
       * @param s    Name of the step part
       */
      void
      SetStepPart(std::string s)
      {
        _part = s;
      }

      /******************************************************/

      /**
       * Sets the actual time.
       *
       * @param time      The actual time.
       * @param interval  The actual interval. Make sure that time
       *                  lies in interval!
       */

      void SetTime(double time, const TimeIterator& interval)
      {
        _OP.SetTime(time, interval);
      }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.CellFunctional(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename DATACONTAINER>
        double
        CellFunctional(const DATACONTAINER& dc)
        {
          return _OP.CellFunctional(dc);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.PointFunctional(...). For more information we refer to
       * the file optproblem.h
       */
      double
      PointFunctional(
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values)
      {
        return _OP.PointFunctional(param_values, domain_values);
      }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.BoundaryFunctional(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        double
        BoundaryFunctional(const FACEDATACONTAINER& fdc)
        {
          return _OP.BoundaryFunctional(fdc);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceFunctional(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        double
        FaceFunctional(const FACEDATACONTAINER& fdc)
        {
          return _OP.FaceFunctional(fdc);
        }
      /******************************************************/

      /**
       * Computes the value of the cell equation which corresponds
       * to the residuum in nonlinear cases. This function
       * itself contains a maximum of four subroutines of cell equations:
       * CellEquation, CellEquationExplicit, TimeEquation, TimeEquationExplicit.
       * So far, three types are needed for fluid-structure interaction problems:
       * CellEquation: implicit terms, like pressure.
       * CellEquationExplicit: stress tensors, fluid convection, etc.
       * TimeEquationExplicit: time derivatives of certain variables which are
       *                       combined with transformations, etc.
       *
       * In fluid problems, the CellEquations terms coincide. However the
       * TimeEquations terms differ:
       * CellTimeEquation: time derivatives, e.g., dt v
       *
       * The function is divided into two parts `old' and `new' which  are given
       * to the Newton solver. Then, the computation is done in two steps: first
       * computation of the old Newton- or time step equation parts. After,
       * computation of the actual parts.
       *
       *
       * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
       *                                 is done by parameters, it is contained in this map at the position "control".
       * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
       *                                 is distributed, it is contained in this map at the position "control". The state may always
       *                                 be found in this map at the position "state"
       * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
       * @param n_q_points               Number of quadrature points on a cell.
       * @param material_id              Material Id of the cell.
       * @param cell_diameter            Diameter of the cell.
       * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
       *                                 on dealii::Vector, please visit, the deal.ii manual pages.
       * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
       */
      template<typename DATACONTAINER>
        void
        CellEquation(const DATACONTAINER & dc,
            dealii::Vector<double> &local_cell_vector, double scale, double)
        {
          if (_part == "New")
            {
              dealii::Vector<double> tmp(local_cell_vector);
              tmp = 0.0;
              _OP.CellEquation(dc, tmp, scale, scale);
              local_cell_vector += tmp;

              tmp = 0.0;
              _OP.CellTimeEquation(dc, tmp,
                  scale / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
              local_cell_vector += tmp;

              _OP.CellTimeEquationExplicit(dc, local_cell_vector,
                  scale / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else if (_part == "Old")
            {
              _OP.CellTimeEquation(
                  dc,
                  local_cell_vector,
                  (-1) * scale
                      / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize());
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

      /**
       * Computes the value of the right-hand side.
       * The function is divided into two parts `old' and `new' which  are given
       * the Newton solver. Then, the computation is done in two steps: first
       * computation of the old Newton- or time step equation parts. After,
       * computation of the actual parts.
       *
       *
       * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
       *                                 is done by parameters, it is contained in this map at the position "control".
       * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
       *                                 is distributed, it is contained in this map at the position "control". The state may always
       *                                 be found in this map at the position "state"
       * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
       * @param n_q_points               Number of quadrature points on a cell.
       * @param material_id              Material Id of the cell.
       * @param cell_diameter            Diameter of the cell.
       * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
       *                                 on dealii::Vector, please visit, the deal.ii manual pages.
       * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
       */
      template<typename DATACONTAINER>
        void
        CellRhs(const DATACONTAINER & dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              _OP.CellRhs(dc, local_cell_vector, scale);
            }
          else if (_part == "Old")
            {
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

      /**
       * Computes the value of the cell matrix which is derived
       * by computing the directional derivatives of the residuum equation of the PDE
       * under consideration.
       * This function itself contains a maximum of four subroutines of matrix equations:
       * CellMatrix, CellMatrixExplicit, CellTimeMatrix, CellTimeMatrixExplicit.
       * So far, all three types are needed for fluid-structure interaction problems:
       * CellMatrix:           implicit terms, like pressure.
       * CellMatrixExplicit:   stress tensors, fluid convection, etc.
       * TimeMatrixExplicit:   time derivatives of certain variables which are
       *                       combined with transformations, etc.
       *
       * In fluid problems, the CellMatrix terms coincide. However the
       * TimeMatrix terms differ:
       * CellTimeMatrix: time derivatives, e.g., dt v in direction \partial v
       *
       * This function is just considered in the `new' part. This is due to that directional
       * derivatives vanish if they are applied to old values which are, of course,
       * already computed and therefore constant.
       *
       * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
       *                                 is done by parameters, it is contained in this map at the position "control".
       * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
       *                                 is distributed, it is contained in this map at the position "control". The state may always
       *                                 be found in this map at the position "state"
       * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
       * @param n_q_points               Number of quadrature points on a cell.
       * @param material_id              Material Id of the cell.
       * @param cell_diameter            Diameter of the cell.
       * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
       *                                 filled by the locally computed values. For more information of its functionality, please
       *                                 search for the keyword `FullMatrix' in the deal.ii manual.
       */
      template<typename DATACONTAINER>
        void
        CellMatrix(const DATACONTAINER & dc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          assert(_part == "New");
          dealii::FullMatrix<double> m(local_entry_matrix);

          _OP.CellMatrix(dc, local_entry_matrix, 1., 1.);

          m = 0.;
          _OP.CellTimeMatrix(dc, m);
          local_entry_matrix.add(
              1.0 / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

          m = 0.;
          _OP.CellTimeMatrixExplicit(dc, m);
          local_entry_matrix.add(
              1.0 / _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize(), m);

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceEquation(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceEquation(const FACEDATACONTAINER & fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
              _OP.FaceEquation(fdc, local_cell_vector, scale);
            }
          else if (_part == "Old")
            {
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceEquation(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceEquation(const FACEDATACONTAINER& dc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
              _OP.InterfaceEquation(dc,local_cell_vector, scale);
            }
          else if (_part == "Old")
            {
            }
          else
            {
              abort();
            }
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceRhs(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceRhs(const FACEDATACONTAINER & fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          _OP.FaceRhs(fdc, local_cell_vector, scale);
        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        FaceMatrix(const FACEDATACONTAINER & fdc,
            dealii::FullMatrix<double> &local_entry_matrix)
        {
          assert(_part == "New");
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
          _OP.FaceMatrix(fdc, local_entry_matrix);

        }

      /******************************************************/
      /**
       * Not implemented so far. Returns just _OP.InterfaceMatrix(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        InterfaceMatrix(const FACEDATACONTAINER& fdc,
            dealii::FullMatrix<double> &local_entry_matrix __attribute__((unused)))
        {
          assert(_part == "New");
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
          _OP.InterfaceMatrix(fdc, local_entry_matrix);

        }

      /******************************************************/

      /**
       * Computes the value of boundary equations. In the actual
       * implementation we just consider values for the actual time step.
       * Therefore, no implementation is necessary of older time steps, etc.
       * An example of a boundary equations is the `do-nothing' outflow
       * condition of fluid flows in a channel when using the symmetric stress tensor.
       * In order to have correct outflow conditions the transposed part of the
       * stress tensor should be subtracted at the outflow boundary.
       *
       *
       * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
       *                                 is done by parameters, it is contained in this map at the position "control".
       * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
       *                                 is distributed, it is contained in this map at the position "control". The state may always
       *                                 be found in this map at the position "state"
       * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
       * @param n_q_points               Number of quadrature points on a cell.
       * @param material_id              Material Id of the cell.
       * @param cell_diameter            Diameter of the cell.
       * @param local_cell_vector        This vector contains the locally computed values of the cell equation. For more information
       *                                 on dealii::Vector, please visit, the deal.ii manual pages.
       * @param scale                    A scaling factor which is -1 or 1 depending on the subroutine to compute.
       */
      template<typename FACEDATACONTAINER>
        void
        BoundaryEquation(const FACEDATACONTAINER & fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          if (_part == "New")
            {
              // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
              _OP.BoundaryEquation(fdc, local_cell_vector, scale);
            }
          else if (_part == "Old")
            {
            }
          else
            {
              abort();
            }

        }

      /******************************************************/

      /**
       * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to
       * the file optproblem.h
       */
      template<typename FACEDATACONTAINER>
        void
        BoundaryRhs(const FACEDATACONTAINER & fdc,
            dealii::Vector<double> &local_cell_vector, double scale = 1.)
        {
          _OP.BoundaryRhs(fdc, local_cell_vector, scale);
        }

      /******************************************************/

      /**
       * Computes the matrix entries of boundary equations.
       * This function is just considered in the `new' part. This is due to that directional
       * derivatives vanish if they are applied to old values which are, of course,
       * already computed and therefore constant.
       * An example of a boundary equations is the `do-nothing' outflow
       * condition of fluid flows in a channel when using the symmetric stress tensor.
       * In order to have correct outflow conditions the transposed part of the
       * stress tensor should be subtracted at the outflow boundary.
       *
       *
       * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
       *                                 is done by parameters, it is contained in this map at the position "control".
       * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
       *                                 is distributed, it is contained in this map at the position "control". The state may always
       *                                 be found in this map at the position "state"
       * @param n_dofs_per_cell          Number of degrees of freedom on a cell.
       * @param n_q_points               Number of quadrature points on a cell.
       * @param material_id              Material Id of the cell.
       * @param cell_diameter            Diameter of the cell.
       * @param local_entry_matrix       The local matrix is quadratic and has size local DoFs times local DoFs and is
       *                                 filled by the locally computed values. For more information of its functionality, please
       *                                 search for the keyword `FullMatrix' in the deal.ii manual.
       */
      template<typename FACEDATACONTAINER>
        void
        BoundaryMatrix(const FACEDATACONTAINER & fdc,
            dealii::FullMatrix<double> &local_cell_matrix)
        {
          assert(_part == "New");
          // Hier nicht mit _OP.GetBaseProblem().GetSpaceTimeHandler()->GetStepSize() multiplizieren, da local_cell_matrix schon skaliert ist
          _OP.BoundaryMatrix(fdc, local_cell_matrix);

        }

      /******************************************************/

      /**
       * A pointer to the whole FESystem
       *
       * @return A const pointer to the FESystem()
       */
      const dealii::SmartPointer<const DOpEWrapper::FiniteElement<dealdim> >
      GetFESystem() const
      {
        return _OP.GetFESystem();
      }

      /******************************************************/

      /**
       * This function determines whether a loop over all faces is required or not.
       *
       * @return Returns whether or not this functional has components on faces between elements.
       *         The default value is false.
       */
      bool
      HasFaces() const
      {
        return _OP.HasFaces();
      }

      /******************************************************/
      /**
       * This function determines whether a loop over all faces is required or not.
       *
       * @return Returns whether or not this functional has components on faces between elements.
       *         The default value is false.
       */
      bool
      HasInterfaces() const
      {
        return _OP.HasInterfaces();
      }

      /******************************************************/

      /**
       * This function returns the update flags for domain values
       * for the computation of shape values, gradients, etc.
       * For detailed explication, please visit `Finite element access/FEValues classes' in
       * the deal.ii manual.
       *
       * @return Returns the update flags to use in a computation.
       */
      dealii::UpdateFlags
      GetUpdateFlags() const
      {
        return _OP.GetUpdateFlags();
      }

      /******************************************************/

      /**
       * This function returns the update flags for face values
       * for the computation of shape values, gradients, etc.
       * For detailed explication, please visit
       * `FEFaceValues< dim, spacedim > Class Template Reference' in
       * the deal.ii manual.
       *
       * @return Returns the update flags for faces to use in a computation.
       */
      dealii::UpdateFlags
      GetFaceUpdateFlags() const
      {
        return _OP.GetFaceUpdateFlags();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of Dirichlet boundaries.
       *
       * @return Returns the Dirichlet Colors.
       */
      const std::vector<unsigned int>&
      GetDirichletColors() const
      {
        return _OP.GetDirichletColors();
      }

      /******************************************************/

      /**
       * A std::vector of boolean values to decide at which parts of the boundary and solutions variables
       * Dirichlet values should be applied.
       *
       * @return Returns a component mask for each boundary color.
       */
      const std::vector<bool>&
      GetDirichletCompMask(unsigned int color) const
      {
        return _OP.GetDirichletCompMask(color);
      }

      /******************************************************/

      /**
       * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply
       * on each boundary part with color 'color'.
       *
       * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
       */
      const dealii::Function<dealdim>&
      GetDirichletValues(
          unsigned int color,
          //							const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler,
          //							const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values) const
      {
        return _OP.GetDirichletValues(color,/*control_dof_handler,state_dof_handler,*/
        param_values, domain_values);
      }

      /******************************************************/

      /**
       * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
       * problem, respectively.
       *
       * @return Returns a dealii::Function of initial values.
       */
      const dealii::Function<dealdim>&
      GetInitialValues() const
      {
        return _OP.GetInitialValues();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of the boundary equation.
       *
       * @return Returns colors for the boundary equation.
       */
      const std::vector<unsigned int>&
      GetBoundaryEquationColors() const
      {
        return _OP.GetBoundaryEquationColors();
      }

      /******************************************************/

      /**
       * A std::vector of integer values which contains the colors of the boundary functionals.
       *
       * @return Returns colors for the boundary functionals.
       */
      const std::vector<unsigned int>&
      GetBoundaryFunctionalColors() const
      {
        return _OP.GetBoundaryFunctionalColors();
      }

      /******************************************************/

      /**
       * This function returns the number of functionals to be considered in the problem.
       *
       * @return Returns the number of functionals.
       */
      unsigned int
      GetNFunctionals() const
      {
        return _OP.GetNFunctionals();
      }

      /******************************************************/

      /**
       * This function gets the number of blocks considered in the PDE problem.
       * Example 1: in fluid problems we have to find velocities and pressure
       * --> number of blocks is 2.
       * Example 2: in FSI problems we have to find velocities, displacements, and pressure.
       *  --> number of blocks is 3.
       *
       * @return Returns the number of blocks.
       */
      unsigned int
      GetNBlocks() const
      {
        return _OP.GetNBlocks();
      }

      /******************************************************/

      /**
       * A function which has the number of degrees of freedom for the block `b'.
       *
       * @return Returns the number of DoFs for block `b'.
       */
      unsigned int
      GetDoFsPerBlock(unsigned int b) const
      {
        return _OP.GetDoFsPerBlock(b);
      }

      /******************************************************/

      /**
       * A std::vector which contains the number of degrees of freedom per block.
       *
       * @return Returns a vector with DoFs.
       */
      const std::vector<unsigned int>&
      GetDoFsPerBlock() const
      {
        return _OP.GetDoFsPerBlock();
      }

      /******************************************************/

      /**
       * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
       *
       * @return Returns a matrix with hanging node constraints.
       */
      const dealii::ConstraintMatrix&
      GetHangingNodeConstraints() const
      {
        return _OP.GetHangingNodeConstraints();
      }

      std::string
      GetType() const
      {
        return _OP.GetType();
      }
      std::string
      GetDoFType() const
      {
        return _OP.GetDoFType();
      }

      /******************************************************/

      /**
       * This function describes what type of Functional is considered
       *
       * @return A string describing the functional, feasible values are "domain", "boundary", "point" or "face"
       *         if it contains domain, or boundary ... parts all combinations of these keywords are feasible.
       *         In time dependent problems use "timelocal" to indicate that
       *         it should only be evaluated at a certain time_point, or "timedistributed" to consider \int_0^T J(t,q(t),u(t))  \dt
       *         only one of the words "timelocal" and "timedistributed" should be considered if not it will be considered to be
       *         "timelocal"
       *
       */
      std::string
      GetFunctionalType() const
      {
        return _OP.GetFunctionalType();
      }

      /******************************************************/

      /**
       * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
       *
       * @return A string. This is the name beeing displayed next to the computed values.
       */
      std::string
      GetFunctionalName() const
      {
        return _OP.GetFunctionalName();
      }

      /******************************************************/

      /**
       * A pointer to the OutputHandler() object.
       *
       * @return The OutputHandler() object.
       */
      DOpEOutputHandler<VECTOR>*
      GetOutputHandler()
      {
        return _OP.GetOutputHandler();
      }

      /******************************************************/

      /**
       * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
       *
       * @return The SpaceTimeHandler() object.
       */
        const SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
            dopedim, dealdim>*
        GetSpaceTimeHandler() const
        {
          return _OP.GetBaseProblem().GetSpaceTimeHandler();
        }
        SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,
            dealdim>*
        GetSpaceTimeHandler()
        {
          return _OP.GetBaseProblem().GetSpaceTimeHandler();
        }

      /******************************************************/

      void
      ComputeSparsityPattern(SPARSITYPATTERN & sparsity) const
      {
        _OP.ComputeSparsityPattern(sparsity);
      }

      bool
      L2ProjectionInitialDataWithDeal() const
      {
        return true;
      }

      /******************************************************/

      /**
       * Access to the private membervariable _fe_values_needed_to_be_initialized which
       * points out the necessity to initialize the fevalues. This is needed in the
       * Solverclass (i.e. statsolver or instatsolver), which holds coordinates the
       * initialization of the fevalues, which are stored in the SpaceTimeHandler.
       *
       */
      bool
      GetFEValuesNeededToBeInitialized() const
      {
        return _OP.GetFEValuesNeededToBeInitialized();
      }

      void
      SetFEValuesAreInitialized()
      {
        _OP.SetFEValuesAreInitialized();
      }

      /******************************************************/
    private:
      OPTPROBLEM& _OP;
      std::string _part;
    };
}

#endif
