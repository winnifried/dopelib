#ifndef NETWORK_
#define NETWORK_

#include <network/networkinterface.h>



class LocalNetwork : public DOpE::Networks::NetworkInterface
{
public:
  LocalNetwork(DOpE::ParameterReader &param_reader)
  {
    param_reader.SetSubsection("localpde parameters");
    win_    = param_reader.get_double("win");
    pin_  = param_reader.get_double("pin");
  }

  unsigned int GetNPipes() const
  {
    return 2;
  }
  unsigned int GetNNodes() const
  {
    return 3;
  }
  void PipeCouplingResidual(dealii::Vector<double> &res,
                            const dealii::Vector<double> &u,
                            const std::vector<bool> &present_in_outflow) const
  {
    //In the case here we have two pipes 1 and 2 with two unknowns per pipe.
    //The first two are the inflow to the first pipe
    //The second two the outflow
    //The third two the inflow to the second pipe
    //The final pair is the outflow of the second pipe
    assert(res.size()==8); //2 Pipes with two components each + four outflow conditions
    assert(u.size()==8); //4 fluxvariabels  (2 per pipe) with two components each.
    //Ordering is as follows pipe_1 left, pipe_2 left  then pipe_1 right, pipe_2 right

    //First four lines are the output coupling
    assert(present_in_outflow.size()==2*GetNPipes()*2);
    for (unsigned int p = 0; p < GetNPipes(); p++)
      {
        for (unsigned int c = 0; c < 2; c++)
          {
            assert(present_in_outflow[p*2+c]||present_in_outflow[GetNPipes()*2+p*2+c]);
            assert(! (present_in_outflow[p*2+c]&&present_in_outflow[GetNPipes()*2+p*2+c]));
            if (present_in_outflow[p*2+c])
              {
                res[p*2+c] = -u[p*2+c];
              }
            else
              {
                res[p*2+c] = -u[GetNPipes()*2+p*2+c];
              }
          }
      }

    //Einströmbedingungen
    res[4] = win_ - u[0];
    res[5] = pin_ - u[7];

    //Kopplung am Mittleren Knoten pipe2 rechts - pipe_1 links = 0
    res[6] = u[4] - u[2];
    res[7] = u[5] - u[3];

    //No conditions at final node (otherwise there are too many)
  }

  void CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                      const std::vector<bool> &present_in_outflow) const
  {
    assert(present_in_outflow.size()==2*GetNPipes()*2);
    //First n_comp*n_pipes lines for the outflow linearization
    for (unsigned int p = 0; p < GetNPipes(); p++)
      {
        for (unsigned int c = 0; c < 2; c++)
          {
            assert(present_in_outflow[p*2+c]||present_in_outflow[GetNPipes()*2+p*2+c]);
            assert(! (present_in_outflow[p*2+c]&&present_in_outflow[GetNPipes()*2+p*2+c]));
            if (present_in_outflow[p*2+c])
              {
                matrix.set(p*2+c,p*2+c,-1);
              }
            else
              {
                assert(present_in_outflow[GetNPipes()*2+p*2+c]);
                matrix.set(p*2+c,GetNPipes()*2+p*2+c,-1);
              }
          }
      }
    //Now the Matrix for the coupling conditions
    matrix.set(4,0,-1);
    matrix.set(5,7,-1);
    matrix.set(6,4, 1);
    matrix.set(6,2,-1);
    matrix.set(7,5, 1);
    matrix.set(7,3,-1);
  }

  void GetFluxSparsityPattern(dealii::SparsityPattern &sparsity) const
  {
    sparsity.reinit(8,8,4); //8 Flux unkonwns with at most two unknowns coupled.
    //( coupling times 2 for symmetrize!)
    //Outflow conditions 2 (n_comp) per Pipe
    for (unsigned int i = 0; i < 2*GetNPipes(); i++)
      {
        //line_i can have outflow q_i^L or q_i^R
        sparsity.add(i,i);
        sparsity.add(i,GetNPipes()+i);
      }
    //Coupling conditions
    sparsity.add(4,0);//First condition q_1^L = ...
    sparsity.add(5,7);// (second component of the above)
    sparsity.add(6,2);//Coupling flux dof 2 and 4
    sparsity.add(6,4);
    sparsity.add(7,3);//Coupling flux dof 3 and 5
    sparsity.add(7,5);
    sparsity.symmetrize();
  }
private:
  double win_, pin_;
};

#endif
