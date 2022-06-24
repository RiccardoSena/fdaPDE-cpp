#ifndef __FEM_STANDARD_SPACE_TIME_SOLVER_H__
#define __FEM_STANDARD_SPACE_TIME_SOLVER_H__

#include "../Assembler.h"
#include "../../utils/Symbols.h"
#include "../PDE.h"

template <typename B, typename I>
class FEMStandardSpaceTimeSolver {
private:
  const I& integrator_; // integrator used to approximate integrals
  const B& basis_;      // basis used as approximation of the infinite dimensional space where the PDE solution is searched
  
  DMatrix solution_;       // vector of coefficients of the approximate solution written in terms of the chosen basis
  std::vector<DVector> forcingVector_;  // right-hand side of the linear system giving the FEM solution
  Eigen::SparseMatrix<double> R1_;      // result of the discretization of the bilinear form, also known as R1.
  Eigen::SparseMatrix<double> R0_;      // mass matrix, needed by components in higher levels of the architecture, known as R0.
 
  // some informations about solution error

  // initializes internal FEM solver status
  template <unsigned int M, unsigned int N, typename E> 
  void init(const PDE<M, N, E>& pde);
  
public:
  // constructor
  FEMStandardSpaceTimeSolver(const B& basis, const I& integrator) : basis_(basis), integrator_(integrator) {};
  // flag used to notify is something was wrong during computation of solution
  bool success = true;

  // solves the PDE using the classical FEM approach: compute stiffness matrix using some finite element basis R1_ and forcing
  // vector b, then solves the linear system R1_*u = b where u is the searched PDE approximation
  template <unsigned int M, unsigned int N, typename E> 
  void solve(const PDE<M, N, E>& pde, double timeHorizon, double deltaT);

  // getters
  DMatrix getSolution() const { return solution_; }
  std::vector<DVector> getForce() const { return forcingVector_; }
  Eigen::SparseMatrix<double> getR1() const { return R1_; }
  Eigen::SparseMatrix<double> getR0() const { return R0_; }
};

// fill all internal data structures required by FEM to solve the problem. These
// operations constitute the core of FEM and should be independent on any specific solver (both space or space-time)
template <typename B, typename I>
template <unsigned int M, unsigned int N, typename E> 
void FEMStandardSpaceTimeSolver<B, I>::init(const PDE<M, N, E>& pde) {
  Assembler<M, N, B, I> assembler(pde.getDomain(), basis_, integrator_); // create assembler object
  R1_ = assembler.assemble(pde.getBilinearForm());       // fill discretization matrix for current operator
  // SparseQR solver needs its matrix in compressed form (see Eigen documentation for details)
  R1_.makeCompressed();

  forcingVector_.resize(pde.getForcingData().size());
  for(const DVector& data : pde.getForcingData()){
    forcingVector_.push_back(assembler.forcingTerm(data)); // fill discretization of rhs for FEM linear system
  }
  
  // R0_ is a mass matrix ([R0]_{ij} = \int_{\Omega} \phi_i \phi_j). This quantity can be obtained by computing
  // the discretization of the Identity() operator
  R0_ = assembler.assemble(Identity());
  return;
}

// use Euler forward to discretize the time derivative. Under this approximation we get a discretization matrix for the PDE operator
// equal to K = [M/deltaT + A] (forward Euler scheme)
template <typename B, typename I>
template <unsigned int M, unsigned int N, typename E> 
void FEMStandardSpaceTimeSolver<B, I>::solve(const PDE<M, N, E>& pde, double timeHorizon, double deltaT) {
  
  // define eigen system solver, use QR decomposition.
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;

  unsigned int timeSteps = timeHorizon/deltaT; // number of iterations for the time loop
  
  solution_.resize(pde.getDomain().getNumberOfNodes(), timeSteps);
  solution_.col(0) = pde.getInitialCondition(); // impose initial condition
  
  DVector rhs = (R0_/deltaT)*pde.getInitialCondition() + pde.getForcingData()[0];

  // execute temporal loop to solve ODE system
  for(std::size_t i = 1; i < timeSteps - 1; ++i){
    // Observe that K is time invariant only for homogeneous boundary conditions. In general we need to recompute K at each time instant
    auto K = R0_/deltaT + R1_;             // build system matrix

    // impose boundary conditions
    for(std::size_t j = 0; j < pde.getDomain().getNumberOfNodes(); ++j){
      if(pde.getDomain().isOnBoundary(j)){
	// boundaryDatum is a pair (nodeID, boundary value)
	double boundaryDatum = pde.getBoundaryData().at(j)[i];
      
	// To impose a Dirichlet boundary condition means to introduce an equation of the kind u_j = b_j where j is the index
	// of the boundary node and b_j is the boundary value we want to impose on this node. This actually removes one degree
	// of freedom from the system. We do so by zeroing out the j-th row of the stiff matrix and set the corresponding
	// diagonal element to 1
	R1_.row(j) *= 0;          // zero all entries of this row
	R1_.coeffRef(j, j) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
	rhs[j] = boundaryDatum;   // impose boundary value
      }
    }
    
    solver.compute(K); // prepare solver
    if(solver.info()!=Eigen::Success){ // stop if something was wrong...
      success = false;
      return;
    }

    DVector u_i = solver.solve(rhs);
    // append time step solution to solution matrix
    solution_.col(i) = u_i;;
    
    // update rhs for next iteration
    rhs = (R0_/deltaT)*u_i + pde.getForcingData()[i+1];
  }
  return;
}

#endif // __FEM_STANDARD_SPACE_TIME_SOLVER_H__