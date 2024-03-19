// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef _WAL_BASE_H_
#define _WALD_BASE_H_

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"

using fdapde::core::SMW;

// add this??
#include "exact_edf.h"


namespace fdapde {
namespace models {

enum CIType {bonferroni,...}

template <typename Model, typename Strategy> class Wald;

struct exact {};
using 

// base class for any regression model
template <typename Model>
class WALD<Model, exact> : public WaldBase<Model> {

    private: 
     Model* m_; 

     
     DMatrix<double> S_ {};  // we rather need the smoothing matrix S instead of Q
     // could extract S directly from exact_edf
     ExactEDF ???

     DMatrix<double> invSigma_ {};

     // matrix of errors sigma^2, should be a n x n matrix
     DMatrix<double> sigma_sq_ {};
     
     // is_empty(Vw_) ritorna true se Vw_ è zero x zero
     DMatrix<double> Vw_ {};

     // matrice C per cui creare un setter
     DMatrix<double> C_ {};
     
     DVector<double> betaw_ {};
     // Maybe to let the compiler decide which type of interval to compute
     // Or don't care and compute all intervals
     std::string intervalType;


    // ci sarà anche una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

    public:
     // deafult constructor
     WALD() = default;
     // starting the constructor
     WALD(Model *m): m_(m) {};

     // computes smoothing matrix S = Q*\Psi*T^{-1}*\Psi^T
     const DMatrix<double>& S() {

        // from exact_edf.h
        // need to check this

        // factorize matrix T
        // Why in the computation of T there's a +P (penalization)
        Eigen::PartialPivLU<DMatrix<double>> invT_ {};
        invT_ = m_.T().partialPivLu();
        DMatrix<double> E_ = m_.PsiTD();    // need to cast to dense for PartialPivLU::solve()
        // penso che questo S_ sia calcolato come Q*\Psi*T^{-1}*\Psi^T quindi va sistemato
        // secondo me va calcolato così
        // S_ = m_Psi() * invT_.solve(E_) * m_.computeQ();
        S_ = m_.lmbQ(m_.Psi() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q 
        return S_;
     }

     const DMatrix<double>& invSigma() {

        // since W is a diagonal matrix we need to square it 
        // what is the best function for transpose and inversion??
        // is it partialPivLu??

        // do we need to check if W has missing values?

        // we could avoid using W.transpose since it is diagonal

        invSigma_ =  (m_.W().transpose()*m_.W()).inverse();
        return invSigma_;
     }

     const DMatrix<double>& sigma_sq() {

        // in gcv.h there is a way to use exact_edf.h which I don't really understand

        // double q = m_.q();            // number of covariates
        // std::size_t n = m_.n_obs();   // number of observations
        // double dor = n - (q + trS);       // residual degrees of freedom

        DMatrix<double> epsilon = m_.y() - m_.fitted();
 

        // don't know how to initialize the ExactEDF object since it only has the deafult constructor
        ExactEDF;
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - ExactEDF::compute())) * (epsilon.transpose()*epsilon);
        return sigma_sq_;
     }

     const DMatrix<double>& Vw() {

        DMatrix<double> ss = S() * S().transpose();
        DMatrix<double> left = invSigma() * m_.W().transpose();
        DMatrix<double> right = m_.W() * invSigma();
        Vw_ = sigma_sq() * (invSigma().transpose() + left * ss * right);

        return Vw_;
     }

     const DVector<double>& betaw() {
      // Is betaw just the beta from the Model???
      betaw_ = m_.beta();
      return betaw_;
     }

     // methods per calcolare p_Value e CI
     // in base al tipo di CI che voglio restituisco una cosa diversa quindi il tipo di CI dove lo faccio passare? in imput alla funzione computeCI?
     DMatrix<double> computeCI(CIType type){
        // SIMULTANEOUS

        //quantile deve cambiare a seconda del confidence interval 
        // magari creare un setter per p e fare p una variabile privata??
        int p = ;
        std::chi_squared_distribution<double> chi_squared(p);
        //quantile livello alpha 
        double quantile = std::quantile(chi_squared, alpha);
        
        
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
        // e che sulle righe della matrice di siano c1, c2, c3...
        DMatrix<double> CVCdiag_ = ((C_ * Vw_) * C_.transpose()).diagonal();
        // della matrice C*V*C^T devo prendere solo la diagonale per i Confidence intervals quindi magari è meglio far calcolare solo la diagonale      
        // oppure la parte con il for la fai così questo è da sistemare 
	     int size = std::min(C_.rows(), Vw_.rows()) // questo lo sai a priori quindi sostituisci con la teoria  
	     DVector<double> diagonal(size);
	
	     //qui C_, Vw_ sono matrici quindi devi capire come accedere a solo le righe o solo le colonne 
	     for(int i=0; i< size;++i){
		      double element = 0;
		      for(int j=0; j< size; ++j){
			      element+=C_(i)*Vw_(j)*C_(i);}
            diagonal(i) = element;
        }


        DVector<double> lowerBound = C_ * betas() - sqrt(quantile * CVCdiag_);
        DVector<double> upperBound = C_ * betas() + sqrt(quantile * CVCdiag_);

        //costruisco la matrice che restituisce i confidence intervals
        DMatrix<double> CIMatrix(m_.n_obs(), 2);
        CIMatrix.col(0) = lowerBound;
        CIMatrix.col(1) = upperBound;

      

        return std::make_pair(lowerBound, upperBound);
     }

     double p_value(){

     }

   // da aggiungere tutti i getters
   // aggiunger destructor?

}  
}  // closing models namespace
}  // closing fdapde namespace
    // method per restituire i risultati