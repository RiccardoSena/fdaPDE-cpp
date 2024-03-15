#ifndef _WALD_H_
#define _WALD_H_

// TUTTI QUESTI INCLUDE SARANNO DA CONTROLLARE 
#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>
#include "../model_macros.h"
#include "../model_traits.h"
#include "../space_only_base.h"
#include "../space_time_base.h"
#include "../space_time_separable_base.h"
#include "../space_time_parabolic_base.h"
#include "../sampling_design.h"
#include "gcv.h"
#include "stochastic_edf.h"

// add this??
#include "exact_edf.h"

// probably can delete this
using fdapde::core::BinaryVector;


namespace fdapde {
namespace models {

// base class for any regression model
template <typename Model>
class WALD {

    private: 
     Model* m_; // in teoria da qui dovremmo avere accesso a tutte le variabili che ci servono per calcolare CI 

     
     DMatrix<double> S_ {};  // we rather need the smoothing matrix S instead of Q
     // could extract S directly from exact_edf
     ExactEDF ???

     DMatrix<double> invSigma_ {};

     // matrix of errors sigma^2, should be a n x n matrix
     DMatrix<double> sigma_sq_ {};
     
     // 
     DMatrix<double> Vw_ {};

     // matrice C da fare
     DMatrix<double> C_ {};
     
     // Maybe to let the compiler decide which type of interval to compute
     // Or don't care and compute all intervals
     std::string intervalType


    // ci sarà anche una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

    public:
     // deafult constructor
     WALD() = default;
     // starting the constructor
     WALD(Model *m): m_(m) {};

     // computes smoothing matrix S = Q*\Psi*T^{-1}*\Psi^T
     const DMatrix<double> &S() {

        // from exact_edf.h
        // need to check this

        // factorize matrix T
        invT_ = m_.T().partialPivLu();
        DMatrix<double> E_ = m_.PsiTD();    // need to cast to dense for PartialPivLU::solve()
        // penso che questo S_ sia calcolato come Q*\Psi*T^{-1}*\Psi^T quindi va sistemato
        S_ = m_.lmbQ(m_.Psi() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q 
        return S_;
     }

     const DMatrix<double> &invSigma() {

        // since W is a diagonal matrix we need to square it 
        // what is the best function for transpose and inversion??
        // is it partialPivLu??

        // do we need to check if W has missing values?

        // we could avoid using W.transpose since it is diagonal

        invSigma_ =  (m_.W.transpose()*m_.W).inverse();
        return invSigma_;
     }

     const DMatrix<double> &sigma_sq() {

        // in gcv.h there is a way to use exact_edf.h which I don't really understand

        // double q = m_.q();            // number of covariates
        // std::size_t n = m_.n_obs();   // number of observations
        // double dor = n - (q + trS);       // residual degrees of freedom

        DMatrix<double> epsilon = m_.y() - m_.fitted();
 

        // don't know how to initialize the ExactEDF object since it only has the deafult constructor
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - ExactEDF::compute())) * (epsilon.transpose()*epsilon);
        return sigma_sq_;
     }

     const DMatrix<double> &Vw {

        DMatrix<double> ss = S_ * S_.transpose();
        DMatrix<double> left = invSigma_ * m_.W_.transpose();
        DMatrix<double> right = m_.W_ * invSigma_;
        Vw_ = sigma_sq_ * (invSigma_.transpose() + left * ss * right);

        return Vw_;
     }


    // methods per calcolare p_Value e CI
    // in base al tipo di CI che voglio restituisco una cosa diversa quindi il tipo di CI dove lo faccio passare? in impu alla funzione computeCI?
    DMatrix<double> computeCI(){
      
        //costruisco lowerBound e upperBound
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
         
        DVector<double> lowerBound = C_*m_.beta();
        DVector<double> upperBound = C_*m_.beta()
        DMatrix<double> BoundMatrix(n_, 2);
        BoundMatrix.col(0) = lowerBound;
        BoundMatrix.col(1) = upperBound;

      

        return std::make_pair(lowerBound, upperBound);
    }

   // da aggiungere tutti i getters e i setters

}  
}  // closing models namespace
}  // closing fdapde namespace
    // method per restituire i risultati