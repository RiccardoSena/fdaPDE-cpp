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

#ifndef __WALD_BASE_H__
#define __WALD_BASE_H__

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"


using fdapde::core::SMW;

namespace fdapde {
namespace models {

enum CIType {bonferroni, simultaneous, one_at_the_time};

// template <typename Model, typename Strategy> class WaldBase
template <typename Model> class WaldBase {

    protected: 

     Model* m_; 

     DMatrix<double> S_ {}; 

     DMatrix<double> invSigma_ {};

     // matrix of errors sigma^2, should be a n x n matrix
     DMatrix<double> sigma_sq_ {};
     
     // is_empty(Vw_) ritorna true se Vw_ è zero x zero
     DMatrix<double> Vw_ {};

     // matrice C per cui creare un setter
     DMatrix<double> C_ {};
     
     DVector<double> betaw_ {};

    // ci sarà anche una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

    public:
     // deafult constructor
     WaldBase() = default;
     // starting the constructor
     WaldBase(Model *m): m_(m) {};


     // check this!!!
     virtual DMatrix<double>& S() = 0;


     const DMatrix<double>& invSigma() {

        // since W is a diagonal matrix we need to square it 
        // what is the best function for transpose and inversion??
        // is it partialPivLu??

        // do we need to check if W has missing values?

        // we could avoid using W.transpose since it is diagonal

        invSigma_ =  (m_.W().transpose() * m_.W()).inverse();

        return invSigma_;
     }

     // bisogna fare override anche di questo metodo visto che si può utilizzare StochasticEDF per
     // calcolare la traccia
     const DMatrix<double>& sigma_sq() {

        // in gcv.h there is a way to use exact_edf.h which I don't really understand

        // double q = m_.q();            // number of covariates
        // std::size_t n = m_.n_obs();   // number of observations
        // double dor = n - (q + trS);       // residual degrees of freedom

        DMatrix<double> epsilon = m_.y() - m_.fitted();
 
        // don't know how to initialize the ExactEDF object since it only has the deafult constructor
        ExactEDF strat;
        strat.set_model(m_);
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - strat.compute())) * (epsilon.transpose() * epsilon);
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
      // Is betaw just the beta from the Model??? In that case could not store betaw_ but use directly m_.beta()
      betaw_ = m_.beta();

      return betaw_;
     }

     DMatrix<double> computeCI(CIType type){ 
        // need to set C first
        if(is_empty(C_)){
         // print an error (need to set C)
        }  
        
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
        // e che sulle righe della matrice di siano c1, c2, c3...
        // devi capire quale è il meodo più veloce facendoti restituire il tempo di esecuzione
        // metodo con libreria eigen 
        DMatrix<double> CVCdiag_ = ((C_ * Vw_) * C_.transpose()).diagonal();
        //metodo con ciclo for per caclolare solo la diagonale e non tutta la matrice 
	     int size = std::min(C_.rows(), Vw_.rows()) // questo lo sai a priori quindi sostituisci con la teoria  
	     DVector<double> diagonal(size);
        for (int i = 0; i < C.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector ci = C.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagonal[i] = ci.transpose() * Vw_* ci;
        }
        DVector<double> lowerBound;
        DVector<double> upperBound;



        if(type == simultaneous){ 
        // SIMULTANEOUS
        int p = C_.rows();
        std::chi_squared_distribution<double> chi_squared(p);
        //quantile livello alpha 
        double quantile = std::quantile(chi_squared, alpha);
        
        lowerBound = C_ * betas() - std::sqrt(quantile * diagonal/m_.n_obs());
        upperBound = C_ * betas() + std::sqrt(quantile * diagonal/m_.n_obs());

        }

        else if (type == bonferroni){
         // Bonferroni
         int p = ;
        //quantile livello alpha 
        double quantile = std::sqrt(2.0) * std::erfinv(1-alpha/(2*p));
        
        lowerBound = C_ * betas() - quantile *std::sqrt( diagonal/m_.n_obs());
        upperBound = C_ * betas() + quantile *std::sqrt( diagonal/m_.n_obs());

        }

        else if (type == one_at_the_time){
        //quantile livello alpha 
        double quantile = std::sqrt(2.0) * std::erfinv(1-alpha/2);
        
        lowerBound = C_ * betas() - quantile *std::sqrt( diagonal/m_.n_obs());
        upperBound = C_ * betas() + quantile *std::sqrt( diagonal/m_.n_obs());

        }

        else{
         // inserire errore: nome intervallo non valido
        }

        //costruisco la matrice che restituisce i confidence intervals
        DMatrix<double> CIMatrix(m_.n_obs(), 2);
        CIMatrix.col(0) = lowerBound;
        CIMatrix.col(1) = upperBound;
        

        return std::make_pair(lowerBound, upperBound);
     }

     double p_value(){

     }
     
     // setter for matrix of combination of coefficients C
     void setC(DMatrix<double> C){
      C_ = C;
     }

   // aggiungere destructor?

}  
}  // closing models namespace
}  // closing fdapde namespace

#endif   // __WALD_BASE_H__