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


namespace fdapde {
namespace models {

enum CIType {bonferroni, simultaneous, one_at_the_time};

template <typename Model> class WaldBase {

    protected: 

     Model* m_; 

     // E è sparsa, ma M?

     DMatrix<double> S_ {};            // smoothing matrix S (n x n) matrix
     double sigma_sq_ = 0;             // sigma^2 
     DMatrix<double> Vw_ {};           // variance matrix of betaw_ (q x q) matrix
     DMatrix<double> C_ {};            // inference matrix C (p x q) matrix
     
     DVector<double> betaw_ {};        // sol of srpde ( q x 1 ) matrix
     DVector<double> beta0_ {};        // inference hypothesis H0 (p x 1) matrix
     int alpha_ = 0;                   // level of the confidence intervals

     // variabili da aggungere: 
     // una variabile che fa il check che il modello sia stato runnato prima di fare inferenza


    public:
     
     WaldBase() = default;             // deafult constructor
     WaldBase(Model *m): m_(m) {};     // starting the constructor

     // check this!!!
     virtual void S() = 0;

     // si potrebbe fare override anche di questo metodo visto che si può utilizzare StochasticEDF per calcolare la traccia
     const double sigma_sq() {

        // double q = m_.q();            // number of covariates
        // std::size_t n = m_.n_obs();   // number of observations
        // double dor = n - (q + trS);       // residual degrees of freedom

        DMatrix<double> epsilon = m_.y() - m_.fitted();

        ExactEDF::ExactEDF strat;
        strat.set_model(m_);
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - strat.compute())) * (epsilon.transpose() * epsilon);
        return sigma_sq_;
     }

     const DMatrix<double>& Vw() {
        if(is_empty(S_)){
            S();
        }
        if(sigma_sq_ = 0){
            sigma_sq();
        }
        DMatrix<double> invSigma_ = inverse(m_.W().transpose() * m_.W());
        DMatrix<double> ss = S_ * S_.transpose();
        DMatrix<double> left = invSigma_ * m_.W().transpose();
        Vw_ = sigma_sq_ * (invSigma_ + left * ss * left.transpose()); 

        return Vw_;
     }

     const DVector<double>& betaw() {
      // Is betaw just the beta from the Model??? 
      //In that case could not store betaw_ but use directly m_.beta()
      betaw_ = m_.beta();
      return betaw_;
     }

     DMatrix<double> computeCI(CIType type){ 
        fdapde_assert(!is_empty(C_))      // throw an exception if condition is not met  
        
        if(alpha_ == 0) {
         setAlpha(0.05);         // default value 5%
        }

        else{

         if(is_empty(Vw_)){
            Vw();
         }
         
         int p = C_.rows();
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente e 
        //che sulle righe della matrice di siano c1, c2, c3...
        
        // devi capire quale è il metodo più veloce facendoti restituire il tempo di esecuzione
        // 1) metodo con libreria eigen 
        DMatrix<double> CVCdiag_ = ((C_ * Vw_) * C_.transpose()).diagonal();
        // 2) metodo con ciclo for per caclolare solo la diagonale e non tutta la matrice 
	     int size = std::min(C_.rows(), Vw_.rows()) // questo lo sai a priori quindi sostituisci con la teoria  
	     DVector<double> diagonal(size);
        for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector<double> ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagonal[i] = ci.transpose() * Vw_* ci;
        }

        DVector<double> lowerBound;
        DVector<double> upperBound;

        if(type == simultaneous){ 
        // SIMULTANEOUS
        std::chi_squared_distribution<double> chi_squared(p);
        double quantile = std::quantile(chi_squared, alpha_);
        
        lowerBound = C_ * betaw_ - std::sqrt(quantile * diagonal/m_.n_obs());
        upperBound = C_ * betaw_ + std::sqrt(quantile * diagonal/m_.n_obs());

        }

        else if (type == bonferroni){
        // Bonferroni
        double quantile = std::sqrt(2.0) * std::erfinv(1-alpha_/(2*p));
        
        lowerBound = C_ * betaw_ - quantile * std::sqrt( diagonal/m_.n_obs());
        upperBound = C_ * betaw_ + quantile * std::sqrt( diagonal/m_.n_obs());

        }

        else if (type == one_at_the_time){
        // One at the time
        double quantile = std::sqrt(2.0) * std::erfinv(1-alpha_/2);
        
        lowerBound = C_ * betaw_ - quantile * std::sqrt( diagonal/m_.n_obs());
        upperBound = C_ * betaw_ + quantile * std::sqrt( diagonal/m_.n_obs());

        }

        else{
         // inserire errore: nome intervallo non valido
        }

        DMatrix<double> CIMatrix(p, 2);      //matrix of confidence intervals
        CIMatrix.col(0) = lowerBound;
        CIMatrix.col(1) = upperBound;
        

        return CIMatrix;
        }
     }


     // this function returns the statistics not the p-values
     // come hanno fatto gli altri nel report 
     DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_))      // throw an exception if condition is not met  

         // is_empty va bene anche per i Vectors?
         if(is_empty(beta0_)){
            // print errore (need to set beta0)
            // inizializzare i beta_0 a 0???
            // default value 0 for all betas
            setBeta0(DVector<double>::Zero(betaw().size()));
         }
         if(is_empty(Vw_)){
            Vw_ = Vw();
         }

         DVector<double> statistics(C_.rows());
         // simultaneous 
         if( type == simultaneous ){
            DVector<double> diff = C_ * m_.beta() - beta0_;
            DMatrix<double> Sigma = C_ * Vw_ * C_.transpose();
            DMatrix<double> Sigmadec_ = inverse(Sigma);

            double stat = diff.adjoint() * Sigmadec_ * diff;

            statistics.resize(C_.rows());
            statistics(0) = stat;

            for(int i = 1; i < C_.rows(); i++){
               statistics(i) = 10e20;
            }
            return statistics; 
         }
         // one at the time
         if ( type == one_at_the_time ){
            int p = C_.rows();
            statistics.resize(p);
            for(int i = 0; i < p; i++){
               DVector<double> col = C_.row(i);
               double diff = col.adjoint()* m_.beta() - beta0_[i];
               double sigma = col.adjoint() * Vw_ *col;
               double stat = diff/std::sqrt(sigma);
               statistics(i) = stat;
            }
            return statistics;
         }
     }
     

     // setter for matrix of combination of coefficients C
     void setC(DMatrix<double> C){
      C_ = C;
     }

     // setter for alpha
     void setAlpha(int alpha){
      fdapde_assert(0 <= alpha && alpha <= 1)      // throw an exception if condition is not met  
      if( 0 <= alpha && alpha <= 1) {
         alpha_ = alpha;
      }
     }

     // setter per i beta0_
     void setBeta0(DVector<double> beta0){
      // funziona così per i Eigen::Vector??
      beta0_ = beta0;
     }

     // funzione ausiliare per invertire una matrice densa in maniera efficiente
     DMatrix<double> inverse(DMatrix<double> M){
      // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
      Eigen::PartialPivLU<DMatrix<double>> Mdec_ {};
      Mdec_ = M.partialPivLu(); 
      DMatrix<double> invM_ = Mdec_.solve(DMatrix::Identity(M.rows(), M.cols()));
      return invM_;
     }

   // aggiungere destructor?

}  
}  // closing models namespace
}  // closing fdapde namespace

#endif   // __WALD_BASE_H__