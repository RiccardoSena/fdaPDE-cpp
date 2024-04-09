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

#ifndef __SPECKMAN_H__
#define __SPECKMAN_H__

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"

// using fdapde::core::SMW;

#include <boost/math/distributions/chi_squared.hpp>


namespace fdapde {
namespace models {

enum CIType {bonferroni, simultaneous, one_at_the_time};

// template <typename Model, typename Strategy> class WaldBase
template <typename Model> class Speckman {

    private:

     struct ExactInverse{
      SpMatrix<double> compute(Model m){
         SpMatrix<double> inverseA_ {};
         inverseA_ =  m_.invA().solve(DMatrix<double>::Identity(m_.n_basis, m_.n_basis));
         return inverseA_;         
      }
     };

     struct NonExactInverse{
      SpMatrix<double> compute(Model m){
        // quali funzioni devo chiamare per far calcolare la inversa alla classe FSPAI solo compute e getInverse
        // FSPAI approx
        //creo oggetto FSPAI( vanno controllati tipi di input e output)
        fdapde::core::FSPAI fspai_R0(m_.R0());

        // questi non so come vadano scelti ho messo nuemri a caso ???
        unsigned alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A
        unsigned beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
        double epsilon = 0.001; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità
        // calcolo inversa di R0
        fspai_R0.compute(alpha, beta, epsilon);
        //getter per l'inversa di R0
        SpMatrix<double> inv_R0 fspai_R0.getInverse();

        //qui non so se è giusto questo lambda
        //caclolo la matrice Atilde
        // Bisogna usare PsiTD()??
        DMatrix<double> tildeA_ = m_.Psi().transpose()* m_.Psi()+ m_.lambda_D() * m_.R1().transpose() * inv_R0 * m_.R1();

        //applico FSPAI su Atilde
        fdapde::core::FSPAI fspai_A(tildeA_);
        fspai_A.compute(alpha, beta, epsilon);

        // inverseA_
        SpMatrix<double> inverseA_ = fspai_A.getInverse();
        return inverseA_;
      }
     };
     
     using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
     Solver s_;   
    
     Model m_;
     
     DMatrix<double> Lambda_ {};          // Speckman correction matrix (n x n) matrix
     DMatrix<double> Vs_ {};              // variance matrix of betas_ (q x q) matrix
     DMatrix<double> C_ {};               // inference matrix C (p x q) matrix               

     DVector<double> betas_ {};            // sol of srpde ( q x 1 ) matrix
     DVector<double> beta0_ {};           // inference hypothesis H0 (p x 1) matrix
     int alpha_ = 0;                      // level of the confidence intervals


    public:

     Speckman() = default;             // constructors
     Speckman(const Model& m): m_(m) {};

     const DMatrix<double>& Lambda() {
        // serve il .block???
        Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * s_.compute(m_).block(0, 0, m_.n_basis, m_.n_basis) * m_.PsiTD();
        return Lambda_;
      }
      
     const DVector<double>& betas() {
        // Wt = Lambda_*\W
        // zt = Lambda_*\z
        // betas_ = (Wt^T*\Wt)^{-1}*\Wt^T*\zt

        DMatrix<double> Wtilde_ = Lambda() * m_.X();
        DMatrix<double> ytilde_ = Lambda() * m_.y();

        //DMatrix<double> temp = (Wtilde_.transpose() * Wtilde_).partialPivLU().solve(DMatrix<double>::Identity(,));
        DMatrix<double> temp = inverse(Wtilde_.transpose() * Wtilde_);
        betas_ = temp * Wtilde_.transpose() * ytilde_;
        return betas_;

     }

     DMatrix<double>& Vs() {
         if(is_empty(Lambda_)){
            Lambda();
         }
        // set U = Wt^T*\W
        // set E = epsilon*\epsilon^T
        // Vs = U^{-1}*\Wt^T*\Lambda*\E*\Lambda^T*\Wt*U^{-1}
        DMatrix<double> Wt_ = Lambda_ * m_.X();
        // DMatrix<double> U_ = Wtilde_.transpose() * Wtilde_; // symmetric
        // DMatrix<double> invU_ = inverse(U_); 
        DMatrix<double> left_ = WaldBase<Model>::inverse(Wt_.transpose() * Wt_) * Wt_.transpose();
        DMatrix<double> epsilon_ = m_.y() - m_.fitted();

        Vs_ = left_ * Lambda_ * epsilon_ * epsilon_.transpose() * Lambda_.transpose() * left_.transpose();
        return Vs_;
     }

      DMatrix<double> computeCI(CIType type){ 
         fdapde_assert(!is_empty(C_)) ;     // throw an exception if condition is not met  
        
         if(alpha_ == 0) {
          setAlpha(0.05);         // default value 5%
         }
        
         else{

            if(is_empty(Vs_)){
               Vs();
            }
         
         int p = C_.rows();
         // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
         // e che sulle righe della matrice di siano c1, c2, c3...
         // devi capire quale è il meodo più veloce facendoti restituire il tempo di esecuzione
         // metodo con libreria eigen 
         DMatrix<double> CVCdiag_ = ((C_ * Vs()) * C_.transpose()).diagonal();
         //metodo con ciclo for per caclolare solo la diagonale e non tutta la matrice 
	      int size = std::min(C_.rows(), Vs().rows()) // questo lo sai a priori quindi sostituisci con la teoria  
	      DVector<double> diagon(size);
         for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagon[i] = ci.transpose() * Vs() * ci;
         }
         DVector<double> lowerBound;
         DVector<double> upperBound;


         if(type == simultaneous){ 
         // SIMULTANEOUS
         boost::math::chi_squared_distribution<double> chi_squared(p);
         double quantile = boost::math::quantile(chi_squared, alpha_);
         
         lowerBound = (C_ * betas_).array() - (quantile * diagon.array()).sqrt();
         upperBound = (C_ * betas_).array() + (quantile * diagon.array()).sqrt();

         }

         else if (type == bonferroni){
         // BONFERRONI
         double quantile = std::sqrt(2.0) * boost::math::erf_inv(1-alpha_/(2*p));
         
         lowerBound = (C_ * betas_).array() - quantile * (diagon.array()).sqrt();
         upperBound = (C_ * betas_).array() + quantile * (diagon.array()).sqrt();

         }

         else if (type == one_at_the_time){
         // ONE AT THE TIME
         double quantile = std::sqrt(2.0) * boost::math::erf_inv(1-alpha_/2);
         
         lowerBound = (C_ * betas_).array() - quantile * (diagon.array()).sqrt();
         upperBound = (C_ * betas_).array() + quantile * (diagon.array()).sqrt();

         }

         else{
            // inserire errore: nome intervallo non valido
         }

         //costruisco la matrice che restituisce i confidence intervals
         DMatrix<double> CIMatrix(p, 2);
         CIMatrix.col(0) = lowerBound;
         CIMatrix.col(1) = upperBound;        
        
         return CIMatrix;
         }
         
         return DMatrix<double>::Zero(1, 1);// questo va cambiato ma se non c'è non runna

      }

     // this function returns the statistics not the p-values
     DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_));     // throw an exception if condition is not met  

         if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            // set beta_0 to 0
            setBeta0(DVector<double>::Zero(betas().size()));
         }

         if(is_empty(Vs_)){
               Vs();
         }

         DVector<double> statistics(C_.rows());
         // simultaneous 
         if( type == simultaneous ){
            DVector<double> diff = C_ * m_.beta() - beta0_;
            DMatrix<double> Sigma = C_ * Vs_ * C_.transpose();
            DMatrix<double> Sigmadec_ = WaldBase<Model>::inverse(Sigma);

            double stat = diff.adjoint() * Sigmadec_ * diff;

            statistics.resize(C_.rows());
            statistics(0) = stat;

            for(int i = 0; i < C_.rows(); i++){
               statistics(i) == 10e20;
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
               double sigma = col.adjoint() * Vs_ *col;
               double stat= diff/std::sqrt(sigma);
               statistics(i) = stat;
            }
            return statistics;
         }
         else{
            //inserire messaggio di errore
            return DVector<double>::Zero(1);
         }
     }

     // setter for matrix of combination of coefficients C
     void setC(DMatrix<double> C){
      C_ = C;
     }

     // setter for alpha
     void setAlpha(int alpha){
      fdapde_assert(0 <= alpha && alpha <= 1);      // throw an exception if condition is not met  
      if( 0 <= alpha && alpha <= 1) {
         alpha_ = alpha;
      }
     }

     // setter per i beta0_
     void setBeta0(DVector<double> beta0){
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


} // end class
} // end fdapde models
} // end fdapde namespace

#endif  //__SPECKMAN_H__

