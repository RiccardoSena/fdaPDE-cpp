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
using fdapde::core::FSPAI;


#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "inference.h"

//#include <boost/math/distributions/chi_squared.hpp>

namespace fdapde {
namespace models {


template <typename Model, typename Strategy> class Speckman {

   private:
   
      struct ExactInverse{

         DMatrix<double> compute(Model m){
            Eigen::PartialPivLU<DMatrix<double>> Adec_ (m.E());
            DMatrix<double> inverseA_ = Adec_.solve(DMatrix<double>::Identity(m.E().rows(), m.E().cols()));
            return inverseA_;       
         }
      };

      struct NonExactInverse{
         DMatrix<double> compute(Model m){
            // quali funzioni devo chiamare per far calcolare la inversa alla classe FSPAI solo compute e getInverse
            // FSPAI approx
            // creo oggetto FSPAI( vanno controllati tipi di input e output)

            SpMatrix<double> decR0_ = lump(m.R0());  

            // fare con 1/R0_ii
            DiagMatrix<double> invR0_(decR0_.rows());
            invR0_.setZero(); 

            for (int i = 0; i < decR0_.rows(); ++i) {
               double diagElement = decR0_.diagonal()[i]; 
               invR0_.diagonal()[i] = 1.0 / diagElement; 
            }

            unsigned alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A
            unsigned beta = 10;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
            double epsilon = 0.05; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità
            
            DMatrix<double> Et_ = m.PsiTD()* m.Psi() + m.lambda_D() * m.R1().transpose() * invR0_ * m.R1();
            
            //Et_ should be stored as a sparse matrix 
            Eigen::SparseMatrix<double> Et_sparse = Et_.sparseView();
            FSPAI fspai_E(Et_sparse);
            fspai_E.compute(alpha, beta, epsilon);
            SpMatrix<double> invE_ = fspai_E.getInverse();
            /*
            //prova di inversa di Et_ con lumping 
            DMatrix<double> decEt_ = lump(Et_);  

            // fare con 1/R0_ii
            DiagMatrix<double> invEt_(decEt_.rows());
            invEt_.setZero(); 

            for (int i = 0; i < decEt_.rows(); ++i) {
               double diagElement = decEt_.diagonal()[i]; 
               invEt_.diagonal()[i] = 1.0 / diagElement; 
            }
            */
            return invE_;
         }
      };
     
      using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
      Solver s_;   
      
      Model m_;
      
      DMatrix<double> Lambda_ {};          // Speckman correction matrix (n x n) matrix
      DMatrix<double> Vs_ {};              // variance matrix of betas_ (q x q) matrix
      DMatrix<double> C_ {};               // inference matrix C (p x q) matrix               
      DVector<double> betas_ {};           // sol of srpde ( q x 1 ) matrix
      DVector<double> beta0_ {};           // inference hypothesis H0 (p x 1) matrix
      double alpha_ = 0;                   // level of the confidence intervals


   public:

      Speckman() = default;                // constructors
      Speckman(const Model& m): m_(m) {};

      // return Lambda_^2
      DMatrix<double> Lambda() {
         DMatrix<double> Lambda = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * s_.compute(m_) * m_.PsiTD();
         //Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * s_.compute(m_) * m_.PsiTD()*DMatrix<double>::Identity(m_.n_obs(), m_.n_obs())-m_.Psi() * s_.compute(m_) * m_.PsiTD();
         DMatrix<double> Lambda_squared = Lambda * Lambda;
         return Lambda_squared;

      }
         
      const DVector<double>& betas() {
         // Wt = Lambda_*\W
         // zt = Lambda_*\z
         // betas_ = (Wt^T*\Wt)^{-1}*\Wt^T*\zt
         if(is_empty(Lambda_)){
            Lambda_ = Lambda();
         }
         DMatrix<double> W = m_.X();
         //Eigen::PartialPivLU<DMatrix<double>> WLW_dec; 
         //WLW_dec.compute(W.transpose()*Lambda_*(W));  
         //betas_ = WLW_dec.solve(W.transpose()*Lambda_*(m_.y()));
         DMatrix<double> invWtW = inverseS(W.transpose() * Lambda_ * (W));      
         betas_ = invWtW * W.transpose() * Lambda_ * (m_.y());
            
            /* implementazione alternativa..
   
         DMatrix<double> Wtilde_ = Lambda_ * m_.X();
         DMatrix<double> ytilde_ = Lambda_ * m_.y();
         DMatrix<double> temp = inverseS(Wtilde_.transpose() * Wtilde_);
         betas_ = temp * Wtilde_.transpose() * ytilde_;
         //std::cout<<"questi sono i beta s che vengono calcolati dentro alla funzione: "<<betas_<<std::endl;
         */
         return betas_;

      }

      DMatrix<double>& Vs() {
         if(is_empty(Lambda_)){
            Lambda_ = Lambda();
         }
         DMatrix<double> W = m_.X();
         DMatrix<double> invWtW = inverseS(W.transpose() * Lambda_ * (W));
         // get the residuals needed
         DVector<double> eps_hat = (m_.y() - m_.fitted());
         // compute squared residuals
         DVector<double> Res2 = eps_hat.array() * eps_hat.array();
            
         // resize the variance-covariance matrix
         // qua non doverbbe essere m_.q() al posto di due
         int q = 2;
         Vs_.resize(q,q);        
            
         DMatrix<double> W_t = W.transpose();
            
         DMatrix<double> diag = Res2.asDiagonal();
            
         //Vs_ = (WLW_dec).solve((W_t)*Lambda_*Res2.asDiagonal()*Lambda_*(W)*(WLW_dec).solve(DMatrix<double>::Identity(q,q)));
         Vs_ = invWtW * (W_t) * Lambda_ * Res2.asDiagonal() * Lambda_ * (W) * invWtW;
            // set U = Wt^T*\W
            // set E = epsilon*\epsilon^T
            // Vs = U^{-1}*\Wt^T*\Lambda*\E*\Lambda^T*\Wt*U^{-1}
            //DMatrix<double> epsilon_ = m_.y() - m_.fitted();
            //DMatrix<double> U=m_.X().transpose()*m_.X();
            //DMatrix<double> E=epsilon_*epsilon_.transpose();
            // Vs_ = inverse(U)*Wt_.transpose()*Lambda_*E*Lambda_.transpose()*Wt_*inverse(U);

            /* implementazione che non va.. 
            DMatrix<double> Wt_ = Lambda_ * m_.X();
            //DMatrix<double> U_ = Wtilde_.transpose() * Wtilde_; // symmetric
            //DMatrix<double> invU_ = inverse(U_); 
            DMatrix<double> left_ = inverseS(Wt_.transpose() * Wt_);
            DMatrix<double> epsilon_ = m_.y() - m_.fitted();
            Vs_ = left_ * Wt_.transpose() * Lambda_ * (epsilon_ * epsilon_.transpose()) * Lambda_.transpose() * Wt_ * left_;
            */
         return Vs_;
      }

      DMatrix<double> computeCI(CIType type){ 
         fdapde_assert(!is_empty(C_)) ;     // throw an exception if condition is not met  
         
         if(alpha_ == 0) {
            setAlpha(0.05);         // default value 5%
         }
         
         if(is_empty(Vs_)){
            Vs();
         }

         if(is_empty(betas_)){
            betas();
         }
            
         int p = C_.rows();
         // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
         // e che sulle righe della matrice di siano c1, c2, c3...
         // devi capire quale è il meodo più veloce facendoti restituire il tempo di esecuzione
         // metodo con libreria eigen 
         DMatrix<double> CVCdiag_ = ((C_ * Vs()) * C_.transpose()).diagonal();
         //metodo con ciclo for per caclolare solo la diagonale e non tutta la matrice 
         int size = std::min(C_.rows(), Vs().rows()); // questo lo sai a priori quindi sostituisci con la teoria  
         DVector<double> diagon(size);
         for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector<double> ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagon[i] = ci.transpose() * Vs() * ci;
         }
         DVector<double> lowerBound;
         DVector<double> upperBound;


         if(type == simultaneous){ 
         // SIMULTANEOUS
            double quantile = inverseChiSquaredCDF(0.95, p);
            
            lowerBound = (C_ * betas_).array() - (quantile * diagon.array()).sqrt();
            upperBound = (C_ * betas_).array() + (quantile * diagon.array()).sqrt();
         
         }

         else if (type == bonferroni){
         // BONFERRONI
            double quantile = normal_standard_quantile(1-alpha_/(2*p));
            
            lowerBound = (C_ * betas_).array() - quantile * (diagon.array()).sqrt();
            upperBound = (C_ * betas_).array() + quantile * (diagon.array()).sqrt();

         }

         else if (type == one_at_the_time){
            // ONE AT THE TIME
            double quantile = normal_standard_quantile(1-alpha_/2);
            
            lowerBound = (C_ * betas_).array() - quantile * (diagon.array()).sqrt();
            upperBound = (C_ * betas_).array() + quantile * (diagon.array()).sqrt();

         }

         else{
               // inserire errore: nome intervallo non valido
            return DMatrix<double>::Zero(1, 1);// questo va cambiato ma se non c'è non runna
            
         }

         //costruisco la matrice che restituisce i confidence intervals
         DMatrix<double> CIMatrix(p, 2);
         CIMatrix.col(0) = lowerBound;
         CIMatrix.col(1) = upperBound;        
         
         return CIMatrix;
            
      }

      DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  

         if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            // set beta0 to 0
            setBeta0(DVector<double>::Zero(betas().size()));
         }
         
         if(is_empty(Vs_)){
            Vs_ = Vs();
         }
         
         int p = C_.rows();
         DVector<double> statistics(p);

         if(type == simultaneous){
            // SIMULTANEOUS
            DVector<double> diff = C_ * betas() - beta0_;           
            DMatrix<double> Sigma = C_ * Vs_ * C_.transpose();
            DMatrix<double> Sigmadec_ = inverseS(Sigma);
            double stat = diff.adjoint() * Sigmadec_ * diff;
            
            statistics.resize(p);
            double pvalue = chi_squared_cdf(stat, p);

            if(pvalue < 0){
               statistics(0)=1;
            }
            if(pvalue > 1){
               statistics(0)=0;
            }
            else{
               statistics(0) = 1 - pvalue;
            }

            for(int i = 1; i < C_.rows(); i++){
               statistics(i) = 10e20;
            }
            return statistics; 
         }

         if (type == one_at_the_time){
            // ONE AT THE TIME 
            int p = C_.rows();
            statistics.resize(p);
            for(int i = 0; i < p; i++){
               DVector<double> col = C_.row(i);
               double diff = col.adjoint() * betas() - beta0_[i];
               double sigma = col.adjoint() * Vs_ * col;
               double stat = diff/std::sqrt(sigma);
               double pvalue = 2 * gaussian_cdf(-std::abs(stat), 0, 1);
               if(pvalue < 0){
                  statistics(i) = 0;
               }
               if(pvalue > 1){
                  statistics(i) = 1;
               }
               else{
                  statistics(i) = pvalue;
               }
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
      void setAlpha(double alpha){
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
      DMatrix<double> inverseS(const DMatrix<double> M){
         //Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
         Eigen::PartialPivLU<DMatrix<double>> Mdec_ {};
         Mdec_ = M.partialPivLu();
         return Mdec_.solve(DMatrix<double>::Identity(M.rows(), M.cols()));
      }


}; // end class
} // end fdapde models
} // end fdapde namespace

#endif  //__SPECKMAN_H__

