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

#ifndef __WALD_H__
#define __WALD_H__

// questi sono da controllare 
#include <cmath>
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
using fdapde::core::FSPAI;
using fdapde::core::lump;

#include "../model_macros.h"
#include "../model_traits.h"
#include "../model_base.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "inference.h"

// #include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <random>

namespace fdapde {
namespace models {

template <typename Model, typename Strategy> class Wald {

   private:  

      struct ExactInverse {
      // forse static
         DMatrix<double> compute(Model m){
            Eigen::PartialPivLU<DMatrix<double>> Tdec_ (m.T());
            DMatrix<double> invT_ = Tdec_.solve(DMatrix<double>::Identity(m.T().rows(), m.T().cols()));
            //DMatrix<double> S = m.Psi() * invT_.block(0, 0, m.n_basis(), m.n_basis()) * m.PsiTD() * m.Q(); 
            DMatrix<double> S = m.Psi() * invT_ * m.PsiTD() * m.Q(); 
            return S;
         }
      };

      struct NonExactInverse{
      // forse static
         DMatrix<double> compute(Model m){
            // FSPAI approx
            // E_tilde = Psi^T*\Psi+lambda*\R
            // making E_tilde sparse
            // Dalla Sub Woodbury decomposition
            // bisogna fare prima un'approssimazione dell'inversa di R0, usando FSPAI
            // R0 should be stored as a sparse matrix
            //FSPAI fspai_R0(m.R0());

            //int alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A (perform alpha steps of approximate inverse update along column k)
            //int beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
            //double epsilon = 0.05; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità (the best improvement is higher than accetable treshold)
            // calcolo inversa di R0
            //fspai_R0.compute(alpha, beta, epsilon);
            //getter per l'inversa di R0
           
            SpMatrix<double> decR0_ = lump(m.R0());  

            // fare con 1/R0_ii
            DiagMatrix<double> invR0_(decR0_.rows());
            invR0_.setZero(); // Imposta tutti gli elementi a zero

            for (int i = 0; i < decR0_.rows(); ++i) {
               double diagElement = decR0_.diagonal()[i]; // Ottieni l'elemento diagonale
               invR0_.diagonal()[i] = 1.0 / diagElement; 
            }


            //Eigen::SparseLU<SpMatrix<double>> sol;
            //sol.analyzePattern(decR0_);
            //sol.factorize(decR0_);
            //SpMatrix<double> id(m.R0().rows(), m.R0().cols());
            //id.setIdentity();
            //SpMatrix<double> invR0_ = sol.solve(id);
            
            //altrimenti calcolo in modo esatto R0^-1 e poi uso il lumping per renderla sparsa:
            //DMatrix<double> R0inv=inverse(m.R0());
            //DMatrix<double> invR0_=lump(R0inv);
            //std::cout<<"questa è invR0 : " <<std::endl;
            //for (int i = 0; i < 4; ++i) {
            //    for(int j=0; j<4;++j){
            //    std::cout << invR0_(i,j)<< " ";
            //}
            //}
            //std::cout << std::endl;

            //calcolo la matrice Atilde
            DMatrix<double> Et_ = m.PsiTD()* m.Psi()+ m.lambda_D() * m.R1().transpose() * invR0_ * m.R1();

            //applico FSPAI su Atilde
            int alpha = 20;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A (perform alpha steps of approximate inverse update along column k)
            int beta = 7;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
            double epsilon = 0.05; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità (the best improvement is higher than accetable treshold)

            //Et_ should be stored as a sparse matrix 
            Eigen::SparseMatrix<double> Et_sparse = Et_.sparseView();
            FSPAI fspai_E(Et_sparse);
            fspai_E.compute(alpha, beta, epsilon);
            SpMatrix<double> invE_ = fspai_E.getInverse();

            // Mt^{-1} = Et^{-1} + Et{-1}*\Ut*\(Ct+Vt*\Et^{-1}*\Ut)^{-1}*\Vt*\Et^{-1}
            // Ut = Psi^T*\W    Nt x q
            // Ct = -(W^T*\W)^{-1}   q x q
            // Vt = W^T*\Psi   q x Nt
            // DMatrix<double> Ut_ = m_.PsiTD() * m_.W();
            DMatrix<double> Ut_ = m.Psi().transpose() * m.X();
            DMatrix<double> Ct_ = - inverse(m.X().transpose() * m.X());
            DMatrix<double> Vt_ = m.X().transpose() * m.Psi();

            SpMatrix<double> invMt_ = invE_ + invE_ * Ut_ * inverse(Ct_ + Vt_ * invE_ * Ut_) * Vt_ * invE_;
            // m_.Psi().transpose() or m_.PsiTD()
            DMatrix<double> S = m.Psi() * invMt_ * m.PsiTD() * m.Q();
            /*
            std::cout<<"questa è S : " <<std::endl;
            std::cout << std::endl;
               for (int i = 0; i < 4; ++i) {
                  for(int j=0; j<4;++j){
                  std::cout << S(i,j)<< " ";
               }
            }
            std::cout << std::endl;*/
            return S;
         }
      };
     
      using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
      Solver s_;   

      Model m_; 

      // DMatrix<double> S_ {};            // smoothing matrix S (n x n) matrix
      DMatrix<double> Vw_ {};           // variance matrix of betaw_ (q x q) matrix
      DMatrix<double> C_ {};            // inference matrix C (p x q) matrix      
      DVector<double> betaw_ {};        // sol of srpde ( q x 1 ) matrix
      DVector<double> beta0_ {};        // inference hypothesis H0 (p x 1) matrix
      double alpha_ = 0;                   // level of the confidence intervals

      // variabili da aggungere: 
      // una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

   public:
     
      Wald() = default;             // deafult constructor
      Wald(const Model& m): m_(m) {};     // starting the constructor


      // si potrebbe fare override anche di questo metodo visto che si può utilizzare StochasticEDF per calcolare la traccia
      double sigma_sq() {
         double sigma_sq_ = 0;             // sigma^2 
         DMatrix<double> epsilon = m_.y() - m_.fitted();

         ExactEDF strat;
         strat.set_model(m_);
         sigma_sq_  = (1/(m_.n_obs() - m_.q() - strat.compute())) * epsilon.squaredNorm();
         return sigma_sq_;
      }

      const DMatrix<double>& Vw() {
         DMatrix<double> invSigma_ = inverse(m_.X().transpose() * m_.X());
         DMatrix<double> ss = s_.compute(m_) * s_.compute(m_).transpose();
         DMatrix<double> left = invSigma_ * m_.X().transpose();
         Vw_ = sigma_sq() * (invSigma_ + left * ss * left.transpose()); 
         std::cout<<"il valore di V è : "<<Vw_<<std::endl;
         return Vw_;
      }

      const DVector<double>& betaw() {
         betaw_ = m_.beta();
         return betaw_;
      }

      DMatrix<double> computeCI(CIType type){ 
         fdapde_assert(!is_empty(C_));     // throw an exception if condition is not met  
         
         if(alpha_ == 0.0) {
            setAlpha(0.05);         // default value 5%
         }
         if(is_empty(Vw_)){
            Vw();
         }
         if(is_empty(betaw_)){
            betaw();
         }
         
         int p = C_.rows();

         // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente e 
         //che sulle righe della matrice di siano c1, c2, c3...
         
         // devi capire quale è il metodo più veloce facendoti restituire il tempo di esecuzione
         // 1) metodo con libreria eigen 
         //DMatrix<double> CVCdiag_ = ((C_ * Vw_) * C_.transpose()).diagonal();
         // 2) metodo con ciclo for per calcolare solo la diagonale e non tutta la matrice 
         int size = std::min(C_.rows(), Vw_.rows());
         DVector<double> diagon(size);
         for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector<double> ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagon[i] = ci.transpose() * Vw_* ci;
         }

         DVector<double> lowerBound(size);
         DVector<double> upperBound(size);

         if(type == simultaneous){ 
            // SIMULTANEOUS
            double quantile = inverseChiSquaredCDF(1-alpha_, p);

            lowerBound = (C_ * betaw_).array() - (quantile * diagon.array()).sqrt();
            upperBound = (C_ * betaw_).array() + (quantile * diagon.array()).sqrt();
         
         }

         else if (type == bonferroni){
            // BONFERRONI
            double quantile = normal_standard_quantile(1-alpha_/(2*p));
            
            lowerBound = (C_ * betaw_).array() - quantile * (diagon.array()).sqrt();
            upperBound = (C_ * betaw_).array() + quantile * (diagon.array()).sqrt();

         }

         else if (type == one_at_the_time){
            // ONE AT THE TIME
            double quantile = normal_standard_quantile(1-alpha_/2);
            
            lowerBound = (C_ * betaw_).array() - quantile * (diagon.array()).sqrt();
            upperBound = (C_ * betaw_).array() + quantile * (diagon.array()).sqrt();

         }

         else{
            // inserire errore: nome intervallo non valido
            return DMatrix<double>::Zero(1, 1);
         }

         DMatrix<double> CIMatrix(p, 2);      //matrix of confidence intervals
         CIMatrix.col(0) = lowerBound;
         CIMatrix.col(1) = upperBound;

         return CIMatrix;
      }


      DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  

         if(is_empty(beta0_)){
            setBeta0(DVector<double>::Zero(betaw().size())); // set beta0 to 0
         }

         if(is_empty(betaw_)){
            betaw_ = betaw();
         }

         if(is_empty(Vw_)){
            Vw_ = Vw();
         }

         int p = C_.rows();
         DVector<double> statistics(p);
         
         if(type == simultaneous){
            // SIMULTANEOUS 
            DVector<double> diff = C_ * betaw() - beta0_;          
            DMatrix<double> Sigma = C_ * Vw_ * C_.transpose();
            DMatrix<double> Sigmadec_ = inverse(Sigma);
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
            std::cout<<"il valore del pvalue è "<<statistics<<std::endl;

            return statistics; 
         }

         else if (type == one_at_the_time){
            //  ONE AT THE TIME 
            int p = C_.rows();
            statistics.resize(p);
            for(int i = 0; i < p; i++){
               DVector<double> col = C_.row(i);
               double diff = col.adjoint()* m_.beta() - beta0_[i];
               double sigma = col.adjoint() * Vw_ *col;
               double stat = diff/std::sqrt(sigma);
               std::cout<<"il valore della stat è "<<stat<<std::endl;
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
      // inverse() è progettata per operare solo sulla matrice passata come argomento e non dipende da alcun altro stato interno della classe Wald, puoi renderla statica
      static DMatrix<double> inverse(DMatrix<double> M){
         Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
         // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
         // Mdec_ = M.partialPivLu(); 
         return Mdec_.solve(DMatrix<double>::Identity(M.rows(), M.cols()));
      }

}; // closing class 
}  // closing models namespace
}  // closing fdapde namespace

#endif   // __WALD_H__