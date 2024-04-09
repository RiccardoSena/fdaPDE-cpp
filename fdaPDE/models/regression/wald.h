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
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
using fdapde::core::FSPAI;

#include "../model_macros.h"
#include "../model_traits.h"
#include "../model_base.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"

// #include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <random>

namespace fdapde {
namespace models {

struct exact {};

enum CIType {bonferroni,simultaneous,one_at_the_time};

template <typename Model, typename Strategy> class Wald {

    private:  

      struct ExactInverse {
      // forse static
      DMatrix<double> compute(Model m){
        Eigen::PartialPivLU<DMatrix<double>> Tdec_ {};
        Tdec_ = m.T().partialPivLu(); 
        DMatrix<double> invT_ = Tdec_.solve(DMatrix<double>::Identity(m.T().rows(), m.T().cols()));
        DMatrix<double> S = m.Psi() * invT_ * m.PsiTD() * m.Q(); 
        return S;
        }
      };

     struct NonExactInverse{
     // forse static
      DMatrix<double> compute(const Model& m){
        // FSPAI approx
        // E_tilde = Psi^T*\Psi+lambda*\R
        // making E_tilde sparse
        // Dalla Sub Woodbury decomposition
        // bisogna fare prima un'approssimazione dell'inversa di R0, usando FSPAI
        // R0 should be stored as a sparse matrix
        FSPAI fspai_R0(m.R0());

        int alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A (perform alpha steps of approximate inverse update along column k)
        int beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
        double epsilon = 0.001; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità (the best improvement is higher than accetable treshold)
        // calcolo inversa di R0
        fspai_R0.compute(alpha, beta, epsilon);
        //getter per l'inversa di R0
        SpMatrix<double> invR0_= fspai_R0.getInverse();

        //calcolo la matrice Atilde
        DMatrix<double> Et_ = m.PsiTD()* m.Psi()+ m.lambda_D() * m.R1().transpose() * invR0_ * m.R1();

        //applico FSPAI su Atilde
        FSPAI fspai_E(Et_);
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
     int alpha_ = 0;                   // level of the confidence intervals

     // variabili da aggungere: 
     // una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

    public:
     
     Wald() = default;             // deafult constructor
     Wald(const Model& m): m_(m) {};     // starting the constructor


     // si potrebbe fare override anche di questo metodo visto che si può utilizzare StochasticEDF per calcolare la traccia
     double sigma_sq() {

        // double q = m_.q();            // number of covariates
        // std::size_t n = m_.n_obs();   // number of observations
        // double dor = n - (q + trS);       // residual degrees of freedom
        
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

        return Vw_;
     }

     const DVector<double>& betaw() {
      // Is betaw just the beta from the Model??? 
      //In that case could not store betaw_ but use directly m_.beta()
      betaw_ = m_.beta();
      return betaw_;
     }

     DMatrix<double> computeCI(CIType type){ 
        fdapde_assert(!is_empty(C_));     // throw an exception if condition is not met  
        
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
        // 2) metodo con ciclo for per calcolare solo la diagonale e non tutta la matrice 
	     int size = std::min(C_.rows(), Vw_.rows());
	     DVector<double> diagon(size);
        for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector<double> ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagon[i] = ci.transpose() * Vw_* ci;
        }

        DVector<double> lowerBound;
        DVector<double> upperBound;

        if(type == simultaneous){ 
        // SIMULTANEOUS
        //boost::math::chi_squared_distribution<double> chi_squared(p);
        // double quantile = boost::math::quantile(chi_squared, alpha_);
        double quantile = chi_squared_quantile(p, 1-alpha_);

        //double quantile = 1;
        
        lowerBound = (C_ * betaw_).array() - (quantile * diagon.array() / m_.n_obs()).sqrt();
        upperBound = (C_ * betaw_).array() + (quantile * diagon.array() / m_.n_obs()).sqrt();

        }

        else if (type == bonferroni){
        // Bonferroni
        //double quantile = std::sqrt(2.0) * boost::math::erf_inv(1-alpha_/(2*p));
         double quantile = normal_standard_quantile(1-alpha_/(2*p));

        //double quantile = 1;
        
        lowerBound = (C_ * betaw_).array() - quantile * (diagon.array() / m_.n_obs()).sqrt();
        upperBound = (C_ * betaw_).array() + quantile * (diagon.array() / m_.n_obs()).sqrt();

        }

        else if (type == one_at_the_time){
        // One at the time
        //double quantile = std::sqrt(2.0) * boost::math::erf_inv(1-alpha_/2);
        double quantile = normal_standard_quantile(1-alpha_/2);

        //double quantile = 1;
        
        lowerBound = (C_ * betaw_).array() - quantile * (diagon.array() / m_.n_obs()).sqrt();
        upperBound = (C_ * betaw_).array() + quantile * (diagon.array() / m_.n_obs()).sqrt();

        }

        else{
         // inserire errore: nome intervallo non valido
         
        }

        DMatrix<double> CIMatrix(p, 2);      //matrix of confidence intervals
        CIMatrix.col(0) = lowerBound;
        CIMatrix.col(1) = upperBound;
        

        return CIMatrix;
        }

        return DMatrix<double>::Zero(1, 1);// questo va cambiato ma se non c'è non runna
     }
     


     // this function returns the statistics not the p-values
     // come hanno fatto gli altri nel report 
     DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  
         std::cout<<"controllo su C avviene correttamente"<<std::endl;

         // is_empty va bene anche per i Vectors?
         if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            // set beta0 to 0
            setBeta0(DVector<double>::Zero(betaw().size()));
         }

         std::cout<<"controllo su beta0 avviene correttamente"<<std::endl;

         std::cout<<"la lunghezza di beta0_ è : "<<betaw().size()<<std::endl;
         std::cout<<"questa è beta0_ : " <<std::endl;
         for (int i = 0; i < betaw().size(); ++i) {
            std::cout << beta0_[i] << " ";
         }

         std::cout<<"questa è betaw : " <<std::endl;
         std::cout << std::endl;
         for (int i = 0; i < betaw().size(); ++i) {
            std::cout << betaw()[i] << " ";
         }
         std::cout << std::endl;

         if(is_empty(Vw_)){
            Vw_ = Vw();
         }
         std::cout<<"controllo su Vw avviene correttamente"<<std::endl;

         DVector<double> statistics(C_.rows());
         // simultaneous 
         if( type == simultaneous ){
            std::cout<<"riesce ad entrare nell'if giusto"<<std::endl;

            std::cout << std::endl;
            for (int i = 0; i < betaw().size(); ++i) {
               std::cout << C_(0,i) << " ";
            }
            std::cout << std::endl;
            // Ottenere le dimensioni di C_
            std::cout<<"numero di righe di C_: "<<C_.rows()<<std::endl;
            std::cout<<"numero di colonne di C_: "<<C_.cols()<<std::endl;

            // Ottenere le dimensioni di m_.beta()
            std::cout<<"numero di righe di betaw: "<<betaw().rows()<<std::endl; 
            std::cout<<"numero di colonne di betaw: "<<betaw().cols()<<std::endl; 

            //C_ * betaw() - beta0_;
            //std::cout<<"la moltiplicazione non è il rpoblema"<<std::endl;

            //DVector<double> diff = C_ * betaw() - beta0_;
            //std::cout<<"creazione diff avviene correttamente"<<std::endl;
            
            DVector<double> diff(1);
            diff << 0.89;
            std::cout << std::endl;
            for (int i = 0; i < Vw_.rows(); ++i) {
               for (int j = 0; j < Vw_.cols(); ++j) {
                  std::cout << Vw_(i,j) << " ";
            }
            std::cout << std::endl;           
            DMatrix<double> Sigma = C_ * Vw_ * C_.transpose();
            std::cout<<"creazione Sigma avviene correttamente"<<std::endl;

            DMatrix<double> Sigmadec_ = inverse(Sigma);
            std::cout<<"creazione Sigmadec_ avviene correttamente"<<std::endl;

            std::cout<<"numero di righe di sigmadec_: "<<Sigmadec_.rows()<<std::endl;
            std::cout<<"numero di colonne di sigmadec_: "<<Sigmadec_.cols()<<std::endl;

            std::cout<<"numero di righe di diff.adj: "<<diff.adjoint().rows()<<std::endl; 
            std::cout<<"numero di colonne di diff.adj: "<<diff.adjoint().cols()<<std::endl; 

            std::cout<<"numero di righe di diff: "<<diff.rows()<<std::endl; 
            std::cout<<"numero di colonne di diff: "<<diff.cols()<<std::endl;

            double stat = diff.adjoint() * Sigmadec_ * diff;
            std::cout<<"creazione stat avviene correttamente"<<std::endl;


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
      // funziona così per i Eigen::Vector??
      beta0_ = beta0;
     }

     // funzione ausiliare per invertire una matrice densa in maniera efficiente
     DMatrix<double> inverse(DMatrix<double> M){
      // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
      Eigen::PartialPivLU<DMatrix<double>> Mdec_ {};
      Mdec_ = M.partialPivLu(); 
      DMatrix<double> invM_ = Mdec_.solve(DMatrix<double>::Identity(M.rows(), M.cols()));
      return invM_;
     }
     
     //questa è da controllare 
     double chi_squared_quantile(double percentile, int degrees_of_freedom) {
      // Percentuale complementare
      double p = 1.0 - percentile;

      // Calcolare il valore z corrispondente al percentile complementare
      double z = std::sqrt(2.0 * degrees_of_freedom);

      // Calcolare il valore del quantile utilizzando la funzione inversa della distribuzione normale standard
      double quantile = z - ((1.0 / 3.0) * (1.0 / z) - 1.0 / (36.0 * z * z * z)) * (1.0 / std::sqrt(2.0)) * std::log(p / std::sqrt(2.0 * M_PI));
      
      // Applicare correzioni successive per gradi di libertà maggiori di 1
      if (degrees_of_freedom > 1) {
         quantile -= (1.0 / (6.0 * z)) * ((1.0 - (2.0 / (9.0 * degrees_of_freedom))) / std::sqrt(2.0 / (9.0 * degrees_of_freedom)) - 1.0) * std::log(p / std::sqrt(2.0 * M_PI));
      }

      // Applicare correzioni successive per gradi di libertà maggiori di 2
      if (degrees_of_freedom > 2) {
         quantile -= (1.0 / (6.0 * z)) * ((1.0 - (2.0 / (9.0 * degrees_of_freedom))) / std::sqrt(2.0 / (9.0 * degrees_of_freedom)) - 1.0) * std::log(p / std::sqrt(2.0 * M_PI));
      }

      return quantile * quantile;
     }

      // questa è da controllare 
     // Funzione per calcolare i quantili di una distribuzione normale standard
     double normal_standard_quantile(double percentile) {
    // Calcola il quantile utilizzando la funzione inversa della distribuzione normale standard
      return std::sqrt(2.0) * inverse_erf(2.0 * percentile - 1.0);     
   }

      //questa è da controllare 
     // Funzione di approssimazione per il calcolo dell'inverso dell'errore
     double inverse_erf(double x) {
      const double epsilon = 1e-10; // Tolleranza per l'approssimazione
      double y = 0.0;
      double delta;
      do {
         delta = (std::erf(y) - x) / (2.0 / std::sqrt(M_PI) * std::exp(-y * y));
         y -= delta;
      } while (std::fabs(delta) > epsilon);
      return y;
   }
      // aggiungere destructor?

} ;
}  // closing models namespace
}  // closing fdapde namespace

#endif   // __WALD_H__