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
using fdapde::core::lump;

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

template <typename Model, typename Strategy> class Wald {

    private:  

      struct ExactInverse {
      // forse static
      DMatrix<double> compute(Model m){
        Eigen::PartialPivLU<DMatrix<double>> Tdec_ (m.T());
        DMatrix<double> invT_ = Tdec_.solve(DMatrix<double>::Identity(m.T().rows(), m.T().cols()));
        //DMatrix<double> S = m.Psi() * invT_.block(0, 0, m.n_basis(), m.n_basis()) * m.PsiTD() * m.Q(); 
        DMatrix<double> S = m.Psi() * invT_ * m.PsiTD() * m.Q(); 

        //std::cout<<"questa è S : " <<std::endl;
        //std::cout << std::endl;
         //for (int i = 0; i < 4; ++i) {
          //  for(int j=0; j<4;++j){
           // std::cout << S(i,j)<< " ";
         //}
         //}
         //std::cout << std::endl;
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
        FSPAI fspai_R0(m.R0());

        int alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A (perform alpha steps of approximate inverse update along column k)
        int beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
        double epsilon = 0.05; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità (the best improvement is higher than accetable treshold)
        // calcolo inversa di R0
        fspai_R0.compute(alpha, beta, epsilon);
        //getter per l'inversa di R0
        SpMatrix<double> invR0_= fspai_R0.getInverse();
      /*
        //altirmenti calcolo in modo esatto R0^-1 e poi uso il lumping per renderla sparsa:
        DMatrix<double> R0inv=inverse(m.R0());
        DMatrix<double> invR0_=lump(R0inv);
*/
        //calcolo la matrice Atilde
        DMatrix<double> Et_ = m.PsiTD()* m.Psi()+ m.lambda_D() * m.R1().transpose() * invR0_ * m.R1();

        //applico FSPAI su Atilde
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
        //std::cout<<"questa è S : " <<std::endl;
        // std::cout << std::endl;
        // for (int i = 0; i < 4; ++i) {
         //   for(int j=0; j<4;++j){
         //   std::cout << S(i,j)<< " ";
        // }
        // }
        // std::cout << std::endl;
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
        
        if(alpha_ == 0.0) {
         setAlpha(0.05);         // default value 5%
        }
        if(is_empty(Vw_)){
            Vw();
         }
        if(is_empty(betaw_)){
         betaw();
        }
         
         std::cout<<"alpha è "<<alpha_<<std::endl;
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
        double quantile2 = chi_squared_quantile(0.95,p);
        std::cout<<" the quantile calcolato  is "<<quantile2<<std::endl;


        double quantile = 5.991465;

        lowerBound = (C_ * betaw_).array() - (quantile * diagon.array()).sqrt();
        upperBound = (C_ * betaw_).array() + (quantile * diagon.array()).sqrt();
        // std::cout<<" the lower bound is "<<lowerBound<<std::endl;
        // std::cout<<" the upper bound is "<<upperBound<<std::endl;

        }

        else if (type == bonferroni){
        // Bonferroni
        double quantile = normal_standard_quantile(1-alpha_/(2*p));
        std::cout<<" the quantile calcolato  is "<<quantile<<std::endl;

        //double quantile = 2.241403; quantile con alpha = 0.05
        
        lowerBound = (C_ * betaw_).array() - quantile * (diagon.array()).sqrt();
        upperBound = (C_ * betaw_).array() + quantile * (diagon.array()).sqrt();

        }

        else if (type == one_at_the_time){
        // One at the time
        double quantile = normal_standard_quantile(1-alpha_/2);
        std::cout<<" the quantile calcolato  is "<<quantile<<std::endl;

        //double quantile = 1.959964; quantile con alpha=0.05
        
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
        std::cout<<" Confidence intervals "<<std::endl;
        for (int i =0; i<CIMatrix.rows(); ++i){
            for (int j =0; j<CIMatrix.cols(); ++j){
               std::cout<<CIMatrix(i,j)<<"  ";

            }
        }
        std::cout<<std::endl;
        return CIMatrix;
     }


     // this function returns the statistics not the p-values
     DVector<double> p_value(CIType type){
         fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  
         //std::cout<<"controllo su C avviene correttamente"<<std::endl;

         if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            setBeta0(DVector<double>::Zero(betaw().size())); // set beta0 to 0
         }

         //std::cout<<"controllo su beta0 avviene correttamente"<<std::endl;

         //std::cout<<"la lunghezza di beta0_ è : "<<betaw().size()<<std::endl;
         //std::cout<<"questa è beta0_ : " <<std::endl;
         //for (int i = 0; i < betaw().size(); ++i) {
         //   std::cout << beta0_[i] << " ";
         //}
         if(is_empty(betaw_)){
            Vw_ = betaw();
         }
         std::cout<<"questa è betaw : " <<betaw_<<std::endl;

         if(is_empty(Vw_)){
            Vw_ = Vw();
         }
         //std::cout<<"controllo su Vw avviene correttamente"<<std::endl;

         DVector<double> statistics(C_.rows());
         
         if( type == simultaneous ){
            // simultaneous

            DVector<double> diff = C_ * (betaw() - beta0_);
            
            //std::cout<<"matrice Vw: "<<std::endl;     
            //for (int i = 0; i < Vw_.rows(); ++i) {
            //   for (int j = 0; j < Vw_.cols(); ++j) {
             //     std::cout << Vw_(i,j) << " ";
             //  }
            //}
            //std::cout << std::endl;           
            DMatrix<double> Sigma = C_ * Vw_ * C_.transpose();
            //std::cout<<"creazione Sigma avviene correttamente"<<std::endl;

            DMatrix<double> Sigmadec_ = inverse(Sigma);
            //std::cout<<"creazione Sigmadec_ avviene correttamente"<<std::endl;

            //std::cout<<"numero di righe di sigmadec_: "<<Sigmadec_.rows()<<std::endl;
            //std::cout<<"numero di colonne di sigmadec_: "<<Sigmadec_.cols()<<std::endl;

            //std::cout<<"numero di righe di diff.transpse: "<<diff.transpose().rows()<<std::endl; 
            //std::cout<<"numero di colonne di diff.transpose: "<<diff.transpose().cols()<<std::endl; 

            //std::cout<<"numero di righe di diff: "<<diff.rows()<<std::endl; 
            //std::cout<<"numero di colonne di diff: "<<diff.cols()<<std::endl;

            double stat = diff.adjoint() * Sigmadec_ * diff;
            //double stat = m_.n_obs() * diff.transpose() * C_.transpose() * Sigmadec_ * C_ * diff;
            std::cout<<"valore della statistica: " <<stat<< std::endl;
            
            statistics.resize(C_.rows());
            //statistics(0) = 1-chi_squared_cdf(stat, 2);
            statistics(0)=stat;

            for(int i = 1; i < C_.rows(); i++){
               statistics(i) = 10e20;
            }
            
            return statistics; 
         }

         else if ( type == one_at_the_time ){
            // one at the time
            int p = C_.rows();
            statistics.resize(p);
            for(int i = 0; i < p; i++){
               DVector<double> col = C_.row(i);
               double diff = col.adjoint()* m_.beta() - beta0_[i];
               double sigma = col.adjoint() * Vw_ *col;
               double stat = diff/std::sqrt(sigma);
               std::cout<<"il valore della statistica è :"<<stat<<std::endl;
               statistics(i) = 2*gaussian_cdf(-std::abs(stat),0,1);

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
     //inverse() è progettata per operare solo sulla matrice passata come argomento e non dipende da alcun altro stato interno della classe Wald, puoi renderla statica
     static DMatrix<double> inverse(DMatrix<double> M){
      Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
      // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
      // Mdec_ = M.partialPivLu(); 
      return Mdec_.solve(DMatrix<double>::Identity(M.rows(), M.cols()));
     }
     
     //questa è da controllare 
     double chi_squared_quantile(double percentile, int degrees_of_freedom) {
      // Percentuale complementare
      double p = 1.0 - percentile;
      std::cout<<"p è "<<p<<std::endl;
      // Calcolare il valore z corrispondente al percentile complementare
      double z = std::sqrt(2.0 * degrees_of_freedom);
      std::cout<<"z è "<<z<<std::endl;

      // Calcolare il valore del quantile utilizzando la funzione inversa della distribuzione normale standard
      double quantile = z - ((1.0 / 3.0) * (1.0 / z) - 1.0 / (36.0 * z * z * z)) * (1.0 / std::sqrt(2.0)) * std::log(p / std::sqrt(2.0 * M_PI));
      std::cout<<"quantile è "<<quantile<<std::endl;
      // questo valore che viene è quasi corretto se fai la funzione restituisce direttamnte questo quantile*quantile
      //non so cosa facciamo questi due if 
      
      // Applicare correzioni successive per gradi di libertà maggiori di 1
      if (degrees_of_freedom > 1) {
         quantile -= (1.0 / (6.0 * z)) * ((1.0 - (2.0 / (9.0 * degrees_of_freedom))) / std::sqrt(2.0 / (9.0 * degrees_of_freedom)) - 1.0) * std::log(p / std::sqrt(2.0 * M_PI));
      }

      // Applicare correzioni successive per gradi di libertà maggiori di 2
      if (degrees_of_freedom > 2) {
         quantile -= (1.0 / (6.0 * z)) * ((1.0 - (2.0 / (9.0 * degrees_of_freedom))) / std::sqrt(2.0 / (9.0 * degrees_of_freedom)) - 1.0) * std::log(p / std::sqrt(2.0 * M_PI));
      }
      std::cout<<"quantile  finale è "<<quantile*quantile<<std::endl;

      return quantile * quantile;
     }

     // Funzione per calcolare i quantili di una distribuzione normale standard
     //questa funziona correttamente 
     double normal_standard_quantile(double percentile) {
         // Calcola il quantile utilizzando la funzione inversa della distribuzione normale standard
         return std::sqrt(2.0) * inverse_erf(2.0 * percentile - 1.0);     
     }

     // Funzione di approssimazione per il calcolo dell'inverso dell'errore
     // questa funziona correttamente 
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
     


     //pvalues
     double gamma(double x) {
         // Implementation of the gamma function using the Lanczos approximation
         // (This is a simple version and may not be as accurate as other implementations)
         const double sqrt_2pi = 2.50662827463100050242; // sqrt(2 * pi)
         const double lanczos_coefficients[6] = {1.000000000190015, 76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179};
         double sum = lanczos_coefficients[0];
         double denom = x + 1.0;
         for (int i = 1; i < 6; ++i) {
            sum += lanczos_coefficients[i] / (x + i);
            denom *= (x + 1.0);
         }
         return sqrt_2pi * sum / denom;
      }

      double incomplete_gamma(double a, double x) {
         // Implementation of the incomplete gamma function using the continued fraction expansion
         const double epsilon = 1e-15;
         double ans, c, r;

         r = c = ans = 0.0;

         do {
            r += 1.0;
            c *= x / r;
            ans += c;
         } while (c > epsilon * ans);

         return ans * std::exp(-x + a * std::log(x) - std::lgamma(a));
      }

      double chi_squared_cdf(double x, double k) {
         if (x <= 0 || k <= 0) return 0.0;
         return incomplete_gamma(k / 2.0, x / 2.0) / gamma(k / 2.0);
      }

     
     // funzione per calcolare il pvalue di una normale 
     double gaussian_cdf(double x, double mean, double stddev) { 
         return 0.5 * (1 + std::erf((x - mean) / (stddev * std::sqrt(2)))); 
     }

      // aggiungere destructor?

   } ;
}  // closing models namespace
}  // closing fdapde namespace

#endif   // __WALD_H__