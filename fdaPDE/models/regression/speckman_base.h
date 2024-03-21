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

#ifndef __SPECKMAN_BASE_H__
#define __SPECKMAN_BASE_H__

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

enum CIType {bonferroni, simultaneous, one_at_the_time};

// template <typename Model, typename Strategy> class WaldBase
template <typename Model> class SpeckmanBase {

    protected:
    
     Model* m_;
     
     // do we need to store these matrixes???
     DMatrix<double> Lambda_ {};

     DVector<double> betas_ {};

     DMatrix<double> Vs_ {};

     DMatrix<double> inverseA_ {};

     DMatrix<double> C_ {};

     int alpha_ = 0;


    public:

     // constructors
     SpeckmanBase() = default;
     SpeckmanBase(Model* m):m_(m) {};

     // questo si specializza in speckman exact e non_exact  
     virtual void inverseA() = 0;


     const DMatrix<double>& Lambda() {
        // I - Psi*\(Psi^T*\Psi + lambda*\R)^{-1}*\Psi^T
        // I - Psi*\A^{-1}*\Psi^T
        // A = Psi^T*\Psi + lambda*\R
        // A^{-1} is potentially large and dense, we only need the northwest block
        // m_.invA().block(0, 0, m_.n_basis, m_.n_basis)
        if(is_empty(inverseA_)){
           inverseA();
        }
        Lambda_ = DMatrix<double>::Identity(m_.n_basis, m_.n_basis) - m_.Psi() * inverseA_ * m_.PsiTD();
        return Lambda_;
      }
      
     const DVector<double>& betas() {
        // Wt = Lambda_*\W
        // zt = Lambda_*\z
        // betas_ = (Wt^T*\Wt)^{-1}*\Wt^T*\zt

        DMatrix<double> Wtilde_ = Lambda() * m_.W();
        DMatrix<double> ytilde_ = Lambda() * m_.y();

        // ????
        // Maybe for temp is ok to do .inverse()
        //DMatrix<double> temp = (Wtilde_.transpose() * Wtilde_).partialPivLU().solve(DMatrix<double>::Identity(,));
        DMatrix<double> temp = inverse(Wtilde_.transpose() * Wtilde_);
        betas_ = temp * Wtilde_.transpose() * ytilde_;
        return betas_;

     }

     DMatrix<double>& Vs() {
        // set U = Wt^T*\W
        // set E = epsilon*\epsilon^T
        // Vs = U^{-1}*\Wt^T*\Lambda*\E*\Lambda^T*\Wt*U^{-1}
        DMatrix<double> Wtilde_ = Lambda() * m_.W();
        DMatrix<double> U_ = Wtilde_.transpose() * Wtilde_;
        DMatrix<double> invU_ = inverse(U_);
        DMatrix<double> epsilon_ = m_.y() - m_.fitted();
        DMatrix<double> E = epsilon_ * epsilon_.transpose();

        Vs_ = invU_ * Wtilde_.tanspose() * Lambda() * E * Lambda().transpose() * Wtilde_ * invU_;
        return Vs_;
     }

      DMatrix<double> computeCI(CIType type){ 
         // need to set C first
         if(is_empty(C_)){
         // print an error (need to set C)
         }  
         else if(alpha_ == 0){
          // print error: "alpha missing"
         }
         else{
         
         int p = C_.rows();
         // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
         // e che sulle righe della matrice di siano c1, c2, c3...
         // devi capire quale è il meodo più veloce facendoti restituire il tempo di esecuzione
         // metodo con libreria eigen 
         DMatrix<double> CVCdiag_ = ((C_ * Vs()) * C_.transpose()).diagonal();
         //metodo con ciclo for per caclolare solo la diagonale e non tutta la matrice 
	      int size = std::min(C_.rows(), Vs().rows()) // questo lo sai a priori quindi sostituisci con la teoria  
	      DVector<double> diagonal(size);
         for (int i = 0; i < C_.rows(); ++i) {
            // ottengo la riga i-esima della matrice C
            DVector ci = C_.row(i);
            // calcolo il prodotto c_i^T * V * c_i
            diagonal[i] = ci.transpose() * Vs()* ci;
         }
         DVector<double> lowerBound;
         DVector<double> upperBound;


         if(type == simultaneous){ 
         // SIMULTANEOUS
         std::chi_squared_distribution<double> chi_squared(p);
         //quantile livello alpha 
         double quantile = std::quantile(chi_squared, alpha_);
         
         lowerBound = C_ * betas() - std::sqrt(quantile * diagonal);
         upperBound = C_ * betas() + std::sqrt(quantile * diagonal);

         }

         else if (type == bonferroni){
            // Bonferroni
         //quantile livello alpha 
         double quantile = std::sqrt(2.0) * std::erfinv(1-alpha_/(2*p));
         
         lowerBound = C_ * betas() - quantile *std::sqrt( diagonal);
         upperBound = C_ * betas() + quantile *std::sqrt( diagonal);

         }

         else if (type == one_at_the_time){
         //quantile livello alpha 
         double quantile = std::sqrt(2.0) * std::erfinv(1-alpha_/2);
         
         lowerBound = C_ * betas() - quantile *std::sqrt( diagonal);
         upperBound = C_ * betas() + quantile *std::sqrt( diagonal);

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
      }

     // this function returns the statistics not the p-values
     double p_value(){

     }

     // setter for matrix of combination of coefficients C
     void setC(DMatrix<double> C){
      C_ = C;
     }

     // setter for alpha
     void setAlpha(int alpha){
      if(alpha > 1 || alpha < 0){
         // print error
      }
      else{
         alpha_ = alpha;
      }
     }

          // funzione ausiliare per invertire una matrice densa in maniera efficiente
     DMatrix<double> inverse(DMatrix<double> M){
      // perchè in ExactEdf non fa il solve con l'identità?
      // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
      Eigen::PartialPivLU<DMatrix<double>> Mdec_ {};
      Mdec_ = M.partialPivLu();
      DMatrix<double> invM_ = Mdec_.solve(DMatrix::Identity(M.rows(), M.cols()));
      return invM_;
     }



} // end class
} // end fdapde models
} // end fdapde namespace

#endif  //__SPECKMAN_BASE_H__

