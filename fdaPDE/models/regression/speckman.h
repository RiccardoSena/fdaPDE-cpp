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

#ifndef _SPECKMAN_H_
#define _SPECKMAN_H_

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

// base class for any regression model
template <typename Model>
class SPECKMAN {

    private:
     Model* m_;
     
     // do we need to store Lambda???
     DMatrix<double> Lambda_ {};

     DVector<double> betas_ {};

     DMatrix<double> Vs_ {};


    public:

     // condtructors
     SPECKMAN() = default;
     SPECKMAN(Model *m):m_(m) {};


     const DMatrix<double>& Lambda() {
        // I - Psi*\(Psi^T*\Psi + lambda*\R)^{-1}*\Psi^T
        // I - Psi*\E^{-1}*\Psi^T
        // E = Psi^T*\Psi + lambda*\R
        // E^{-1} is potentially large and dense, we only need the northwest block
        // The northwest block should just be Psi^T*\Psi (+P (penalization)???)
        DMatrix<double> E_ = m_.PsiTD() * m_.Psi();

        // ????
        DMatrix<double> Einv_ = E_.partialPivLu();

        // Matrice Identità????????? Is it D????
        Lambda_ = DMatrix<double>::Identity(m_.n_basis, m_.n_basis) - m_.Psi() * Einv_ * m_.PsiTD();

        return Lambda_;
     }
     
     //questo è quello che farei io 
     //const DMatrix<double>& Lambda() {
        // I - Psi*\(Psi^T*\Psi + lambda*\R)^{-1}*\Psi^T
        // I - Psi*\A^{-1}*\Psi^T
        // A = Psi^T*\Psi + lambda*\R
        // A^{-1} is potentially large and dense, we only need the northwest block
        // invA() contiene la fattorizzazione LU di A quindi va comunque calcolata la sua inversa 
        //Lambda_ = DMatrix<double>::Identity(m_.n_basis, m_.n_basis) - m_.Psi() * m_.invA().inverse()* m_.PsiTD();
        //return Lambda_;
     //}
      
     const DVector<double>& betas() {
        // Wt = Lambda_*\W
        // zt = Lambda_*\z
        // betas_ = (Wt^T*\Wt)^{-1}*\Wt^T*\zt

        DMatrix<double> Wtilde_ = Lambda() * m_.W();
        DMatrix<double> ytilde = Lambda() * m_.y();

        // ????
        DMatrix<double> temp = (Wtilde_.transpose() * Wtilde_).partialPivLU();


        betas_ = temp * Wtilde_.transpose() * ytilde_;
        return betas_;

     }

     DMatrix<double>& Vs() {
        // set U = Wt^T*\W
        // set E = epsilon*\epsilon^T
        // Vs = U^{-1}*\Wt^T*\Lambda*\E*\Lambda^T*\Wt*U^{-1}
        DMatrix<double> Wtilde_ = Lambda() * m_.W();
        DMatrix<double> U_ = Wtilde_.transpose() * Wtilde_;
        DMatrix<double> epsilon_ = m_.y() - m_.fitted();
        DMatrix<double> E = epsilon_ * epsilon_.transpose();

        // inversione U con pivLU???
        Vs_ = U_.inverse() * Wtilde_.tanspose() * Lambda() * E * Lambda().transpose() * Wtilde_ * U_.inverse();
        return Vs_;
     }



     DMatrix<double> computeCI(){

        // SIMULTANEOUS

        //quantile deve cambiare a seconda del confidence interval 
        // magari creare un setter per p e fare p una variabile privata??
        int p = ;
        std::chi_squared_distribution<double> chi_squared(p);
        //quantile livello alpha 
        double quantile = std::quantile(chi_squared, alpha);
        
        
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
        // e che sulle righe della matrice di siano c1, c2, c3...
        DMatrix<double> CVC_ = C * Vs() * C.transpose();
        // della matrice C*V*C^T devo prendere solo la diagonale per i Confidence intervals quindi magari è meglio far calcolare solo la diagonale      
        DVector<double> CVCdiag_ = CVC_.diagonal();

        DVector<double> lowerBound = C_ * betas() - sqrt(quantile * CVCdiag_);
        DVector<double> upperBound = C_ * betas() + sqrt(quantile * CVCdiag_);
        
        //costruisco la matrice che restituisce i confidence intervals
        DMatrix<double> CIMatrix(m_.n_obs(), 2);
        CIMatrix.col(0) = lowerBound;
        CIMatrix.col(1) = upperBound;

      
        return CIMatrix;
     }

     double p_value(){

     }


} // end class

} // end models namespace
} // end fdapde namespace
