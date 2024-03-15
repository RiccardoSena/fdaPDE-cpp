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


namespace fdapde {
namespace models {

// base class for any regression model
template <typename Model>
class SPECKMAN {

    private:
     Model* m_;

     DMatrix<double> Lambda_ {};

     DVector<double> betas_ {};


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
        Lambda_ = I - m_.Psi() * Einv_ * m_.PsiTD();

        return Lambda_;
     }

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

    DMatrix<double> computeCI(){
      
        //costruisco lowerBound e upperBound
        // supponendo che abbiamo la matrice C che in teoria dovrebbe essere inserita dall'utente
         
        DVector<double> lowerBound = C_*m_.betas();
        DVector<double> upperBound = C_*m_.betas();
        DMatrix<double> BoundMatrix(n_, 2);
        BoundMatrix.col(0) = lowerBound;
        BoundMatrix.col(1) = upperBound;

      

        return std::make_pair(lowerBound, upperBound);
    }

    double p_value(){

    }






} // end class

} // end models namespace
} // end fdapde namespace
