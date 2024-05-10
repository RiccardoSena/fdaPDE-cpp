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

#ifndef __WALD2_H__
#define __WALD2_H__

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
#include "stochastic_edf.h"
#include "inference_base.h"
#include "inference.h"

// per salvare la matrice con savemarket
#include <unsupported/Eigen/SparseExtra> 

namespace fdapde {
namespace models {

template <typename Model, typename Strategy> class Wald2: public InferenceBase<Model> {

    private: 
     struct ExactInverse {
        DMatrix<double> compute(Model m){
            return inverse(m.T());
        }
     };
     struct NonExactInverse {
        SpMatrix<double> compute(Model m){
            DMatrix<double> Ut_ = m.Psi().transpose() * m.X();
            DMatrix<double> Ct_ = - inverse(m.X().transpose() * m.X());
            DMatrix<double> Vt_ = m.X().transpose() * m.Psi();
            SpMatrix<double> invE_ = Base::invE_approx(m);


            //questo serve per fare confronto con la forma esatta 
            DMatrix<double> invEesatta=inverse(m.E());
            Eigen::saveMarket(invEesatta, "invEexact.mtx");
 
            SpMatrix<double> invMt_ = invE_ + invE_ * Ut_ * inverse(Ct_ + Vt_ * invE_ * Ut_) * Vt_ * invE_;
            return invMt_;            
        }       
     };


    public: 
     using Base = InferenceBase<Model>;
     using Base::m_;
     using Base::V_;
     using Base::beta_;
     using Base::invE_approx;
     using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
     Solver s_; 

     // constructors
     Wald2() = default;                   // deafult constructor
     Wald2(const Model& m): Base(m) {};     // constructor    

     void beta() override{
        beta_ = m_.beta();
     }

     // si potrebbe fare override anche di questo metodo visto che si può utilizzare StochasticEDF per calcolare la traccia
     double sigma_sq() {
        double sigma_sq_ = 0;             // sigma^2 
        DMatrix<double> epsilon = m_.y() - m_.fitted();
        ExactEDF strat;
        strat.set_model(m_);
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - strat.compute())) * epsilon.squaredNorm();
        return sigma_sq_;
     }

     void V() override{
        DMatrix<double> invSigma_ = inverse(m_.X().transpose() * m_.X());
        DMatrix<double> S = m_.Psi() * s_.compute(m_) * m_.PsiTD() * m_.Q(); 
        DMatrix<double> ss = S * S.transpose();
        DMatrix<double> left = invSigma_ * m_.X().transpose();
        V_ = sigma_sq() * (invSigma_ + left * ss * left.transpose()); 
     }

};

} // namespace models
} // namespace fdapde

#endif   // __WALD2_H__