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

#ifndef __EIGEN_SIGN_FLIP_H__
#define __EIGEN_SIGN_FLIP_H__

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"

// questo va aggiunto per funzione count???
#include <algorithm>



namespace fdapde {
namespace models {

template <typename Model> class EigenSignFlip {

    private:

     Model* m_;


    public:
     EigenSignFlip() = default;                 // constructors
     EigenSignFlip(Model* m): m_(m) {};

     int n;                                     // dimension of matrix Pi_
     const DiagMatrix<double> Pi_ {};           // diagonal matrix for sign flip (n x n) matrix
     const DVector<double> beta0_ {};           // vector of beta 0

     const DiagMatrix<double>& Pi() {  
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1, 1);
        for(int i = 0; i < n; ++i){
            Pi_.diagonal()[i] = dist(gen);
        }
        return Pi_;
     }

     DVector<double> pvalues(){
        // T  = W^T * Pi(z − W * beta − Psi* f )
        //compute score components under H0
        
        if(is_empty(beta0_)){
            // print error
            // default value is vector of full zeros
            setBeta0(DVector<double>::Zero(m_.beta().size()));            
        }
        
        //Compute the observed test statistic Tobs
        DVector<double> Tobs_= m_.W().transpose() * Eigen::Identity(m_.n_obs(), m_.n_obs()) * (m_.y()- m_.W() * m_.beta())
        DMatrix<double> T_ {};
        // M lo voliam oimpostare noi???
        for (int i = 0; i < M; ++i) {
            // generate matrix Pi_
            DiagMatrix<double> Pi_ = Pi();
            // compute test statistic Ti with Pi_ just generated 
            // ie costruisco un vettore di Ti che contiene tutti i valori della statistica per ogni Pi diverso 
            T_.col(i)= m_.W().transpose() * Pi_ * (m_.y() - m_.W() * beta0_);
        }
        // pvalue= sum( Ti> Tobs)/M
        DVector<int> count = (T_.array() > T_obs.replicate(1, M).array()).rowwise().count();
        return count/M;
     }

     void setBeta0(DVector<double> beta0){
      // funziona così per i Eigen::Vector??
      beta0_ = beta0;
     }


} // closing EigenSignFlip class


} // closing models namespace
} // closing fdapde namespace

#endif // __EIGEN_SIGN_FLIP_H__