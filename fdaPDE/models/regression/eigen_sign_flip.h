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


namespace fdapde {
namespace models {

template <typename Model> class EigenSignFlip {

    private:

     Model* m_;


    public:

     // constructors
     EigenSignFlip() = default;
     EigenSignFlip(Model* m): m_(m) {};
    
     // dimension of matrix Pi_
     int n;
     // diagonal matrix
     const DiagMatrix<double> Pi_ {};

     const DiagMatrix<double>& Pi() {
        // setting random seed
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(-1, 1);
        for(int i = 0; i < n; ++i){
            Pi_.diagonal()[i] = dist(rng);
        }
        return Pi_;
     }



} // closing EigenSignFlip class


} // closing models namespace
} // closing fdapde namespace

#endif // __EIGEN_SIGN_FLIP_H__