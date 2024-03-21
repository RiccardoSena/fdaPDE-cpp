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

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"

using fdapde::core::SMW;

// add this??
#include "exact_edf.h"


namespace fdapde {
namespace models {

struct exact {};

template <typename Model>

// SPECKMAN model signature, guarda in strpde.h
template <typename Model, typename Strategy> class SPECKMAN;


class SPECKMAN<Model, exact> : public SpeckmanBase<Model> {

    public: 
     // is this necessary
     using Base = SpeckmanBase<Model>;

     // constructor
     SPECKMAN() = default;
     SPECKMAN(Model* m): Base(m) {};

     void inverseA() override{
            inverseA_ =  m_.invA().solve(DMatrix<double>::Identity(m_.n_basis, m_.n_basis));
     }

}

class SPECKMAN<Model, non_exact> : public SpeckmanBase<Model> {

    public: 
     using Base = SpeckmanBase<Model>;

     SPECKMAN() = default;
     SPECKMAN(Model* m): Base(m) {};
     
     DMatrix<double>& inverseA() override{
        // FSPAI approx
     }

}

}  // closing models namespace
}  // closing fdapde namespace


#endif  //__SPECKMAN_H__

