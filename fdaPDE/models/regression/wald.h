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

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "wald_base.h"


using fdapde::core::SMW;

namespace fdapde {
namespace models {

// why do we need this
struct exact {};

template <typename Model>

// WALD model signature, guarda in strpde.h
template <typename Model, typename Strategy> class WALD;

// class WALD<Model, exact> : public WaldBase<Model> 
class WALD<Model, exact> : public WaldBase<Model> {

    public: 
     // is this necessary
     using Base = WaldBase<Model>;
     
     // constructor
     WALD() = default;
     WALD(Model* m): Base(m) {};

     
     // perchè la funzione return S e non inizializza direttamente S_ ???
     void S() override{
            DMatrix<double> invT_ = inverse(m_.T());
            S_ = m_.Psi() * invT_.block(0, 0, m_.n_basis, m_.n_basis) * m_.Psi().transpose() * m_.Q();
     }

}

class WALD<Model, non_exact> : public WaldBase<Model> {

    public: 
     using Base = WaldBase<Model>;

     // constructor 
     WALD() = default;
     WALD(Model* m): Base(m) {};

     DMatrix<double>& S() override{
        // FSPAI approx
     }

}

}  // closing models namespace
}  // closing fdapde namespace

# endif   //__WALD_H__

