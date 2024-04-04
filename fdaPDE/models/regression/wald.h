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
#include <fdaPDE/linear_algebra/fspai.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "wald_base.h"

//#include "../fspai.h" questo l'ho aggiunto in linear_algebra.h

// do we need this to use FSPAI??
//using fdapde::core::FSPAI

namespace fdapde {
namespace models {

enum class Strategy{exact, non_exact};

// Dichiarazione anticipata del template di classe Model
template <typename Model>
class WaldBase;

template <typename Model, Strategy S> class Wald;
// class WALD<Model, exact> : public WaldBase<Model> 
template <typename Model>
class Wald<Model, Strategy::exact> : public WaldBase<Model> {

    public: 
     // is this necessary
     using Base = WaldBase<Model>;
     
     
     Wald() = default;                  // constructor
     Wald(Model* m): Base(m) {}

     
     void S() override{
            DMatrix<double> invT_ = Base::inverse(Base::m_->T());
            Base::S_ = Base::m_->Psi() * invT_ * Base::m_->PsiTD() * Base::m_->Q();
     };

};

template <typename Model>
class Wald<Model, Strategy::non_exact> : public WaldBase<Model> {

    public: 
     using Base = WaldBase<Model>;

     Wald() = default;              // constructor 
     Wald(Model* m): Base(m) {}

     void S() override{
        // FSPAI approx
        // E_tilde = Psi^T*\Psi+lambda*\R
        // making E_tilde sparse
        // Dalla Sub Woodbury decomposition
        // bisogna fare prima un'approssimazione dell'inversa di R0, usando FSPAI
        // R0 should be stored as a sparse matrix
        fdapde::core::FSPAI fspai_R0(Base::m_->R0());

        int alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A (perform alpha steps of approximate inverse update along column k)
        int beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
        double epsilon = 0.001; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità (the best improvement is higher than accetable treshold)
        // calcolo inversa di R0
        fspai_R0.compute(alpha, beta, epsilon);
        //getter per l'inversa di R0
        Eigen::SparseMatrix<double> invR0_= fspai_R0.getInverse();

        //calcolo la matrice Atilde
        DMatrix<double> Et_ = Base::m_->PsiTD()* Base::m_->Psi()+ Base::m_->lambda_D() * Base::m_->R1().transpose() * invR0_ * Base::m_->R1();

        //applico FSPAI su Atilde
        fdapde::core::FSPAI fspai_E(Et_);
        fspai_E.compute(alpha, beta, epsilon);
        Eigen::SparseMatrix<double> invE_ = fspai_E.getInverse();

        // Mt^{-1} = Et^{-1} + Et{-1}*\Ut*\(Ct+Vt*\Et^{-1}*\Ut)^{-1}*\Vt*\Et^{-1}

        // Bisogna capire se sono matrici dense o sparse in modo da ottimizzare

        // Ut = Psi^T*\W    Nt x q
        // Ct = -(W^T*\W)^{-1}   q x q
        // Vt = W^T*\Psi   q x Nt

        // DMatrix<double> Ut_ = m_.PsiTD() * m_.W();
        DMatrix<double> Ut_ = Base::m_->Psi().transpose() + Base::m_->X();
        DMatrix<double> Ct_ = - Base::inverse(Base::m_->X().transpose() * Base::m_->X());
        DMatrix<double> Vt_ = Base::m_->X().transpose() * Base::m_->Psi();

        Eigen::SparseMatrix<double> invMt_ = invE_ + invE_ * Ut_ * Base::inverse(Ct_ + Vt_ * invE_ * Ut_) * Vt_ * invE_;

        // m_.Psi().transpose() or m_.PsiTD()
        Base::S_ = Base::m_->Psi() * invMt_ * Base::m_->PsiTD() * Base::m_->Q();

     };


};


}  // closing models namespace
} // closing fdapde namespace

# endif   //__WALD_H__

