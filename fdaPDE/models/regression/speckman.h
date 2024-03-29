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
#include "exact_edf.h"
#include ".../fspai.h"

// we need this to use FSPAI
using fdapde::core::FSPAI


namespace fdapde {
namespace models {

enum class Strategy{ exact, non_exact };

template <typename Model>

template <typename Model, Strategy S> class SPECKMAN;

class SPECKMAN<Model, Strategy::exact> : public SpeckmanBase<Model> {

    public: 
     using Base = SpeckmanBase<Model>;

     // constructor
     SPECKMAN() = default;
     SPECKMAN(Model* m): Base(m) {};

     void inverseA() override{
            inverseA_ =  m_.invA().solve(DMatrix<double>::Identity(m_.n_basis, m_.n_basis));
     }

}

class SPECKMAN<Model, Strategy::non_exact> : public SpeckmanBase<Model> {

    public: 
     using Base = SpeckmanBase<Model>;

     SPECKMAN() = default;
     SPECKMAN(Model* m): Base(m) {};
     
     void inverseA() override{
        // quali funzioni devo chiamare per far calcolare la inversa alla classe FSPAI solo compute e getInverse
        // FSPAI approx
        //creo oggetto FSPAI( vanno controllati tipi di input e output)
        FSPAI fspai_R0(m_.R0());

        // questi non so come vadano scelti ho messo nuemri a caso ???
        unsigned alpha = 10;    // Numero di aggiornamenti del pattern di sparsità per ogni colonna di A
        unsigned beta = 5;      // Numero di indici da aggiungere al pattern di sparsità di Lk per ogni passo di aggiornamento
        double epsilon = 0.001; // Soglia di tolleranza per l'aggiornamento del pattern di sparsità
        // calcolo inversa di R0
        fspai_R0.compute(alpha, beta, epsilon);
        //getter per l'inversa di R0
        Eigen::SparseMatrix<double> inv_R0 fspai_R0.getInverse();

        //qui non so se è giusto questo lambda
        //caclolo la matrice Atilde
        // Bisogna usare PsiTD()??
        DMatrix<double> tildeA_ = m_.Psi().transpose()* m_.Psi()+ m_.lambda_D() * m_.R1().transpose() * inv_R0 * m_.R1();

        //applico FSPAI su Atilde
        FPSAI fspai_A(tildeA_);
        fspai_A.compute(alpha, beta, epsilon);

        // inverseA_
        inverseA_ = fspai_A.getInverse();
     
     }

}

}  // closing models namespace
}  // closing fdapde namespace


#endif  //__SPECKMAN_H__

