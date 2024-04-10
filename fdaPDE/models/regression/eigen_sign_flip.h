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

     Model m_;


    public:
     EigenSignFlip() = default;                 // constructors
     EigenSignFlip(const Model& m): m_(m) {};

     int n;                                     // dimension of matrix Pi_
     //DiagMatrix<double> Pi_ {};           // diagonal matrix for sign flip (n x n) matrix
     DVector<double> beta0_ {};           // inference hypothesis H0 (p x 1) matrix

     DiagMatrix<double> Pi() {  

        std::random_device rd;
        //std::cout<<"device correttamente"<<std::endl;
        std::mt19937 gen(rd());
        //std::cout<<"gen correttamente"<<std::endl;
        std::uniform_real_distribution<double> dist(-1, 1);
        //std::cout<<"distribution correttamente"<<std::endl;
        DiagMatrix<double> Pi_(m_.n_obs());
        for(int i = 0; i < m_.n_obs(); ++i){
            Pi_.diagonal()[i] = dist(gen);
        }
        //std::cout<<"pi viene calcolata correttamente"<<std::endl;
        //std::cout<<"il numero di righe di Pi è "<<Pi_.rows()<<std::endl;
        //std::cout<<"il numero di colonne di Pi è "<<Pi_.cols()<<std::endl;
        return Pi_;
     }

     DVector<double> p_value(){
        // T  = W^T * Pi(z − W * beta − Psi* f )
        //compute score components under H0
        
        if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            // set beta0 to 0
            setBeta0(DVector<double>::Zero(m_.beta().size()));
        }
        std::cout<<"controllo su beta0 avviene correttamente"<<std::endl;

        std::cout<<"la lunghezza di beta0_ è : "<<beta0_.size()<<std::endl;
        std::cout<<"questa è beta0_ : " <<std::endl;
        for (int i = 0; i < beta0_.size(); ++i) {
           std::cout << beta0_[i] << " ";
        }
        
        //Compute the observed test statistic Tobs
        DVector<double> Tobs_= m_.X().transpose() * DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) * (m_.y()- m_.X() * m_.beta());
        std::cout<<"la test statistic è : "<<std::endl;
        for (int i = 0; i < Tobs_.size(); ++i) {
           std::cout << Tobs_[i] << " ";
        }
        int M = 20;
        DMatrix<double> T_(m_.X().cols(), M);
        std::cout<<"T viene inizializzato vuoto correttamente"<<std::endl;
        // M lo vogliamo impostare noi???
        for (int i = 0; i < M; ++i) {
            // generate matrix Pi_
            DiagMatrix<double> Pi_ = Pi();
            // compute test statistic Ti with Pi_ just generated 
            // ie costruisco un vettore di Ti che contiene tutti i valori della statistica per ogni Pi diverso 
            DMatrix<double> Pi_Matrix = DMatrix<double>::Zero(Pi_.rows(), Pi_.cols());
            Pi_Matrix.diagonal() = Pi_.diagonal();
            /*
            std::cout<<"Pi è diventata una matrice densa"<<std::endl;
            std::cout<<"il numero di righe di X è "<<m_.X().rows()<<std::endl;
            std::cout<<"il numero di colonne di X è "<<m_.X().cols()<<std::endl;
            std::cout<<"il numero di righe di Pi è "<<Pi_Matrix.rows()<<std::endl;
            std::cout<<"il numero di colonne di Pi è "<<Pi_Matrix.cols()<<std::endl;
            std::cout<<"il numero di righe di y è "<<m_.y().rows()<<std::endl;
            std::cout<<"il numero di colonne di y è "<<m_.y().cols()<<std::endl;
            */
            T_.col(i) = m_.X().transpose() * Pi_Matrix * (m_.y() - m_.X() * beta0_);
        }
        std::cout<<"calcolo di T avviene correttamente e T è :"<<std::endl;
        for (int i = 0; i < T_.rows(); ++i) {
            for (int j = 0; j < T_.cols(); ++j) {
                std::cout << T_(i, j) << " ";
            }
            std::cout << std::endl; // Aggiungi una nuova riga dopo ogni riga di elementi
        }
        // pvalue= sum( Ti> Tobs)/M
        // Replicare Tobs_ M volte
        auto replicated_Tobs = Tobs_.replicate(1, M);
        std::cout << "Replicated Tobs_:" << std::endl << replicated_Tobs << std::endl;

        // Effettua il confronto elemento per elemento tra T_ e replicated_Tobs
        auto comparison = T_.array() > replicated_Tobs.array();
        std::cout << "Comparison:" << std::endl << comparison << std::endl;

        // Calcola il conteggio delle occorrenze di ciascuna riga in base al confronto
        auto rowwise_count = comparison.rowwise().count();
        std::cout << "Row-wise count:" << std::endl << rowwise_count << std::endl;

        // Cast dei risultati in double
        DVector<double> count = rowwise_count.cast<double>();
        std::cout << "Count (double cast):" << std::endl << count << std::endl;

        //DVector<double> count = (T_.array() > Tobs_.replicate(1, M).array()).rowwise().count().cast<double>();
        std::cout<<"count  è : "<<std::endl;
        for (int i = 0; i < count.size(); ++i) {
           std::cout << count[i] << " ";
        }
        return count/M;
     }

      // setter per i beta0_
     void setBeta0(DVector<double> beta0){
      beta0_ = beta0;
     }


}; // closing EigenSignFlip class


} // closing models namespace
} // closing fdapde namespace

#endif // __EIGEN_SIGN_FLIP_H__