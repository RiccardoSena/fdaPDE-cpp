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
     DMatrix<double> C_ {};               // inference matrix C (p x q) matrix               

     int n;                                     // dimension of matrix Pi_
     //DiagMatrix<double> Pi_ {};           // diagonal matrix for sign flip (n x n) matrix
     DVector<double> beta0_ {};           // inference hypothesis H0 (p x 1) matrix
     DMatrix<double> Lambda_ {};          // Speckman correction matrix (n x n) matrix


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

/*
     DVector<double> p_value(){
        // T  = W^T * Pi(z − W * beta − Psi* f )
        //compute score components under H0
        
        if(is_empty(beta0_)){
            // print errore (need to set beta0)???
            // set beta0 to 0
            setBeta0(DVector<double>::Zero(m_.beta().size()));
        }
        //std::cout<<"controllo su beta0 avviene correttamente"<<std::endl;

        //std::cout<<"la lunghezza di beta0_ è : "<<beta0_.size()<<std::endl;
        //std::cout<<"questa è beta0_ : " <<std::endl;
        //for (int i = 0; i < beta0_.size(); ++i) {
        //   std::cout << beta0_[i] << " ";
        //}
        
        //Compute the observed test statistic Tobs
        DVector<double> Tobs_= m_.X().transpose() * DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) * (m_.y()- m_.X() * m_.beta());
        //std::cout<<"la test statistic è : "<<std::endl;
        //for (int i = 0; i < Tobs_.size(); ++i) {
        //   std::cout << Tobs_[i] << " ";
        //}
        int M = 1000;
        DMatrix<double> T_(m_.X().cols(), M);
        std::cout<<"T viene inizializzato vuoto correttamente"<<std::endl;
        DVector<double> count(Tobs_.size());
        // M lo vogliamo impostare noi???
        for (int i = 0; i < M; ++i) {
            // generate matrix Pi_
            DiagMatrix<double> Pi_ = Pi();
            // compute test statistic Ti with Pi_ just generated 
            // ie costruisco un vettore di Ti che contiene tutti i valori della statistica per ogni Pi diverso 
            DMatrix<double> Pi_Matrix = DMatrix<double>::Zero(Pi_.rows(), Pi_.cols());
            Pi_Matrix.diagonal() = Pi_.diagonal();
            
            std::cout<<"Pi è diventata una matrice densa"<<std::endl;
            std::cout<<"il numero di righe di X è "<<m_.X().rows()<<std::endl;
            std::cout<<"il numero di colonne di X è "<<m_.X().cols()<<std::endl;
            std::cout<<"il numero di righe di Pi è "<<Pi_Matrix.rows()<<std::endl;
            std::cout<<"il numero di colonne di Pi è "<<Pi_Matrix.cols()<<std::endl;
            std::cout<<"il numero di righe di y è "<<m_.y().rows()<<std::endl;
            std::cout<<"il numero di colonne di y è "<<m_.y().cols()<<std::endl;
            
            T_.col(i) = m_.X().transpose() * Pi_Matrix * (m_.y() - m_.X() * beta0_);
            DVector<double> count_temp = (T_.col(i).array() > Tobs_.array()).cast<double>();
            count += count_temp;
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
    */




     DVector<double> p_value(CIType type){
  
        // extract matrix C 
        // (in the eigen-sign-flip case we cannot have linear combinations, but we can have at most one 1 for each column of C) 
        fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  
        std::cout<<"controllo su C avviene correttamente"<<std::endl;

        // is_empty va bene anche per i Vectors?
        if(is_empty(beta0_)){
                    // print errore (need to set beta0)???
                    // set beta0 to 0
            setBeta0(DVector<double>::Zero(m_.beta().size()));
        }
        int p = C_.rows(); 
        DMatrix<double> C=C_;
        // declare the vector that will store the p-values
        DVector<double> result;
        // get the number of flips
        int n_flip=1000;
        
        // compute Lambda
        //facciamo finta di avere lambda 
        //DMatrix<double> Lambda=DMatrix<double>::Identity(m_.n_obs(), m_.n_obs());
        DMatrix<double> Lambda=Lambda_;
        
        // compute eigenvectors and eigenvalues of Lambda
        Eigen::EigenSolver<DMatrix<double>> solver(Lambda);

        // Ottieni gli autovalori e gli autovettori
        DMatrix<std::complex<double>> eigenvalues_complex = solver.eigenvalues();
        DMatrix<std::complex<double>> eigenvectors_complex = solver.eigenvectors();

        DMatrix<double> eigenvalues=eigenvalues_complex.real();
        DMatrix<double> eigenvectors=eigenvectors_complex.real();




        // Store beta_hat
        DVector<double> beta_hat = m_.beta();
        DVector<double> beta_hat_mod=beta_hat;
        
        // simultaneous test
        if(type == simultaneous){    
            // extract the current betas in test
            for(int i=0; i<p; i++){
            for(int j=0; j<C.cols(); j++){
                if(C(i,j)>0){
                    beta_hat_mod[j]=beta0_[j];
                    }
                }
            }
            
            // compute the partial residuals
            DMatrix<double> Partial_res_H0 = m_.y() - m_.X() * beta_hat_mod;
            
            // compute the vectors needed for the statistic
            DMatrix<double> TildeX = (C * m_.X().transpose()) * eigenvectors*eigenvalues.asDiagonal();   	// W^t * V * D
            DVector<double> Tilder = eigenvectors.transpose()*Partial_res_H0;   			        		// V^t * partial_res_H0
                
            //int n_obs = m_.n_obs();

            // Prepare vectors for enhanced-ESF if requested
            DVector<double> Tilder_hat = eigenvectors.transpose()* (m_.y() - m_.X()* beta_hat); // This vector represents Tilder using only beta_hat, needed for bias estimation

            // Estimate the standard error
            //DVector<double> eps_hat = (m_.y() - m_.X()*m_.beta()); 
            //double SS_res = eps_hat.squaredNorm();
            //double Sigma_hat = std::sqrt(SS_res/(n_obs-1));

            //double threshold = 10*Sigma_hat; // This threshold is used to determine how many components will not be flipped: we drop those that show large alpha_hat w.r.t. the expected standar error
            //int N_Eig_Out=0; // It will store the number of biased components that will be kept fixed if enhanced-ESF is required
            
            // Initialize observed statistic and sign-flipped statistic
            DVector<double> stat=TildeX*Tilder;
            DVector<double> stat_flip=stat;
            
            //Random sign-flips
            std::random_device rd; 
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<> distr{0,1}; // Bernoulli(1/2)
            int count_Up=0;
            int count_Down=0;

            DVector<double> Tilder_perm=Tilder;
            
            for(int i=0;i<n_flip;i++){
            //N_Eig_Out=0;
                for(int j=0;j<TildeX.cols();j++){
                    int flip;
                    //if((fabs(Tilder_hat(j))>threshold)){ // If enhanced-ESF is required and component is biased
                    //    flip=1; // Fix the biased component
                //++N_Eig_Out;
                //}else{
                    flip=2*distr(eng)-1;
                    Tilder_perm(j)=Tilder(j)*flip;
                }
                stat_flip=TildeX*Tilder_perm; // Flipped statistic
                if(is_Unilaterally_Greater(stat_flip,stat)){ ++count_Up;}
                else{ //Here we use the custom-operator defined in Eigen_Sign_Flip.h
                if(is_Unilaterally_Smaller(stat_flip,stat)){ ++count_Down;} //Here we use the custom-operator defined in Eigen_Sign_Flip.h 
                }
            }
            
            double pval_Up = count_Up/n_flip;
            double pval_Down = count_Down/n_flip;

            result.resize(p); // Allocate more space so that R receives a well defined object (different implementations may require higher number of pvalues)
            result(0) = 2*std::min(pval_Up,pval_Down); // Obtain the bilateral p_value starting from the unilateral
            for(int k=1;k<p;k++){
            result(k)=10e20;
            }
        }
        else{
            
            // one-at-the-time tests    
            DMatrix<double> Partial_res_H0(Lambda.cols(), p);
            for(int i=0; i<p; ++i){
            // Extract the current beta in test
            beta_hat_mod = beta_hat;

            for(int j=0; j<C.cols(); j++){
                if(C(i,j)>0){
                    beta_hat_mod[j]=beta0_[j];}
            }
            // compute the partial residuals
            Partial_res_H0.col(i) = m_.y() - m_.X()* beta_hat_mod; // (z-W*beta_hat(non in test)-W*beta_0(in test))
            }
            // compute the vectors needed for the statistic
            DMatrix<double> TildeX = (C *m_.X().transpose()) * eigenvectors*eigenvalues.asDiagonal();   	// W^t * V * D
            DMatrix<double> Tilder = eigenvectors.transpose()*Partial_res_H0;   			        		// V^t * partial_res_H0
            
            //int n_obs = m_.n_obs();

            // Seclect eigenvalues that will not be flipped basing on the estimated bias carried
            DVector<double> Tilder_hat = eigenvectors.transpose()* (m_.y() - m_.X()* beta_hat); // This vector represents Tilder using only beta_hat, needed for bias estimation

            // Estimate the standard error
            //DVector<double> eps_hat = (m_.y() - m_.X()*m_.beta()); 
            //double SS_res = eps_hat.squaredNorm();
            //double Sigma_hat = std::sqrt(SS_res/(n_obs-1));

            //double threshold = 10*Sigma_hat; // This threshold is used to determine how many components will not be flipped: we drop those that show large alpha_hat w.r.t. the expected standar error
            //UInt N_Eig_Out=0; // It will store the number of biased components that will be kept fixed if enhanced-ESF is required

            // Observed statistic
            DMatrix<double> stat=TildeX*Tilder;
            DMatrix<double> stat_flip=stat;

            // Random sign-flips
            std::random_device rd; 
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<> distr{0,1}; // Bernoulli(1/2)
            DVector<double> count_Up = DVector<double>::Zero(p);
            DVector<double> count_Down = DVector<double>::Zero(p);
            
            DMatrix<double> Tilder_perm=Tilder;
            
            for(int i=0;i<n_flip;i++){
            //N_Eig_Out=0;
                for(int j=0;j<TildeX.cols();j++){
                    int flip;
                //if((this->inf_car.getInfData()->get_enhanced_inference()[this->pos_impl]==true) && (N_Eig_Out<n_obs/2) && (fabs(Tilder_hat(j))>threshold)){ // If enhanced-ESF is required and component is biased
                    //flip=1; // Fix the biased component
                //++N_Eig_Out;
            //}else{
                    flip=2*distr(eng)-1;
            //}
                    Tilder_perm.row(j)=Tilder.row(j)*flip;
                 }
                stat_flip=TildeX*Tilder_perm; // Flipped statistic
            
                for(int k=0; k<p; ++k){
                    if(stat_flip(k,k) > stat(k,k)){
                        ++count_Up(k);
                    }else{
                        ++count_Down(k);
                    }
                } 
            }
            
            DVector<double> pval_Up = count_Up/n_flip;
            DVector<double> pval_Down = count_Down/n_flip;
            
            result.resize(p);
            //result = 2*min(pval_Up,pval_Down); // Obtain the blateral p_value starting from the unilateral
        } 
        return result;
     }
    
      // setter per i beta0_
     void setBeta0(DVector<double> beta0){
      beta0_ = beta0;
     }

    inline bool is_Unilaterally_Greater (DVector<double> v, DVector<double> u){
    int q=v.size();
    //if(u.size()!=q){
      //  Rprintf("Error: in Eigen-Sign-Flip procedure two vectors of different length have been compared");
       // return false;
   // }
    for (int i=0; i< q; i++){
        if(v(i)<=u(i)){
        return false;
        }
    }
    return true;
    };
    inline bool is_Unilaterally_Smaller (DVector<double> v, DVector<double> u){
    int q=v.size();
    //if(u.size()!=q){
        //Rprintf("Error: in Eigen-Sign-Flip procedure two vectors of different length have been compared");
      //  return false;
    //}
    for (int i=0; i< q; i++){
        if(v(i)>=u(i)){
        return false;
        }
    }
    return true;
    };

    void setC(DMatrix<double> C){
      C_ = C;
     }





    DMatrix<double> Lambda() {
        //Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * s_.compute(m_).block(0, 0, m_.n_basis(), m_.n_basis()) * m_.PsiTD();
        DMatrix<double> inverseA_ {};
        inverseA_ =  - m_.invA().solve(DMatrix<double>::Identity(2 * m_.n_basis(),2 * m_.n_basis()));
        Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * inverseA_.block(0, 0, m_.n_basis(), m_.n_basis()) * m_.PsiTD();
        return Lambda_;
    }

}; // closing EigenSignFlip class


} // closing models namespace
} // closing fdapde namespace

#endif // __EIGEN_SIGN_FLIP_H__