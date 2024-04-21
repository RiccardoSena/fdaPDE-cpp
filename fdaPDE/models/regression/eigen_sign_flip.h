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
#include "inference.h"

// questo va aggiunto per funzione count???
#include <algorithm>



namespace fdapde {
namespace models {

template <typename Model> class EigenSignFlip {

    private:

        Model m_;
        int n_flip = 1;


    public:
        EigenSignFlip() = default;                 // constructors
        EigenSignFlip(const Model& m): m_(m) {};
        DMatrix<double> C_ {};               // inference matrix C (p x q) matrix               

        int n;                                     // dimension of matrix Pi_
        //DiagMatrix<double> Pi_ {};           // diagonal matrix for sign flip (n x n) matrix
        DVector<double> beta0_ {};           // inference hypothesis H0 (p x 1) matrix
        DMatrix<double> Lambda_ {};          // Speckman correction matrix (n x n) matriX

    DVector<double> p_value(CIType type){
        // extract matrix C (in the eigen-sign-flip case we cannot have linear combinations, but we can have at most one 1 for each column of C) 
        fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  

        if(is_empty(beta0_)){
            setBeta0(DVector<double>::Zero(m_.beta().size()));
        }

        int p = C_.rows(); 

        DVector<double> result;     // declare the vector that will store the p-values
        
        DMatrix<double> Lambda=Lambda_;         // compute Lambda
        Eigen::EigenSolver<DMatrix<double>> solver(Lambda);        // compute eigenvectors and eigenvalues of Lambda

        // get real part of eigenvalues and eigenvectors
        DMatrix<std::complex<double>> eigenvalues_complex = solver.eigenvalues();
        DMatrix<std::complex<double>> eigenvectors_complex = solver.eigenvectors();

        DMatrix<double> eigenvalues=eigenvalues_complex.real();
        DMatrix<double> eigenvectors=eigenvectors_complex.real();

        // Store beta_hat
        DVector<double> beta_hat = m_.beta();
        DVector<double> beta_hat_mod=beta_hat;
        
        if(type == simultaneous){  
            // SIMULTANEOUS   
            // extract the current betas in test
            for(int i=0; i<p; i++){
                for(int j=0; j<C_.cols(); j++){
                    if(C_(i,j)>0){
                        beta_hat_mod[j]=beta0_[j];
                    }
                }
            }
            
            // compute the partial residuals
            DMatrix<double> Partial_res_H0 = m_.y() - m_.X() * beta_hat_mod;
            
            // compute the vectors needed for the statistic
            DMatrix<double> TildeX = (C_ * m_.X().transpose()) * eigenvectors * eigenvalues.asDiagonal();   	// W^t * V * D
            DVector<double> Tilder = eigenvectors.transpose()*Partial_res_H0;   			        		// V^t * partial_res_H0

            // Initialize observed statistic and sign-flipped statistic
            DVector<double> stat=TildeX*Tilder;
            DVector<double> stat_flip=stat;
            std::cout<<"questo è stat observed: "<<stat<<std::endl;
            //Random sign-flips
            std::random_device rd; 
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<> distr{0,1}; // Bernoulli(1/2)
            int count_Up=0;
            int count_Down=0;

            DVector<double> Tilder_perm=Tilder;
            
            for(int i=0;i<n_flip;i++){
                for(int j=0;j<TildeX.cols();j++){
                    int flip=2*distr(eng)-1;
                    Tilder_perm(j)=Tilder(j)*flip;
                }
                stat_flip=TildeX*Tilder_perm; // Flipped statistic
                //std::cout<<"questo è stat flip: "<<stat_flip<<std::endl;

                if(is_Unilaterally_Greater(stat_flip,stat)){ 
                    count_Up=count_Up+1;
                    //std::cout<<"count up è: "<<count_Up<<std::endl;
                }
                else{ 
                if(is_Unilaterally_Smaller(stat_flip,stat)){ 
                    count_Down=count_Down+1;
                    //std::cout<<"count down è: "<<count_Down<<std::endl;
                    }                    
                }
            }
            
            double pval_Up = static_cast<double>(count_Up) / n_flip;
            double pval_Down = static_cast<double>(count_Down) / n_flip;
            //std::cout<<"il valore di pvalup è : "<<pval_Up<<std::endl;
            //std::cout<<"il valore di pvaldown è : "<<pval_Down<<std::endl;

            result.resize(p); // Allocate more space so that R receives a well defined object (different implementations may require higher number of pvalues)
            result(0) = 2*std::min(pval_Up,pval_Down); // Obtain the bilateral p_value starting from the unilateral
            for(int k=1;k<p;k++){
            result(k)=10e20;
            }
        }
        else{
            // ONE AT THE TIME   
            DMatrix<double> Partial_res_H0(Lambda.cols(), p);
            for(int i=0; i<p; ++i){
            // Extract the current beta in test
            beta_hat_mod = beta_hat;

            for(int j=0; j<C_.cols(); j++){
                if(C_(i,j)>0){
                    beta_hat_mod[j]=beta0_[j];
                }
            }
            // compute the partial residuals
            Partial_res_H0.col(i) = m_.y() - m_.X()* beta_hat_mod; // (z-W*beta_hat(non in test)-W*beta_0(in test))
            }
            // compute the vectors needed for the statistic
            DMatrix<double> TildeX = (C_ *m_.X().transpose()) * eigenvectors*eigenvalues.asDiagonal();   	// W^t * V * D
            DMatrix<double> Tilder = eigenvectors.transpose()*Partial_res_H0;   			        		// V^t * partial_res_H0

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
                for(int j=0;j<TildeX.cols();j++){
                    int flip=2*distr(eng)-1;
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
            
            DVector<double> pval_Up = count_Up.array() / static_cast<double>(n_flip);
            DVector<double> pval_Down = count_Down.array() / static_cast<double>(n_flip);

            //std::cout<<"il valore di pvalup è : "<<pval_Up<<std::endl;
            //std::cout<<"il valore di pvaldown è : "<<pval_Down<<std::endl;
            result.resize(p);
            result = 2*min(pval_Up,pval_Down); // Obtain the blateral p_value starting from the unilateral
        } 
        return result;
     }
    
      // setter per i beta0_
    void setBeta0(DVector<double> beta0){
      beta0_ = beta0;
    }

    inline bool is_Unilaterally_Greater (DVector<double> v, DVector<double> u){
        int q=v.size();
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

    inline DVector<double> min(const DVector<double> & v, const DVector<double> & u){
        DVector<double> result;
        result.resize(v.size());
        for(int i=0; i<v.size(); ++i){
            result(i)=std::min(v(i),u(i));
        }
        return result;
    }

    void setC(DMatrix<double> C){
      C_ = C;
    }

    void setNflip(int m){
        n_flip = m;
    }

    DMatrix<double> Lambda() {
        DMatrix<double> inverseA_ {};
        inverseA_ =  - m_.invA().solve(DMatrix<double>::Identity(2 * m_.n_basis(),2 * m_.n_basis()));
        Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * inverseA_.block(0, 0, m_.n_basis(), m_.n_basis()) * m_.PsiTD();
        return Lambda_;
    }

}; // closing EigenSignFlip class
} // closing models namespace
} // closing fdapde namespace

#endif // __EIGEN_SIGN_FLIP_H__