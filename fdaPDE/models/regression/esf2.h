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

#ifndef __ESF2_H__
#define __ESF2_H__

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "inference_base.h"
#include "inference.h"

#include <algorithm>



namespace fdapde {
namespace models {

template <typename Model> class Esf2: public InferenceBase<Model>{

    private:
     int n_flip = 1; //default value
     DMatrix<double> Lambda_ {};

    public:
     using Base = InferenceBase<Model>;
     using Base::m_;
     using Base::beta_;  
     using Base::C_;
     using Base::beta0_;
     
     // constructors
     Esf2() = default;                   // deafult constructor
     Esf2(const Model& m): Base(m) {};     // constructor      

     DVector<double> p_value(CIType type) override{
        // extract matrix C (in the eigen-sign-flip case we cannot have linear combinations, but we can have at most one 1 for each column of C) 
        fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  

        if(is_empty(beta0_)){
            Base::setBeta0(DVector<double>::Zero(m_.beta().size()));
        }

        int p = C_.rows(); 

        DVector<double> result;     // declare the vector that will store the p-values
        
        // compute Lambda
        if(is_empty(Lambda_)){
            V();
        }

        Eigen::EigenSolver<DMatrix<double>> solver(Lambda_);        // compute eigenvectors and eigenvalues of Lambda

        DMatrix<std::complex<double>> eigenvalues_complex = solver.eigenvalues();
        DMatrix<std::complex<double>> eigenvectors_complex = solver.eigenvectors();

        DMatrix<double> eigenvalues = eigenvalues_complex.real();
        DMatrix<double> eigenvectors = eigenvectors_complex.real();

        // Store beta_hat
        DVector<double> beta_hat = m_.beta();
        DVector<double> beta_hat_mod = beta_hat;
        
        if(type == simultaneous){  
            // SIMULTANEOUS   
            // extract the current betas in test
            for(int i = 0; i < p; ++i){
                for(int j = 0; j < C_.cols(); j++){
                    if(C_(i,j) > 0){
                        beta_hat_mod[j] = beta0_[j];
                    }
                }
            }
            
            // partial residuals
            DMatrix<double> res_H0 = m_.y() - m_.X() * beta_hat_mod; 
            // W^t * V * D
            DMatrix<double> Xt = (C_ * m_.X().transpose()) * eigenvectors * eigenvalues.asDiagonal();   
            DVector<double> Tilder = eigenvectors.transpose() * res_H0;   

            // Initialize observed statistic and sign-flipped statistic
            DVector<double> stat = Xt * Tilder;
            DVector<double> stat_flip = stat;
            //std::cout<<"questo è stat observed: "<<stat<<std::endl;
            //Random sign-flips
            std::random_device rd; 
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<> distr{0,1}; 
            int up = 0;
            int down = 0;

            DVector<double> Tilder_perm = Tilder;
            
            for(int i = 0; i < n_flip; i++){
                for(int j = 0; j < Xt.cols(); ++j){
                    int flip = 2 * distr(eng) - 1;
                    Tilder_perm(j) = Tilder(j) * flip;
                }
                stat_flip = Xt * Tilder_perm; // Flipped statistic
                //std::cout<<"questo è stat flip: "<<stat_flip<<std::endl;

                if(is_Unilaterally_Greater(stat_flip, stat)){ 
                    up = up + 1;
                    //std::cout<<"count up è: "<<up<<std::endl;
                }
                else{ 
                if(is_Unilaterally_Smaller(stat_flip, stat)){ 
                    down = down + 1;
                    //std::cout<<"count down è: "<<dwon<<std::endl;
                    }                    
                }
            }
            
            double pval_up = static_cast<double>(up) / n_flip;
            double pval_down = static_cast<double>(down) / n_flip;
            //std::cout<<"il valore di pvalup è : "<<pval_up<<std::endl;
            //std::cout<<"il valore di pvaldown è : "<<pval_down<<std::endl;

            result.resize(p); // Allocate more space so that R receives a well defined object (different implementations may require higher number of pvalues)
            result(0) = 2 * std::min(pval_up, pval_down); // Obtain the bilateral p_value starting from the unilateral
            for(int k = 1; k < p; k++){
            result(k) = 0.0;
            }
        }
        else{
            // ONE AT THE TIME   
            DMatrix<double> res_H0(Lambda_.cols(), p);
            for(int i = 0; i < p; ++i){
            // Extract the current beta in test
            beta_hat_mod = beta_hat;

            for(int j = 0; j < C_.cols(); ++j){
                if(C_(i,j)>0){
                    beta_hat_mod[j] = beta0_[j];
                }
            }
            // compute the partial residuals
            res_H0.col(i) = m_.y() - m_.X()* beta_hat_mod;
            }
            // compute the vectors needed for the statistic
            DMatrix<double> Xt = (C_ * m_.X().transpose()) * eigenvectors * eigenvalues.asDiagonal();   	// W^t * V * D
            DMatrix<double> Tilder = eigenvectors.transpose() * res_H0;   			        		// V^t * partial_res_H0

            // Observed statistic
            DMatrix<double> stat = Xt * Tilder;
            DMatrix<double> stat_flip = stat;

            // Random sign-flips
            std::random_device rd; 
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<> distr{0,1}; // Bernoulli(1/2)
            DVector<double> up = DVector<double>::Zero(p);
            DVector<double> down = DVector<double>::Zero(p);
            
            DMatrix<double> Tilder_perm = Tilder;
            
            for(int i = 0; i < n_flip; ++i){
                for(int j = 0; j < Xt.cols(); ++j){
                    int flip = 2 * distr(eng) - 1;
                    Tilder_perm.row(j) = Tilder.row(j) * flip;
                 }
                stat_flip = Xt * Tilder_perm; // Flipped statistic
            
                for(int k = 0; k < p; ++k){
                    if(stat_flip(k, k) > stat(k, k)){
                        ++up(k);
                    }else{
                        ++down(k);
                    }
                } 
            }
            
            DVector<double> pval_up = up.array() / static_cast<double>(n_flip);
            DVector<double> pval_down = down.array() / static_cast<double>(n_flip);

            //std::cout<<"il valore di pvalup è : "<<pval_up<<std::endl;
            //std::cout<<"il valore di pvaldown è : "<<pval_down<<std::endl;
            result.resize(p);
            result = 2 * min(pval_up, pval_down); // Obtain the blateral p_value starting from the unilateral
        } 
        return result;
     }   

    
     void V() override{
        DMatrix<double> inverseA_ {};
        inverseA_ =  - m_.invA().solve(DMatrix<double>::Identity(2 * m_.n_basis(),2 * m_.n_basis()));
        Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * inverseA_.block(0, 0, m_.n_basis(), m_.n_basis()) * m_.PsiTD();
     }   

     inline bool is_Unilaterally_Greater (DVector<double> v, DVector<double> u){
        int q = v.size();
        for (int i = 0; i < q; ++i){
            if(v(i) <= u(i)){
                return false;
            }
        }
        return true;
     };

     inline bool is_Unilaterally_Smaller (DVector<double> v, DVector<double> u){
      int q = v.size();
      for (int i = 0; i < q; ++i){
        if(v(i) >= u(i)){
        return false;
        }
     }
    return true;
    };

     inline DVector<double> min(const DVector<double> & v, const DVector<double> & u){
        DVector<double> result;
        result.resize(v.size());
        for(int i = 0; i < v.size(); ++i){
            result(i) = std::min(v(i), u(i));
        }
        return result;
     } 
    
     void setNflip(int m){
        n_flip = m;
     }  

};

} // namespace models
} // namespace fdapde

#endif // __ESF2_H__
