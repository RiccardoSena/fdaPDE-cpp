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

#ifndef __PESF_H__
#define __PESF_H__

// questi sono da controllare 
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "../model_base.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "stochastic_edf.h" 
#include "inference_base.h"
#include "inference.h"

#include <algorithm>

// for parallelization
#include <omp.h>


namespace fdapde {
namespace models {

template <typename Model, typename Strategy> class PESF: public InferenceBase<Model>{

    private:
     struct ExactInverse{
         DMatrix<double> compute(Model m){
            return inverse(m.E());       
         }
      };

      struct NonExactInverse{
         SpMatrix<double> compute(Model m){
            return Base::invE_approx(m);  
         }
      };

     int n_flip = 1000; //default value
     int set_seed=0;

     DMatrix<double> Lambda_ {};
     
     // variabili aggiunte per confidence intervals 
     bool is_speckman_aux_computed = false;
     DVector<double> Speckman_aux_ranges;                         //Speckman auxiliary CI ranges needed for CI method initialization (for beta)
     // vairabili aggiunte per inferenza su f 
     DMatrix<double> Qp_ {};
     int p_l_ = 0;   // number of locations for inference on f
     SpMatrix<double> Psi_p_ {};   // Psi only in the locations for inference
     DMatrix<double> Qp_dec_ {}; // Decomposition of Qp matrix
     DVector<double> mesh_nodes_ {};  // if inference is performed on a subset of mesh nodes

     
    public:
     using Base = InferenceBase<Model>;
     using Base::m_;
     using Base::beta_;  
     using Base::f0_;
     using Base::C_;
     using Base::locations_f_;
     using Base::alpha_f_;
     using Base::beta0_;
     using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
     Solver s_;
     // aggiunta per CI 
     using Base::V_;

     // constructors
     PESF() = default;                   // deafult constructor
     PESF(const Model& m): Base(m) {};     // constructor      

     

     DVector<double> p_value_serial(CIType type){
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

        Eigen::SelfAdjointEigenSolver<DMatrix<double>> solver(Lambda_); // compute eigenvectors and eigenvalues of Lambda

        DMatrix<double> eigenvalues = solver.eigenvalues();
        DMatrix<double> eigenvectors = solver.eigenvectors();

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
            int n_obs=m_.n_obs();

            DVector<double> Tilder_hat= eigenvectors.transpose()*(m_.y() - m_.X()* beta_hat);

            // Estimate the standard error
            DVector<double> eps_hat =  m_.y() - m_.fitted();
            double SS_res = eps_hat.squaredNorm();
            double Sigma_hat = std::sqrt(SS_res/(n_obs-1));

            double threshold = 10*Sigma_hat; // This threshold is used to determine how many components will not be flipped: we drop those that show large alpha_hat w.r.t. the expected standar error
            int N_Eig_Out=0; // It will store the number of biased components that will be kept fixed if enhanced-ESF is required
            
            // Initialize observed statistic and sign-flipped statistic
            DVector<double> stat = Xt * Tilder;
            DVector<double> stat_flip = stat;

            //Random sign-flips
             //Random sign-flips
            std::default_random_engine eng;
            std::uniform_int_distribution<int> distr(0, 1); 

            //if we have a set seed 
            if(set_seed != 0) {
                eng.seed(set_seed);
            } else {
                std::random_device rd; 
                eng.seed(rd()); // random seed 
            }
            int up = 0;
            int down = 0;

            DVector<double> Tilder_perm = Tilder;
            
             for(int i = 0; i < n_flip; ++i){
                N_Eig_Out=0;
                for(int j = 0; j < Xt.cols(); ++j){
                    int flip;
                    if((N_Eig_Out<n_obs/2) && (fabs(Tilder_hat(j))>threshold)){
                        flip=1;
                    }else{
                        flip=2 * distr(eng) - 1;
                    }
                    Tilder_perm.row(j) = Tilder.row(j) * flip;
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

            int n_obs=m_.n_obs();

            DVector<double> Tilder_hat= eigenvectors.transpose()*(m_.y() - m_.X()* beta_hat);

            // Estimate the standard error
            DVector<double> eps_hat =  m_.y() - m_.fitted();
            double SS_res = eps_hat.squaredNorm();
            double Sigma_hat = std::sqrt(SS_res/(n_obs-1));
          
            double threshold = 10*Sigma_hat; // This threshold is used to determine how many components will not be flipped: we drop those that show large alpha_hat w.r.t. the expected standar error
            int N_Eig_Out=0; // It will store the number of biased components that will be kept fixed if enhanced-ESF is required
            
            
            // Observed statistic
            DMatrix<double> stat = Xt * Tilder;
            DMatrix<double> stat_flip = stat;

            // Random sign-flips
            std::default_random_engine eng;
            std::uniform_int_distribution<int> distr(0, 1); 

            //if we have a set seed 
            if(set_seed != 0) {
                eng.seed(set_seed);
            } else {
                std::random_device rd; 
                eng.seed(rd()); // random seed 
            }
            DVector<double> up = DVector<double>::Zero(p);
            DVector<double> down = DVector<double>::Zero(p);
            
            DMatrix<double> Tilder_perm = Tilder;
            
            for(int i = 0; i < n_flip; ++i){
                N_Eig_Out=0;
                for(int j = 0; j < Xt.cols(); ++j){
                    int flip;
                    if((N_Eig_Out<n_obs/2) && (fabs(Tilder_hat(j))>threshold)){
                        flip=1;
                    }else{
                        flip=2 * distr(eng) - 1;
                    }
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

DMatrix<double> computeCI_serial(CIType type){
        // compute Lambda
        if(is_empty(Lambda_)){
            V();
        }

        // Store beta_hat
        DVector<double> beta_hat = m_.beta();
        DVector<double> beta_hat_mod = beta_hat;
        
        fdapde_assert(!is_empty(C_));      // throw an exception if condition is not met  
        int p = C_.rows(); 

        DVector<int> beta_in_test; // In this vector are stored the respective positions of the beta we are testing in the actual test (ie le posizioni dei beta che vengono testate perchè in C abbiamo un 1 nella corrispondente posizione)
        beta_in_test.resize(p);
        for(int i=0; i<p; i++){
            for(int j=0; j<C_.cols(); j++){
                if(C_(i,j)>0){
                    beta_in_test[i]=j;
                }
            }
        }

        // compute eigenvectors and eigenvalues of Lambda
 Eigen::SelfAdjointEigenSolver<DMatrix<double>> solver(Lambda_); // compute eigenvectors and eigenvalues of Lambda

        DMatrix<double> eigenvalues = solver.eigenvalues();
        DMatrix<double> eigenvectors = solver.eigenvectors();


        // declare the matrix that will store the intervals
        DMatrix<double> result;
        result.resize(p, 2);

        // compute the initial ranges from speckman's CI (initial guess for CI) 
        if(!is_speckman_aux_computed){
            Compute_speckman_aux();
        }

        // this vector will store the tolerance for each interval upper/lower limit
        // QUI NON SO SE 0.1 O 0.2 PER LA TOLLERANZA MASSIMA 
        DVector<double> ESF_bisection_tolerances = 0.1*Speckman_aux_ranges; // 0.1 of the speckman CI as maximum tolerance
        

        // define storage structures for bisection algorithms
        DVector<double> UU; // Upper limit for Upper bound
        UU.resize(p);
        DVector<double> UL; // Lower limit for Upper bound
        UL.resize(p);
        DVector<double> LU; // Upper limit for Lower bound
        LU.resize(p);
        DVector<double> LL; // Lower limit for Lower bound
        LL.resize(p);



        // ESF initialization with Speckman intervals as initial guess 
        for(int i = 0; i < p; ++i){
            double half_range = Speckman_aux_ranges(i);

            // compute the limits of the interval
            // QUI NON SO SE DEVO CONSIDERARE COME LIMITI 1/2 O 3/2 DEL HALF_RANGE
            result(i,0) = beta_hat(beta_in_test[i]) - half_range;
            LU(i)=result(i,0)+0.5*half_range;
            LL(i)=result(i,0)-0.5*half_range;
            result(i,1) = beta_hat(beta_in_test[i]) + half_range;
            UU(i)=result(i,1) +0.5*half_range;
            UL(i)=result(i,1) -0.5*half_range; 	
        }


        // define booleans used to understand which CI need to be computed forward on
        std::vector<bool> converged_up(p,false);
        std::vector<bool> converged_low(p,false);
        bool all_betas_converged=false;

        // matrix that stores p_values of the bounds at actual step
        DMatrix<double> local_p_values;
        local_p_values.resize(4,p);
        
        // compute the vectors needed for the statistic
        DMatrix<double> TildeX = (C_ * m_.X().transpose()) * eigenvectors * eigenvalues.asDiagonal();   	// W^t * V * D
        DMatrix<double> Tilder_star = eigenvectors.transpose();   			        		// V^t
        // Select eigenvalues that will not be flipped basing on the estimated bias carried
        DVector<double> Tilder_hat = eigenvectors.transpose()* (m_.y() - (m_.X())* beta_hat); // This vector represents Tilder using only beta_hat, needed for bias estimation
        DVector<double> Partial_res_H0_CI;
        Partial_res_H0_CI.resize(Lambda_.cols());
        

        // fill the p_values matrix
        for (int i=0; i<p; i++){
            DMatrix<double> TildeX_loc = TildeX.row(beta_in_test[i]);
            beta_hat_mod = beta_hat;

            // compute the partial residuals and p value
            beta_hat_mod(beta_in_test[i])=UU(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
            Partial_res_H0_CI = m_.y() - (m_.X()) * (beta_hat_mod); // (y-W*beta_hat(non in test)-W*UU[i](in test))
            local_p_values(0,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

            // compute the partial residuals and p value
            beta_hat_mod(beta_in_test[i])=UL(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
            Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (y-W*beta_hat(non in test)-W*UL[i](in test))
            local_p_values(1,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

            // compute the partial residuals and p value
            beta_hat_mod(beta_in_test[i])=LU(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
            Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (y-W*beta_hat(non in test)-W*LU[i](in test))
            local_p_values(2,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

            // compute the partial residuals and p value
            beta_hat_mod(beta_in_test[i])=LL(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
            Partial_res_H0_CI = m_.y() - (m_.X()) * (beta_hat_mod); // (y-W*beta_hat(non in test)-W*LL[i](in test))
            local_p_values(3,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

        }

        // extract the CI significance (1-confidence)
        double alpha=0.05;

        if(type == one_at_the_time){
            alpha=0.5*alpha;
        }else{
            alpha=0.5/p*alpha;
        }
    
        int Max_Iter=50;
        int Count_Iter=0;
        while((!all_betas_converged) & (Count_Iter<Max_Iter)){

        
            // Compute all p_values (only those needed)
            for (int i=0; i<p; i++){

            DMatrix<double> TildeX_loc= TildeX.row(beta_in_test[i]);
            beta_hat_mod = beta_hat;

            if(!converged_up[i]){
                if(local_p_values(0,i)>alpha){ // Upper-Upper bound excessively tight

                UU(i)=UU(i)+0.5*(UU(i)-UL(i));
            
                // compute the partial residuals
                beta_hat_mod(beta_in_test[i])=UU(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (z-W*beta_hat(non in test)-W*UU[i](in test))
                local_p_values(0,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);
        
                }else{
        
                    if(local_p_values(1,i)<alpha){ // Upper-Lower bound excessively tight
                        UL(i)=beta_hat(beta_in_test[i])+0.5*(UL(i)-beta_hat(beta_in_test[i]));
                
                        // compute the partial residuals
                        beta_hat_mod(beta_in_test[i])=UL(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                        Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (z-W*beta_hat(non in test)-W*UL[i](in test))
                        local_p_values(1,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

                    }else{//both the Upper bounds are well defined

                        if(UU(i)-UL(i)<ESF_bisection_tolerances(i)){

                        converged_up[i]=true;

                        }else{

                            double proposal=0.5*(UU(i)+UL(i));
        
                            // compute the partial residuals
                            beta_hat_mod(beta_in_test[i])=proposal; // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                            Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (z-W*beta_hat(non in test)-W*proposal)
                            double prop_p_val=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

                            if(prop_p_val<=alpha){UU(i)=proposal; local_p_values(0,i)=prop_p_val;}else{UL(i)=proposal;local_p_values(1,i)=prop_p_val;}
                        }
                    }
                }
            }  // end if


            if(!converged_low[i]){
	            if(local_p_values(2,i)<alpha){ // Lower-Upper bound excessively tight

	                LU(i)=beta_hat(beta_in_test[i])-0.5*(beta_hat(beta_in_test[i])-LU(i));
  
                    // compute the partial residuals
                    beta_hat_mod(beta_in_test[i])=LU(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                    Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (z-W*beta_hat(non in test)-W*LU[i](in test))
                    local_p_values(2,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);
                
	            }else{
 
                    if(local_p_values(3,i)>alpha){ // Lower-Lower bound excessively tight
                        LL(i)=LL(i)-0.5*(LU(i)-LL(i));
                
                        // compute the partial residuals
                        beta_hat_mod(beta_in_test[i])=LL(i); // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                        Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod);// (z-W*beta_hat(non in test)-W*LL[i](in test))
                        local_p_values(3,i)=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

                    }else{//both the Upper bounds are well defined

	                    if(LU(i)-LL(i)<ESF_bisection_tolerances(i)){

	                        converged_low[i]=true;

	                    }else{

	                        double proposal=0.5*(LU(i)+LL(i));
   
	                        // compute the partial residuals
                            beta_hat_mod(beta_in_test[i])=proposal; // beta_hat_mod(i) = beta_hat(i) if i not in test; beta_HP otherwise
                            Partial_res_H0_CI =  m_.y() - (m_.X()) * (beta_hat_mod); // (z-W*beta_hat(non in test)-W*proposal)
                            double prop_p_val=compute_CI_aux_beta_pvalue(Partial_res_H0_CI, TildeX_loc, Tilder_star);

                            if(prop_p_val<=alpha){LL(i)=proposal; local_p_values(3,i)=prop_p_val;}else{LU(i)=proposal;local_p_values(2,i)=prop_p_val;}
                        }
                    }
                }
            }  // end if
        }  // end for
        all_betas_converged =true;
        for(int j=0; j<p; j++){

            if(!converged_up[j] || !converged_low[j]){
	            all_betas_converged=false;
            }
        }

        Count_Iter++;

        }
        result.resize(p,2);
        // for each row of C matrix
        for(int i=0; i<p; ++i){
                 
            if(Count_Iter < Max_Iter){ // No discrepancy between beta_hat(i) and ESF, bisection converged
                // Limits of the interval
                result(i,0) = 0.5*(LU(i)+LL(i));
                result(i,1) = 0.5*(UU(i)+UL(i)); 
                }else{ // Not converged in time, give a warning in R
                // Limits of the interval
                result(i,0) = 10e20;
                result(i,1) = 10e20; 
            }
        }
        
        return result;
        
    };

    void Compute_speckman_aux(void){
        // questo è il calcolo di Speckman intervals per initial guess per CI di ESF 
        // COSTRUITA ESATTAMENTE COME LA NOSTRA 
        fdapde_assert(!is_empty(C_));  

        double alpha_=0.05;
        if(is_empty(V_)){
           V();
        }

        int p = C_.rows();
        int size = std::min(C_.rows(), V_.rows());
        DVector<double> diagon(size);
        for (int i = 0; i < C_.rows(); ++i) {
           DVector<double> ci = C_.row(i);
           diagon[i] = ci.transpose() * V_ * ci;
        }

        // ONE AT THE TIME
        double quantile = normal_standard_quantile(1 - alpha_/2);            
         
        Speckman_aux_ranges.resize(p);
        Speckman_aux_ranges=quantile * (diagon.array()).sqrt();

        is_speckman_aux_computed = true; 
        return;
    
    }

  double compute_CI_aux_beta_pvalue(const DVector<double> & partial_res_H0_CI, const DMatrix<double> & TildeX,  const  DMatrix<double> & Tilder_star) const {
        // declare the vector that will store the p-values
        double result;
    
        // compute the vectors needed for the statistic 
        DVector<double> Tilder = Tilder_star * partial_res_H0_CI;

        // Initialize observed statistic and sign_flipped statistic
        DMatrix<double> stat_temp = TildeX*Tilder;
        double stat=stat_temp(0);
        double stat_flip=stat;

        // Random sign-flips
            std::default_random_engine eng;
            std::uniform_int_distribution<int> distr(0, 1); 

            //if we have a set seed 
            if(set_seed != 0) {
                eng.seed(set_seed);
            } else {
                std::random_device rd; 
                eng.seed(rd()); // random seed 
            }

        double count_Up = 0;   // Counter for the number of flipped statistics that are larger the observed statistic
        double count_Down = 0; // Counter for the number of flipped statistics that are smaller the observed statistic
            
        DVector<double> Tilder_perm=Tilder;
    
        // get the number of flips
        int nflip=n_flip;

        for(int i=0;i<nflip;i++){
            for(int j=0;j<TildeX.cols();j++){
                int flip;
                flip=2*distr(eng)-1;
                Tilder_perm(j)=Tilder(j)*flip;
            }
                DMatrix<double> stat_flip_temp = TildeX*Tilder_perm; 
                stat_flip= stat_flip_temp(0);// Flipped statistic
                if(stat_flip > stat){ ++count_Up;}else{ 
                    if(stat_flip < stat){ ++count_Down;}  
                }
        }
            
        double pval_Up = count_Up/n_flip;     
        double pval_Down = count_Down/n_flip; 

        result = std::min(pval_Up, pval_Down); // select the correct unilateral p_value 

        return result;
        
    };
    
        





    void Psi_p(){
      // case in which the locations are extracted from the observed ones
      if(is_empty(locations_f_) && is_empty(mesh_nodes_)){
        Psi_p_ = m_.Psi();
        if(p_l_ == 0)
           p_l_ = Psi_p_.rows();
      }
      else if(is_empty(mesh_nodes_)){
      int m = locations_f_.size();
      SpMatrix<double> Psi = m_.Psi();
      Psi_p_.resize(m, Psi.cols());
      for(int j = 0; j < m; ++j) {
        int row = locations_f_[j];
        for(SpMatrix<double>::InnerIterator it(Psi, row); it; ++it) {
            Psi_p_.insert(j, it.col()) = it.value();
        }
      }
      }
      
      else {
      // pay attention that this has the opposite dimensions of the other Psi
      int m = mesh_nodes_.size();
      SpMatrix<double> Psi = m_.PsiESF();
      Psi_p_.resize(m, Psi.cols());
      for(int j = 0; j < m; ++j) {
        int row = mesh_nodes_[j];
        for(SpMatrix<double>::InnerIterator it(Psi, row); it; ++it) {
            Psi_p_.insert(j, it.col()) = it.value();
        }       
      }
      }
      
      Psi_p_.makeCompressed();
      if(p_l_ == 0)
        p_l_ = Psi_p_.rows();

    }

    DVector<double> yp(){
      if(is_empty(locations_f_)  && is_empty(mesh_nodes_))
        return m_.y();

      else if (is_empty(mesh_nodes_)){
        
      int m = locations_f_.size();
      DVector<double> y = m_.y();
      DVector<double> yp;
      yp.resize(m);
      for(int j = 0; j < m; ++j) {
        int row = locations_f_[j];
        yp.row(j) = y.row(row);
      }
      return yp;
      }
      else{
      int m = mesh_nodes_.size();
      DVector<double> y = m_.y();
      DVector<double> yp;
      yp.resize(m);
      for(int j = 0; j < m; ++j) {
        int row = mesh_nodes_[j];
        yp.row(j) = y.row(row);
      }
      return yp;
      }
    }

    DMatrix<double> Xp(){
      // case in which the locations are extracted from the observed ones
      if(is_empty(locations_f_) && is_empty(mesh_nodes_)){
        return m_.X();
      }
      if(!m_.has_covariates())
        return m_.X();

      if(is_empty(mesh_nodes_)){
      int m = locations_f_.size();
      DMatrix<double> X = m_.X();
      DMatrix<double> Xp;
      Xp.resize(m, X.cols());
      for(int j = 0; j < m; ++j) {
        int row = locations_f_[j];
        Xp.row(j) = X.row(row);
      }
      return Xp;
      }
      int m = mesh_nodes_.size();
      DMatrix<double> X = m_.X();
      DMatrix<double> Xp;
      Xp.resize(m, X.cols());
      for(int j = 0; j < m; ++j) {
        int row = mesh_nodes_[j];
        Xp.row(j) = X.row(row);
      }
      return Xp;      
    }

    DMatrix<double> Wp(){
        // how to deal with this
        // for now set it as a identity matrix
        if(is_empty(Psi_p_)){
            Psi_p();
        }   
        DMatrix<double> id = DMatrix<double>::Identity(p_l_, p_l_);
        return id;
    }

    // computes matrix Q = W(I - X*(X^\top*W*X)^{-1}*X^\top*W)
    void Qp() {
        if (!m_.has_covariates()){
            Qp_ = Wp() * DMatrix<double>::Identity(p_l_, p_l_);
            return;
        }   
        DMatrix<double> Xp_ = Xp();    
        DMatrix<double> Wp_ = Wp();     
        DMatrix<double> XptWp = Xp_.transpose() * Wp_;   // X^\top*W,   q x p_l_ 
        DMatrix<double> invXptWpXp = inverse(Xp_.transpose() * Wp_ * Xp_);    // q x q       
        // perchè unica richiesta di solve per PartialPivLU è che il numero di righe di XtWX e v sia uguale
        // compute W - W*X*z = W - (W*X*(X^\top*W*X)^{-1}*X^\top*W) = W(I - H) = Q
        Qp_ =  Wp_ * DMatrix<double>::Identity(p_l_, p_l_) - Wp_ * Xp_ * invXptWpXp * XptWp;

    }

    void Qp_dec(){
        // compute the Q only on the locations in which you want inference
        if(is_empty(Qp_))
           Qp();

        // Q * (y - epislon) = Q * r
        // Eigen sign flip implementation
        Eigen::SelfAdjointEigenSolver<DMatrix<double>> Q_eigen(Qp_);  

        // matrix V is the matrix p_l_ x (p_l_-q) of the nonzero eigenvectors of Q
        DMatrix<double> Q_eigenvectors = Q_eigen.eigenvectors();
        Qp_dec_ = Q_eigenvectors.rightCols(p_l_ - m_.X().cols());
    }



    


     void V() override{
        // questa è quella modificata per FSPAI
        //DMatrix<double> inverseA_ {};
        //inverseA_ =  - m_.invA().solve(DMatrix<double>::Identity(2 * m_.n_basis(),2 * m_.n_basis()));
        //Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * inverseA_.block(0, 0, m_.n_basis(), m_.n_basis()) * m_.PsiTD();
        Lambda_ = DMatrix<double>::Identity(m_.n_obs(), m_.n_obs()) - m_.Psi() * s_.compute(m_) * m_.PsiTD();

        //aggiunto per CI 
        DMatrix<double> W = m_.X();
        DMatrix<double> invWtW = inverse(W.transpose() * Lambda_ * (W));
        DVector<double> eps_ = (m_.y() - m_.fitted());
        DVector<double> Res2 = eps_.array() * eps_.array();            
        // resize the variance-covariance matrix
        V_.resize(m_.q(), m_.q());                   
        DMatrix<double> W_t = W.transpose();           
        DMatrix<double> diag = Res2.asDiagonal();
        V_ = invWtW * (W_t) * Lambda_ * Res2.asDiagonal() * Lambda_ * (W) * invWtW;
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
     };
     void setseed(int k){
        set_seed=k;
     }
     void setMesh_loc(DVector<double> m_nodes){
        mesh_nodes_ = m_nodes;
     }

};

} // namespace models
} // namespace fdapde

#endif // __PESF_H__
