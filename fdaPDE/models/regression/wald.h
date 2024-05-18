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

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
using fdapde::core::FSPAI;
using fdapde::core::lump;

#include "../model_macros.h"
#include "../model_traits.h"
#include "../model_base.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"
#include "stochastic_edf.h"
#include "inference_base.h"
#include "inference.h"

// per salvare la matrice con savemarket
#include <unsupported/Eigen/SparseExtra> 

namespace fdapde {
namespace models {

template <typename Model, typename Strategy> class Wald: public InferenceBase<Model> {

    private: 
     struct ExactInverse {
        DMatrix<double> compute(Model m){
            return inverse(m.T());
        }
     };
     struct NonExactInverse {
        SpMatrix<double> compute(Model m){
            DMatrix<double> Ut_ = m.Psi().transpose() * m.X();
            DMatrix<double> Ct_ = - inverse(m.X().transpose() * m.X());
            DMatrix<double> Vt_ = m.X().transpose() * m.Psi();
            SpMatrix<double> invE_ = Base::invE_approx(m);
            
            SpMatrix<double> invMt_ = invE_ + invE_ * Ut_ * inverse(Ct_ + Vt_ * invE_ * Ut_) * Vt_ * invE_;
            return invMt_;            
        }       
     };


    public: 
     using Base = InferenceBase<Model>;
     using Base::m_;
     using Base::V_;
     using Base::f0_;
     using Base::alpha_f_;
     using Base::beta_;
     using Base::invE_approx;
     using Solver = typename std::conditional<std::is_same<Strategy, exact>::value, ExactInverse, NonExactInverse>::type;
     Solver s_; 
     DMatrix<double> Vf_ {};
     DMatrix<double> Psi_p_ {};
     DVector<double> f_p_ {};

     // constructors
     Wald() = default;                   // deafult constructor
     Wald(const Model& m): Base(m) {};     // constructor    

     void beta() override{
        beta_ = m_.beta();
     }

     // si potrebbe fare override anche di questo metodo visto che si può utilizzare StochasticEDF per calcolare la traccia
     double sigma_sq() {
        double sigma_sq_ = 0;             // sigma^2 
        DMatrix<double> epsilon = m_.y() - m_.fitted();
        ExactEDF strat;
        strat.set_model(m_);
        sigma_sq_  = (1/(m_.n_obs() - m_.q() - strat.compute())) * epsilon.squaredNorm();
        return sigma_sq_;
     }

     void V() override{
        DMatrix<double> invSigma_ = inverse(m_.X().transpose() * m_.X());
        DMatrix<double> S = m_.Psi() * s_.compute(m_) * m_.PsiTD() * m_.Q(); 
        DMatrix<double> ss = S * S.transpose();
        DMatrix<double> left = invSigma_ * m_.X().transpose();
        V_ = sigma_sq() * (invSigma_ + left * ss * left.transpose()); 
     }

     void Psi_p(){
     }

     void f_p(){
      if(is_empty(Psi_p_))
         Psi_p();
      f_p_ = Psi_p_ * m_.f(); 
     }

     void Vf(){
      // covariance matrice of f^
      // still difference in exact and non exact when computing S
      DMatrix<double> S_psiT = s_.compute(m_) * m_.PsiTD(); // is it Psi.transpose or PsiTD???
      DMatrix<double> Vff = sigma_sq() * S_psiT * m_.Q() * S_psiT.transpose(); 

      // need to create a new Psi: matrix of basis evaluation in the set of observed locations
      // belonging to the chosen portion Omega_p

      // for now just Psi
      Psi_p_ = m_.Psi();
      Vf_ = Psi_p_ * Vff * Psi_p_.transpose();
     }

     DMatrix<double> invVf(){
      if(is_empty(Vf_))
         Vf();
      // reduction of matrix
      // discard eigenvalues that are too small
      // Vw is a covariance matrix, hence it is symmetric
      // A = V * D * V^{-1} = V * D * V^T
      // V is the matrix of eigenvectors and D is the diagonal matrix of the eigenvalues

      // to retrieve eigenvalues and eigenvectors of the Vw matrix
      Eigen::SelfAdjointEigenSolver<DMatrix<double>> Vw_eigen(Vf_);
      // eigenvalues
      DVector<double> eigenvalues = Vw_eigen.eigenvalues();

      // now we need to discard the one really close to 0
      // the eigenvalues are in increasing oreder and since the covariance matrix is spd
      // there won't be any negative values

      // fix a threshold
      double thresh = 1e-7;
      
      // we need to get the index for the first eigenvalue greater than the threshold
      int flag = 0;
      int it = 0;
      while(flag == 0 && it < eigenvalues.size()){
         if(eigenvalues(it) > thresh)
            flag = 1;
         ++it;
      }
      
      // rank
      int r = eigenvalues.size() - it + 1;
      // consider only the significant eigenvalues and create the diagonal matrix
      DVector<double> imp_eigval = eigenvalues.tail(r);
      // consider only the significant eigenvectors
      DMatrix<double> imp_eigvec = Vw_eigen.eigenvectors().rightCols(r);

      // now we can compute the r-rank pseudoinverse
      DVector<double> temp = imp_eigval.array().inverse();
      DiagMatrix<double> inv_imp_eigval = temp.asDiagonal();
      DMatrix<double> invVf = imp_eigvec * inv_imp_eigval * imp_eigvec.transpose();

      return invVf;      
     }

      double f_p_value(){ 
         if(is_empty(Vf_))
            Vf();
         if(is_empty(f0_))
            Base::setf0(DVector<double>::Zero(m_.f().size()));
         if(is_empty(f_p_))
            f_p();
         // compute the test statistic
         // should only consider the f of the considered locations!!!!!
         double stat = (f_p_ - f0_).transpose() * invVf() * (f_p_ - f0_);
         double pvalue = 0;
         // distributed as a chi squared of r degrees of freedom
         double p = chi_squared_cdf(stat, Vf_.rows());
         if(p < 0){
            pvalue = 1;
         }
         if(p > 1){
            pvalue = 0;
         }
         else{
            pvalue = 1 - p;
         }
         return pvalue;
      }

      DMatrix<double> f_CI(){

         if(is_empty(Vf_))
            Vf();
         
         if(alpha_f_ == 0)
            Base::setAlpha_f(0.05);

         if(is_empty(f_p_))
            f_p();

         // Psi_p_ should be p x n, where n is the number of basis and 
         // p the locations in which you want inference
         int p = Vf_.rows();
         DVector<double> diagon = Vf_.diagonal();
         DVector<double> lowerBound(p);
         DVector<double> upperBound(p);
         double quantile = normal_standard_quantile(1 - alpha_f_/2);            
         lowerBound = f_p_.array() - quantile * (diagon.array()).sqrt();
         upperBound = f_p_.array() + quantile * (diagon.array()).sqrt();

         DMatrix<double> CIMatrix(p, 2);      //matrix of confidence intervals
         CIMatrix.col(0) = lowerBound;
         CIMatrix.col(1) = upperBound;
         return CIMatrix;
      }



};

} // namespace models
} // namespace fdapde

#endif   // __WALD_H__