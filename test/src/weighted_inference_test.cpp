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


// QUESTO FILE VA SPOSTATO IN FDAPDE-CORE/TEST/SRC
// NEL MAIN.CPP DELLA REPOSITORY FDAPDE-CORE/TEST VA AGGIUNTO 
// #include inference_test.cpp
// e la parte che chiama la classe inference_test 


// questi sono da controllare 
#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework
#include <cstddef>
#include <chrono>

#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"

#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;

using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;


#include "../../fdaPDE/models/regression/wald.h"
#include "../../fdaPDE/models/regression/speckman.h"
#include "../../fdaPDE/models/regression/esf.h"
#include "../../fdaPDE/models/regression/pesf.h"

#include <../../../fdaPDE-core/fdaPDE/core.h>
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::DiscretizedVectorField;



// TEST WEIGHTED 
    

TEST(weighted_inference_test, simulations_beta_weighted1) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("TIME/20nodes");
    DMatrix<double> locs = read_csv<double>("../data/mesh/TIME/20nodes/points.csv");
    
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define  model
    double lambda = 0.01;
    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
 
    //set beta_H1
    DVector<double> beta_H1_list(11);
    beta_H1_list(0)=0.0;
    for (int i = 1; i < beta_H1_list.size(); ++i) {
        beta_H1_list(i)=beta_H1_list(i-1)+0.01;
                }            

    int max_iter=100;
    DMatrix<DVector<double>> pvaluesW(beta_H1_list.size(),max_iter);

    for (int i = 0; i < beta_H1_list.size(); ++i) {
        for (int iter = 0; iter < max_iter; ++iter) {
            DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_testweight1/observations_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_testweight1/covariates_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DVector<double> weights    = read_csv<double>("../data/models/srpde/2D_testweight1/variances_" + std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DMatrix<double> weightsMatrix = DMatrix<double>::Map(weights.data(), weights.size(), 1);
            
            df.insert(OBSERVATIONS_BLK, y);
            df.insert(DESIGN_MATRIX_BLK, X);
            df.insert(WEIGHTS_BLK, weightsMatrix);

            model.set_data(df);
            // solve smoothing problem
            model.init();
            model.solve();

            // define inference objects
            fdapde::models::PESF<SRPDE, fdapde::models::exact> inferenceWald(model);

            df.insert(OBSERVATIONS_BLK, y);
            df.insert(DESIGN_MATRIX_BLK, X);
            df.insert(WEIGHTS_BLK, weightsMatrix);

            model.set_data(df);
            // solve smoothing problem
            model.init();
            model.solve();

            int cols = model.beta().size();
            DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
            inferenceWald.setC(C);
        
            DVector<double> beta0(1);
            beta0(0)=0.0;
            inferenceWald.setBeta0(beta0);
            inferenceWald.setseed(46);
            inferenceWald.setNflip(1000);

            DVector<double> pvalueswald = inferenceWald.p_value_serial(fdapde::models::one_at_the_time);
            pvaluesW(i,iter)=pvalueswald;
            std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;
        }
    }

    std::vector<double> power_mat(beta_H1_list.size()); // 11 righe, 3 colonne


    for (int i = 0; i < beta_H1_list.size(); ++i) {
        double count = 0;  // Inizializza il contatore per la riga corrente
        for (int j = 0; j < max_iter; ++j) {
            // Verifica se il primo elemento della riga è minore di 0.05
            if (pvaluesW(i,j)(0) < 0.05) {
                count++;  // Incrementa il contatore se la condizione è vera
            }
        }
        power_mat[i]=count/max_iter;
    }

    // Stampa del risultato (per verificare)
    for (int i = 0; i < beta_H1_list.size() ; ++i) {
        
            std::cout << power_mat[i] << " ";
        
        std::cout << std::endl;
    }

}








TEST(weighted_inference_test, simulations_beta_weighted2) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("TIME/20nodes");
    DMatrix<double> locs = read_csv<double>("../data/mesh/TIME/20nodes/points.csv");
        //std::cout<<"dopo modello matrix ok"<<std::endl;
       // if (locs.rows() == 0 || locs.cols() != 2) {
  //  std::cerr << "Errore: 'locs' non ha il numero corretto di righe o colonne!" << std::endl;
   
//}

//std::cout << "locs ha " << locs.rows() << " righe e " << locs.cols() << " colonne." << std::endl;



    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define  model
    double lambda = 0.01;
    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
           // std::cout<<"dopo MODELLO  ok"<<std::endl;

    DVector<double> beta_H1_list(11);
    beta_H1_list(0)=0.0;
    for (int i = 1; i < beta_H1_list.size(); ++i) {
        beta_H1_list(i)=beta_H1_list(i-1)+0.01;
                }            
    //std::cout<<"dopo h1  ok"<<beta_H1_list<<std::endl;

    int max_iter=100;
    DMatrix<DVector<double>> pvaluesW(beta_H1_list.size(),max_iter);
          //  std::cout<<"dopo pvalues matrix ok"<<std::endl;

    for (int i = 0; i < beta_H1_list.size(); ++i) {
        for (int iter = 0; iter < max_iter; ++iter) {
            DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_testweight/2observations_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_testweight/2covariates_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DVector<double> weights    = read_csv<double>("../data/models/srpde/2D_testweight/2variances_" + std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DMatrix<double> weightsMatrix = DMatrix<double>::Map(weights.data(), weights.size(), 1);
                  //  std::cout<<"dopo osservazioni matrix ok"<<std::endl;

// Controllo se 'locs' è corretta prima di procedere
//if (locs.size() == 0 || locs.cols() != domain.mesh.embed_dim) {
   // std::cerr << "Errore: 'locs' non ha il numero corretto di colonne o è vuota!" << std::endl;
   // continue;  // Skip questo ciclo se 'locs' non è valida
//}

            df.insert(OBSERVATIONS_BLK, y);
            df.insert(DESIGN_MATRIX_BLK, X);
            df.insert(WEIGHTS_BLK, weightsMatrix);

            model.set_data(df);
            // solve smoothing problem
            model.init();
            model.solve();

            // Ciclo per stampare gli elementi della matrice dei pesi 
            //for (int i = 0; i < 10; ++i) { // Itera sulle righe
              //      std::cout << "Elemento della matrice del modello  [" << i << "] = " << model.W().diagonal()(i) << std::endl;
              //   }

           // std::cout<<"beta stimati dal modello"<<model.beta()<<std::endl;

            // define inference objects
            fdapde::models::PESF<SRPDE, fdapde::models::exact> inferenceWald(model);

    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    df.insert(WEIGHTS_BLK, weightsMatrix);

    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

  //  std::cout<<"dopo modello ok"<<std::endl;

// Ciclo per stampare gli elementi della matrice dei pesi 
    for (int k = 0; k < 2; ++k) { // Itera sulle righe
            std::cout << "Elemento della matrice del modello  [" << k << "] = " << model.W().diagonal()(k) << std::endl;
   }

//std::cout<<"beta stimati dal modello"<<model.beta()<<std::endl;


  int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
   

    DVector<double> beta0(1);
    beta0(0)=0.0;
    inferenceWald.setBeta0(beta0);
    inferenceWald.setseed(46);
    inferenceWald.setNflip(1000);
   

    

    DVector<double> pvalueswald = inferenceWald.p_value_serial(fdapde::models::one_at_the_time);
    pvaluesW(i,iter)=pvalueswald;
   // std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;


        }
    }

//for (int i = 0; i < beta_H1_list.size(); ++i) {
     //   for (int iter = 0; iter < max_iter; ++iter) {


  //      std::cout<<"pvalues H1 "<<beta_H1_list(i)<<"iterazione "<<iter<<std::endl;
  //      std::cout<<pvaluesW(i,iter)<<std::endl;
      //  }
//}

    std::vector<double> power_mat(beta_H1_list.size()); // 11 righe, 3 colonne


for (int i = 0; i < beta_H1_list.size(); ++i) {
        double count = 0;  // Inizializza il contatore per la riga corrente
        for (int j = 0; j < max_iter; ++j) {
            // Verifica se il primo elemento della riga è minore di 0.05
            if (pvaluesW(i,j)(0) < 0.05) {
                count++;  // Incrementa il contatore se la condizione è vera
            }
        }

        // Stampa il risultato per la riga i
    //    std::cout << "Per la riga " << i + 1 << ", il numero di valori < 0.05 è: " << count << std::endl;
     //   std::cout<<count/max_iter<<std::endl;
        power_mat[i] = count/max_iter;

    }




    // Stampa del risultato (per verificare)
    for (int i = 0; i < beta_H1_list.size() ; ++i) {
        
            std::cout << power_mat[i] << " ";
        
        std::cout << std::endl;
    }

}




TEST(weighted_inference_test, power_f){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("power_f2D");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/power_f2D/locs.csv");
    DMatrix<double> f(locs.rows(), 1);
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/power_f2D/X.csv");

    auto gamSim_2 = [](double x, double y) {
    double pi = M_PI; 
    double term1 = 1.2 * exp(-pow(x - 0.2, 2) / pow(0.3, 2) - pow(y - 0.3, 2) / pow(0.4, 2));
    double term2 = 0.8 * exp(-pow(x - 0.7, 2) / pow(0.3, 2) - pow(y - 0.8, 2) / pow(0.4, 2));
    return (0.4 * pow(pi, 0.3)) * (term1 + term2);
    };

    for (int i = 0; i < locs.rows(); ++i){
        f(i, 0) = gamSim_2(locs(i,0), locs(i,1));
    }

    int m = 21;
    DVector<double> scales(m);
    scales << -0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0,
    0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2;
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;

    int n_loc = 117;
    DVector<int> loc_indexes(n_loc);
    loc_indexes << 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 16, 19, 22, 24, 25, 26, 28, 30, 34, 36, 38, 
39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 55, 58, 59, 61, 63, 69, 73, 75, 76, 77, 80, 82, 85, 
88, 89, 96, 98, 103, 106, 114, 116, 117, 120, 122, 124, 127, 131, 132, 133, 134, 135, 139, 141, 
142, 143, 144, 146, 148, 149, 152, 154, 155, 156, 157, 158, 162, 163, 164, 166, 169, 172, 173, 
176, 178, 179, 181, 183, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 198, 199, 200, 
201, 202, 203, 204, 206, 208, 213, 215, 216, 218, 219, 222, 223;



    DVector<double> f_0(n_loc);
    for(int p = 0; p < n_loc; ++p){
        f_0(p) = gamSim_2(locs(loc_indexes(p), 0), locs(loc_indexes(p), 1));
    }

    int rep = 10;

    //DMatrix<double> res_wald(rep, m);
    DMatrix<double> res_sf(rep, m);
    DMatrix<double> res_esf(rep, m);

    double sd = 0.1;
    
    for(int i = 0; i < m; ++i){
        //DVector<double> pval_wald(rep);
        DVector<double> pval_sf(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            // Distribuzioni per errore
            std::normal_distribution<double> normal_dist(0, sd);  // Per errore normale
std::gamma_distribution<double> gamma_dist(2.0, sd);  // Gamma distribution correzione per varianza 
            
            DVector<double> random_vector(f.size());
    DVector<double> weights(f.size());  // Vettore dei pesi
            std::cout << "f size: " << f.size() << std::endl;


    for (int j = 0; j < random_vector.size(); ++j) {
        // Calcola la deviazione standard eteroschedastica per ciascun punto
        double ran_min = f.minCoeff();  // Ottieni il valore minimo di f
        double ran_max = f.maxCoeff();  // Ottieni il valore massimo di f

        double ran_range = ran_max - ran_min;  // Calcola il range dei valori di f
        std::cout << "Range: " << ran_range << std::endl;

        // Correzione esponenziale per varianza
        double err_correction = gamma_dist(generator);  // Esegui la correzione esponenziale per la varianza
        std::cout << "Correzione varianza errore: " << err_correction << std::endl;

        // Deviazione standard eteroschedastica basata sul range dei valori di f
        double dynamic_sd = (ran_range * 0.10) / std::sqrt(err_correction);  // Deviazione standard eteroschedastica
        std::cout << "Deviazione standard dinamica: " << dynamic_sd << std::endl;

        // Genera errore con deviazione eteroschedastica
        random_vector[j] = normal_dist(generator) * dynamic_sd;

        // Calcola i pesi basati sulla deviazione eteroschedastica
        weights[j] = dynamic_sd;  // Pesi basati sulla deviazione standard eteroschedastica
    }
    DMatrix<double> weightsMatrix = DMatrix<double>::Map(weights.data(), weights.size(), 1);
                std::cout << "creazione varianza  matrice del modello  "<< std::endl;


            DMatrix<double> observations = X + (1 + scales[i]) * f + random_vector;
                            std::cout << "creazione osservazioni  matrice del modello  "<< std::endl;

        
            SRPDE model(problem, Sampling::pointwise);
            model.set_lambda_D(lambda);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
           df.insert(WEIGHTS_BLK, weightsMatrix);
                    std::cout << "modello ok : " << std::endl;

 

            model.set_data(df);
            model.init();
            model.solve();
        
            fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
            fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

            inferenceWald.setLocationsF(loc_indexes);
            inferenceESF.setLocationsF(loc_indexes);
            inferenceWald.setf0(f_0);
            inferenceESF.setf0(f_0);
            inferenceESF.setNflip(1000);
            //pval_wald[k-1] = inferenceWald.f_p_value();
            pval_sf[k-1] = inferenceESF.sign_flip_p_value();
            pval_esf[k-1] = inferenceESF.f_p_value();

            //std::cout << "Wald: " << pval_wald[k-1] << std::endl;
            std::cout << "SF: " << pval_sf[k-1] << std::endl;
            std::cout << "ESF: " << pval_esf[k-1] << std::endl;

        }
        //res_wald.col(i) = pval_wald;
        res_sf.col(i) = pval_sf;
        res_esf.col(i) = pval_esf;

    }

    //DVector<double> power_matrix_wald(m);
    DVector<double> power_matrix_sf(m);
    DVector<double> power_matrix_esf(m);

    // compute the power
    double threshold = 0.05;
    for (int j = 0; j < m; ++j) {  
        //int count_wald = 0;
        int count_sf = 0;
        int count_esf = 0;
        for (int i = 0; i < res_esf.rows(); ++i){
            //if (res_wald(i, j) < threshold){
            //    count_wald++;
            //}
            if (res_sf(i, j) < threshold){
                count_sf++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        //power_matrix_wald[j] = static_cast<double> (count_wald) / rep; 
        power_matrix_sf[j] = static_cast<double> (count_sf) / rep;   
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;   
    }

    //std::cout << "Power Wald:" << std::endl;
    //std::cout << power_matrix_wald << std::setprecision(7) << std::endl;
    std::cout << "Power Sign Flip:" << std::endl;
    std::cout << power_matrix_sf << std::setprecision(7) << std::endl;
    std::cout << "Power Eigen Sign Flip:" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;

}







TEST(weighted_inference_test, power_f){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("power_f2D");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/power_f2D/locs.csv");
    DMatrix<double> f(locs.rows(), 1);
    DMatrix<double> X = read_csv<double>("../data/models/srpde/power_f2D/X.csv");

    auto gamSim_2 = [](double x, double y) {
        double pi = M_PI;
        double term1 = 1.2 * exp(-pow(x - 0.2, 2) / pow(0.3, 2) - pow(y - 0.3, 2) / pow(0.4, 2));
        double term2 = 0.8 * exp(-pow(x - 0.7, 2) / pow(0.3, 2) - pow(y - 0.8, 2) / pow(0.4, 2));
        return (0.4 * pow(pi, 0.3)) * (term1 + term2);
    };

    for (int i = 0; i < locs.rows(); ++i){
        f(i, 0) = gamSim_2(locs(i, 0), locs(i, 1));
    }

    int m = 21;
    DVector<double> scales(m);
    scales << -0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0,
    0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2;

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // define statistical model
    double lambda = 0.2201047;

    int n_loc = 117;
    DVector<int> loc_indexes(n_loc);
    loc_indexes << 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 16, 19, 22, 24, 25, 26, 28, 30, 34, 36, 38, 
    39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 55, 58, 59, 61, 63, 69, 73, 75, 76, 77, 80, 82, 85, 
    88, 89, 96, 98, 103, 106, 114, 116, 117, 120, 122, 124, 127, 131, 132, 133, 134, 135, 139, 141, 
    142, 143, 144, 146, 148, 149, 152, 154, 155, 156, 157, 158, 162, 163, 164, 166, 169, 172, 173, 
    176, 178, 179, 181, 183, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 198, 199, 200, 
    201, 202, 203, 204, 206, 208, 213, 215, 216, 218, 219, 222, 223;

    DVector<double> f_0(n_loc);
    for(int p = 0; p < n_loc; ++p){
        f_0(p) = gamSim_2(locs(loc_indexes(p), 0), locs(loc_indexes(p), 1));
    }

    int rep = 100;

    //DMatrix<double> res_wald(rep, m);
    DMatrix<double> res_sf(rep, m);
    DMatrix<double> res_esf(rep, m);

    double sd = 0.1;
    
    // Define a weight vector for heteroscedasticity (e.g., based on some function of the location)
    DVector<double> weights(locs.rows());
    for (int i = 0; i < locs.rows(); ++i) {
        weights(i) = 1.0 + 0.1 * pow(locs(i, 0) - 0.5, 2) + 0.1 * pow(locs(i, 1) - 0.5, 2);  // Example weight function

    }
DMatrix<double> weightsMatrix = DMatrix<double>::Map(weights.data(), weights.size(), 1);
    for(int i = 0; i < m; ++i){
        DVector<double> pval_sf(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution(0, sd);
            DVector<double> random_vector(f.size());
            for (int j = 0; j < random_vector.size(); ++j) {
                // Generate heteroscedastic errors by scaling by weights
                random_vector[j] = distribution(generator) / sqrt(weights(j)); // Adjust the error based on weights
            }

            DMatrix<double> observations = X + (1 + scales[i]) * f + random_vector;

            SRPDE model(problem, Sampling::pointwise);
            model.set_lambda_D(lambda);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
            df.insert(WEIGHTS_BLK, weightsMatrix);


            model.set_data(df);
            model.init();
            model.solve();

            fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
            fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

            inferenceWald.setLocationsF(loc_indexes);
            inferenceESF.setLocationsF(loc_indexes);
            inferenceWald.setf0(f_0);
            inferenceESF.setf0(f_0);
            inferenceESF.setNflip(1000);
            
            pval_sf[k-1] = inferenceESF.sign_flip_p_value();
            pval_esf[k-1] = inferenceESF.f_p_value();

            std::cout << "SF: " << pval_sf[k-1] << std::endl;
            std::cout << "ESF: " << pval_esf[k-1] << std::endl;
        }

        res_sf.col(i) = pval_sf;
        res_esf.col(i) = pval_esf;
    }

    DVector<double> power_matrix_sf(m);
    DVector<double> power_matrix_esf(m);

    double threshold = 0.05;
    for (int j = 0; j < m; ++j) {  
        int count_sf = 0;
        int count_esf = 0;
        for (int i = 0; i < res_esf.rows(); ++i){
            if (res_sf(i, j) < threshold){
                count_sf++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        power_matrix_sf[j] = static_cast<double> (count_sf) / rep;   
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;   
    }

    std::cout << "Power Sign Flip:" << std::endl;
    std::cout << power_matrix_sf << std::setprecision(7) << std::endl;
    std::cout << "Power Eigen Sign Flip:" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;
}




