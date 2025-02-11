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




// test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(qsrpde_test, laplacian_semiparametric_samplingatlocations) {
    // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/qsrpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/qsrpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/qsrpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double alpha = 0.9; // quantile 
    double lambda = 3.162277660168379 * std::pow(0.1, 4);   // use optimal lambda to avoid possible numerical issues
    QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    std::cout<<"beta stimato: "<<model.beta()<<std::endl;
    //std::cout<<"f stimato: "<<model.f()<<std::endl;
    //std::cout<<"osservazioni modello "<< X.rows()<<std::endl;

    fdapde::models::Wald<QSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inference.setC(C);
    DVector<double> beta0(2);
    beta0(0)=model.beta()(0)+0.1;
    beta0(1)=model.beta()(1)+0.1;
    //beta0(0)=model.beta()(0)+0.1; //questo non mi da risultati sperati 
    //beta0(1)=model.beta()(1)+0.1; //questo non mi da risultati sperati 
    inference.setBeta0(beta0);

    DVector<double> f_0=model.f();
    //DVector<double> f_0=model.f()*(1+0.2); // questo non mi da risultati sperati 

    inference.setf0(f_0);

     DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
     std::cout << "il valore dei pvalue beta è " << pvalues <<std::endl;
     std::cout << " dovrebbe venire 1 perchè ho impostato i beta0 uguale ai beta stimati del modello"<< std::endl;
     inference.setf0(f_0);

    double pvalues_f=inference.f_p_value();
    std::cout << "il valore dei pvalue f è " << pvalues_f <<std::endl;
    std::cout << " dovrebbe venire 1 perchè ho impostato i f0 uguale agli f stimati del modello"<< std::endl;

    
    // test correctness
    EXPECT_TRUE(almost_equal(model.f()   , "../data/models/qsrpde/2D_test2/sol.mtx" ));
    EXPECT_TRUE(almost_equal(model.beta(), "../data/models/qsrpde/2D_test2/beta.mtx"));
}


// test 4
//    domain:       c-shaped
//    sampling:     areal
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(qsrpde_test, laplacian_semiparametric_samplingareal) {
    // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("c_shaped_areal");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/qsrpde/2D_test4/y.csv");
    DMatrix<double> X = read_csv<double>("../data/models/qsrpde/2D_test4/X.csv");
    DMatrix<double> subdomains = read_csv<double>("../data/models/qsrpde/2D_test4/incidence_matrix.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double alpha = 0.5;
    double lambda = 5.623413251903491 * std::pow(0.1, 3);   // use optimal lambda to avoid possible numerical issues
    QSRPDE<SpaceOnly> model(problem, Sampling::areal, alpha);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(subdomains);
    // set model data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    std::cout<<"beta stimato: "<<model.beta()<<std::endl;
   // std::cout<<"f stimato: "<<model.f()<<std::endl;
    //std::cout<<"osservazioni modello "<< X.rows()<<std::endl;

    fdapde::models::Wald<QSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inference.setC(C);
    DVector<double> beta0(1);
    beta0(0)=model.beta()(0);
    //beta0(0)=model.beta()(0)+0.1; // questo non mi da risultati sperati 
    inference.setBeta0(beta0);

    DVector<double> f_0=model.f();
    //DVector<double> f_0=model.f()*(1+0.2); // questo non mi da risultati sperati 
    inference.setf0(f_0);

     DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
     std::cout << "il valore dei pvalue beta è " << pvalues <<std::endl;
     std::cout << " dovrebbe venire 1 perchè ho impostato i beta0 uguale ai beta stimati del modello"<< std::endl;
     inference.setf0(f_0);

    double pvalues_f=inference.f_p_value();
    std::cout << "il valore dei pvalue f è " << pvalues_f <<std::endl;
    std::cout << " dovrebbe venire 1 perchè ho impostato i f0 uguale agli f stimati del modello"<< std::endl;

    
    // test correctness
    EXPECT_TRUE(almost_equal(model.f()   , "../data/models/qsrpde/2D_test4/sol.mtx" ));
    EXPECT_TRUE(almost_equal(model.beta(), "../data/models/qsrpde/2D_test4/beta.mtx"));
}





/*

TEST(qsrpde_test, power_f){
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
    scales= scales*10;
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
    //DMatrix<double> res_esf(rep, m); 

    double sd = 0.1;
      
    for(int i = 0; i < m; ++i){
        DVector<double> pval_sf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution(0, sd);
            DVector<double> random_vector(f.size());
            for (int j = 0; j < random_vector.size(); ++j) {
                random_vector[j] = distribution(generator) ; 
            }

            DMatrix<double> observations = X + (1 + scales[i]) * f + random_vector;
            double alpha = 0.9;
            QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
            model.set_lambda_D(lambda);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);

            model.set_data(df);
            model.init();
            model.solve();

            fdapde::models::Wald<QSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);

            inference.setLocationsF(loc_indexes);
            inference.setf0(f_0);
            
            pval_sf[k-1] = inference.f_p_value();

            std::cout << "SF: " << pval_sf[k-1] << std::endl;
        }

        res_sf.col(i) = pval_sf;
    }

    DVector<double> power_matrix_sf(m);

    double threshold = 0.05;
    for (int j = 0; j < m; ++j) {  
        int count_sf = 0;
        for (int i = 0; i < res_sf.rows(); ++i){
            if (res_sf(i, j) < threshold){
                count_sf++;
            }
            
        }   
        power_matrix_sf[j] = static_cast<double> (count_sf) / rep;   
    }

    std::cout << "Power Sign Flip:" << std::endl;
    std::cout << power_matrix_sf << std::setprecision(7) << std::endl;
}




TEST(quantile_inference_test, quantile_simulatons_beta) {
   // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/qsrpde/2D_test2/locs.csv");
    //DMatrix<double> y    = read_csv<double>("../data/models/qsrpde/2D_test2/y.csv");
    //DMatrix<double> X    = read_csv<double>("../data/models/qsrpde/2D_test2/X.csv");
   // DMatrix<double> y = read_csv<double>("../data/models/qsrpde/2D_beta_power/2quantileobservations_1_rep_1.csv");
  //  DMatrix<double> X    = read_csv<double>("../data/models/qsrpde/2D_beta_power/2quantilecovariates_1_rep_1.csv");
            
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double alpha = 0.9;
    double lambda = 3.162277660168379 * std::pow(0.1, 4);   // use optimal lambda to avoid possible numerical issues
    QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);



    DMatrix<double> lambdas(13, 1);

    for (int i = 0; i < 13; ++i) { lambdas(i, 0) = std::pow(10, -6.0 + 0.25 * i); }
    // optimize GCV
    auto GCV = model.gcv<fdapde::models::ExactEDF>();
    fdapde::core::Grid<fdapde::core::Dynamic> opt;
    opt.optimize(GCV, lambdas);
    std::cout<<"lambda ottimale: "<<opt.optimum()<<std::endl;*/
    /*model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
 
    //set beta_H1
    DVector<double> beta_H1_list(11);
    beta_H1_list(0)=2.0;
    for (int i = 1; i < beta_H1_list.size(); ++i) {
        beta_H1_list(i)=beta_H1_list(i-1)+0.1;
                }             

    int max_iter=10;
    DMatrix<double> pvaluesW(beta_H1_list.size(),max_iter); 

    for (int i = 0; i < beta_H1_list.size(); ++i) {
        for (int iter = 0; iter < max_iter; ++iter) {
            DMatrix<double> y = read_csv<double>("../data/models/qsrpde/2D_beta_power/2quantileobservations_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            DMatrix<double> X    = read_csv<double>("../data/models/qsrpde/2D_beta_power/2quantilecovariates_"+ std::to_string(i+1) + "_rep_" + std::to_string(iter+1) + ".csv");
            double alpha = 0.9;
            double lambda = 3.162277660168379 * std::pow(0.1, 4);   // use optimal lambda to avoid possible numerical issues
            QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
            model.set_lambda_D(lambda);
            model.set_spatial_locations(locs);

            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, y);
            df.insert(DESIGN_MATRIX_BLK, X);

            model.set_data(df);
            // solve smoothing problem
            model.init();
            model.solve();
            std::cout<<"beta modello  "<<model.beta()<<std::endl;
            std::cout<<"beta 1 : "<<beta_H1_list(i)<<std::endl;


            // define inference objects
            fdapde::models::Wald<QSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);

            int cols = model.beta().size();
            DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
            inference.setC(C);
        
            DVector<double> beta0(2);
            beta0(0)=2.0;
            beta0(1)=-1.0;

            inference.setBeta0(beta0);
            std::cout<<"beta 0 : "<<beta0<<std::endl;
            std::cout << "Dimensioni beta_H1_list: " << beta_H1_list.size() << std::endl;
            std::cout << "Dimensioni pvaluesW: " << pvaluesW.rows() << " x " << pvaluesW.cols() << std::endl;
            std::cout << "i=" << i << ", iter=" << iter << std::endl;

            DVector<double> pvalueswald = inference.p_value(fdapde::models::one_at_the_time);
                        std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;

            pvaluesW(i,iter)=pvalueswald(0);
        }
    }

    std::vector<double> power_mat(beta_H1_list.size()); // 11 righe, 3 colonne


    for (int i = 0; i < beta_H1_list.size(); ++i) {
        double count = 0;  // Inizializza il contatore per la riga corrente
        for (int j = 0; j < max_iter; ++j) {
            // Verifica se il primo elemento della riga è minore di 0.05
            if (pvaluesW(i,j) < 0.05) {
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



// test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(qsrpde_test, laplacian_semiparametric_samplingatlocations) {
    // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/qsrpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/qsrpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/qsrpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double alpha = 0.9;
    double lambda = 3.162277660168379 * std::pow(0.1, 4);   // use optimal lambda to avoid possible numerical issues
    QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    std::cout<<"beta stimato: "<<model.beta()<<std::endl;
    std::cout<<"f stimato: "<<model.f()<<std::endl;
                               std::cout<<"osservazioni modello "<< X.rows()<<std::endl;

DVector<double> f_0=model.f();
for (int i = 0; i < f_0.size(); ++i) {
  f_0[i] += 1e-10; // Aggiungi 0.0000000001 a ogni elemento
}

                               std::cout<<"f0 ok  "<< std::endl;


//    beta stimato:  1.99072
       // -0.98117
 // test correctness WALD
    fdapde::models::ESF<QSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);
                               std::cout<<"inference model  ok  "<< std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inference.setC(C);
                                   std::cout<<"set c ok  "<< std::endl;

    DVector<double> beta0(2);
    beta0(0)=1.99072;
    beta0(1)=-0.98117;
    inference.setBeta0(beta0);
                                   std::cout<<"set beta 0 ok  "<< std::endl;


           // inference.setf0(f_0);

    //double pvalues=inference.f_p_value();
   // std::cout << "il valore dei pvalue è" << pvalues <<std::endl;
     DVector<double> pvalues=inference.signFlipTest();
     std::cout << "il valore dei pvalue beta è" << pvalues <<std::endl;

     inference.setf0(f_0);

    //double pvalues_f=inference.f_p_value();
   //std::cout << "il valore dei pvalue f è" << pvalues_f <<std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    //EXPECT_TRUE(almost_equal(pvalues(0), 0.4119913 , 1e-7));

    // test correctness
    EXPECT_TRUE(almost_equal(model.f()   , "../data/models/qsrpde/2D_test2/sol.mtx" ));
    EXPECT_TRUE(almost_equal(model.beta(), "../data/models/qsrpde/2D_test2/beta.mtx"));
}
*/