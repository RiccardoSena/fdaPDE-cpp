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

#include "../../fdaPDE/models/regression/gsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::GSRPDE;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::SpaceOnly;
using fdapde::models::Sampling;
using fdapde::models::Poisson;
using fdapde::models::Bernulli;
using fdapde::models::Exponential;
using fdapde::models::Gamma;
using fdapde::models::Gaussian;

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


// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   random gaussian mean 0 sd 0.1
//    BC:           no
//    order FE:     1
//    distribution: poisson


/*
TEST(glm_inference, poisson) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_medium");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/gsrpde/2D_test1/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/gsrpde/2D_test1/y.csv"   );
    DMatrix<double> X(y.rows(), 1);
    std::default_random_engine generator(42);
    std::normal_distribution<double> distribution(3, 0.5);
    for (int i = 0; i < X.rows(); ++i) {
        X(i, 0) = distribution(generator);
    }
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-3;
    GSRPDE<SpaceOnly> model(problem, Sampling::pointwise, Poisson());
    model.set_lambda_D(lambda_D);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    fdapde::models::Wald<GSRPDE<SpaceOnly>, fdapde::models::exact> inference(model);
    fdapde::models::ESF<GSRPDE<SpaceOnly>, fdapde::models::exact> inference_sf(model);
    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);       
    inference.setC(C);
    inference_sf.setC(C);
    inference.setBeta0(beta0);
    inference_sf.setBeta0(model.beta());

    inference_sf.setNflip(1000);

    //std::cout << "Estimate of beta: " << model.beta() << std::endl;
    //std::cout << "Wald: " << inference.p_value(fdapde::models::one_at_the_time) << std::endl;
    DVector<double> res_sf = inference_sf.p_value(fdapde::models::simultaneous);
    DVector<double> f = model.f();
    std::cout << "f" << std::endl;
    for(int l = 0; l < 4; ++l){
        std::cout << f(l) << std::endl;
    }
    std::cout << "SF: " << res_sf << std::endl;
    //DVector<int> locat(3);
    //locat << 1, 2, 3;
    //inference.setLocationsF(locat);
    //std::cout << inference.f_p_value() << std::endl;
}
*/


/*
TEST(glm_inference, power){

    MeshLoader<Triangulation<2, 2>> domain("unit_square_medium");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/gsrpde/2D_test1/locs.csv");
    DMatrix<double> f    = read_csv<double>("../data/models/gsrpde/2D_test1/f.csv"   );
    DMatrix<double> X(f.rows(), 1);
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-3;

    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);    

    DVector<double> H1(1);
    //H1 << 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1;
    H1 << 0;
    // repetition for the simulations
    int rep = 1;

    DMatrix<double> res_wald(rep, H1.size());
    DMatrix<double> res_esf(rep, H1.size());

    double sd = 0.5;
    double mean = 3;

    DMatrix<double> observations(X.size(), 1);
    
    for(int i = 0; i < H1.size(); ++i){
        DVector<double> pval_wald(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution2(mean, sd);
            std::poisson_distribution<> poisson_dist;
            for (int j = 0; j < X.rows(); ++j) {
                X(j, 0) = distribution2(generator);
                double lambda = std::exp(X(j, 0) * H1[i] + f(j)) ;
                poisson_dist.param(std::poisson_distribution<>::param_type(lambda));
                observations(j, 0) = poisson_dist(generator);
            }
        
            GSRPDE<SpaceOnly> model(problem, Sampling::pointwise, Poisson());
            model.set_lambda_D(lambda_D);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
            model.set_data(df);
            model.init();
            model.solve();

            DVector<double> f = model.Psi() * model.f();
            std::cout << "f" << std::endl;
            for(int l = 0; l < 4; ++l){
                std::cout << f(l) << std::endl;
            }
        
            fdapde::models::Wald<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_wald(model);
            inf_wald.setBeta0(beta0);
            inf_wald.setC(C);

            fdapde::models::ESF<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_esf(model);
            inf_esf.setBeta0(model.beta());
            inf_esf.setC(C);
            inf_esf.setNflip(10000);

            pval_wald[k-1] = inf_wald.p_value(fdapde::models::one_at_the_time)(0);
            pval_esf[k-1] = inf_esf.p_value(fdapde::models::simultaneous)(0);

            std::cout << "Rep " << k << std::endl;
            std::cout << pval_wald[k-1] << std::endl;
            std::cout << pval_esf[k-1] << std::endl;

        }
        res_wald.col(i) = pval_wald;
        res_esf.col(i) = pval_esf;

        //std::cout << "H1 " << i << std::endl;

    }

    DVector<double> power_matrix_wald(H1.size());
    DVector<double> power_matrix_esf(H1.size());

    // compute the power
    double threshold = 0.05;
    for (int j = 0; j < H1.size(); ++j) {  
        int count_wald = 0;
        int count_esf = 0;
        for (int i = 0; i < res_wald.rows(); ++i){
            if (res_wald(i, j) < threshold){
                count_wald++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        power_matrix_wald[j] = static_cast<double> (count_wald) / rep; 
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;    
    }

    std::cout << "Power Wald:" << std::endl;
    std::cout << power_matrix_wald << std::setprecision(7) << std::endl;
    std::cout << "Power ESF" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;

}
*/



TEST(glm_inference, gaussian){
    
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> f(locs.rows(), 1);
    const double pi = M_PI;
    const double a1 = -1.5;
    const double a2 = 0.4;
    auto z = [a1, a2, pi](const DVector<double>& p) -> double {
        return a1 * std::sin(2 * pi * p[0]) * std::cos(2 * pi * p[1]) +
               a2 * std::sin(3 * pi * p[0]) + 2;
    };
    for (int i = 0; i < locs.rows(); ++i){
        f(i, 0) = z(locs.row(i));
    }
    DMatrix<double> X(f.rows(), 1);
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-3;

    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);    

    DVector<double> H1(11);
    //H1 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11;
    H1 << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1;
    //H1 << 0;
    // repetition for the simulations
    int rep = 50;

    DMatrix<double> res_wald(rep, H1.size());
    DMatrix<double> res_esf(rep, H1.size());

    double sd = 0.5;
    double mean = 0;

    DMatrix<double> observations(X.size(), 1);
    
    for(int i = 0; i < H1.size(); ++i){
        DVector<double> pval_wald(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution2(mean, sd);
            for (int j = 0; j < X.rows(); ++j) {
                X(j, 0) = distribution2(generator);
                std::normal_distribution<double> dist(X(j, 0) * H1[i] + f(j), 1);
                observations(j, 0) = dist(generator);
            }
        
            GSRPDE<SpaceOnly> model(problem, Sampling::pointwise, Gaussian());
            model.set_lambda_D(lambda_D);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
            model.set_data(df);
            model.init();
            model.solve();

            //DVector<double> f = model.f();
            //std::cout << "f" << std::endl;
            //for(int l = 0; l < 4; ++l){
            //    std::cout << f(l) << std::endl;
            //}
        
            fdapde::models::Wald<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_wald(model);
            inf_wald.setBeta0(beta0);
            inf_wald.setC(C);

            fdapde::models::ESF<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_esf(model);
            inf_esf.setBeta0(beta0);
            inf_esf.setC(C);
            inf_esf.setNflip(10000); 

            pval_wald[k-1] = inf_wald.p_value(fdapde::models::one_at_the_time)(0);
            pval_esf[k-1] = inf_esf.p_value(fdapde::models::simultaneous)(0);

            //std::cout << "Rep " << k << std::endl;
            //std::cout << pval_wald[k-1] << std::endl;
            //std::cout << pval_esf[k-1] << std::endl;

        }
        res_wald.col(i) = pval_wald;
        res_esf.col(i) = pval_esf;

        //std::cout << "H1 " << i << std::endl;

    }

    DVector<double> power_matrix_wald(H1.size());
    DVector<double> power_matrix_esf(H1.size());

    // compute the power
    double threshold = 0.05;
    for (int j = 0; j < H1.size(); ++j) {  
        int count_wald = 0;
        int count_esf = 0;
        for (int i = 0; i < res_wald.rows(); ++i){
            if (res_wald(i, j) < threshold){
                count_wald++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        power_matrix_wald[j] = static_cast<double> (count_wald) / rep; 
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;    
    }

    std::cout << "Power Wald:" << std::endl;
    std::cout << power_matrix_wald << std::setprecision(7) << std::endl;
    std::cout << "Power ESF" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;
 
}


/*
TEST(glm_inference, gaussian2){
    
    MeshLoader<Triangulation<2, 2>> domain("unit_square_medium");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/gsrpde/2D_test1/locs.csv");
    DMatrix<double> f(locs.rows(), 1);
    const double pi = M_PI;
    const double a1 = -1.5;
    const double a2 = 0.4;
    auto z = [a1, a2, pi](const DVector<double>& p) -> double {
        return a1 * std::sin(2 * pi * p[0]) * std::cos(2 * pi * p[1]) +
               a2 * std::sin(3 * pi * p[0]) + 2;
    };
    for (int i = 0; i < locs.rows(); ++i){
        f(i, 0) = z(locs.row(i));
    }
    DMatrix<double> X(f.rows(), 1);
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-3;

    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);    

    DVector<double> H1(11);
    //H1 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11;
    H1 << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1;
    // repetition for the simulations
    int rep = 30;

    DMatrix<double> res_wald(rep, H1.size());
    DMatrix<double> res_esf(rep, H1.size());

    double sd = 0.5;
    double mean = 3;

    DMatrix<double> observations(X.size(), 1);
    
    for(int i = 0; i < H1.size(); ++i){
        DVector<double> pval_wald(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution2(mean, sd);
            for (int j = 0; j < X.rows(); ++j) {
                X(j, 0) = distribution2(generator);
                std::normal_distribution<double> dist(X(j, 0) * H1[i] + f(j), 1);
                observations(j, 0) = dist(generator);
            }
        
            GSRPDE<SpaceOnly> model(problem, Sampling::pointwise, Gaussian());
            model.set_lambda_D(lambda_D);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
            model.set_data(df);
            model.init();
            model.solve();

        
            fdapde::models::Wald<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_wald(model);
            inf_wald.setBeta0(beta0);
            inf_wald.setC(C);

            fdapde::models::ESF<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_esf(model);
            inf_esf.setBeta0(beta0);
            inf_esf.setC(C);
            inf_esf.setNflip(10000); 

            pval_wald[k-1] = inf_wald.p_value(fdapde::models::one_at_the_time)(0);
            pval_esf[k-1] = inf_esf.p_value(fdapde::models::simultaneous)(0);

            //std::cout << "Rep " << k << std::endl;
            std::cout << pval_wald[k-1] << std::endl;
            std::cout << pval_esf[k-1] << std::endl;

        }
        res_wald.col(i) = pval_wald;
        res_esf.col(i) = pval_esf;

        //std::cout << "H1 " << i << std::endl;

    }

    DVector<double> power_matrix_wald(H1.size());
    DVector<double> power_matrix_esf(H1.size());

    // compute the power
    double threshold = 0.05;
    for (int j = 0; j < H1.size(); ++j) {  
        int count_wald = 0;
        int count_esf = 0;
        for (int i = 0; i < res_wald.rows(); ++i){
            if (res_wald(i, j) < threshold){
                count_wald++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        power_matrix_wald[j] = static_cast<double> (count_wald) / rep; 
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;    
    }

    std::cout << "Power Wald:" << std::endl;
    std::cout << power_matrix_wald << std::setprecision(7) << std::endl;
    std::cout << "Power ESF" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;
 
}
*/


/*
TEST(glm_inference, poisson){
    
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> f(locs.rows(), 1);
    const double pi = M_PI;
    const double a1 = -1.5;
    const double a2 = 0.4;
    auto z = [a1, a2, pi](const DVector<double>& p) -> double {
        return a1 * std::sin(2 * pi * p[0]) * std::cos(2 * pi * p[1]) +
               a2 * std::sin(3 * pi * p[0]) + 2;
    };
    for (int i = 0; i < locs.rows(); ++i){
        f(i, 0) = z(locs.row(i));
    }
    DMatrix<double> X(f.rows(), 1);
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-3;

    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);    

    DVector<double> H1(12);
    H1 << 0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.36;
    //H1 << 0;
    // repetition for the simulations
    int rep = 100;

    DMatrix<double> res_wald(rep, H1.size());
    DMatrix<double> res_esf(rep, H1.size());

    double sd = 0.5;
    double mean = 0;

    DMatrix<double> observations(X.size(), 1);
    
    for(int i = 0; i < H1.size(); ++i){
        DVector<double> pval_wald(rep);
        DVector<double> pval_esf(rep);

        for(int k = 1; k < rep + 1; ++k){
            std::default_random_engine generator(k);
            std::normal_distribution<double> distribution2(mean, sd);
            std::poisson_distribution<> poisson_dist;
            for (int j = 0; j < X.rows(); ++j) { 
                X(j, 0) = distribution2(generator);
                double lambda = std::exp(X(j, 0) * H1[i] + f(j)) ;
                poisson_dist.param(std::poisson_distribution<>::param_type(lambda));
                observations(j, 0) = poisson_dist(generator);
            }

            GSRPDE<SpaceOnly> model(problem, Sampling::pointwise, Poisson());
            model.set_lambda_D(lambda_D);
            model.set_spatial_locations(locs);
            // set model's data
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, observations);
            df.insert(DESIGN_MATRIX_BLK, X);
            model.set_data(df);
            model.init();
            model.solve();
            

            //DVector<double> f = model.f();
            //std::cout << "f" << std::endl;
            //for(int l = 0; l < 4; ++l){
            //    std::cout << f(l) << std::endl;
            //}
        
            fdapde::models::Wald<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_wald(model);
            inf_wald.setBeta0(beta0);
            inf_wald.setC(C);

            fdapde::models::ESF<GSRPDE<SpaceOnly>, fdapde::models::exact> inf_esf(model);
            inf_esf.setBeta0(beta0);
            inf_esf.setC(C);
            inf_esf.setNflip(10000);

            pval_wald[k-1] = inf_wald.p_value(fdapde::models::one_at_the_time)(0);
            pval_esf[k-1] = inf_esf.p_value(fdapde::models::simultaneous)(0);
            //pval_esf[k-1] = 0;
            std::cout << "Rep " << k << std::endl;
            std::cout << pval_wald[k-1] << std::endl;
            std::cout << pval_esf[k-1] << std::endl;

        }
        res_wald.col(i) = pval_wald;
        res_esf.col(i) = pval_esf;
        //std::cout << "H1 " << i << std::endl;

    }

    DVector<double> power_matrix_wald(H1.size());
    DVector<double> power_matrix_esf(H1.size());

    // compute the power
    double threshold = 0.05;
    for (int j = 0; j < H1.size(); ++j) {  
        int count_wald = 0;
        int count_esf = 0;
        for (int i = 0; i < res_wald.rows(); ++i){
            if (res_wald(i, j) < threshold){
                count_wald++;
            }
            if (res_esf(i, j) < threshold){
                count_esf++;
            }
        }   
        power_matrix_wald[j] = static_cast<double> (count_wald) / rep; 
        power_matrix_esf[j] = static_cast<double> (count_esf) / rep;    
    }

    std::cout << "Power Wald:" << std::endl;
    std::cout << power_matrix_wald << std::setprecision(7) << std::endl;
    std::cout << "Power ESF" << std::endl;
    std::cout << power_matrix_esf << std::setprecision(7) << std::endl;
 
}

*/