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

using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Newton;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::dt;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::Mesh;
using fdapde::core::spline_order;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/gcv.h"
#include "../../fdaPDE/models/sampling_design.h"
#include "../../fdaPDE/models/regression/regression_type_erasure.h"
using fdapde::models::SRPDE;
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
using fdapde::models::RegressionView;
#include "../../fdaPDE/calibration/gcv.h"

#include "../../fdaPDE/models/regression/strpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::STRPDE;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;

#include "../../fdaPDE/models/regression/wald.h"
#include "../../fdaPDE/models/regression/speckman.h"
#include "../../fdaPDE/models/regression/esf.h"




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

#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <../../../fdaPDE-core/fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1

// input penso dovrebbe essere tipo: Nome_insieme_dei_test, Nome_specifico_del_test
// esempio: TEST(Inference, WaldExactInference)
// Poi Inference penso vada usato anche per i test successivi

/*
// bisognerà poi controllare tutti i parametri dei modelli da confrontare (pde utlizzata,...)
TEST(inferenceTest, WaldExactSRPDE){
        // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 5.623413 * std::pow(0.1, 5);
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness
    WALD<SRPDE, Strategy::exact> inference(model);
    inference.computeCI(CItype::simultaneous);
    int cols = model.beta().size();
    DMatrix<double> C(1, cols);
    C.setOnes(); // matrice C ha una sola riga di tutti 1
    inference.setC(C);
    EXPECT_TRUE(almost_equal(inference.computeCI(CItype::simultaneous), file della vecchia libreria che contiene risultati di CI simultaneous))
    
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test1/sol.mtx"));
}

TEST(inferenceTest, WaldNonExactSRPDE){
        // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 5.623413 * std::pow(0.1, 5);
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness
    WALD<SRPDE, Strategy::non_exact> inference(model);
    inference.computeCI(CItype::simultaneous);
    int cols = model.beta().size();
    DMatrix<double> C(1, cols);
    C.setOnes(); // matrice C ha una sola riga di tutti 1
    inference.setC(C);
    EXPECT_TRUE(almost_equal(inference.computeCI(CItype::simultaneous), file della vecchia libreria che contiene risultati di CI simultaneous))
    
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test1/sol.mtx"));
}

TEST(inferenceTest, SpeckmanNonExactSRPDE){
        // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 5.623413 * std::pow(0.1, 5);
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness
    SPECKMAN<SRPDE, Strategy::non_exact> inference(model);
    inference.computeCI(CItype::simultaneous);
    int cols = model.beta().size();
    DMatrix<double> C(1, cols);
    C.setOnes(); // matrice C ha una sola riga di tutti 1
    inference.setC(C);
    EXPECT_TRUE(almost_equal(inference.computeCI(CItype::simultaneous), file della vecchia libreria che contiene risultati di CI simultaneous))
    
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test1/sol.mtx"));
}


TEST(inferenceTest, SpeckmanNonExactSRPDE){
        // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 5.623413 * std::pow(0.1, 5);
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness
    SPECKMAN<SRPDE, Strategy::non_exact> inference(model);
    inference.computeCI(CItype::simultaneous);
    int cols = model.beta().size();
    DMatrix<double> C(1, cols);
    C.setOnes(); // matrice C ha una sola riga di tutti 1
    inference.setC(C);
    // il file va messo in cartella inference 
    EXPECT_TRUE(almost_equal(inference.computeCI(CItype::simultaneous), file della vecchia libreria che contiene risultati di CI simultaneous))
    
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test1/sol.mtx"));
}
*/


// test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1

/*
TEST(inference_test, WaldExact27Sim) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
    // test correctness
    //EXPECT_TRUE(almost_equal(model.f()   , "../data/models/srpde/2D_test2/sol.mtx" ));
    //EXPECT_TRUE(almost_equal(model.beta(), "../data/models/srpde/2D_test2/beta.mtx"));


    
     // test correctness WALD
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    std::cout << "C è: " << std::endl;
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        }
    }
    std::cout << std::endl;
    
    inference.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);
    std::cout << "beta0 è: " << beta0<<std::endl;

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(inference.p_value(fdapde::models::simultaneous)(0), 0.4119913 , 1e-7));
}


TEST(inference_test, WaldExact27Oat) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
    // test correctness
    //EXPECT_TRUE(almost_equal(model.f()   , "../data/models/srpde/2D_test2/sol.mtx" ));
    //EXPECT_TRUE(almost_equal(model.beta(), "../data/models/srpde/2D_test2/beta.mtx"));


    
     // test correctness WALD
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    std::cout << "C è: " << std::endl;
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        }
    }
    std::cout << std::endl;
    
    inference.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);
    std::cout << "beta0 è: " << beta0<<std::endl;

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues(0) << std::endl;
    std::cout<< pvalues(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.1872113 , 1e-7));
    EXPECT_TRUE(almost_equal(pvalues(1), 0.9015565 , 1e-7));

}




TEST(inference_test, WaldExact28sim) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
    
     // test correctness WALD
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    C(0,0)=1;
    C(0,1)=1;
    C(1,0)=1;
    C(1,1)=-1;
    
    //c(1,1,1,-1)
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    std::cout << "C è: " << std::endl;
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        }
    }
    std::cout << std::endl;

    inference.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);
    //std::cout << "beta0 è: " << beta0<<std::endl;

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.0 , 1e-7));

}


TEST(inference_test, WaldExact28oat) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
    
     // test correctness WALD
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    C(0,0)=1;
    C(0,1)=1;
    C(1,0)=1;
    C(1,1)=-1;
    
    //c(1,1,1,-1)
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    std::cout << "C è: " << std::endl;
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        }
    }
    std::cout << std::endl;

    inference.setC(C);
    //std::cout << "set C" << std::endl;
    //DVector<double> beta0(2);
    //beta0(0)=2;
    //beta0(1)=-1;
    //inference.setBeta0(beta0);
    //std::cout << "beta0 è: " << beta0<<std::endl;

    //DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inference.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
    //std::cout << "il valore dei pvalue è" << std::endl;
    //std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 1.03099e-02 , 1e-7));
    EXPECT_TRUE(almost_equal(pvalues(1), 2.76879e-17 , 1e-7));

}


TEST(inference_test, SpeckmanExact27oat){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    // test correctness SPECKMAN
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);
    //std::cout << "creato elemento inference" << std::endl;
    //std::cout<<" questi sono i beta del modello "<<model.beta()<<std::endl;
    inferenceSpeck.betas();
    std::cout<<std::endl;
    //std::cout<<" questi sono i beta di speckman "<<inferenceSpeck.betas()<<std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        std::cout<<std::endl;
        }
    }

    inferenceSpeck.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues1=inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout<< pvalues1(0) << std::endl;
    std::cout<< pvalues1(1) << std::endl;

    
    std::cout << "ora inizia il test speckman 27" << std::endl;
    //DMatrix<double> matrix(1, 1);
    //matrix << 0.00002458211564814289 ;
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0), 0.08680236, 1e-7));
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(1), 0.48107956, 1e-7));
} 


TEST(inference_test, SpeckmanExact28oat){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    // test correctness SPECKMAN
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);
    //std::cout << "creato elemento inference" << std::endl;
    //std::cout<<" questi sono i beta del modello "<<model.beta()<<std::endl;
    inferenceSpeck.betas();
    std::cout<<std::endl;
    //std::cout<<" questi sono i beta di speckman "<<inferenceSpeck.betas()<<std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    C(0,0)=1;
    C(0,1)=1;
    C(1,0)=1;
    C(1,1)=-1;


    inferenceSpeck.setC(C);
    //std::cout << "set C" << std::endl;
    //DVector<double> beta0(2);
    //beta0(0)=2;
    //beta0(1)=-1;
    //inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    //DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::simultaneous);
    //std::cout << "il valore dei pvalue è" << std::endl;
    //std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;

    
    std::cout << "ora inizia il test speckman sim" << std::endl;
    std::cout << "statistic: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0) << std::endl;
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0), 0.8937451158, 1e-7));
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(1), 0.0008644394, 1e-7));
}   

*/
TEST(inference_test, SpeckmanExact27sim){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    // test correctness SPECKMAN
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);
    //std::cout << "creato elemento inference" << std::endl;
    //std::cout<<" questi sono i beta del modello "<<model.beta()<<std::endl;

    //std::cout<<" questi sono i beta di speckman "<<inferenceSpeck.betas()<<std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        std::cout<<std::endl;
        }
    }

    inferenceSpeck.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues(0) << std::endl;
    std::cout<< pvalues(1) << std::endl;

    
    std::cout << "ora inizia il test speckman sim" << std::endl;
   // std::cout << "statistic: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0) << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.1574313, 1e-6));
}   

/*
TEST(inference_test, SpeckmanExact28sim){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    // test correctness SPECKMAN
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);
    //std::cout << "creato elemento inference" << std::endl;
    //std::cout<<" questi sono i beta del modello "<<model.beta()<<std::endl;
    inferenceSpeck.betas();
    std::cout<<std::endl;
    //std::cout<<" questi sono i beta di speckman "<<inferenceSpeck.betas()<<std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    C(0,0)=1;
    C(0,1)=1;
    C(1,0)=1;
    C(1,1)=-1;


    inferenceSpeck.setC(C);
    //std::cout << "set C" << std::endl;
    //DVector<double> beta0(2);
    //beta0(0)=2;
    //beta0(1)=-1;
    //inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::simultaneous)<<std::endl;

    //DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::simultaneous);
    //std::cout << "il valore dei pvalue è" << std::endl;
    //std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;

    
    std::cout << "ora inizia il test speckman sim" << std::endl;
    std::cout << "statistic: " << inferenceSpeck.p_value(fdapde::models::simultaneous)(0) << std::endl;
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::simultaneous)(0), 0.0 , 1e-7));
}   
*/

/*
TEST(inference_test, EigenSignFlip27sim){
// 50 volte stesso test per eigensignflip 
DVector<double> solutions(50);
    for(int test = 0; test < 50; ++test) {

        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
   
    // test correctness EigenSignFlip
    fdapde::models::EigenSignFlip<SRPDE > inferenceESF(model);
    inferenceESF.Lambda();
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    inferenceESF.setC(C);


    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceESF.setBeta0(beta0);

    // set the number of flips
    inferenceESF.setNflip(1000000);

    DVector<double> pvalues=inferenceESF.p_value(fdapde::models::simultaneous);
    //std::cout << "valore pvalue: " << std::endl;
    //std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    solutions(test)=pvalues(0);

}
std::cout<<"il vettore dei pvalues è : "<<solutions<<std::endl;
}



TEST(inference_test, EigenSignFlip27oat){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
   
    // test correctness EigenSignFlip
    fdapde::models::EigenSignFlip<SRPDE > inferenceESF(model);
    inferenceESF.Lambda();
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    inferenceESF.setC(C);


    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceESF.setBeta0(beta0);
    //std::cout << "set beta0 completato correttamente" << std::endl;

    //inference.computeCI(fdapde::models::simultaneous);
    //std::cout << "computed CI: " << inference.computeCI(fdapde::models::simultaneous)<<std::endl;

    DVector<double> pvalues3=inferenceESF.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues3(0) << std::endl;
    std::cout<< pvalues3(1) << std::endl;
    //std::cout << "ora inizia il test wald " << std::endl;
    //DMatrix<double> matrix(1, 1);
    //matrix << 0.00002458211564814289 ;
   // EXPECT_TRUE(almost_equal(pvalues(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalues(1), 0.924 , 1e-7));

}



TEST(inference_test, WaldNonExact27Sim) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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
    // test correctness
    //EXPECT_TRUE(almost_equal(model.f()   , "../data/models/srpde/2D_test2/sol.mtx" ));
    //EXPECT_TRUE(almost_equal(model.beta(), "../data/models/srpde/2D_test2/beta.mtx"));


    
     // test correctness WALD
    fdapde::models::Wald<SRPDE, fdapde::models::nonexact> inference(model);
    //std::cout << "creato elemento inference" << std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inference.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);
    std::cout << "beta0 è: " << beta0<<std::endl;

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues4=inference.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues4(0) << std::endl;
    std::cout<< pvalues4(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues4(0), 0.1261320 , 1e-7));
}


TEST(inference_test, SpeckmanNonExact27oat){
        // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    // test correctness SPECKMAN
    fdapde::models::Speckman<SRPDE, fdapde::models::nonexact> inferenceSpeck(model);
    //std::cout << "creato elemento inference" << std::endl;
    //std::cout<<" questi sono i beta del modello "<<model.beta()<<std::endl;
    //std::cout<<" questi sono i beta di speckman "<<inferenceSpeck.betas()<<std::endl;

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    //DMatrix<double> C(1,cols);
    //C.setOnes(); // matrice C ha una sola riga di tutti 1
    
    for (int i = 0; i < model.beta().size(); ++i) {
        for (int j = 0; j < model.beta().size(); ++j) {
        std::cout << C(i,j) << " ";
        std::cout<<std::endl;
        }
    }

    inferenceSpeck.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues5=inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues5(0) << std::endl;
    std::cout<< pvalues5(1) << std::endl;

    
    std::cout << "ora inizia il test speckman 27" << std::endl;
    //DMatrix<double> matrix(1, 1);
    //matrix << 0.00002458211564814289 ;
    EXPECT_TRUE(almost_equal(pvalues5(0), 0.1194335, 1e-7));
    EXPECT_TRUE(almost_equal(pvalues5(1), 0.0902682, 1e-7));
} 
*/

/*
TEST(inference_test, chrono27) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    inferenceWald.p_value(fdapde::models::one_at_the_time);
    inferenceSpeck.p_value(fdapde::models::simultaneous);
    inferenceESF.p_value(fdapde::models::simultaneous);
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // chrono end

    std::cout << "mean execution time (seconds) for " << n_it << " iterations: " << duration.count()/n_it << std::endl;

}
*/

TEST(inference_test, chronoWald) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);

    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceWald.setBeta0(beta0);

    inferenceWald.p_value(fdapde::models::simultaneous);
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // chrono end

    std::cout << "mean Wald execution time (seconds) for " << n_it << " iterations: " << duration.count()/n_it << std::endl;

}

TEST(inference_test, chronoSpeckman) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeckman(model);

    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    
    inferenceSpeckman.setC(C);
    inferenceSpeckman.setBeta0(beta0);

    inferenceSpeckman.p_value(fdapde::models::one_at_the_time);
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // chrono end

    std::cout << "mean Speckman execution time (seconds) for " << n_it << " iterations: " << duration.count()/n_it << std::endl;

}

TEST(inference_test, chronoESF) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    
    inferenceESF.setC(C);
    inferenceESF.setBeta0(beta0);

    inferenceESF.p_value(fdapde::models::one_at_the_time);
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // chrono end

    std::cout << "mean ESF execution time (seconds) for " << n_it << " iterations: " << duration.count()/n_it << std::endl;

}

TEST(inference_test, inference27) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;

    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    int n = 1000;
    inferenceESF.setNflip(n);

    DVector<double> pvalueswald = inferenceWald.p_value(fdapde::models::one_at_the_time);
    //std::cout << "valore pvalue wald con " << n << "flips: " << std::endl;
    //std::cout<< pvalueswald(0) << std::endl;
    //std::cout<< pvalueswald(1) << std::endl;
    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::simultaneous);
    //std::cout << "valore pvalue speck con " << n << "flips: " << std::endl;
    //std::cout<< pvaluesspeck(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::simultaneous);
    //std::cout << "valore pvalue ESF con " << n << "flips: " << std::endl;
    //std::cout<< pvaluesesf(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;


    // test correctness Wald
    EXPECT_TRUE(almost_equal(inferenceWald.p_value(fdapde::models::simultaneous)(0), 0.4119913 , 1e-7));

    // test correctness Speckman
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0), 0.08680236, 1e-7));
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(1), 0.48107956, 1e-7));

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));


}

/*
TEST(inference_test, inference28) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    C(0,0)=1;
    C(0,1)=1;
    C(1,0)=1;
    C(1,1)=-1;
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);

    // test correctness Wald
    EXPECT_TRUE(almost_equal(inferenceWald.p_value(fdapde::models::one_at_the_time)(0), 1.03099e-02 , 1e-7));
    EXPECT_TRUE(almost_equal(inferenceWald.p_value(fdapde::models::one_at_the_time)(1), 2.76879e-17 , 1e-7));
    
    // test correctness Speckman
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(0), 0.8937451158, 1e-7));
    EXPECT_TRUE(almost_equal(inferenceSpeck.p_value(fdapde::models::one_at_the_time)(1), 0.0008644394, 1e-7));

}


TEST(inference_test, WaldNonExact) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::nonexact> inferenceWald(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceWald.setBeta0(beta0);

    std::cout << "p-value non exact: " << inferenceWald.p_value(fdapde::models::simultaneous)(0) << std::endl;
    std::cout << "p-value non exact: " << inferenceWald.p_value(fdapde::models::simultaneous)(1) << std::endl;


}


TEST(inference_test, exact27) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);

    fdapde::models::ESF<SRPDE> inferenceESF(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    int n = 1000;
    inferenceESF.setNflip(n);

    DVector<double> pvalueswald = inferenceWald.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues wald: "<<std::endl;
    std::cout<< pvalueswald(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout<<"pvalues speckman: "<<std::endl;
    std::cout<< pvaluesspeck(0) << std::endl;
    std::cout<< pvaluesspeck(1) << std::endl;

    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::simultaneous);
    std::cout<<"pvalues esf: "<<std::endl;
    std::cout<< pvaluesesf(0) << std::endl;
    std::cout<< pvaluesesf(1) << std::endl;

    // test correctness Wald
    EXPECT_TRUE(almost_equal(pvalueswald(0), 0.2266538 , 1e-7));
    
    // test correctness Speckman
    EXPECT_TRUE(almost_equal(pvaluesspeck(0), 0.1194335, 1e-7));
    EXPECT_TRUE(almost_equal(pvaluesspeck(1), 0.0902682, 1e-7));

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));

}

TEST(inference_test, nonexact27) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
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

    fdapde::models::Wald<SRPDE, fdapde::models::nonexact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::nonexact> inferenceSpeck(model);

    //fdapde::models::ESF<SRPDE > inferenceESF(model);

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    //inferenceESF.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    //inferenceESF.setBeta0(beta0);

    //int n = 1000;
    //inferenceESF.setNflip(n);

    DVector<double> pvalueswald = inferenceWald.p_value(fdapde::models::simultaneous);
    std::cout<<"pvalues wald: "<<std::endl;
    std::cout<< pvalueswald(0) << std::endl;
    std::cout<< pvalueswald(1) << std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues speckman: "<<std::endl;
    std::cout<< pvaluesspeck(0) << std::endl;
    //std::cout<< pvaluesspeck(1) << std::endl;

    //DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::simultaneous);
    //std::cout<<"pvalues esf: "<<std::endl;
    //std::cout<< pvaluesesf(0) << std::endl;
    //std::cout<< pvaluesesf(1) << std::endl;

    // test correctness Wald
    EXPECT_TRUE(almost_equal(pvalueswald(0), 0.2266538 , 1e-7));
    
    // test correctness Speckman
    EXPECT_TRUE(almost_equal(pvaluesspeck(0), 0.1194335, 1e-7));
    //EXPECT_TRUE(almost_equal(pvaluesspeck(1), 0.0902682, 1e-7));

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));

}

*/


