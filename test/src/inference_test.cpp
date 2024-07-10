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

#include <../../../fdaPDE-core/fdaPDE/core.h>
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::DiscretizedVectorField;


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
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inference.setC(C);
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << pvalues <<std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.4119913 , 1e-7));

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
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);   
    inference.setC(C);
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" <<pvalues<< std::endl;

    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.1872113 , 1e-7));
    EXPECT_TRUE(almost_equal(pvalues(1), 0.9015565 , 1e-7));

}


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
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    inferenceSpeck.setC(C);
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" <<pvalues<< std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.1574313, 1e-6));
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
    inferenceSpeck.betas();
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    inferenceSpeck.setC(C);
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    //inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" <<pvalues<< std::endl;

    EXPECT_TRUE(almost_equal(pvalues(0), 0.08680236, 1e-7));
    EXPECT_TRUE(almost_equal(pvalues(1), 0.48107956, 1e-7));
} 


TEST(inference_test, EigenSignFlip27sim){
// 50 volte stesso test per eigensignflip  per fare i boxplot 
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
    //std::cout << "valore pvalue: " << pvalues<<std::endl;
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

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inferenceESF.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceESF.setBeta0(beta0);

    DVector<double> pvalues=inferenceESF.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" << pvalues<<std::endl;

}


//TEST NON EXACT 2.7
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

    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);
    inference.setC(C);

    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);

    DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" << pvalues<<std::endl;
    
    EXPECT_TRUE(almost_equal(pvalues(0), 0.1261320 , 1e-7));
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
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);

    inferenceSpeck.setC(C);
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inferenceSpeck.setBeta0(beta0);

    inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time)<<std::endl;

    DVector<double> pvalues=inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout << "il valore dei pvalue è" <<pvalues<< std::endl;

    EXPECT_TRUE(almost_equal(pvalues(0), 0.1194335, 1e-7));
    EXPECT_TRUE(almost_equal(pvalues(1), 0.0902682, 1e-7));
} 


*/


/*

// RIASSUNTO TESTS 2.7 EXACT E NON EXACT 
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
    fdapde::models::ESF<SRPDE,fdapde::models::exact> inferenceESF(model);

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
    std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;
    DMatrix<double> CIwald_=inferenceWald.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " <<std::fixed << std::setprecision(15)<< CIwald_<<std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues speckman: "<<std::fixed << std::setprecision(15)<<pvaluesspeck<<std::endl;
    DMatrix<double> CIspeck_=inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << std::fixed << std::setprecision(15)<<CIspeck_<<std::endl;

    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues esf: "<<pvaluesesf<<std::endl;

    DMatrix<double> CIESF_=inferenceESF.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << CIESF_<<std::endl;

    // test correctness Wald
    EXPECT_TRUE(almost_equal(pvalueswald(0), 0.411991314607044 , 1e-7));
    
    // test correctness Speckman
    //EXPECT_TRUE(almost_equal(pvaluesspeck(0), 0.0868023617435293, 1e-8));
    //EXPECT_TRUE(almost_equal(pvaluesspeck(1), 0.4810795610695496, 1e-78);

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));

}

*/



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
    fdapde::models::ESF<SRPDE,fdapde::models::nonexact > inferenceESF(model);
 
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
    std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;
    DMatrix<double> CIwald_=inferenceWald.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " <<std::fixed << std::setprecision(15)<< CIwald_<<std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues speckman: "<<std::fixed << std::setprecision(15)<<pvaluesspeck<<std::endl;
    DMatrix<double> CIspeck_=inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << std::fixed << std::setprecision(15)<<CIspeck_<<std::endl;

    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues esf: "<<pvaluesesf<<std::endl;

    DMatrix<double> CIESF_=inferenceESF.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << CIESF_<<std::endl;

    // test correctness Wald
    EXPECT_TRUE(almost_equal(pvalueswald(0), 0.2266538 , 1e-7));
    
    // test correctness Speckman
    //EXPECT_TRUE(almost_equal(pvaluesspeck(0), 0.1194335, 1e-7));
    //EXPECT_TRUE(almost_equal(pvaluesspeck(1), 0.0902682, 1e-7));

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));

}














/*
// TEST EXACT ON 2.8 
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
TEST TEMPI DI ESECUZIONE 
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

    int cols = beta0.size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    
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

    int cols = beta0.size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeckman(model);
    
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

    int cols = beta0.size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);

    // chrono start
    using namespace std::chrono;

    int n_it = 100;

    auto start = high_resolution_clock::now();

    for(int i = 0; i < n_it; ++i){

    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);
    
    inferenceESF.setC(C);
    inferenceESF.setBeta0(beta0);

    inferenceESF.p_value(fdapde::models::one_at_the_time);
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // chrono end

    std::cout << "mean ESF execution time (seconds) for " << n_it << " iterations: " << duration.count()/n_it << std::endl;

}


TEST(inference_test, non_exactESF27) {
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

    int cols = beta0.size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);


    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    fdapde::models::ESF<SRPDE, fdapde::models::nonexact> inferenceESF(model);
    
    inferenceESF.setC(C);
    inferenceESF.setBeta0(beta0);

    std::cout << "ESF non exact: " << inferenceESF.p_value(fdapde::models::simultaneous) << std::endl;

}
*/


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
*/


/*

TEST(inference_test, inference29) {
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
    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);
    DVector<double> f0(171);
    f0 <<   1.806250e-01,  2.678915e-01,  3.551579e-01,  4.424244e-01,  5.296909e-01,  6.169573e-01,
    7.042238e-01,  7.914902e-01,  8.787567e-01,  9.660232e-01,  1.132690e+00,  1.299356e+00,
   1.466023e+00,  1.632690e+00,  1.799356e+00,  1.966023e+00,  2.132690e+00,  2.299356e+00,
   2.466023e+00,  2.632690e+00,  2.799356e+00,  2.966023e+00,  3.132690e+00,  3.299356e+00,
   3.466023e+00,  3.632690e+00,  3.799356e+00,  3.966023e+00,  4.100597e+00,  4.173578e+00,
   4.200893e+00,  4.205423e+00,  4.197678e+00,  4.167638e+00,  4.092837e+00,  3.957623e+00,
   3.790956e+00,  3.624290e+00,  3.457623e+00,  3.290956e+00,  3.124290e+00,  2.957623e+00,
   2.790956e+00,  2.624290e+00,  2.457623e+00,  2.290956e+00,  2.124290e+00,  1.957623e+00,
   1.790956e+00,  1.624290e+00,  1.457623e+00,  1.290956e+00,  1.124290e+00,  9.576232e-01,
   6.123032e-17, -6.131732e-01, -7.798398e-01, -9.465065e-01, -1.113173e+00, -1.279840e+00,
  -1.446506e+00, -1.613173e+00, -1.779840e+00, -1.946506e+00, -2.113173e+00, -2.279840e+00,
  -2.446506e+00, -2.613173e+00, -2.779840e+00, -2.946506e+00, -3.113173e+00, -3.279840e+00,
  -3.446506e+00, -3.613173e+00, -3.799414e+00, -3.997128e+00, -4.149177e+00, -4.205373e+00,
  -4.145962e+00, -3.991188e+00, -3.791653e+00, -3.604773e+00, -3.438106e+00, -3.271440e+00,
  -3.104773e+00, -2.938106e+00, -2.771440e+00, -2.604773e+00, -2.438106e+00, -2.271440e+00,
  -2.104773e+00, -1.938106e+00, -1.771440e+00, -1.604773e+00, -1.438106e+00, -1.271440e+00,
  -1.104773e+00, -9.381065e-01, -7.714398e-01, -6.047732e-01, -5.175067e-01, -4.302402e-01,
 -3.429738e-01, -2.557073e-01, -1.684409e-01, -8.117439e-02,  6.092075e-03,  9.335854e-02,
 -3.926423e-01,  3.927558e-01, -8.687065e-01, -1.368706e+00, -2.035373e+00,  8.687565e-01,
  1.368756e+00,  2.035423e+00, -2.535373e+00, -3.035373e+00, -3.535373e+00,  2.535423e+00,
  3.035423e+00,  3.535423e+00, -3.858656e+00,  3.863918e+00, -3.634290e+00, -3.702101e+00,
  3.707318e+00,  3.759368e+00,  3.104989e-16, -1.971619e-01, -6.164095e-01, -6.091183e-01,
 -4.231790e-01, -7.395054e-01,  2.778881e-01,  6.685202e-01,  6.467463e-01,  8.264030e-01,
  5.074438e-01,  2.051823e-01,  3.732377e-01,  4.049210e-02, -1.740386e-01, -1.101676e+00,
 -1.085963e+00, -1.695505e+00, -1.872100e+00, -1.461439e+00, -1.560887e+00, -1.812219e+00,
  1.138499e+00,  1.126456e+00,  1.710344e+00,  1.946548e+00,  1.535887e+00,  1.608877e+00,
  1.869888e+00, -2.268343e+00, -2.252630e+00, -2.768343e+00, -2.752630e+00, -3.268343e+00,
 -3.252630e+00, -3.419606e+00,  2.305165e+00,  2.293123e+00,  2.805165e+00,  2.793123e+00,
  3.305165e+00,  3.293123e+00,  3.511451e+00;

    //inferenceWald.setf0(model.Psi() * model.f());
    inferenceWald.setf0(f0); 
    inferenceESF.setf0(f0);
    //DVector<int> loc_indexes(3);
    //loc_indexes << 1, 5, 7;
    //inferenceESF.setLocationsF(loc_indexes);
    //inferenceESF.setNflip(10);

    std::cout << "Wald f p value: " << inferenceWald.f_p_value() << std::endl;
    //std::cout << "Wald f CI: " << inferenceWald.f_CI() << std::endl;
    std::cout << "Esf p value: " << inferenceESF.f_p_value() << std::endl;

    //std::cout << "Esf CI: " << inferenceESF.f_CI() << std::endl;

}


TEST(inference_test, inference2999) {
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
    
    DVector<int> loc_indexes(3);
    loc_indexes << 1, 5, 7;
    inferenceWald.setLocationsF(loc_indexes);
    std::cout << "Wald f p value 2: " << inferenceWald.f_p_value() << std::endl;

}



TEST(inference_test, inference210) {
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
    DMatrix<double> new_locs(63, 2);

    new_locs << -0.148227359, -3.482274e-01,
 -0.148227359,  3.482274e-01,
  0.283333333, -5.050000e-01,
  0.783333333, -5.050000e-01,
  1.450000000, -5.050000e-01,
  0.283333333,  5.050000e-01,
  0.783333333,  5.050000e-01,
  1.450000000,  5.050000e-01,
  1.950000000, -5.050000e-01,
  2.450000000, -5.050000e-01,
  2.950000000, -5.050000e-01,
  1.950000000,  5.050000e-01,
  2.450000000,  5.050000e-01,
  2.950000000,  5.050000e-01,
  3.276399495, -5.560485e-01,
  3.276399495,  4.539515e-01,
  3.087316098, -6.960219e-01,
  3.143644997, -3.358601e-01,
  3.087316098,  3.139781e-01,
  3.143644997,  6.741399e-01,
 -0.246951237,  2.775558e-16,
 -0.065975618, -1.368007e-01,
  0.104523204, -3.248428e-01,
  0.002567691, -6.004744e-01,
 -0.220951271, -5.628989e-01,
  0.194977900, -7.057364e-01,
 -0.065975618,  1.368007e-01,
  0.104523204,  3.248428e-01,
  0.005697717,  6.017445e-01,
  0.197803447,  7.062970e-01,
 -0.218650015,  5.644721e-01,
 -0.402602658,  2.322358e-01,
 -0.378528884,  4.359502e-01,
 -0.485975618,  8.077704e-03,
 -0.403828688, -2.522852e-01,
  0.533333333, -3.694048e-01,
  0.522233333, -6.472024e-01,
  1.116666667, -4.190079e-01,
  1.321997201, -3.121302e-01,
  0.911336132, -3.121302e-01,
  1.001032346, -6.598220e-01,
  1.257352762, -6.747338e-01,
  0.533333333,  6.405952e-01,
  0.522233333,  3.627976e-01,
  1.116666667,  5.909921e-01,
  1.321997201,  6.978698e-01,
  0.911336132,  6.978698e-01,
  1.001032346,  3.501780e-01,
  1.257352762,  3.352662e-01,
  1.700000000, -3.694048e-01,
  1.688900000, -6.472024e-01,
  2.200000000, -3.694048e-01,
  2.188900000, -6.472024e-01,
  2.700000000, -3.694048e-01,
  2.688900000, -6.472024e-01,
  2.882272361, -7.192372e-01,
  1.700000000,  6.405952e-01,
  1.688900000,  3.627976e-01,
  2.200000000,  6.405952e-01,
  2.188900000,  3.627976e-01,
  2.700000000,  6.405952e-01,
  2.688900000,  3.627976e-01,
  2.882272361,  2.907628e-01;

    inferenceWald.setNewLocations_f(new_locs);

    std::cout << "Wald f p value: " << inferenceWald.f_p_value() << std::endl;
    std::cout << "Wald f CI: " << inferenceWald.f_CI() << std::endl;
 
}

*/ 




