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
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
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




// RIASSUNTO TESTS 2.7 EXACT E NON EXACT 
/*
TEST(inference_test, exact27) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
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

    DVector<double> pvalueswald = inferenceWald.p_value(fdapde::models::simultaneous);
    //std::cout<<"pvalues wald: "<<std::fixed << std::setprecision(15)<<pvalueswald<<std::endl;
   // DMatrix<double> CIwald_=inferenceWald.computeCI(fdapde::models::one_at_the_time);
   // std::cout << "computed CI: " <<std::fixed << std::setprecision(15)<< CIwald_<<std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    //std::cout<<"pvalues speckman: "<<std::fixed << std::setprecision(15)<<pvaluesspeck<<std::endl;
   // DMatrix<double> CIspeck_=inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
   // std::cout << "computed CI: " << std::fixed << std::setprecision(15)<<CIspeck_<<std::endl;

    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::one_at_the_time);
    //std::cout<<"pvalues esf: "<<pvaluesesf<<std::endl;

    //DMatrix<double> CIESF_=inferenceESF.computeCI(fdapde::models::one_at_the_time);
   // std::cout << "computed CI: " << CIESF_<<std::endl;

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

/*
TEST(inference_test, exact27) {
    // define problem specifics
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    ...
    //define model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
    ...
    // solve smoothing problem
    model.init();
    model.solve();

    // define inference objects
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

    inferenceWald.p_value(fdapde::models::one_at_the_time);
    inferenceWald.computeCI(fdapde::models::one_at_the_time);

    inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    inferenceSpeck.computeCI(fdapde::models::one_at_the_time);

    DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues esf: "<<pvaluesesf<<std::endl;

    DMatrix<double> CIESF_=inferenceESF.computeCI(fdapde::models::one_at_the_time);
    std::cout << "computed CI: " << CIESF_<<std::endl;
*/


/*

TEST(inference_test, nonexact27) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
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
    //fdapde::models::ESF<SRPDE,fdapde::models::nonexact > inferenceESF(model);
 
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

   // int n = 1000;
    //inferenceESF.setNflip(n);

    DVector<double> pvalueswald = inferenceWald.p_value(fdapde::models::simultaneous);
    std::cout<<"pvalues wald sim: "<<std::fixed << std::setprecision(15)<<pvalueswald(0)<<std::endl;
    DVector<double> pvalueswald_oat = inferenceWald.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues wald oat: "<<std::fixed << std::setprecision(15)<<pvalueswald_oat(0)<<std::endl;
   // DMatrix<double> CIwald_=inferenceWald.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " <<std::fixed << std::setprecision(15)<< CIwald_<<std::endl;

    DVector<double> pvaluesspeck = inferenceSpeck.p_value(fdapde::models::simultaneous);
    std::cout<<"pvalues speckman sim: "<<std::fixed << std::setprecision(15)<<pvaluesspeck(0)<<std::endl;
    DVector<double> pvaluesspeck_oat = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    std::cout<<"pvalues speckman oat: "<<std::fixed << std::setprecision(15)<<pvaluesspeck_oat(0)<<std::endl;
   // DMatrix<double> CIspeck_=inferenceSpeck.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << std::fixed << std::setprecision(15)<<CIspeck_<<std::endl;

    //DVector<double> pvaluesesf = inferenceESF.p_value(fdapde::models::simultaneous);
   //std::cout<<"pvalues esf: "<<pvaluesesf<<std::endl;

    //DMatrix<double> CIESF_=inferenceESF.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << CIESF_<<std::endl;

    // test correctness Wald
    //EXPECT_TRUE(almost_equal(pvalueswald(0), 0.2368866 , 1e-6));
    
    // test correctness Speckman
    //EXPECT_TRUE(almost_equal(pvaluesspeck(0), 0.2351342, 1e-6));
    //EXPECT_TRUE(almost_equal(pvaluesspeck(1), 0.0902682, 1e-7));

    // test correctness ESF
    //EXPECT_TRUE(almost_equal(inferenceESF.p_value(fdapde::models::one_at_the_time)(0), 0.164 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalinferenceESF.p_value(fdapde::models::one_at_the_time)(1), 0.924 , 1e-7));

}


*/











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



*/

/*
TEST(inference_test, chronoWald) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
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

*/


TEST(inference_test, chrono) {
    
    std::vector<std::string> Nodes = {
        "2nodes",
        "3nodes",
        "5nodes",
        "10nodes",
        "15nodes",
        "20nodes",
        "25nodes",
        "30nodes",
        "35nodes",
        "40nodes"
    };

    std::string prefix1 = "TIME/";
    std::string prefix2 = "../data/models/TIME/";
    std::string suffix1 = "/y.csv";
    std::string suffix2 = "/X.csv";

    for(std::size_t i = 0; i < Nodes.size(); ++i){

    std::string mesh_str = prefix1 + Nodes[i];
    std::string y_str = prefix2 + Nodes[i] + suffix1;
    std::string X_str = prefix2 + Nodes[i] + suffix2;

    MeshLoader<Triangulation<2,2>> domain(mesh_str);
    // import data from files
    DMatrix<double> y    = read_csv<double>(y_str);
    DMatrix<double> X    = read_csv<double>(X_str);
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.01;

    DVector<double> beta0(1);
    beta0(0) = 0;
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);

    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    
    // set model's data
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // chrono start
    using namespace std::chrono;

    int n_it = 20;
    std::chrono::microseconds total_duration(0);

    for(int i = 0; i < n_it; ++i){

    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    
    inference.setC(C);
    inference.setBeta0(beta0);

    auto start = high_resolution_clock::now();

    inference.p_value(fdapde::models::one_at_the_time);

    auto end = high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    total_duration += duration;

    }
    
    auto average_duration = total_duration / n_it;

    std::cout << "Mean time of " << Nodes[i] << " is: " << average_duration << std::endl;

    }

}



/*

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
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    //model.set_spatial_locations(locs);
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

    //inferenceWald.setf0(f0); 
    //inferenceESF.setf0(f0);
    inferenceESF.setNflip(10000);
    //DVector<int> loc_indexes(7);
    //loc_indexes << 0, 1, 2, 3, 4, 5, 6;

    //inferenceWald.setLocationsF(loc_indexes);
    //inferenceESF.setLocationsF(loc_indexes);

    std::cout << "Wald f p value: " << inferenceWald.f_p_value() << std::endl;
    //std::cout << "Wald f CI: " << inferenceWald.f_CI() << std::endl;
    std::cout << "Esf p value: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign flip p value: " << inferenceESF.sign_flip_p_value() << std::endl;

    //std::cout << "Esf CI: " << inferenceESF.f_CI() << std::endl;

}

*/


/*
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



/*
TEST(inference_test, inference_f_) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
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

    DVector<int> loc_indexes(6);
    loc_indexes << 1, 5, 7, 8, 9, 10;
    inferenceWald.setLocationsF(loc_indexes);
    inferenceESF.setLocationsF(loc_indexes);
    inferenceESF.setNflip(10000);

    std::cout << "Wald f p value: " << inferenceWald.f_p_value() << std::endl;
    std::cout << "Esf p value: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign p value: " << inferenceESF.sign_flip_p_value() << std::endl;

    //std::cout << "Esf CI: " << inferenceESF.f_CI() << std::endl;

}

*/

/*

TEST(inference_test, inference292) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
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
    model.init_psi_esf(problem);
    model.solve();

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);
    //fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);

    //inferenceESF.setNflip(100000);

    DVector<double> mesh_loc (3);
    mesh_loc << 1, 2, 3;
    inferenceESF.setMesh_loc(mesh_loc);

    std::cout << "Esf p value: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign flip p value: " << inferenceESF.sign_flip_p_value() << std::endl;


}

*/

/*

TEST(inference_test, inference29) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
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

    DVector<double> f0 = read_csv<double>("../data/models/srpde/2D_test2/f0.csv");

    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);

    inferenceESF.setNflip(100000);
    inferenceESF.setf0(f0);
    inferenceWald.setf0(f0);

    std::cout << "Wald p value: " << inferenceWald.f_p_value() << std::endl;
    std::cout << "Esf p value: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign flip p value: " << inferenceESF.sign_flip_p_value() << std::endl;


}

*/


/*
TEST(inference_test, inference37){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test3/locs.csv");
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test3/y2.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test3/X.csv");
    // define regularizing PDE
    SMatrix<2> K;
    K << 1, 0, 0, 4;
    auto L = -diffusion<FEM>(K);   // anisotropic diffusion
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define  model
    double lambda = 10;
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

    inferenceESF.setNflip(10000);

    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    
    inferenceWald.setC(C);
    inferenceSpeck.setC(C);

    DVector<double> beta0(2);
    beta0(0) = 2.5;
    beta0(1) = 1.5;

    inferenceWald.setC(C);
    inferenceSpeck.setC(C);

    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);

    DVector<int> locs_ind(3);
    locs_ind << 0, 1, 2;
    inferenceWald.setLocationsF(locs_ind);
    inferenceESF.setLocationsF(locs_ind);


    // test correctness Wald
    //std::cout << "Wald p val: " << inferenceWald.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Wald CI oat: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    //std::cout << "Wald CI sim: " << inferenceWald.computeCI(fdapde::models::simultaneous) << std::endl; 
    //std::cout << "Speck p val: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Speck CI: " << inferenceSpeck.computeCI(fdapde::models::bonferroni) << std::endl;

    std::cout << "Wald f p val: " << inferenceWald.f_p_value() << std::endl;
    std::cout << "ESF f p val: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign flip f p val: " << inferenceESF.sign_flip_p_value() << std::endl;

}

*/


/*

TEST(inference_test, inference44){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("quasi_circle");
    // import data from files
    DMatrix<double, Eigen::RowMajor> K_data  = read_csv<double>("../data/models/srpde/2D_test4/K.csv");
    DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>("../data/models/srpde/2D_test4/b.csv");
    DMatrix<double> subdomains = read_csv<double>("../data/models/srpde/2D_test4/incidence_matrix.csv");
    DMatrix<double> u1 = read_csv<double>("../data/models/srpde/2D_test4/force.csv");
    DMatrix<double> u = DMatrix<double>::Zero(u1.rows(), u1.cols());
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test4/y2.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test4/X.csv");
    // define regularizing PDE
    DiscretizedMatrixField<2, 2, 2> K(K_data);
    DiscretizedVectorField<2, 2> b(b_data);
    auto L = -diffusion<FEM>(K) + advection<FEM>(b);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = std::pow(0.1, 3);
    SRPDE model(problem, Sampling::areal);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(subdomains);
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

    DVector<double> beta0(1);
    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    beta0(0) = 3;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    std::cout << "Model f: " << model.f() << std::endl;
    std::cout << "Model beta: " << model.beta() << std::endl;

    std::cout << "Wald p val: " << inferenceWald.p_value(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "Wald CI: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    std::cout << "Speck p val: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "Speck CI: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "ESF p val: " << inferenceESF.p_value(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "ESF CI: " << inferenceESF.computeCI(fdapde::models::one_at_the_time) << std::endl;
}
*/



/*
TEST(inference_test, inference25D){
    MeshLoader<Triangulation<2, 3>> domain("horsehoe2.5D");
    // import data from files
    //DMatrix<double> locs = read_csv<double>("../data/models/srpde/25D_test1/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/25D_test1/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/25D_test1/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    double lambda = 0.1;
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    //model.set_spatial_locations(locs);
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

    DVector<double> beta0(1);
    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    beta0(0) = 1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    inferenceESF.setNflip(10000);

    
    //std::cout << "Wald p val: " << inferenceWald.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Wald CI: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    //std::cout << "Speck p val: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Speck CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "ESF p val: " << inferenceESF.p_value(fdapde::models::one_at_the_time) << std::endl;
    
    DVector<double> Wald_beta_p = inferenceWald.p_value(fdapde::models::one_at_the_time);
    DMatrix<double> Wald_beta_CI = inferenceWald.computeCI(fdapde::models::one_at_the_time);
    DVector<double> Speck_beta_p = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
    DMatrix<double> Speck_beta_CI = inferenceSpeck.computeCI(fdapde::models::one_at_the_time);

    EXPECT_TRUE(almost_equal(Wald_beta_p(0), 0.01282658 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_beta_CI(0)(0), 1.026778179 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_beta_CI(0)(1), 1.225362393 , 1e-7));
    EXPECT_TRUE(almost_equal(Speck_beta_p(0), 0.02520083 , 1e-7));
    EXPECT_TRUE(almost_equal(Speck_beta_CI(0)(0), 1.013816481 , 1e-7));
    EXPECT_TRUE(almost_equal(Speck_beta_CI(0)(1), 1.208392922 , 1e-7));
    
    

}


*/


/*
TEST(inference_test, inference15D){
    //
    MeshLoader<Triangulation<1, 2>> domain("c_shaped1.5D");
    // import data from files
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/15D_test1/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/15D_test1/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    double lambda = 0.001;
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    //model.set_spatial_locations(locs);
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

    DVector<double> beta0(1);
    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    beta0(0) = 1;
    inferenceWald.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    inferenceWald.setC(C);
    inferenceESF.setC(C);

    inferenceESF.setNflip(10000);

    DVector<double> f0(201);
    f0 << -2.00000000, -3.00000000, -1.00000000, 2.00000000, -2.01250000, -2.02500000, -2.03750000, -2.05000000,
-2.06250000, -2.07500000, -2.08750000, -2.10000000, -2.11250000, -2.12500000, -2.13750000, -2.15000000,
-2.16250000, -2.17500000, -2.18750000, -2.20000000, -2.21250000, -2.22500000, -2.23750000, -2.25000000,
-2.26250000, -2.27500000, -2.28750000, -2.30000000, -2.31250000, -2.32500000, -2.33750000, -2.35000000,
-2.36250000, -2.37500000, -2.38750000, -2.40000000, -2.41250000, -2.42500000, -2.43750000, -2.45000000,
-2.46250000, -2.47500000, -2.48750000, -2.50000000, -2.51250000, -2.52500000, -2.53750000, -2.55000000,
-2.56250000, -2.57500000, -2.58750000, -2.60000000, -2.61250000, -2.62500000, -2.63750000, -2.65000000,
-2.66250000, -2.67500000, -2.68750000, -2.70000000, -2.71250000, -2.72500000, -2.73750000, -2.75000000,
-2.76250000, -2.77500000, -2.78750000, -2.80000000, -2.81250000, -2.82500000, -2.83750000, -2.85000000,
-2.86250000, -2.87500000, -2.88750000, -2.90000000, -2.91250000, -2.92500000, -2.93750000, -2.95000000,
-2.96250000, -2.97500000, -2.98750000, -1.97500000, -1.95000000, -1.92500000, -1.90000000, -1.87500000,
-1.85000000, -1.82500000, -1.80000000, -1.77500000, -1.75000000, -1.72500000, -1.70000000, -1.67500000,
-1.65000000, -1.62500000, -1.60000000, -1.57500000, -1.55000000, -1.52500000, -1.50000000, -1.47500000,
-1.45000000, -1.42500000, -1.40000000, -1.37500000, -1.35000000, -1.32500000, -1.30000000, -1.27500000,
-1.25000000, -1.22500000, -1.20000000, -1.17500000, -1.15000000, -1.12500000, -1.10000000, -1.07500000,
-1.05000000, -1.02500000, -1.00000000, -0.99999999, -0.99998976, -0.99971351, -0.99788659, -0.99199585,
-0.97928944, -0.95776667, -0.92651932, -0.88560002, -0.83572406, -0.77798078, -0.71361540, -0.64388557,
-0.56997694, -0.49296005, -0.41377357, -0.33322330, -0.25198991, -0.17064086, -0.08964387, -0.00938033,
0.06984208, 0.14777809, 0.22423480, 0.29906307, 0.37215010, 0.44341316, 0.51279424, 0.58025567,
0.64577630, 0.70934847, 0.77097536, 0.83066883, 0.88844766, 0.94433600, 0.99836222, 1.05055781,
1.10095657, 1.14959393, 1.19650636, 1.24173089, 1.28530476, 1.32726508, 1.36764860, 1.40649150,
1.44382921, 1.47969631, 1.51412643, 1.54715213, 1.57880493, 1.60911517, 1.63811206, 1.66582361,
1.69227667, 1.71749687, 1.74150870, 1.76433543, 1.78599919, 1.80652096, 1.82592058, 1.84421678,
1.86142720, 1.87756837, 1.89265580, 1.90670393, 1.91972618, 1.93173496, 1.94274172, 1.95275689,
1.96178997, 1.96984950, 1.97694310, 1.98307745, 1.98825833, 1.99249062, 1.99577828, 1.99812441,
1.99953121;

    inferenceWald.setf0(f0);
    inferenceESF.setf0(f0);


    //std::cout << "Wald p val: " << inferenceWald.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Wald CI: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    std::cout << "ESF p val: " << inferenceESF.p_value(fdapde::models::one_at_the_time) << std::endl;

    //std::cout << "Wald f p val: " << inferenceWald.f_p_value() << std::endl;
    //std::cout << "Wald f CI: " << inferenceWald.f_CI() << std::endl;  
    std::cout << "ESF f p val: " << inferenceESF.f_p_value() << std::endl;

    DVector<double> Wald_beta_p = inferenceWald.p_value(fdapde::models::one_at_the_time);
    DVector<double> Wald_beta_CI = inferenceWald.p_value(fdapde::models::one_at_the_time);
    double Wald_f_p = inferenceWald.f_p_value();
    DVector<double> Wald_f_CI= inferenceWald.f_CI();

    
    EXPECT_TRUE(almost_equal(Wald_beta_p(0), 0.007322027 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_beta_CI(0)(0), 0.866035 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_beta_CI(0)(1), 0.9791665 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_f_p, 0.00009238833 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_f_CI(0)(0), -1.9497310811 , 1e-7));
    EXPECT_TRUE(almost_equal(Wald_f_CI(0)(1), -1.51643021 , 1e-7));


}
*/



/*

TEST(inference_test, inference3D){

    MeshLoader<Triangulation<3,3>> domain("unit_sphere3D");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/3D_test1/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/3D_test1/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 4, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    double lambda = 0.01;
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    //model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);

    // solve smoothing problem
    model.init();
    model.init_psi_esf(problem);
    model.solve();   
        
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inferenceWald(model);
    fdapde::models::Speckman<SRPDE, fdapde::models::exact> inferenceSpeck(model);
    fdapde::models::ESF<SRPDE, fdapde::models::exact> inferenceESF(model);

    DVector<double> beta0(2);
    int cols = model.beta().size();
    DMatrix<double> C = DMatrix<double>::Identity(cols, cols);
    beta0(0) = 2;
    beta0(1) = -1;
    inferenceWald.setBeta0(beta0);
    inferenceSpeck.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    inferenceWald.setC(C);
    inferenceSpeck.setC(C);
    inferenceESF.setC(C);

    inferenceESF.setNflip(10000);

    DVector<int> locs_ind(10);
    locs_ind << 1, 6, 8, 11, 16, 18, 20, 21, 23, 24;
    DVector<double> f0(10);
    f0 << -1.6945137, -1.4288656, -0.9956891, -1.8706411, -1.6348375, -2.1668145, -1.0486633,
    -1.3630456, -1.8352489, -2.0174226;
    inferenceWald.setLocationsF(locs_ind);
    inferenceESF.setMesh_loc(locs_ind);

    inferenceWald.setf0(f0);
    inferenceESF.setf0(f0);

    //std::cout << "Wald p val: " << inferenceWald.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Wald CI: " << inferenceWald.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    //std::cout << "Speck p val: " << inferenceSpeck.p_value(fdapde::models::one_at_the_time) << std::endl;
    //std::cout << "Speck CI: " << inferenceSpeck.computeCI(fdapde::models::one_at_the_time) << std::endl;  
    std::cout << "ESF p val: " << inferenceESF.p_value(fdapde::models::simultaneous)(0) << std::endl;

    std::cout << "Wald f p val: " << inferenceWald.f_p_value() << std::endl;
    //std::cout << "Wald f CI: " << inferenceWald.f_CI() << std::endl;  
    std::cout << "ESF f p val: " << inferenceESF.f_p_value() << std::endl;
    std::cout << "Sign flip p val: " << inferenceESF.sign_flip_p_value() << std::endl;


    DVector<double> Wald_beta_p = inferenceWald.p_value(fdapde::models::simultaneous);
    //DVector<double> Wald_beta_CI = inferenceWald.p_value(fdapde::models::one_at_the_time);
    //double Wald_f_p = inferenceWald.f_p_value();
    //DVector<double> Wald_f_CI= inferenceWald.f_CI();
    DVector<double> Speck_beta_p = inferenceSpeck.p_value(fdapde::models::one_at_the_time);
 
    
    EXPECT_TRUE(almost_equal(Wald_beta_p(0), 0.9684002 , 1e-7));
    EXPECT_TRUE(almost_equal(Speck_beta_p(0), 0.6479218 , 1e-7));
    EXPECT_TRUE(almost_equal(Speck_beta_p(1), 0.4182482 , 1e-7));
    //EXPECT_TRUE(almost_equal(Wald_f_p, 0.00003363157 , 1e-7));


}


*/



