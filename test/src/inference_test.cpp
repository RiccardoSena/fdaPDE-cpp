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

using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Newton;
using fdapde::core::laplacian;
using fdapde::core::PDE;

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

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;

#include "../../fdaPDE/models/regression/wald.h"


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
TEST(inference_test, WaldExact) {
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


    
     // test correctness
    fdapde::models::Wald<SRPDE, fdapde::models::exact> inference(model);
    std::cout << "creato elemento inference" << std::endl;

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

    std::cout << std::endl;
    inference.setC(C);
    std::cout << "set C" << std::endl;
    DVector<double> beta0(2);
    beta0(0)=2;
    beta0(1)=-1;
    inference.setBeta0(beta0);

    inference.computeCI(fdapde::models::simultaneous);
    std::cout << "computed CI: " << inference.computeCI(fdapde::models::simultaneous)<<std::endl;

    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< inference.p_value(fdapde::models::simultaneous)(0) << std::endl;
    
    std::cout << "ora inizia il test" << std::endl;
    DMatrix<double> matrix(1, 1);
    //matrix << 0.00002458211564814289 ;
    EXPECT_TRUE(almost_equal(inference.p_value(fdapde::models::simultaneous)(0), 0.3355454 , 1e-7));
    
    
}