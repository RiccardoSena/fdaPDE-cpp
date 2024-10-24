#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::dt;
using fdapde::core::FEM;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;
using fdapde::core::spline_order;

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


// questi sono da controllare 
#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework
#include <cstddef>

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


// test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    time penalization: separable (mass penalization)

/*
TEST(inferencetime_test, Exact24) {
    // define temporal and spatial domain
    Triangulation<1, 1> time_mesh(0, fdapde::testing::pi, 4);
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/strpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/strpde/2D_test2/X.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
    // define model
    double lambda_D = 0.01;
    double lambda_T = 0.01;
    STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    df.stack(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness WALD
    fdapde::models::Wald<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceW(model);
    fdapde::models::Speckman<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceS(model);
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inferenceW.setC(C);
    inferenceS.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(1);
    beta0(0)=2;
    //beta0(1)=-1;
    inferenceW.setBeta0(beta0);
    inferenceS.setBeta0(beta0);

    //DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << confidence_intervals<<std::endl;
    
    EXPECT_TRUE(almost_equal(inferenceW.p_value(fdapde::models::one_at_the_time)(0), 0.7660934 , 1e-7));
    EXPECT_TRUE(almost_equal(inferenceS.p_value(fdapde::models::one_at_the_time)(0), 0.715712 , 1e-7));

}

*/


/*
TEST(inferencetime_test2, Exact242) {
    // define temporal and spatial domain
    Mesh<1, 1> time_mesh(0, fdapde::testing::pi, 4);
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/strpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/strpde/2D_test2/X.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<Mesh<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Mesh<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
    // define model
    double lambda_D = 0.01;
    double lambda_T = 0.01;
    STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    df.stack(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness 
    fdapde::models::Wald2<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceW(model);
    fdapde::models::Speckman2<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceS(model);
    int cols = model.beta().size();
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inferenceW.setC(C);
    inferenceS.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(1);
    beta0(0)  = 2;
    //beta0(1)=-1;
    inferenceW.setBeta0(beta0);
    inferenceS.setBeta0(beta0);

    //DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << confidence_intervals<<std::endl;

    EXPECT_TRUE(almost_equal(inferenceW.p_value(fdapde::models::one_at_the_time)(0), 0.7660934 , 1e-7));
    EXPECT_TRUE(almost_equal(inferenceS.p_value(fdapde::models::one_at_the_time)(0), 0.715712 , 1e-7));

}

*/

/*
TEST(inferencetime_test, nonparametric24) {
    // define temporal and spatial domain
    Triangulation<1, 1> time_mesh(0, fdapde::testing::pi, 4);
    MeshLoader<Triangulation<2, 2>> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/strpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/strpde/2D_test2/X.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
    // define model
    double lambda_D = 0.01;
    double lambda_T = 0.01;
    STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    df.stack(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness 
    fdapde::models::Wald<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceW(model);
    //fdapde::models::ESF<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceESF(model);
    std::cout << "p-value: " << inferenceW.f_p_value() << std::endl;
    std::cout << "CI: " << std::endl;
    std::cout << inferenceW.f_CI() << std::endl;
    //std::cout << "ESF p val: " << inferenceESF.f_p_value() << std::endl;
    //std::cout << "Sign flip p val: " << inferenceESF.sign_flip_p_value() << std::endl;

}

*/

TEST(inferencetime_test, spacetime25D) {
    // define temporal and spatial domain
    Triangulation<1, 1> time_mesh(0, 4, 4);
    MeshLoader<Triangulation<2, 3>> domain("hub2.5D");
    // import data from files
    //DMatrix<double> locs = read_csv<double>("../data/models/strpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/strpde/25D_test1/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/strpde/25D_test1/X.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<decltype(time_mesh), decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
    // define model
    double lambda_D = 0.00001;
    double lambda_T = 0.00001;
    STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    df.stack(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness 
    fdapde::models::Wald<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceW(model);
    fdapde::models::Speckman<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceS(model);
    fdapde::models::ESF<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inferenceESF(model);

    // set H0
    DVector<double> beta0(1);
    beta0 << 0.45;
    inferenceW.setBeta0(beta0);
    inferenceS.setBeta0(beta0);
    inferenceESF.setBeta0(beta0);

    // set C
    DMatrix<double> C = DMatrix<double>::Identity(1, 1);
    inferenceW.setC(C);
    inferenceS.setC(C);
    inferenceESF.setC(C);

    // set N flips
    inferenceESF.setNflip(10000);

    // set locations for inference on f
    DVector<int> loc_ind(5);
    loc_ind << 0, 476, 476*2, 476*3, 476*4;
    inferenceW.setLocationsF(loc_ind);
    
    std::cout << "Wald beta p val: " << inferenceW.p_value(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "Wald beta CI: " << inferenceW.computeCI(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "Speckman beta p val: " << inferenceS.p_value(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "Speckman beta CI: " << inferenceS.computeCI(fdapde::models::one_at_the_time) << std::endl;
    std::cout << "ESF beta p val: " << inferenceESF.p_value(fdapde::models::one_at_the_time) << std::endl;

    std::cout << "Wald f p-value: " << inferenceW.f_p_value() << std::endl;
    std::cout << "Wald f CI: " << std::endl;
    for (int i = 0; i < loc_ind.size(); ++i) {
        std::cout << "Row " << i + 1 << ": " << inferenceW.f_CI().row(i) << std::endl;
    }

    //std::cout << "Length of estimate of f: " << model.f().size() << std::endl;

}


