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
using fdapde::core::Mesh;
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
#include "../../fdaPDE/models/regression/eigen_sign_flip.h"


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
#include "../../fdaPDE/models/regression/eigen_sign_flip.h"



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

TEST(inferencetime_test, WaldExact27Sim) {
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



    // test correctness WALD
    fdapde::models::Wald<STRPDE<SpaceTimeSeparable, fdapde::monolithic>, fdapde::models::exact> inference(model);
    int cols = model.beta().size();
    std::cout<<"il valore dei beta è: "<<model.beta()<<std::endl;
    DMatrix<double> C=DMatrix<double>::Identity(cols, cols);    
    inference.setC(C);
    //std::cout << "set C" << std::endl;
    DVector<double> beta0(1);
    beta0(0)=2;
    //beta0(1)=-1;
    inference.setBeta0(beta0);
    std::cout << "beta0 è: " << beta0<<std::endl;

    //DMatrix<double> confidence_intervals=inference.computeCI(fdapde::models::one_at_the_time);
    //std::cout << "computed CI: " << confidence_intervals<<std::endl;

    DVector<double> pvalues=inference.p_value(fdapde::models::one_at_the_time);
    std::cout << "il valore dei pvalue è" << std::endl;
    std::cout<< pvalues(0) << std::endl;
    //std::cout<< pvalues(1) << std::endl;
    
    //std::cout << "ora inizia il test wald " << std::endl;
    EXPECT_TRUE(almost_equal(pvalues(0), 0.5290765 , 1e-7));
    //EXPECT_TRUE(almost_equal(pvalues(1), 0.4119913 , 1e-7));

}