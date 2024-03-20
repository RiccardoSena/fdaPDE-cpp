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
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
#include <fdaPDE/models/regression/wald_base.h>
#include <fdaPDE/models/regression/wald.h>
#include <fdaPDE/models/regression/speckman_base.h>
#include <fdaPDE/models/regression/speckman.h>

#include "../model_macros.h"
#include "../model_traits.h"
#include "srpde.h"
#include "strpde.h"
#include "exact_edf.h"


using fdapde::core::SMW;

//DOMANDE 
// dove va costruito il modello? passa in input alla funzione test?

template <typename E> class inference_test : public ::testing::Test {
   public:
    

    // constructor
    inference_test() = default;


    TEST(non so quale input ci vada) {
      Wald<SRPDE> inference(model);
      inference.computeCI(CItype::Simultaneous);
      EXPECT_TRUE(almost_equal(inference.computeCI(CItype::Simultaneous), file della vecchia libreria che contiene risultati di CI simultaneous))

    }




}