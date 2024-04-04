#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

//li ho commentati tutti perchè così facciamo solo il test di inference ma poi saranno da controllare se davvero tutti vadano commetati
// regression test suite
//#include "src/srpde_test.cpp"
//#include "src/strpde_test.cpp"
//#include "src/gsrpde_test.cpp"
//#include "src/qsrpde_test.cpp"
//#include "src/gcv_srpde_test.cpp"
//#include "src/gcv_qsrpde_test.cpp"
//#include "src/gcv_srpde_newton_test.cpp"
// solo questpo era già commentato
// #include "src/kcv_srpde_test.cpp"
// functional test suite
//#include "src/fpca_test.cpp"
//#include "src/fpls_test.cpp"
//#include "src/centering_test.cpp"

#include "src/inference_test.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
