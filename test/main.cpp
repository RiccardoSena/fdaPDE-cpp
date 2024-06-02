#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

//li ho commentati tutti perchè così facciamo solo il test di inference ma poi saranno da controllare se davvero tutti vadano commetati
// regression test suite
/*
#include "src/srpde_test.cpp"
#include "src/strpde_test.cpp"
#include "src/gsrpde_test.cpp"
#include "src/qsrpde_test.cpp"
#include "src/gcv_srpde_test.cpp"
#include "src/gcv_qsrpde_test.cpp"
#include "src/gcv_srpde_newton_test.cpp"
// #include "src/kcv_srpde_test.cpp"
// functional test suite
#include "src/fpca_test.cpp"
#include "src/fpls_test.cpp"
#include "src/centering_test.cpp"
*/


#include "src/inference_test.cpp"
//#include "src/inferencetime_test.cpp"
#include <unsupported/Eigen/SparseExtra> 
#include <fstream>



int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);


/* 


  SpMatrix<double> invE_nonexact;
    SpMatrix<double> invE_exact;
    SpMatrix<double> precond;

    // Carica la matrice dal file Matrix Market
    Eigen::loadMarket(invE_exact, "../build/invEexact.mtx");

    // Carica la matrice dal file Matrix Market
    Eigen::loadMarket(invE_nonexact, "../build/inversaE2.mtx");

    Eigen::loadMarket(precond, "../build/precond3.mtx");

    DMatrix<double> invE_nonexact_densa = invE_nonexact;

    DMatrix<double> invE_exact_densa=invE_exact;

    DMatrix<double> risultato_FSPAI=precond*precond.transpose();
    std::cout<<"numero righe FSPAI: "<<risultato_FSPAI.rows()<<std::endl;
    std::cout<<"numero righe inversa: "<<invE_exact_densa.rows()<<std::endl;
    std::cout<<"numero righe inversa non esatta: "<<invE_nonexact_densa.rows()<<std::endl;

    SpMatrix<double> risultato = risultato_FSPAI.sparseView();
    Eigen::saveMarket(risultato, "risultatoFSPAI.mtx");

    // Calcolo la differenza tra le due matrici
    DMatrix<double> differenza_nostra = invE_nonexact_densa - invE_exact_densa;
    DMatrix<double> differenza_FSPAI = risultato_FSPAI - invE_exact_densa;

    // Calcolo la norma di Frobenius della differenza
    double norma_frobenius_nostra = differenza_nostra.norm();
    double norma_frobenius_FSPAI = differenza_FSPAI.norm();
    std::cout<<"questa è la frobenius norm della differenza nostra:   "<<norma_frobenius_nostra<<std::endl;
    std::cout<<"questa è la frobenius norm della differenza FSPAI:   "<<norma_frobenius_FSPAI<<std::endl;


    
*/

  return RUN_ALL_TESTS();
}
