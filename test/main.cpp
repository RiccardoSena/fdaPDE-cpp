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

// questa serve per leggere dai file quindi l'ho aggiunta io 
#include <fstream>


int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);


    SpMatrix<double> invE_nonexact;
    SpMatrix<double> invE_exact;

    // Carica la matrice dal file Matrix Market
    Eigen::loadMarket(invE_exact, "../build/invE_exact.mtx");

    // Verifica che la matrice sia stata caricata correttamente
    std::cout << "Numero di righe: " << invE_exact.rows() << std::endl;
    std::cout << "Numero di colonne: " << invE_exact.cols() << std::endl;


    // Carica la matrice dal file Matrix Market
    Eigen::loadMarket(invE_nonexact, "../build/invE_nonexact.mtx");

    // Verifica che la matrice sia stata caricata correttamente
    std::cout << "Numero di righe: " << invE_nonexact.rows() << std::endl;
    std::cout << "Numero di colonne: " << invE_nonexact.cols() << std::endl;


    DMatrix<double> invE_nonexact_densa = invE_nonexact;
    DMatrix<double> invE_exact_densa=invE_exact;

    // Calcolo la differenza tra le due matrici
    DMatrix<double> differenza = invE_nonexact_densa - invE_exact_densa;

    // Calcolo la norma di Frobenius della differenza
    double norma_frobenius = differenza.norm();
    std::cout<<"questa è la frobenius norm della differenza:   "<<norma_frobenius<<std::endl;


  
  return RUN_ALL_TESTS();
}
