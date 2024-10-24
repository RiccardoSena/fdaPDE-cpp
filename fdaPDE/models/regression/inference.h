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

#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include <cmath>
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>



#include <cmath>
#include <random>

namespace fdapde {
namespace models {

    // oggetti comuni a Wald e Speckman
    struct exact {};
    struct nonexact {};
    enum CIType {bonferroni,simultaneous,one_at_the_time};


    // P-VALUES

    // calcolo del pvalue di una chiquadro 
    double gamma(double x) {
        if (x <= 0) {
            return 0; // Valore non valido, restituisci 0
        }            
        if (x == 1) {
            return 1; // Caso base: gamma(1) = 1
        }         
        if (x > 1) {
            //std::cout<<"x è maggiore di 1 "<<std::endl;
            double logGamma = 0.5 * log(2 * M_PI * x) + (x - 0.5) * log(x) - x + 1.0 / (12 * x);
            return exp(logGamma);
        } else {
            //std::cout<<"x è 1 "<<x<<std::endl;
            // Formula di riflessione di Euler
            return M_PI / (sin(M_PI * x) * gamma(1 - x));
        }
    }

    double integrand(double t, double a) {
        return pow(t, a - 1) * exp(-t);
    }

    double gamma_incompleta(double a, double x, int numIntervals = 1000) {
        double sum = 0.0;
        double intervalWidth = x / numIntervals;

        for (int i = 0; i < numIntervals; ++i) {
            double left = i * intervalWidth;
            double right = (i + 1) * intervalWidth;
            sum += (integrand(right, a) + integrand(left, a)) / 2.0 * (right - left);
        }
        // std::cout<<"gamma_incompleta restituisce"<<sum<<std::endl;

        return sum;
      }

    double chi_squared_cdf(double chiSquaredStat, int degreesOfFreedom) {
        //std::cout<<"gamma restituisce "<<gamma(degreesOfFreedom / 2.0)<<std::endl;
        double pValue = gamma_incompleta(degreesOfFreedom/2.0,chiSquaredStat/2.0)/gamma(degreesOfFreedom / 2.0);
        // std::cout<<"pvalue è "<<pValue<<std::endl;

        return pValue;
    }
     
    // funzione per calcolare il pvalue di una normale 
    double gaussian_cdf(double x, double mean, double stddev) { 
        return 0.5 * (1 + std::erf((x - mean) / (stddev * std::sqrt(2)))); 
    }



    // quantili 
    // calcolo di quantile di una chiquadro
    double inverseChiSquaredCDF(double alpha, int degreesOfFreedom, double tolerance = 1e-6) {
        double low = 0.0;
        double high = 1000.0; 

        // applica la ricerca binaria fino a raggiungere la precisione desiderata
        while (high - low > tolerance) {
            double mid = (low + high) / 2.0;
            double pValue = chi_squared_cdf(mid, degreesOfFreedom);

            if (pValue < alpha) {
                low = mid;
            } else {
                high = mid;
            }
        }

        return (low + high) / 2.0;
    }

    // Funzione di approssimazione per il calcolo dell'inverso dell'errore
    double inverse_erf(double x) {
        const double epsilon = 1e-10; // Tolleranza per l'approssimazione
        double y = 0.0;
        double delta;
        do {
            delta = (std::erf(y) - x) / (2.0 / std::sqrt(M_PI) * std::exp(-y * y));
            y -= delta;
        } while (std::fabs(delta) > epsilon);
        return y;
    }
    
    // Funzione per calcolare i quantili di una distribuzione normale standard
    double normal_standard_quantile(double percentile) {
        // calcola il quantile utilizzando la funzione inversa della distribuzione normale standard
        return std::sqrt(2.0) * inverse_erf(2.0 * percentile - 1.0);     
    }


 // function that returns the exact inverse of a matrix   
    DMatrix<double> inverse(DMatrix<double> M){
        Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
        // Eigen::PartialPivLU<DMatrix<double>> Mdec_ (M);
        // Mdec_ = M.partialPivLu(); 
        return Mdec_.solve(DMatrix<double>::Identity(M.rows(), M.cols()));
    }


} // closing models namespace
} // closing fdapde namespace

#endif   // __INFERENCE_H__