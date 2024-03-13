#ifndef _WALD_H_
#define _WALD_H_

// TUTTI QUESTI INCLUDE SARANNO DA CONTROLLARE 
#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>
#include "../model_macros.h"
#include "../model_traits.h"
#include "../space_only_base.h"
#include "../space_time_base.h"
#include "../space_time_separable_base.h"
#include "../space_time_parabolic_base.h"
#include "../sampling_design.h"
#include "gcv.h"
#include "stochastic_edf.h"
using fdapde::core::BinaryVector;

//  i namespace così sono da lasciare 
namespace fdapde {
namespace models {

// base class for any regression model
template <typename Model>
class WALD {
    Model* m_; // in teoria da qui dovremmo avere accesso a tutte le variabili che ci servono per calcolare CI 
    // qua dovrai salvare Q e G

    // ci sarà anche una variabile che fa il check che il modello sia stato runnato prima di fare inferenza

    // funzione che calcola M^-1
    matriceM inverseM(){
        //M^{-1}=A^{-1}-A^{-1}U(G)^{-1}VA^{-1}
        //per ora facciamo finta di avere A^-1
        // direttamente dal modello abbiamo m_.U_ e m_.V_
        // G viene calcolata ma non salvata in decomposition of Woodbury (fdaPDE-core/fdaPDE/linear_algebra/smw.h) quindi va salvata in questa classe 
        //( forse viene anche già calcolata la sua inversa) 
        
        return invM;
    
    }

    // methods per calcolare p_Value e CI
    // in base al tipo di CI che voglio restituisco una cosa diversa quindi il tipo di CI dove lo faccio passare? in impu alla funzione computeCI?
    std::pair<double, double> computeCI(){
        // costruisco sigma grande con W e W^-1

        // costruisco sigma con epsilon, n, q e traccia di S

        //costruisco S con PSI, M^-1 e Q
        
        //costruisco V


        //costruisco lowerBound e upperBound


        return std::make_pair(lowerBound, upperBound);
    }

}  
}  // closing models namespace
}  // closing fdapde namespace
    // method per restituire i risultati