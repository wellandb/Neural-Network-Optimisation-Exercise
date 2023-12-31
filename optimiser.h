#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

void initialise_optimiser(double learning_rate, int batch_size, int total_epochs, double intial_learning_rate, double final_learning_rate, double alpha);
void run_optimisation(void);
double evaluate_objective_function(unsigned int sample);
void numerical_solution();

#endif /* OPTMISER_H */
