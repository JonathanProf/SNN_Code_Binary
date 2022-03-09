#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

void getWeights(float* weights, const std::string fileName, const uint32_t pixel, const uint32_t neuron );
void getTheta(float *theta, const std::string fileName , const uint32_t numNeurons);
void getInputSample(uint32_t *input, const std::string fileName, const int row, const int col );
void getAssignments(uint8_t *assignments, std::string fileName , const uint32_t numNeurons);

#endif // FUNCTIONS_H
