#ifndef __NN_H__
#define __NN_H__

#include "mbed.h"
#include "arm_math.h"
#include "parameter.h"
#include "weights.h"
#include "arm_nnfunctions.h"

float32_t run_nn(q7_t* input_data, q7_t* output_data, q7_t* LSH_wt, uint8_t options);
float32_t run_nn_layer(q7_t* input_data, q7_t* output_data, q7_t* LSH_wt, uint8_t options);

#endif
