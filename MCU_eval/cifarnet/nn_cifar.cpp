#include "nn.h"
#include "arm_math.h"
#include "dsp/matrix_functions.h"
#include <chrono>
#include <cstdio>

//Timer t;

static uint8_t mean[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM] = MEAN_DATA;

static q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;
static q7_t proj_bias[LSH_H] = {0};

//static q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_wt[32*CONV2_KER_DIM*CONV2_KER_DIM*32] = CONV2_WT;
//static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;
static q7_t conv2_bias[32] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IN_CH*CONV3_KER_DIM*CONV3_KER_DIM*CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static q7_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;

//Add input_data and output_data in top main.cpp file
//uint8_t input_data[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM];
//q7_t output_data[IP1_OUT_DIM];

q7_t col_buffer[6400]; // 6400
q7_t scratch_buffer[60960];

void mean_subtract(q7_t* image_data) {
  for(int i=0; i<DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM; i++) {
    image_data[i] = (q7_t)__SSAT( ((int)(image_data[i] - mean[i]) >> DATA_RSHIFT), 8);
  }
}

float32_t run_nn(q7_t* input_data, q7_t* output_data, q7_t* LSH_wt, uint8_t options) {
  using namespace std::chrono;

  Timer t;
  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 32768;

  //for (int i = 0; i < 3072; i ++) printf("%hhd ", input_data[i]); puts(""); puts("");

  mean_subtract(input_data);

#ifdef DEBUG_ON
  for (int j = 0; j < 3; j ++, puts(""))
    for (int i = 0; i < 3 * 32; i ++) 
      printf("%hhd ", input_data[j * 3 * 32 + i]); puts("");
#endif

  /* 这一部分是模拟CMSIS的NN看能不能得到好的效果  */
  
  uint32_t cmsis_nn_prog_s = us_ticker_read();
  uint16_t Round = 512;
  for (int i = 0; i < Round; i++){
    buffer1 = arm_nn_mat_mult_kernel_q7_q15(
        LSH_wt, (q15_t*)col_buffer, LSH_H, LSH_L, 0, 0,proj_bias, buffer1);
  } // 1024次 slice 0.01ms 切2
  uint32_t cmsis_nn_prog_e = us_ticker_read();
  float cmsis_nn_prog_d = (float)(cmsis_nn_prog_e-cmsis_nn_prog_s)/1000;
  printf("CMSIS-NN proj: %.2f ms\r\n", cmsis_nn_prog_d);


    //return 0; seed,不一样 input固定
     t.start(); 
    // Deep-reuse version.
   arm_convolve_HWC_q7_RGB_cluster(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, LSH_wt, LSH_L, LSH_H, (q15_t*)col_buffer);
  t.stop();
  // CMSIS-NN implemented version. L5 inputsize 3*5*5
    //arm_convolve_HWC_q7_RGB(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
 //t.stop();
#ifdef DEBUG_ON
  for (int i = 0; i < 32; i ++)
    printf("%d ", buffer1[i]); puts("");
#endif
    
  arm_maxpool_q7_HWC(buffer1, POOL1_IN_DIM, POOL1_IN_CH, POOL1_KER_DIM, POOL1_PAD, POOL1_STRIDE, POOL1_OUT_DIM, col_buffer, buffer2);
  arm_relu_q7(buffer2, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  
  
  //arm_convolve_HWC_q7_basic(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PAD, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  //arm_convolve_HWC_q7_fast(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PAD, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  
  //arm_convolve_HWC_q7_RGB_cluster(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PAD, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, LSH_wt, LSH_L2, LSH_H2, (q15_t*)col_buffer);
  


  /*for (int i = 0; i < 32; i ++)
    printf("%d ", buffer2[i]); puts("");
  */
  
  arm_relu_q7(buffer1, RELU2_OUT_DIM*RELU2_OUT_DIM*RELU2_OUT_CH);
  arm_avepool_q7_HWC(buffer1, POOL2_IN_DIM, POOL2_IN_CH, POOL2_KER_DIM, POOL2_PAD, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, buffer2);
  arm_convolve_HWC_q7_fast(buffer2, CONV3_IN_DIM, CONV3_IN_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PAD, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, buffer1, CONV3_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1, RELU3_OUT_DIM*RELU3_OUT_DIM*RELU3_OUT_CH);
  arm_avepool_q7_HWC(buffer1, POOL3_IN_DIM, POOL3_IN_CH, POOL3_KER_DIM, POOL3_PAD, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, buffer2);
  arm_fully_connected_q7_opt(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, output_data, (q15_t*)col_buffer);
  
  
  return duration<float>{t.elapsed_time()}.count()*1000.0;
}
