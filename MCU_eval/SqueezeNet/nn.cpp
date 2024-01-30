#include "nn.h"
#include "arm_math.h"
#include "dsp/matrix_functions.h"

#include "FLASH_SECTOR_F4.h"

// Timer t;

static uint8_t mean[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM] = MEAN_DATA;

//static q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_wt_old[3*5*5*32] = CONV1_WT;
static q7_t conv1_wt[3*3*3*CONV1_OUT_CH];
//static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;
static q7_t conv1_bias_old[32] = CONV1_BIAS;
static q7_t conv1_bias[CONV1_OUT_CH];
static q7_t proj_bias[LSH_H] = {0};



#define SINGLE_LAYER
#ifndef SINGLE_LAYER
//static q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_wt[32*5*5*32] = CONV2_WT;
//static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;
static q7_t conv2_bias[32] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IN_CH*CONV3_KER_DIM*CONV3_KER_DIM*CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

//static q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static q7_t ip1_wt[1024*10] = IP1_WT;
static q7_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;
#endif

const q7_t conv2_wt[CONV2_OUT_CH*CONV2_S1_KX*CONV2_S1_KY]=CONV2_WT1;
//const q7_t conv2_ds_bias[CONV1_OUT_CH]=CONV2_BIAS;
const q7_t conv2_bias[CONV1_OUT_CH]={0};
const q7_t conv2_pw_wt[CONV2_OUT_CH*CONV1_OUT_CH]=CONV2_WT2;
const q7_t conv2_pw_bias[CONV2_OUT_CH]=CONV2_BIAS;
const q7_t conv3_ds_wt[CONV2_OUT_CH*CONV3_DS_KX*CONV3_DS_KY]=CONV3_WT1;
const q7_t conv3_ds_bias[CONV2_OUT_CH]=CONV2_BIAS;
const q7_t conv3_pw_wt[CONV3_OUT_CH*CONV2_OUT_CH]=CONV3_WT2;
const q7_t conv3_pw_bias[CONV3_OUT_CH]=CONV2_BIAS;
const q7_t conv4_ds_wt[CONV3_OUT_CH*CONV4_DS_KX*CONV4_DS_KY]=CONV4_WT1;
const q7_t conv4_ds_bias[CONV3_OUT_CH]=CONV2_BIAS;
const q7_t conv4_pw_wt[CONV4_OUT_CH*CONV3_OUT_CH]=CONV4_WT2;
const q7_t conv4_pw_bias[CONV4_OUT_CH]=CONV2_BIAS;
const q7_t conv5_ds_wt[CONV4_OUT_CH*CONV5_DS_KX*CONV5_DS_KY]=CONV5_WT1;
const q7_t conv5_ds_bias[CONV4_OUT_CH]=CONV2_BIAS;
const q7_t conv5_pw_wt[CONV5_OUT_CH*CONV4_OUT_CH]=CONV5_WT2;
const q7_t conv5_pw_bias[CONV5_OUT_CH]=CONV2_BIAS;
const q7_t final_fc_wt[CONV5_OUT_CH*OUT_DIM]=CONV5_WT1;
const q7_t final_fc_bias[OUT_DIM]={0};

//Add input_data and output_data in top main.cpp file
//uint8_t input_data[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM];
//q7_t output_data[IP1_OUT_DIM];

q7_t col_buffer[6400]; // 6400
q7_t scratch_buffer[32*32*32*2*2]; // kernels * H * W

void mean_subtract(q7_t* image_data) {
  for(int i=0; i<DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM; i++) {
    image_data[i] = (q7_t)__SSAT( ((int)(image_data[i] - mean[i]) >> DATA_RSHIFT), 8);
  }
}

uint32_t data[5] = {0, 1, 2, 3, 4};
uint32_t Rx_data[600]; // 3*5*5*32

void bypass(int dim_x, int dim_y, int channels, q7_t *input_1, q7_t *input_2, q7_t *output) {
  q7_t *cur_output = output, *cur_input_1 = input_1, *cur_input_2 = input_2;
  
  for (int i = 0; i < dim_x; i ++)
    for (int j = 0; j < dim_y; j ++)
      for (int k = 0; k < channels; k ++) {
        *cur_output = *cur_input_1 + *cur_input_2;
        cur_output ++;
        cur_input_1 ++;
        cur_input_2 ++;
      }

}

float32_t run_nn(q7_t* input_data, q7_t* output_data, q7_t* LSH_wt, uint8_t options) {
  using namespace std::chrono;

  Timer t;
  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 32768;
  q7_t* buffer3 = buffer1;

  //for (int i = 0; i < 3072; i ++) printf("%hhd ", input_data[i]); puts(""); puts("");

  //mean_subtract(input_data);

#ifdef DEBUG_ON
  for (int j = 0; j < 3; j ++, puts(""))
    for (int i = 0; i < 3 * 32; i ++) 
      printf("%hhd ", input_data[j * 3 * 32 + i]); puts("");
#endif
  for (int i = 0; i < 3*3*3*CONV1_OUT_CH; i ++)
    conv1_wt[i] = conv1_wt_old[i % (3*3*3*CONV1_OUT_CH)]; 

  /*
  uint32_t cmsis_nn_prog_s = us_ticker_read();
  uint16_t Round = 512;
  for (int i = 0; i < Round; i++){
    buffer1 = arm_nn_mat_mult_kernel_q7_q15(
        LSH_wt, (q15_t*)col_buffer, LSH_H, LSH_L, 0, 0,proj_bias, buffer1);
  } // 1024次 slice 0.01ms 切2
  uint32_t cmsis_nn_prog_e = us_ticker_read();
  float cmsis_nn_prog_d = (float)(cmsis_nn_prog_e-cmsis_nn_prog_s)/1000;
  printf("CMSIS-NN proj: %.2f ms\r\n", cmsis_nn_prog_d);
*/

  
    // Deep-reuse version.
  //arm_convolve_HWC_q7_RGB_cluster(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, LSH_wt, LSH_L, LSH_H, (q15_t*)col_buffer);
  
  // CMSIS-NN implemented version. L5 inputsize 3*5*5


  //Flash_Write_Data(0x08020000, (uint32_t *)conv1_wt, 100);
  //Flash_Write_Data(0x08030000, data, 5);
  //t.start();
  //Flash_Read_Data(0x08020000, Rx_data, 600);
  //t.stop();
  //return duration<float>{t.elapsed_time()}.count()*1000.0;
  t.start();
    arm_convolve_HWC_q7_RGB(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
  
  


//#ifdef DEBUG_ON
    /*
  for (int i = 0; i < 32; i ++)
    printf("%d ", conv1_wt[i]); puts("");
  for (int i = 0; i < 32; i ++)
    printf("%d ", input_data[i]); puts("");
    */
//#endif

  
  /*for (int i = 0; i < 32 * 3; i ++)
      if (buffer2[i]) printf("buffer [%d] = %d ", i, buffer2[i]); puts("");
  */

  arm_relu_q7(buffer1,CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH);

  //Complex bypass with 1x1 kernel
  arm_convolve_HWC_q7_basic(buffer1, CONV1_IN_DIM, CONV1_OUT_CH, conv2_wt, CONV2_OUT_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV1_OUT_X, (q15_t*)col_buffer, NULL);
  

  //CONV2 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, 5, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV2_OUT_X*CONV2_OUT_X*CONV2_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV2_OUT_X*CONV2_OUT_X*CONV2_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV2_OUT_X*CONV2_OUT_X*CONV2_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV2_IN_X, CONV2_IN_Y, CONV2_OUT_S_CH, 1, buffer2, CONV2_OUT_CH, i * CONV2_OUT_E_CH);

  //Bypass
  //bypass(CONV2_IN_X, CONV2_IN_Y, CONV2_OUT_CH, buffer1, buffer3, buffer2);

  //CONV3 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV3_IN_DIM, CONV3_IN_CH, conv2_wt, CONV3_OUT_S_CH, 1, 0, 1, conv2_bias, CONV3_BIAS_LSHIFT, 9, buffer2, CONV3_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV3_OUT_X*CONV3_OUT_X*CONV3_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV3_IN_DIM, CONV3_IN_CH, conv2_wt, CONV3_OUT_E_CH, 1, 0, 1, conv2_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, buffer1, CONV3_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV3_OUT_X*CONV3_OUT_X*CONV3_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV3_IN_DIM, CONV3_IN_CH, conv2_wt, CONV3_OUT_E_CH, 3, 1, 1, conv2_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, buffer1, CONV3_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV3_OUT_X*CONV3_OUT_X*CONV3_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV3_IN_X, CONV3_IN_Y, CONV3_OUT_S_CH, 1, buffer2, CONV3_OUT_CH, i * CONV3_OUT_E_CH);
  

  //Bypass
  //bypass(CONV3_IN_X, CONV3_IN_Y, CONV3_OUT_CH, buffer1, buffer3, buffer2);


  //Complex bypass with 1x1 kernel
  arm_convolve_HWC_q7_basic(buffer1, CONV3_IN_DIM, CONV3_IN_CH, conv2_wt, CONV4_OUT_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV3_OUT_X, (q15_t*)col_buffer, NULL);
  

  //CONV4 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV3_OUT_X, CONV3_OUT_X, conv2_wt, CONV4_OUT_S_CH, 1, 0, 2, conv2_bias, CONV2_BIAS_LSHIFT, 9, buffer2, CONV4_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV4_OUT_X*CONV4_OUT_X*CONV4_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV4_IN_DIM, CONV4_IN_CH, conv2_wt, CONV4_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV4_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV4_OUT_X*CONV4_OUT_X*CONV4_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV4_IN_DIM, CONV4_IN_CH, conv2_wt, CONV4_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV4_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV5_OUT_X*CONV5_OUT_X*CONV5_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV4_IN_X, CONV4_IN_Y, CONV4_OUT_S_CH, 1, buffer2, CONV4_OUT_CH, i * CONV4_OUT_E_CH);

  //Bypass
 // bypass(CONV4_IN_X, CONV4_IN_Y, CONV4_OUT_CH, buffer1, buffer3, buffer2);
  
  //CONV5 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV5_IN_DIM, CONV5_IN_CH, conv2_wt, CONV5_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV5_OUT_RSHIFT, buffer2, CONV5_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV5_OUT_X*CONV5_OUT_X*CONV5_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV5_IN_DIM, CONV5_IN_CH, conv2_wt, CONV5_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV5_OUT_RSHIFT, buffer1, CONV5_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV5_OUT_X*CONV5_OUT_X*CONV5_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV5_IN_DIM, CONV5_IN_CH, conv2_wt, CONV5_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV5_OUT_RSHIFT, buffer1, CONV5_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV5_OUT_X*CONV5_OUT_X*CONV5_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV5_IN_X, CONV5_IN_Y, CONV5_OUT_S_CH, 1, buffer2, CONV5_OUT_CH, i * CONV5_OUT_E_CH);

  //Bypass
  //bypass(CONV5_IN_X, CONV5_IN_Y, CONV5_OUT_CH, buffer1, buffer3, buffer2);

  //Complex bypass with 1x1 kernel
  arm_convolve_HWC_q7_basic(buffer1, CONV5_IN_DIM, CONV5_OUT_CH, conv2_wt, CONV6_OUT_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV5_OUT_X, (q15_t*)col_buffer, NULL);



  //CONV6 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV6_IN_DIM, CONV6_IN_CH, conv2_wt, CONV6_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV6_OUT_RSHIFT, buffer2, CONV6_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV6_OUT_X*CONV6_OUT_X*CONV6_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV6_IN_DIM, CONV6_IN_CH, conv2_wt, CONV6_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV6_OUT_RSHIFT, buffer1, CONV6_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV6_OUT_X*CONV6_OUT_X*CONV6_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV6_IN_DIM, CONV6_IN_CH, conv2_wt, CONV6_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV6_OUT_RSHIFT, buffer1, CONV6_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV6_OUT_X*CONV6_OUT_X*CONV6_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV6_IN_X, CONV6_IN_Y, CONV6_OUT_S_CH, 1, buffer2, CONV6_OUT_CH, i * CONV6_OUT_E_CH);

  //Bypass
  //bypass(CONV6_IN_X, CONV6_IN_Y, CONV6_OUT_CH, buffer1, buffer3, buffer2);

  //CONV7 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV7_IN_DIM, CONV7_IN_CH, conv2_wt, CONV7_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, 9, buffer2, CONV7_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV7_OUT_X*CONV7_OUT_X*CONV7_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV7_IN_DIM, CONV7_IN_CH, conv2_wt, CONV7_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV7_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV7_OUT_X*CONV7_OUT_X*CONV7_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV7_IN_DIM, CONV7_IN_CH, conv2_wt, CONV7_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV7_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV7_OUT_X*CONV7_OUT_X*CONV7_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV7_IN_X, CONV7_IN_Y, CONV7_OUT_S_CH, 1, buffer2, CONV7_OUT_CH, i * CONV7_OUT_E_CH);
  
  //Bypass
  //bypass(CONV7_IN_X, CONV7_IN_Y, CONV7_OUT_CH, buffer1, buffer3, buffer2);

  //Complex bypass with 1x1 kernel
  arm_convolve_HWC_q7_basic(buffer1, CONV7_IN_DIM, CONV7_OUT_CH, conv2_wt, CONV8_OUT_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV7_OUT_X, (q15_t*)col_buffer, NULL);


  //CONV8 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV8_IN_DIM, CONV8_IN_CH, conv2_wt, CONV8_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, 9, buffer2, CONV8_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV8_OUT_X*CONV8_OUT_X*CONV8_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV8_IN_DIM, CONV8_IN_CH, conv2_wt, CONV8_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV8_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV8_OUT_X*CONV8_OUT_X*CONV8_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV8_IN_DIM, CONV5_IN_CH, conv2_wt, CONV8_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV8_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV8_OUT_X*CONV8_OUT_X*CONV8_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV8_IN_X, CONV8_IN_Y, CONV8_OUT_S_CH, 1, buffer2, CONV8_OUT_CH, i * CONV8_OUT_E_CH);

  //Bypass
  //bypass(CONV8_IN_X, CONV8_IN_Y, CONV8_OUT_CH, buffer1, buffer3, buffer2);
  
  //CONV9 : 1x1 squeeze + 1x1 expand + 3x3 expand conv

  //1x1 squeeze
  //arm_convolve_HWC_q7_basic(buffer1, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer2, CONV2_OUT_X, (q15_t*)col_buffer, NULL);
  arm_convolve_HWC_q7_basic(buffer1, CONV8_OUT_X, CONV9_IN_CH, conv2_wt, CONV9_OUT_S_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV9_OUT_RSHIFT, buffer2, CONV9_OUT_X, (q15_t*)col_buffer, NULL);
  
  arm_relu_q7(buffer2,CONV9_OUT_X*CONV9_OUT_X*CONV9_OUT_S_CH);

  //1x1 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV9_IN_DIM, CONV9_IN_CH, conv2_wt, CONV9_OUT_E_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV9_OUT_RSHIFT, buffer1, CONV9_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV9_OUT_X*CONV9_OUT_X*CONV9_OUT_E_CH);

  //3x3 expand
  arm_convolve_HWC_q7_basic(buffer2, CONV9_IN_DIM, CONV9_IN_CH, conv2_wt, CONV9_OUT_E_CH, 3, 1, 1, conv2_bias, CONV2_BIAS_LSHIFT, CONV9_OUT_RSHIFT, buffer1, CONV9_OUT_X, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV9_OUT_X*CONV9_OUT_X*CONV9_OUT_E_CH);

  for (int i = 0; i < 2; i ++)
    arm_concatenation_s8_z(buffer1, CONV9_IN_X, CONV9_IN_Y, CONV9_OUT_S_CH, 1, buffer2, CONV9_OUT_CH, i * CONV9_OUT_E_CH);

  //Bypass
  //bypass(CONV9_IN_X, CONV9_IN_Y, CONV9_OUT_CH, buffer1, buffer3, buffer2);

  // CONV10
  arm_convolve_HWC_q7_basic(buffer1, CONV10_IN_DIM, CONV10_IN_CH, conv2_wt, CONV10_OUT_CH, 1, 0, 1, conv2_bias, CONV2_BIAS_LSHIFT, 13, output_data, CONV10_OUT_DIM, (q15_t*)col_buffer, NULL);
  
  //arm_relu_q7(output_data,CONV10_IN_DIM*CONV10_IN_DIM*CONV10_OUT_CH);
  
  return duration<float>{t.elapsed_time()}.count()*1000.0;
  


  //
  printf("insert \n\n");
  
  for (int i = 0; i < 32 * 3; i ++)
    printf("%d ", buffer1[i]); puts("");
  //for (int i = 0; i < 32*32*25; i ++) if(buffer2[i]) printf("buffer2 [%d] = %d ", i, buffer2[i]); puts("");
    
   
  for (int i = 0; i < 32 * 3; i ++)
    printf("%d ", buffer2[i]); puts("");
  
  //arm_fully_connected_q7(buffer2, final_fc_wt, CONV2_OUT_CH, OUT_DIM, FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT, final_fc_bias, output_data, (q15_t*)col_buffer);

  for (int i = 0; i < 32 * 3; i ++)
    printf("%d ", buffer1[i]); puts("");
  

  return duration<float>{t.elapsed_time()}.count()*1000.0;


  /*

  #ifdef VOID
  arm_maxpool_q7_HWC(buffer1, POOL1_IN_DIM, POOL1_IN_CH, POOL1_KER_DIM, POOL1_PAD, POOL1_STRIDE, POOL1_OUT_DIM, col_buffer, buffer2);
  
  arm_relu_q7(buffer2, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  
  arm_convolve_HWC_q7_fast(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PAD, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1, RELU2_OUT_DIM*RELU2_OUT_DIM*RELU2_OUT_CH);
  
  arm_avepool_q7_HWC(buffer1, POOL2_IN_DIM, POOL2_IN_CH, POOL2_KER_DIM, POOL2_PAD, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, buffer2);
  arm_convolve_HWC_q7_fast(buffer2, CONV3_IN_DIM, CONV3_IN_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PAD, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, buffer1, CONV3_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1, RELU3_OUT_DIM*RELU3_OUT_DIM*RELU3_OUT_CH);
  arm_avepool_q7_HWC(buffer1, POOL3_IN_DIM, POOL3_IN_CH, POOL3_KER_DIM, POOL3_PAD, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, buffer2);
  
  arm_fully_connected_q7_opt(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, output_data, (q15_t*)col_buffer);
  #endif
  
  return duration<float>{t.elapsed_time()}.count()*1000.0;
  */
}



float32_t run_nn_layer(q7_t* input_data, q7_t* output_data, q7_t* LSH_wt, uint8_t options) {
  using namespace std::chrono;

  Timer t;
  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 32768;

  //for (int i = 0; i < 3072; i ++) printf("%hhd ", input_data[i]); puts(""); puts("");

  //mean_subtract(input_data);

#ifdef DEBUG_ON
  for (int j = 0; j < 3; j ++, puts(""))
    for (int i = 0; i < 3 * 32; i ++) 
      printf("%hhd ", input_data[j * 3 * 32 + i]); puts("");
#endif

  /*
    padding the additional spaces of conv1_wt and conv1_bias.
  */

  for (int i = 0; i < 3*5*5*CONV1_OUT_CH; i ++)
    conv1_wt[i] = conv1_wt_old[i % 3*5*5*32];
  for (int i = 0; i < CONV1_OUT_CH; i ++)
    conv1_bias[i] = conv1_bias_old[i % 32];

  t.start();
    // Deep-reuse version.
   arm_convolve_HWC_q7_RGB_cluster(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, LSH_wt, LSH_L, LSH_H, (q15_t*)col_buffer);
  
  // CMSIS-NN implemented version. L5 inputsize 3*5*5
    //arm_convolve_HWC_q7_RGB(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PAD, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
  t.stop();

  
  return duration<float>{t.elapsed_time()}.count()*1000.0;
}
