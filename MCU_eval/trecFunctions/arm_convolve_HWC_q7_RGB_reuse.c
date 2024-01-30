#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "arm_math.h"
#include "parameter.h"
#include "dsp/matrix_functions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Q7 convolution function for RGB image
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   bufferB     pointer to buffer space for output
 * @param[in]       L           length of sub-vector
 * @param[in]       H           power for Number of cluster
 * @return     The function returns either
 * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * @details
 *
 * <b>Buffer size:</b>
 *
 * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
 *
 * bufferB size: 0
 *
 * <b>Input dimension constraints:</b>
 *
 * ch_im_in equals 3
 *
 * This kernel is written exclusively for convolution with ch_im_in
 * equals 3. This applies on the first layer of CNNs which has input
 * image with RGB format.
 */
// q15_t sub_im2col_buf[(32*32)*LSH_L]; 
//q7_t sub_im2col_buf[(32*32)*LSH_L];  // 32 * 32 * 5


#define PROFILE // oprion flag for profiling information
//#define DEBUG_ON
#define PRINT_TIME

//static q7_t buf[32 * 32 * 3 * 5 * 5];
static q7_t buf[16 * 16 * 16 * 5 * 7];

static q7_t proj_buf[(32*32)*LSH_H2];
static uint16_t clusterID_buf[(16*16)*1];
static q15_t cnt_buf[LSH_2_to_H*1 + 16];
static uint16_t reverseID_buf[(LSH_2_to_H + 16) * 1]; //
static q7_t cen_buf[(LSH_2_to_H + 16) * 130]; // * LSH_L
static q7_t tempRes_buf[(LSH_2_to_H + 16) * 16];


// struct timeval start, end;
arm_status arm_convolve_HWC_q7_RGB_cluster(const q7_t *Im_in,
                                   const uint16_t dim_im_in,
                                   const uint16_t ch_im_in,
                                   const q7_t *wt,
                                   const uint16_t ch_im_out,
                                   const uint16_t dim_kernel,
                                   const uint16_t padding,
                                   const uint16_t stride,
                                   const q7_t *bias,
                                   const uint16_t bias_shift,
                                   const uint16_t out_shift,
                                   q7_t *Im_out,
                                   const uint16_t dim_im_out,
                                   const q7_t* LST_wt,
                                   const uint16_t L,
                                   const uint16_t H,
                                   q15_t *bufferA)
{
    // (void)bufferB;
    /* Run the following code for Cortex-M4 and Cortex-M7 */
    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */
    // q15_t *pBuffer = bufferA;

    // q15_t *pBuffer = sub_im2col_buf;
    // q7_t *pOut = Im_out;
    const q7_t *pwt = wt;

    // check if number of input channels is 3
    if (ch_im_in != 3)
    {
        //return ARM_MATH_SIZE_MISMATCH;
    }

    // This part implements the partial im2col CONV function
    //uint16_t n_sub = N_SUB_VEC;
    uint16_t n_sub = dim_kernel * dim_kernel * ch_im_in / LSH_L;
    #ifdef PROFILE
	printf("n_sub = %d !!!!!!!!!\n", n_sub);
    #endif

    // for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    // {
    //     for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
    //     {
    //         // im2col: store one sliding window into a column
    //         for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
    //         {
    //             for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
    //             {
    uint32_t all_s = us_ticker_read();
    float all_d = 0;
    float sub_d = 0;
    float slice_d = 0;
    float proj_d = 0;
    float transID_d = 0;
    float centroid_d = 0;
    float tempRs_d = 0;
    float recover_d = 0;

       	// for (int i = 0; i < 100; i ++) printf("%hhd ", Im_in[i]); printf("\n");

	q7_t *data_row = buf;
#ifdef PROFILE
    uint32_t im2col_s = us_ticker_read();
#endif

	int channels = ch_im_in;
	//for (int w_out = 0; w_out < dim_im_in; w_out ++) for (int h_out = 0; h_out < dim_im_in; h_out ++) {
    for (int w_out = 0; w_out < dim_im_out; w_out ++) for (int h_out = 0; h_out < dim_im_out; h_out ++) {
		int h_in = h_out * stride - padding;
		int w_in = w_out * stride - padding;

		//q7_t *im = Im_in[h_in][w_in][channel_in]
		q7_t *im = Im_in + (h_in * dim_im_in + w_in) * ch_im_in;
		//printf("init = %d\n", (h_in * dim_im_in + w_in) * ch_im_in);
		
		int kernel_height = dim_kernel, kernel_width = dim_kernel;
         
        #define FASTER
		#ifdef FASTER

        int matrix_offset = 0, matrix_id = 0;

        q7_t *tmp_data = data_row + (h_out * dim_im_in + w_out) * L;
		q7_t *tmp_im = im;

        for (int i = 0; i < kernel_height; i ++) {
			tmp_im = im + i * dim_im_in * channels;
			for (int j = 0; j < kernel_width; j ++) {

				int h = h_in + i;
				int w = w_in + j;

                
				for (int channel_in = 0; channel_in < ch_im_in; channel_in ++) {

					//int row_offset = (i * kernel_width + j) * channels + channel_in;

                    /*
					data_row[((matrix_id * dim_im_in + h_out) * dim_im_in + w_out) * L + matrix_offset]
						=
                        */ 

                    *tmp_data = (h >= 0 && w >= 0 && h < dim_im_in && w < dim_im_in)
						//? im[(i * dim_im_in + j) * channels + channel_in] // im[i * width + j]
						? *tmp_im // im[i * width + j]
						: 0;

					tmp_im ++;

                    matrix_offset ++;
                    tmp_data ++;
                    
                    if (matrix_offset == L) {
                        matrix_offset = 0;
                        matrix_id ++;
                        tmp_data -= L;
                        tmp_data += dim_im_in * dim_im_in * L;
                    }
				}
			}
		}


        #else
    
        for (int i = 0; i < kernel_height; i ++)
			for (int j = 0; j < kernel_width; j ++) {

				int h = h_in + i;
				int w = w_in + j;

                
				for (int channel_in = 0; channel_in < ch_im_in; channel_in ++) {

					//int row_offset = data_row[i][j][channel_in];
					int row_offset = (i * kernel_width + j) * channels + channel_in;
					int matrix_offset = row_offset % L;
					int matrix_id = row_offset / L;

					/*
					int matrix_offset = row_offset;
					int matrix_id = 0;
					*/

					//printf("data_row [ matrix_offset = %d][ matrix_id = %d ] = im [ i = %d ][ j = %d ][ channel_in = %d ] = %d\n", matrix_offset, matrix_id, i, j, channel_in, im[((i * dim_im_in + j) * channels + channel_in)]);


					//data_row[matrix_id][h_out][w_out][i][j][channel_in] = im[i][j][channel_in];
					//data_row[matrix_id][h_out][w_out][matrix_offset] = im[i][j][channel_in];
					//printf ("%d %d %d %d\n", h, w, h < dim_im_in, w < dim_im_in);
					/*
					data_row[((matrix_id * dim_im_in + h_out) * dim_im_in + w_out) * L + matrix_offset]
						= (h >= 0 && w >= 0 && h < dim_im_in && w < dim_im_in)
						? im[(i * dim_im_in + j) * channels + channel_in] // im[i * width + j]
						: 0;
						*/

					data_row[((matrix_id * dim_im_in + h_out) * dim_im_in + w_out) * L + matrix_offset]
						= (h >= 0 && w >= 0 && h < dim_im_in && w < dim_im_in)
						? im[(i * dim_im_in + j) * channels + channel_in] // im[i * width + j]
						: 0;


					//printf("[i] = %d\n", ((matrix_id * dim_im_in + h_out) * dim_im_in + w_out) * L + matrix_offset );
					//printf("im [i] = %d\n", (i * dim_im_in + j) * channels + channel_in);
					//printf("data_row [] = %d\n", data_row[((matrix_id * dim_im_in + h_out) * dim_im_in + w_out) * L + matrix_offset] );
					//printf("%d\n", data_row[0]);
				}
			}
                #endif
	}


	/*
	for (int i = 0 ; i < 5; i ++, puts("")) for(int j = 0; j < 3 * 5 * 5; j ++)
		printf("%d ", data_row[i * 3 * 5 * 5 + j]);
		*/

	 q7_t status;
     /*
		arm_matrix_instance_q7 A;
        arm_mat_init_q7(&A, LSH_L, ch_im_out, pwt);
        arm_matrix_instance_q7 B;
        arm_mat_init_q7(&B, 32*32, LSH_L, (q7_t *)data_row);
        arm_matrix_instance_q7 C;
        arm_mat_init_q7(&C, 32*32, ch_im_out, (q7_t *)Im_out);
        arm_mat_mult_q7(&B, &A, &C, &status);
    */
		/*
	if (status == ARM_MATH_SUCCESS)
		printf("Success!!\n");
	else 
		printf("Failure!!\n");
		*/

    //return (ARM_MATH_SUCCESS);

#ifdef PROFILE
    uint32_t im2col_e = us_ticker_read();
#endif

#ifdef DEBUG_ON
    printf("im2col completed!\n");
#endif
	

	//for (int i = 0; i < 100; i ++) printf("%hhd ", Im_in[i]); printf("\n");

	/*
	for (int i = 0; i < 32 * 32; i ++, puts(""))
		for (int j = 0; j < L; j ++)
			printf("%d ", (int)data_row[L * i + j]);

    return (ARM_MATH_SUCCESS);
	*/

	/*
	for (int i = 0; i < L; i ++) {
		if (i % 15 == 0)
			puts("");
		printf("%d ", (int)data_row[L * 2 + i]); 
	}
	puts("");
	*/
	
	//printf("im2row done!\n");
	

	/*
	   im2row reconstructing done
	 */

#ifdef DEBUG_ON
	printf("n_sub = %d\n", n_sub);
#endif
    for (int n = 0; n < n_sub; n++){

#ifdef PROFILE
        uint32_t sub_s = us_ticker_read();
#endif
		arm_memset_q7((q7_t *)proj_buf, 0, 32 * 32 * LSH_H * sizeof(q7_t));
        arm_memset_q7((q7_t *)cnt_buf, 0, (LSH_2_to_H + 16) * sizeof(q15_t));
        arm_memset_q7((q7_t *)reverseID_buf, 0, (LSH_2_to_H + 16) * sizeof(q15_t));

		arm_memset_q7((q7_t *)cen_buf, 0, (LSH_2_to_H + 16) * LSH_L * sizeof(q7_t));

		arm_memset_q7((q7_t *)tempRes_buf, 0, (LSH_2_to_H + 16) * ch_im_out * sizeof(q7_t));
#ifdef DEBUG_ON
		printf("n = %d\n ----------------\n", n);
#endif

#ifdef PROFILE
        uint32_t slice_s = us_ticker_read();
#endif
		/*
        for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++){ // dim_im_out = 32
            for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++){ // dim_im_out = 32
                for (int k = 0; k < dim_kernel; k++){ // dim_kernel = 5
                    i_ker_y = i_out_y * stride - padding + (n*dim_kernel) + k;
                    i_ker_x = i_out_x * stride - padding + (n*dim_kernel) + k;
                    // Zero Padding
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        // Equivalent to arm_fill_q15(0, pBuffer, ch_im_in) with assumption: ch_im_in = 3 
                        // arm_memset_q7((q7_t *)pBuffer, (q7_t)0, 3 * sizeof(q15_t));
                        arm_memset_q7(pBuffer, (q7_t)0, 3 * sizeof(q7_t));
                        pBuffer += 3;
                    }
                    else
                    {
                        // Equivalent to:  arm_q7_to_q15_no_shift( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*3, pBuffer, 3);
                        const q7_t *pPixel = Im_in + (i_ker_y * dim_im_in + i_ker_x) * 3;
                        arm_memset_q7(pBuffer++, *pPixel, sizeof(q7_t));
                        arm_memset_q7(pBuffer++, *(pPixel+1), sizeof(q7_t));
                        arm_memset_q7(pBuffer++, *(pPixel+2), sizeof(q7_t));
                    }
                }
            }
        }
		*/

		//sub_im2col_buf = data_row[n][];

		//q7_t *sub_im2col_buf = data_row + n * 32 * 32 * L; // dim_im_in
        q7_t *sub_im2col_buf = data_row + n * dim_im_in * dim_im_in * L;

#ifdef DEBUG_ON
		puts("im2col [1] = \n");
		for (int i = 0; i < L; i ++)
			printf("%d ", sub_im2col_buf[i]); puts("");
#endif

		/*
		for (int i = 0; i < L; i ++) {
			if (i % 15 == 0) 
				puts("");
			printf("%d ", (int)sub_im2col_buf[i]);
		}
		puts("");
		*/

#ifdef DEBUG_ON
		printf("slice done!\n");
#endif

#ifdef PROFILE
        uint32_t slice_e = us_ticker_read();
        slice_d += (float)(slice_e-slice_s)/1000;
#endif
        // for (int i = 0; i < 32*32; i++){
        //     uint16_t id = 0;
        //     for (int j = 0; j < LSH_L; j++){
        //         printf("%d ", sub_im2col_buf[i*LSH_L+j]);
        //         // if (featuremap_buf[i*LSH_H+j] > 0) id |= (1 << j);
        //     }
        //     // clusterID_buf[i] = id;
        //     printf("\n");
        // }
        // deep-reuse pre-processing
#ifdef PROFILE
        uint32_t proj_s = us_ticker_read();
#endif

    #define CMSIS_PROJ
    #ifndef CMSIS_PROJ
        arm_matrix_instance_q7 w; 
        arm_matrix_instance_q7 im2col;
        arm_matrix_instance_q7 fp;
        arm_mat_init_q7(&w, LSH_L2, LSH_H2, (q7_t *)LST_wt);
        arm_mat_init_q7(&im2col, dim_im_in * dim_im_in, LSH_L2, (q7_t *)sub_im2col_buf);
        arm_mat_init_q7(&fp, dim_im_in * dim_im_in, LSH_H2, (q7_t *)proj_buf);
        arm_mat_mult_q7(&im2col, &w, &fp, &status);

    #else

        //float reader_d = 0;

        q15_t *pBuffer = bufferA;
        q7_t *projection_out = proj_buf;
        //for (int out_x = 0; out_x < 1024; out_x ++) {
        for (int out_x = 0; out_x < dim_im_in * dim_im_in; out_x ++) {

//#define AGGRESIVE
#ifdef AGGRESIVE
            for (int out_y = 0; out_y < LSH_L / 4 + 1; out_y ++ ){
                const q7_t *pPixel = sub_im2col_buf + out_x * LSH_L + out_y * 4;
                
                //uint32_t reader_s = us_ticker_read();
                q31_t buf = arm_nn_read_q7x4(pPixel);
                //uint32_t reader_e = us_ticker_read();
                //reader_d += (float)(reader_e-reader_s)/1000;

                union arm_nnword top;
                union arm_nnword bottom;

                top.word = __SXTB16(buf);
                bottom.word = __SXTB16(__ROR(buf, 8));

                /*
                    *  little-endian, | 4th | 3rd  | 2nd  | 1st  |
                    *                MSB                         LSB
                    *   top | 3rd | 1st |; bottom | 4th | 2nd |
                    *
                    *  version 1, need to swap 2nd and 3rd weight
                    * *__SIMD32(pBuffer) = top.word;
                    * *(pBuffer+2) = bottom.half_words[0];
                    *
                    *  version 2, no weight shuffling required
                    */

                /*
                *pBuffer++ = top.half_words[0];
                int32_t packed_word = __PKHBT(bottom.word, top.word, 0);
                arm_memcpy_q7((q7_t *)pBuffer, (q7_t *)&packed_word, 4);
                */

                int32_t packed_word1 = __PKHTB(top.word, bottom.word, 16);
                int32_t packed_word2 = __PKHBT(bottom.word, top.word, 16);

                arm_memcpy_q7((q7_t *)pBuffer, (q7_t *)&packed_word1, 4);
                arm_memcpy_q7((q7_t *)pBuffer, (q7_t *)&packed_word2, 4);
                pBuffer += 4;
            }

            pBuffer --;

            if (pBuffer == bufferA + 2 * LSH_L) {
                projection_out = arm_nn_mat_mult_kernel_q7_q15(
                    LST_wt, bufferA, LSH_H, LSH_L, bias_shift, out_shift, bias, projection_out
                ); 
                /* counter reset */
                pBuffer = bufferA;
            }
        }



#else
            for (int out_y = 0; out_y < LSH_L / 3; out_y ++ ){
                const q7_t *pPixel = sub_im2col_buf + (out_x * LSH_L / 3 + out_y) * 3;
                
                //uint32_t reader_s = us_ticker_read();
                q31_t buf = arm_nn_read_q7x4(pPixel);
                //uint32_t reader_e = us_ticker_read();
                //reader_d += (float)(reader_e-reader_s)/1000;

                union arm_nnword top;
                union arm_nnword bottom;

                top.word = __SXTB16(buf);
                bottom.word = __SXTB16(__ROR(buf, 8));

#ifndef ARM_MATH_BIG_ENDIAN
                /*
                    *  little-endian, | omit | 3rd  | 2nd  | 1st  |
                    *                MSB                         LSB
                    *   top | 3rd | 1st |; bottom | omit | 2nd |
                    *
                    *  version 1, need to swap 2nd and 3rd weight
                    * *__SIMD32(pBuffer) = top.word;
                    * *(pBuffer+2) = bottom.half_words[0];
                    *
                    *  version 2, no weight shuffling required
                    */
                *pBuffer++ = top.half_words[0];
                int32_t packed_word = __PKHBT(bottom.word, top.word, 0);
                arm_memcpy_q7((q7_t *)pBuffer, (q7_t *)&packed_word, 4);
#else
                /*
                    *  big-endian,    | 1st  | 2nd  | 3rd  | omit |
                    *                MSB                         LSB
                    *  top | 2nd | omit |; bottom | 1st | 3rd |
                    *
                    *  version 1, need to swap 2nd and 3rd weight
                    * *__SIMD32(pBuffer) = bottom.word;
                    * *(pBuffer+2) = top.half_words[1];
                    *
                    *  version 2, no weight shuffling required
                    */
                *pBuffer++ = bottom.half_words[0];
                int32_t packed_word = __PKHTB(top.word, bottom.word, 0);
                arm_memcpy_q7((q7_t *)pBuffer, (q7_t *)&packed_word, 4);
#endif
                pBuffer += 2;
            }

            if (pBuffer == bufferA + 2 * LSH_L) {
                projection_out = arm_nn_mat_mult_kernel_q7_q15(
                    LST_wt, bufferA, LSH_H, LSH_L, bias_shift, out_shift, bias, projection_out
                ); 
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    #endif


    #endif    
        
#ifdef PROFILE
        uint32_t proj_e = us_ticker_read();
        proj_d += (float)(proj_e-proj_s)/1000;
#endif

#ifdef DEBUG_ON
		printf("projection done!\n");
#endif


        // q7_t* fp = featuremap_buf;
        // for (int i = 0; i < 32*32/2; i++){
        //     fp = arm_nn_mat_mult_kernel_q7_q15(LST_wt, sub_im2col_buf, LSH_H, LSH_L, 0, 0, 0, fp);
        // }

        // transfrom featuremap_buffer to clusterID
#ifdef PROFILE
        uint32_t transID_s = us_ticker_read();
#endif
        //for (int i = 0; i < 32*32; i++){
        for (int i = 0; i < dim_im_out*dim_im_out; i++){
            uint16_t id = 0;
            for (int j = 0; j < LSH_H; j++){
                if (proj_buf[i*LSH_H+j] > 0) id |= (1 << j);
            }
            clusterID_buf[i] = id;

			//clusterID_buf[i] = i; // !!!
        }
        #ifdef PROFILE
        uint32_t transID_e = us_ticker_read();
        transID_d += (float)(transID_e-transID_s)/1000;
        #endif

#ifdef DEBUG_ON
		printf("clustering done!\n");
#endif

        #ifdef PROFILE
        uint32_t centroid_s = us_ticker_read();
        #endif 

        /* count first met vector as centroid */
        uint16_t* cid = clusterID_buf;
        q15_t* cnt = cnt_buf;
        q7_t* in2 = cen_buf;
        q15_t num = 0;

        q15_t stack_pointer = 0;
		q7_t *centroid_pointer = cen_buf;

        for (int i = 0; i < dim_im_out * dim_im_out; i ++) {

            /* initial sub_im2col */
            q7_t  *vector_pointer = sub_im2col_buf + (i * LSH_L);
            uint16_t cluster_id = clusterID_buf[i];

            /* reverse index */
            uint16_t reverse_id = reverseID_buf[cluster_id];
            //printf("cluster_id = %d %d\n", cluster_id, reverse_id);

            /* two stacks */

            if (!reverse_id) {
                arm_add_q7(vector_pointer, centroid_pointer, centroid_pointer, LSH_L);
				cnt_buf[stack_pointer] = cluster_id;
                reverseID_buf[cluster_id] = stack_pointer;

				stack_pointer ++;
				centroid_pointer += LSH_L;
            } 
            else arm_add_q7(vector_pointer, centroid_pointer, centroid_pointer, LSH_L);
            //arm_add_q7(vector_pointer, centroid_pointer, centroid_pointer, LSH_L);

        }


        num = stack_pointer;

// centroid computing
        int count_div = 0;
        for (int i = 0; i < num; i ++) {
            for (int j = 0; j < LSH_L; j ++) {
                *centroid_pointer = (*centroid_pointer) / 3;
                count_div ++;
            }
        }
        

        //printf("num = %d\n", count_div);
        /*
        for (int i = 0; i < 32*32; i++){ // cid contains the cluster id of the i-th row
            // printf("%d\n", *cid); 
            q7_t* in1 = sub_im2col_buf;
            q15_t* rev = reverseID_buf;
            in1 += (*cid)*LSH_L;  // in2 stores the centroid vector
            rev += (*cid);        // rev stores the index of the centroid vector whose id = cluster_id[i];
            if (*rev == 0){
                arm_add_q7(in1, in2, in2, LSH_L);
                *cnt = *cid;
                *rev = (*cnt)+1; // /////
                cnt+=1;
                num+=1;
                in2+=1;
            }
            cid++;
        }
        */

       

#ifdef DEBUG_ON
		printf("stack pointer = %d num = %d\n", stack_pointer, num);
		printf("centroid done\n");
#endif
		/*
		break;
	}

	for (int n = 0; n < 5; n ++)

	{
	*/
        #ifdef PROFILE
        uint32_t centroid_e = us_ticker_read();
        centroid_d += (float)(centroid_e-centroid_s)/1000;

        uint32_t tempRs_s = us_ticker_read();
        #endif 
        // centorid matrix multply sub weight matrix

        arm_matrix_instance_q7 wt_p;
        arm_mat_init_q7(&wt_p, LSH_L, ch_im_out, pwt+(ch_im_out*n*LSH_L));
        arm_matrix_instance_q7 cen_mat;
        arm_mat_init_q7(&cen_mat, num, LSH_L, (q7_t *)cen_buf);
        arm_matrix_instance_q7 tempRes_mat;
        arm_mat_init_q7(&tempRes_mat, num, ch_im_out, (q7_t *)tempRes_buf);

        arm_mat_mult_q7(&cen_mat, &wt_p, &tempRes_mat, &status);

        #ifdef PROFILE
        uint32_t tempRs_e = us_ticker_read();
        tempRs_d += (float)(tempRs_e-tempRs_s)/1000;
        #endif

		//printf("centroid multiplication done\n");

		/**
		 * recovery stage.
		 */
        #ifdef PROFILE
        uint32_t recover_s = us_ticker_read();
        #endif
        cid = clusterID_buf;
        q7_t* pOut = Im_out;
		/*
        for (int i = 0; i < 32*32; i++){
            q7_t* tempRes_p = tempRes_buf;
            q15_t* rev = reverseID_buf;


            rev += *cid;
            uint16_t idx = (*rev)-1;
            tempRes_p += idx*32;
            arm_add_q7(tempRes_p, pOut, pOut, 32);
            cid++;pOut+=32;
        }
		*/
		for (int i = 0; i < dim_im_out * dim_im_out; i ++) {
			q7_t *tmp_result = tempRes_buf;
		 	uint16_t cluster_id = clusterID_buf[i];
			uint16_t reverse_id = reverseID_buf[cluster_id];

#ifdef DEBUG_ON
			if (i == 0) {
				puts("pOut [1] = \n");
				for (int j = 0; j < ch_im_out; j ++)
					printf("%d ", pOut[j]);
				puts("");
			}
#endif

			tmp_result += reverse_id * ch_im_out;
			arm_add_q7(tmp_result, pOut, pOut, ch_im_out);

#ifdef DEBUG_ON
			if (i == 0) {
				puts("pOut [1] = \n");
				for (int j = 0; j < ch_im_out; j ++)
					printf("%d ", pOut[j]);
				puts("");
			}
#endif

			pOut += ch_im_out;
		}

#ifdef DEBUG_ON
		printf("recovery done!\n");
#endif

        // printf("%d\n", t);
        #ifdef PROFILE
        uint32_t recover_e = us_ticker_read();
        recover_d += (float)(recover_e-recover_s)/1000;


        uint32_t sub_e = us_ticker_read();
        sub_d += (float)(sub_e-sub_s)/1000;
        #endif
        // arm_nn_mat_mult_kernel_q7_q15(wt, featuremap_buf, ch_im_out, LSH_L, bias_shift, out_shift, bias, Im_out);
    }

	/*
	puts("Out matrix ------ \n");
	for (int i = 0; i < 32; i ++)
		printf("%d ", Im_out[i]); puts("");
		*/
    uint32_t all_e = us_ticker_read();
    all_d = (float)(all_e-all_s)/1000;

    //#undef PROFILE
    #ifdef PROFILE
	printf("im2col = %d ms\n", (int)((float)(im2col_e - im2col_s) / 1000));
    
    printf("All %.2f ms; Each sub: %.2f ms; slice: %.2f ms; projection: %.2f ms; trans ID: %.2f ms; build centroid: %.2f ms; centroid mult sub wt: %.2f ms; recovery: %.2f ms; \r\n", 
            all_d,
            sub_d/n_sub,
            slice_d/n_sub,
            proj_d/n_sub,
            transID_d/n_sub,
            centroid_d/n_sub,
            tempRs_d/n_sub,
            recover_d/n_sub
    );
    /*
   printf("All %d ms; Each sub: %d ms; slice: %d ms; projection: %d ms; trans ID: %d ms; build centroid: %d ms; centroid mult sub wt: %d ms; recovery: %d ms; \r\n", 
            (int)all_d,
            (int)sub_d/n_sub,
            (int)slice_d/n_sub,
            (int)proj_d/n_sub,
            (int)transID_d/n_sub,
            (int)centroid_d/n_sub,
            (int)tempRs_d/n_sub,
            (int)recover_d/n_sub
    );
    */
    #else
        //printf("All %.2f ms\n", all_d);
    #endif
        
    #ifdef PRINT_TIME
        printf("All %.2f ms\n", all_d);
    #endif

    return (ARM_MATH_SUCCESS);
    // for (int n = 0; n < n_sub; n++){
    //     // slice im2col into (32x32)x(LSH_L)
    //     q15_t* idx = featuremap_buffer;
    //     for (int i = 0; i < 32*32; i++){
    //         for (int j = 0; j < LSH_L; j++){
    //             arm_memcpy_q7((q7_t*)idx, (q7_t*)&(im2col[(3*5*5)*i+n*LSH_L+j]), 2); 
    //             // printf("%d ", *idx);
    //             idx++;
    //         }
    //         // printf("\n");
    //     }
    //     // slice wt into (LSH_L)x(32x32)
    //     // idx = 0;
    //     // for (int i = 0; i < LSH_L; i++){
    //     //     for (int j = 0; j < CONV1_OUT_CH; j++){
    //     //         wt_buffer[idx++] = wt[(n*LSH_L+i)*CONV1_OUT_CH+j];
    //     //     }
    //     // }
        
    //     pOut = arm_nn_mat_mult_kernel_q7_q15(wt, featuremap_buffer, ch_im_out, LSH_L, bias_shift, out_shift, bias, pOut);

    //     printf("\n");
    // }
    // pOut = arm_nn_mat_mult_kernel_q7_q15(
    //     wt, im2col, ch_im_out, 3*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
    // gettimeofday(&end, NULL);
    // printf("%0.8f\n", (end.tv_sec - start.tv_sec) + 1e-6*(end.tv_usec - start.tv_usec));
    return (ARM_MATH_SUCCESS);
}

// std::vector<at::Tensor> cluster_LSH(torch::Tensor conv_in, torch::Tensor w) {
//     auto L = w.size(0);
//     auto H = w.size(1);
//     auto projections = torch::mm(conv_in, w);
// //    std::cout << "projections.size(): " << projections.size(0) << " " << projections.size(1) << std::endl;

//     std::unordered_map<std::string, std::vector<int>> hash_table;
//     for (int i=0; i<conv_in.size(0); i++) {
//         // compute label
//         std::string ind = "";
//         for(int j=0; j<H; j++) {
//             char c;
//             if (projections[i][j].item<float>() > 0) {
//                 c = '1';
//             } else {
//                 c = '0';
//             }
//             ind += c;
//         }
//         //        std::stoi

//         hash_table[ind].push_back(i);
//     }

//     int num_cluster = hash_table.size();
//     int ind_c = 0;

// //    std::cout << "hash_table.size(): " << hash_table.size() << std::endl;

//     auto label = torch::zeros(torch::IntArrayRef({conv_in.size(0)}));
//     auto centroid = torch::zeros(torch::IntArrayRef({num_cluster, L}));

//     std::unordered_map<std::string, std::vector<int>>::iterator k;
//     for (k = hash_table.begin(); k != hash_table.end(); k++) {
//         std::vector<int> &cluster = k->second;
//         for(int v : cluster) {
//             centroid[ind_c] += conv_in[v];
//             label[v] = ind_c;
//         }
//         int size = static_cast<int>(cluster.size());
//         centroid[ind_c] = torch::mul(centroid[ind_c], (1/size));
//         ind_c ++;
//     }

// //    std::cout << "centroid.size(): " << centroid.size(0) << " " << centroid.size(1) << std::endl;
// //    std::cout << "label.size(): " << label.size(0) << std::endl;

//     std::vector<at::Tensor> result;
//     result.push_back(centroid);
//     result.push_back(label);
//     return result;
// }


// at::Tensor Conv2d_dr_forward(
//     torch::Tensor input,
//     torch::Tensor weight,
//     torch::Tensor bias,
//     torch::Tensor w){

//     int L = w.size(0);

//     int n_sub = input.size(2) / L;
//     int total_num_cluster = 0;

//     auto conv_out = torch::zeros(torch::IntArrayRef({input.size(0), input.size(1), weight.size(1)}));

//     for (int _B = 0; _B < input.size(0); _B++) {
//         for (int i = 0; i < n_sub; i++){
//             auto result = cluster_LSH(input[_B].slice(1, i*L, (i+1)*L, 1), w);
//             auto centroid = result[0];
//             auto label = result[1];

//             int num_cluster = centroid.size(0);
//             total_num_cluster += num_cluster;

// //            std::cout << "centroid get: " << centroid.size(0) << " " << centroid.size(1) << std::endl;
// //            std::cout << "label get: " << label.size(0) << std::endl;

//             auto centroid_out = torch::mm(centroid, weight.slice(0, i*L, (i+1)*L, 1));

// //            std::cout<< "centroid_out.size(): " << centroid_out.size(0) << " " << centroid_out.size(1) << std::endl;

//             auto conv_out_temp = torch::zeros(torch::IntArrayRef({input.size(1), weight.size(1)}));
//             for (int l = 0; l < label.size(0); l++) {
//                 conv_out_temp[l] += centroid_out[label[l].item<int>()];
//             }
//             conv_out[_B] += conv_out_temp;

// //            std::cout<< "conv_out.size(): " << conv_out.size(0) " " << conv_out.size(1) << std::endl;
//         }
//     }
//     conv_out.add_(bias);
//     float avg_num_cluster = total_num_cluster / (n_sub * input.size(0));
//     float rc = avg_num_cluster / input.size(1);

//     std::cout << "avg_num_cluster: " << avg_num_cluster << std::endl;
//     std::cout << "rc: " << rc << std::endl;

//     return conv_out;
// }


/**
 * @} end of NNConv group
 */
