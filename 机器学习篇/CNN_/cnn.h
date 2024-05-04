#ifndef CNN_H
#define CNN_H

#include <stdint.h>

typedef struct 
{
    /* data */
    int8_t in_channels, out_channels;
    float *weight;
    float *bias;
    int8_t kernel_size;
    int8_t stride;
    int8_t padding;
}cnn_kernel_t;

typedef struct 
{
    int pool_size;
    int stride;
}maxpooling_t;


float activate_relu(float x);
void activate_relu_array(float* array, int size);
void convolve_2d(float *input_data, const cnn_kernel_t *conv_params, float *output_data, int input_height, int input_width, const char in_place_flag);
void maxpooling2d(  float *input_data, 
                    const int input_height, 
                    const int input_width, 
                    const int input_channels,
                    const maxpooling_t *paras, float *output_data);
// 矩阵-向量乘法
void matrix_vector_multiply(float* matrix, 
                            float* vector,
                            float* result, 
                            int rows, int cols);

// 向量加法
void vector_add(float* vector1, float* vector2, int length);

void full_connect_layer(float* input_data,
                        float* weight, 
                        float* bias,
                        int rows, int cols,
                        float* output_data);

void softmax(float *input, int length);

#endif 