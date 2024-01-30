#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <sys/time.h>

double timestamp(){
    timeval t;
    gettimeofday( &t, NULL);
    return t.tv_sec+1e-6*t.tv_usec;
}


void print_tensor_float2D(torch::Tensor input) {
    int64_t height = input.size(0);
    int64_t width = input.size(1);
    auto input_acc = input.accessor<float, 2>();
    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            printf("%f ", input_acc[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


void print_tensor_float1D(torch::Tensor input) {
    int64_t len = input.size(0);
    auto input_acc = input.accessor<float, 1>();
    for (int i = 0; i < len; i ++) {
        printf("%f ", input_acc[i]);
    }
    printf("\n\n");
}


void print_tensor_int1D(torch::Tensor input) {
    int64_t len = input.size(0);
    auto input_acc = input.accessor<int, 1>();
    for (int i = 0; i < len; i ++) {
        printf("%d ", input_acc[i]);
    }
    printf("\n\n");
}


void print_tensor_float3D(torch::Tensor input) {
    int64_t n = input.size(0);
    int64_t height = input.size(1);
    int64_t width = input.size(2);
    auto input_acc = input.accessor<float, 3>();
    for (int k = 0; k < n; k ++) {
        for (int i = 0; i < height; i ++) {
            for (int j = 0; j < width; j ++) {
                printf("%f ", input_acc[k][i][j]);
            }
            printf("\n");
        }
        printf("-----------------------------\n");
    }
    printf("\n");
}


void print_tensor_int2D(torch::Tensor input) {
    int64_t height = input.size(0);
    int64_t width = input.size(1);
    auto input_acc = input.accessor<int, 2>();
    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            printf("%d ", input_acc[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


void print_tensor_float4D(torch::Tensor input) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);
    auto input_acc = input.accessor<float, 4>();
    for (int n = 0; n < N; n ++) {
        for (int k = 0; k < C; k ++) {
            for (int i = 0; i < height; i ++) {
                for (int j = 0; j < width; j ++) {
                    printf("%f ", input_acc[n][k][i][j]);
                }
                printf("\n");
            }
            printf("-----------------------------\n");
        }
        printf("=========================================\n");
    }
    printf("\n");
}