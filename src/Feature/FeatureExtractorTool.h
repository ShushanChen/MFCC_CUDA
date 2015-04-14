#pragma once
#ifndef __SpeechRecognitionSystem__FeatureExtractionTool__
#define __SpeechRecognitionSystem__FeatureExtractionTool__

#include <iostream>
#include <complex>
#include "mathtool.h"

#define d_type double
#define BLOCK_SIZE 32

#define K_UNROLL_STEP 32 // [1,2,4,8,16]

#if K_UNROLL_STEP > BLOCK_SIZE
#undef K_UNROLL_STEP
#define K_UNROLL_STEP BLOCK_SIZE
#endif

#define COL_STEP 4

#define ty (threadIdx.y)
#define tx (threadIdx.x)

#define by (blockIdx.y)
#define bx (blockIdx.x)

#define dy (blockDim.y)
#define dx (blockDim.x)

//#define e 2.718281828459
//typedef std::complex<double> cp;
//#ifndef __SpeechRecongnitionSystem__PI__
//#define __SpeechRecongnitionSystem__PI__
//const double PI = std::acos(-1);
//#endif

__global__ 
void matrix_mul_kernel(d_type *sq_matrix_1, d_type *sq_matrix_2, d_type *sq_matrix_result, int dimension);
    
__global__
void windowFFT_cu(cp *d_SpeechSignal, int frameNum, int frameSize, int f, int selIdx, double arg=PI);

__global__ 
void fft_cu_part(cp *d_SpeechSignal, int n, int f, double arg=PI);

__device__ 
void mulComplex(cp *output, cp *input1, cp *input2);

__device__ 
void addComplex(cp *output, cp *input1, cp *input2);

__device__
void getRealImag(double& real, double& imag, const cp *input);

__device__
void getPolarValue(double length, double angle, double* output);

#endif
