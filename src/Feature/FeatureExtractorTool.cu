
#include "FeatureExtractorTool.h"

__global__ 
void matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int dimension) 
{
    /** 
     *  A cuda thread will calculate 4 results, result[row][col[0~4]]
     */
    int row = by * dy + ty;
    int col[COL_STEP]; 

    col[0] = COL_STEP * bx * dx + tx;
    col[1] = col[0] + BLOCK_SIZE;
    col[2] = col[1] + BLOCK_SIZE;
    col[3] = col[2] + BLOCK_SIZE;

    /**
     *  One shared copy of sq_matrix_1 can be used to calculate 4 blocks,
     *  increase the utilization of share memory
     */
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE][COL_STEP];

    float val[COL_STEP] = {0.0, 0.0, 0.0, 0.0};

    int rowIdx = row * dimension;

    /**
     * pre calculate index of matrixes. 
     */
    int sq_matrix_1_index = rowIdx + tx;
    int sq_matrix_2_index = (ty) * dimension; 
    int sq_matrix_1_step = BLOCK_SIZE;
    int sq_matrix_2_step = BLOCK_SIZE * dimension;

    int ks, k;

    /**
     * pre fetch values into local memory. 
     */
    float preFetchA;
    float * sb_sq_matrix_p = &(s_b[ty][tx][0]);

    int K_STEP = COL_STEP * BLOCK_SIZE;
    int ceil_ks;

    float *s_b_sq_matrix_used;

    /**
     * ks is BLOCK_SIZE step length, and one iteration will calculate 4 block
     */
    for(ks = 0; ks <= dimension-BLOCK_SIZE; ks += BLOCK_SIZE, sq_matrix_1_index += sq_matrix_1_step, sq_matrix_2_index += sq_matrix_2_step) {
        /**
         * fetch matrix1 and matrix2 into shared memory
         */
        s_a[ty][tx] = sq_matrix_1[sq_matrix_1_index];

        sb_sq_matrix_p[0] = sq_matrix_2[sq_matrix_2_index + col[0]];
        sb_sq_matrix_p[1] = sq_matrix_2[sq_matrix_2_index + col[1]];
        sb_sq_matrix_p[2] = sq_matrix_2[sq_matrix_2_index + col[2]];
        sb_sq_matrix_p[3] = sq_matrix_2[sq_matrix_2_index + col[3]];

        __syncthreads();

        s_b_sq_matrix_used = &(s_b[0][tx][0]);

        for(k = 0; k < BLOCK_SIZE; k++) {
            preFetchA = s_a[ty][k]; 
            val[0] += preFetchA * s_b_sq_matrix_used[0];  
            val[1] += preFetchA * s_b_sq_matrix_used[1]; 
            val[2] += preFetchA * s_b_sq_matrix_used[2];
            val[3] += preFetchA * s_b_sq_matrix_used[3];
            s_b_sq_matrix_used += K_STEP; 
        }

        __syncthreads();
    }

    /**
     * because the dimension is not always power of 2, we need to add a tail for the rest calculation
     */
    if(ks < dimension) {
        if(col[0] < dimension && row < dimension)
            s_a[ty][tx] = sq_matrix_1[sq_matrix_1_index];

        if(col[0] < dimension && row < dimension) 
            sb_sq_matrix_p[0] = sq_matrix_2[sq_matrix_2_index + col[0]];
        if(col[1] < dimension && row < dimension) 
            sb_sq_matrix_p[1] = sq_matrix_2[sq_matrix_2_index + col[1]];
        if(col[2] < dimension && row < dimension) 
            sb_sq_matrix_p[2] = sq_matrix_2[sq_matrix_2_index + col[2]];
        if(col[3] < dimension && row < dimension) 
            sb_sq_matrix_p[3] = sq_matrix_2[sq_matrix_2_index + col[3]];

        __syncthreads();
        s_b_sq_matrix_used = &(s_b[0][tx][0]) - K_STEP;

        ceil_ks = dimension-ks;

#pragma unroll 32
        for(k=0; k < ceil_ks; k++) {
            preFetchA = s_a[ty][k]; 
            s_b_sq_matrix_used += K_STEP; 
            val[0] += preFetchA * s_b_sq_matrix_used[0];  
            val[1] += preFetchA * s_b_sq_matrix_used[1]; 
            val[2] += preFetchA * s_b_sq_matrix_used[2];
            val[3] += preFetchA * s_b_sq_matrix_used[3];
        }
        __syncthreads();
    }

    /**
     * Write the results back to global memory
     */
    if(row >= dimension) return;

    if(col[0] < dimension)
        sq_matrix_result[rowIdx + col[0]] = val[0];
    if(col[1] < dimension)
        sq_matrix_result[rowIdx + col[1]] = val[1];
    if(col[2] < dimension)
        sq_matrix_result[rowIdx + col[2]] = val[2];
    if(col[3] < dimension)
        sq_matrix_result[rowIdx + col[3]] = val[3];
}

__global__
void windowFFT_cu(cp *d_SpeechSignal, int frameNum, int frameSize, int f, int selIdx, double arg){
    extern __shared__ char s_SpeechSignal[];
    int p, i, j, rollIdx=0, oldRollIdx;
    size_t innerIdx = threadIdx.x % frameSize, 
           frame_offset = blockDim.x*blockIdx.x+(threadIdx.x/frameSize)*frameSize;
    double temp_cp[2], temp_wm[2], temp_w[2];
    cp *temp = (cp *) temp_cp, 
       *wm = (cp*)temp_wm, 
       *w = (cp*)temp_w; 
    //cp *d_signal[2];
    cp *s_signal[2]; 

    size_t sharedSize = blockDim.x * sizeof(cp);
    s_signal[0] = (cp *)s_SpeechSignal;
    s_signal[1] = (cp *)&s_SpeechSignal[sharedSize];
    //d_signal[0] = d_SpeechSignal+frame_offset;
    //d_signal[1] = d_signal[0]+frameNum*frameSize;

    *(s_signal[0]+innerIdx) = *(d_SpeechSignal+frame_offset+innerIdx);
    __syncthreads();

    for(int k = frameSize>>1; k; k>>=1, arg*=0.5){
        rollIdx ^= 1;
        oldRollIdx = rollIdx^1;

        getPolarValue(1, f*arg, temp_wm);
        *temp_w = 1;
        *(temp_w+1) = 0;

        i = innerIdx/k;
        j = innerIdx%k;
        for(int t=0; t<i; t++){
            //w = w*wm;
            mulComplex(w,wm,w);
        }
        i = i*k;
        p = i<<1;
        if(p>=frameSize) p-=frameSize;

        //mulComplex(temp, w, d_signal[oldRollIdx]+(p+k+j)); 
        //addComplex(d_signal[rollIdx]+(i+j), temp, d_signal[oldRollIdx]+(p+j));

        mulComplex(temp, w, s_signal[oldRollIdx]+(p+k+j)); 
        addComplex(s_signal[rollIdx]+(i+j), temp, s_signal[oldRollIdx]+(p+j));
        __syncthreads();
    }
    d_SpeechSignal[frame_offset+innerIdx] = *(s_signal[selIdx]+innerIdx);
}

__global__ 
void fft_cu_part(cp *d_SpeechSignal, int n, int f, double arg){
    int p, i, j, idx, rollIdx=0, oldRollIdx;
    cp* d_signal[2]; 
    d_signal[0] = d_SpeechSignal;
    d_signal[1] = &d_SpeechSignal[n];
    
    int *finalRollIdx = (int *) &d_SpeechSignal[2*n];
    
    idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    //double arg = pi;
    double temp_cp[2], temp_wm[2], temp_w[2];
    cp *temp = (cp *) temp_cp, *wm = (cp*)temp_wm, *w = (cp*)temp_w;
    for(int k = n>>1; k; k>>=1, arg*=0.5){
        rollIdx ^= 1;
        oldRollIdx = rollIdx^1;
        
        //cp wm = std::polar(1.0,f*arg), w(1,0);
        getPolarValue(1, f*arg, temp_wm);
        *temp_w = 1;
        *(temp_w+1) = 0;
        
        i = idx/k;
        j = idx%k;
        for(int t=0; t<i; t++){
            //w = w*wm;
            mulComplex(w,wm,w);
        }
        i = i*k;
        p = i<<1;
        if(p>=n) p-=n;
    
        //d_signal[rollIdx][i+j] = d_signal[oldRollIdx][p+j] + w*d_signal[oldRollIdx][p+k+j];
        mulComplex(temp, w, &d_signal[oldRollIdx][p+k+j]); 
        addComplex(&d_signal[rollIdx][i+j], temp, &d_signal[oldRollIdx][p+j]);
        __syncthreads();
    }
    if(idx==0)
        *finalRollIdx = rollIdx;
}

__device__ 
void mulComplex(cp *output, cp *input1, cp *input2){
    double real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    double *ptr_output = (double *)output;
    *ptr_output = real1*real2-imag1*imag2;
    *(ptr_output+1) = real1*imag2+imag1*real2;
    //output = cp( real1*real2-imag1*imag2 , real1*imag2+imag1*real2 );
}

__device__
void addComplex(cp *output, cp *input1, cp *input2){
    double real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    double *ptr_output = (double *)output;
    *ptr_output = real1+real2;
    *(ptr_output+1) = imag1+imag2;
    //output = cp( real1+real2, imag1+imag2 );
}

__device__
void getRealImag(double& real, double& imag, const cp *input){
    double *comp = (double *)input;
    real = *comp;
    imag = *(comp+1);
}

__device__
void getPolarValue(double rho, double theta, double *output){
    *output = rho*cos(theta);
    *(output+1) = rho*sin(theta);
}

