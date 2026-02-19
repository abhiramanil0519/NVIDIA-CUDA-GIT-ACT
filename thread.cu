#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define EPOCHS 50
#define LR 0.5

__device__ float sigmoid(float x)
{
    return 1.0f/(1.0f+expf(-x));
}

__device__ float dsigmoid(float y)
{
    return y*(1-y);
}

__global__ void forward(float *w1, float *w2, float *input, float *hidden, float *output)
{
    int i = threadIdx.x;

    if(i < 2)
    {
        hidden[i] = sigmoid(input[0]*w1[i*2] + input[1]*w1[i*2+1]);
        printf("[FORWARD] Hidden[%d] = %f\n", i, hidden[i]);
    }

    __syncthreads();

    if(i==0)
    {
        *output = sigmoid(hidden[0]*w2[0] + hidden[1]*w2[1]);
        printf("[FORWARD] Output = %f\n", *output);
    }
}

__global__ void train(float *w1, float *w2, float *input, float target)
{
    float hidden[2];
    float out;

    // forward
    hidden[0] = sigmoid(input[0]*w1[0] + input[1]*w1[1]);
    hidden[1] = sigmoid(input[0]*w1[2] + input[1]*w1[3]);
    out = sigmoid(hidden[0]*w2[0] + hidden[1]*w2[1]);

    float error = target - out;

    printf("[TRAIN] target=%f output=%f error=%f\n", target, out, error);

    // backprop output
    float d_out = error * dsigmoid(out);

    // update output weights
    w2[0] += LR * d_out * hidden[0];
    w2[1] += LR * d_out * hidden[1];

    // backprop hidden
    float d_h0 = d_out * w2[0] * dsigmoid(hidden[0]);
    float d_h1 = d_out * w2[1] * dsigmoid(hidden[1]);

    // update hidden weights
    w1[0] += LR * d_h0 * input[0];
    w1[1] += LR * d_h0 * input[1];
    w1[2] += LR * d_h1 * input[0];
    w1[3] += LR * d_h1 * input[1];
}

int main()
{
    std::cout << "===== CUDA NEURAL NETWORK TRAINER =====\n";
    std::cout << "Model: 2-2-1 XOR Network\n\n";

    float h_w1[4] = {0.5,-0.3,0.8,0.2};
    float h_w2[2] = {0.1,-0.4};

    float *d_w1,*d_w2,*d_in;
    cudaMalloc(&d_w1,4*sizeof(float));
    cudaMalloc(&d_w2,2*sizeof(float));
    cudaMalloc(&d_in,2*sizeof(float));

    cudaMemcpy(d_w1,h_w1,4*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2,h_w2,2*sizeof(float),cudaMemcpyHostToDevice);

    float dataset[4][3] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,0}
    };

    for(int e=0;e<EPOCHS;e++)
    {
        std::cout << "\n===== EPOCH " << e << " =====\n";

        for(int i=0;i<4;i++)
        {
            cudaMemcpy(d_in,dataset[i],2*sizeof(float),cudaMemcpyHostToDevice);

            train<<<1,1>>>(d_w1,d_w2,d_in,dataset[i][2]);
            cudaDeviceSynchronize();
        }
    }

    std::cout << "\nTraining complete.\n";
}
