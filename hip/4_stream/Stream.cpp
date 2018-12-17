#include <iostream>
#include <iomanip> // for cout format output
#include <string>

// hip header after installing ROCm
#include "hip/hip_runtime.h"

#define WIDTH   16
#define NUM (WIDTH * WIDTH)

#define THREAD_PER_BLOCK_X  4
#define THREAD_PER_BLOCK_Y  4

#define NUM_STREAM  2

/*
 * Stream launches kernels to the device like a command queue.
 *
 * Create streams:
 *
 *   hipStream_t streams[NUM_STREAM];
 *   for (int i = 0; i < NUM_STREAM; i++)
 *       hipStreamCreate(&streams[i]);
 *
 * Set the stream when kernel launch, 0 is default stream.
 *
 *   hipLaunchKernel(MatrixAddStaticShared,
 *                   dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
 *                   dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
 *                   0;// size of dynamic shared memory
 *                   streams[0], // stream
 *                   B_d,
 *                   A_d,
 *                   WIDTH);
 *
 *   hipLaunchKernel(MatrixAddDynamicShared,
 *                   dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
 *                   dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
 *                   sizeof(float) * WIDTH * WIDTH, // size of dynamic shared memory
 *                   streams[1], // stream
 *                   B_d,
 *                   A_d,
 *                   WIDTH);
 *
 * Copy memory via specific stream:
 *
 *      hipMemcpyAsync(A_d[i], A_h, NUM * sizeof(float), hipMemcpyHostToDevice,
 *                     streams[i]);
 */

__global__ void MatrixAddStaticShared(hipLaunchParm lp, float *out, float *in, unsigned int width)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    __shared__ float sharedMem[WIDTH * WIDTH];

    sharedMem[y * WIDTH + x] = in[y * WIDTH + x];

    __syncthreads();

    out[y * WIDTH + x] = sharedMem[y * WIDTH + x] + 10;
}

__global__ void MatrixAddDynamicShared(hipLaunchParm lp, float *out, float *in, unsigned int width)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    HIP_DYNAMIC_SHARED(float, sharedMem);

    sharedMem[y * WIDTH + x] = in[y * WIDTH + x];

    __syncthreads();

    out[y * WIDTH + x] = sharedMem[y * WIDTH + x] + 10;
}

void MatrixAddByCPU(float *out, float *in, unsigned int width)
{
    for (int i = 0; i < width * width; i++)
        out[i] = in[i] + 10;
}

void MultipleStream(float *A_h, float **A_d, float **B_h, float **B_d, int width)
{
    hipStream_t streams[NUM_STREAM];

    for (int i = 0; i < NUM_STREAM; i++)
        hipStreamCreate(&streams[i]);

    for (int i = 0; i < NUM_STREAM; i++) {
        hipMalloc((void **)&A_d[i], NUM * sizeof(float));
        hipMalloc((void **)&B_d[i], NUM * sizeof(float));
        hipMemcpyAsync(A_d[i], A_h, NUM * sizeof(float), hipMemcpyHostToDevice,
                       streams[i]);
    }

    hipLaunchKernel(MatrixAddStaticShared,
                    dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
                    dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
                    0, // size of dynamic shared memory
                    0, // stream
                    B_d[0],
                    A_d[0],
                    WIDTH);
    hipLaunchKernel(MatrixAddDynamicShared,
                    dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
                    dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
                    sizeof(float) * WIDTH * WIDTH, // size of dynamic shared memory
                    0, // stream
                    B_d[1],
                    A_d[1],
                    WIDTH);

    for (int i = 0; i < NUM_STREAM; i++)
        hipMemcpyAsync(B_h[i], B_d[i], NUM * sizeof(float), hipMemcpyHostToDevice,
                       streams[i]);
}

int main()
{
    float *A_h, *A_d[2], *A_r;
    float *B_h[2], *B_d[2];
    int i;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name: " << devProp.name << std::endl;

    A_h    = (float*)malloc(NUM * sizeof(float));
    B_h[0] = (float*)malloc(NUM * sizeof(float));
    B_h[1] = (float*)malloc(NUM * sizeof(float));
    A_r    = (float*)malloc(NUM * sizeof(float));
    for (i = 0; i < NUM; i++) {
        A_h[i] = (float)i + 1.0;
        B_h[0][i] = 0;
        B_h[1][i] = 0;
    }

    MultipleStream(A_h, A_d, B_h, B_d, WIDTH);

    // verify the results
    int errors = 0;
    MatrixAddByCPU(A_r, A_h, WIDTH);
    for (int s = 0; s < NUM_STREAM; s++) {
        for (i = 0; i < NUM; i++) {
            if (A_r[i] != B_h[s][i]) {
                //printf("stream %i: %f != %f !\n", s, B_h[s][i], A_r[i]);
                std::cout << "stream " << s << ": " << B_h[s][i] << " != " << A_r[i] << '\n';
                errors++;
            }
        }
    }

    if (errors != 0)
        std::cout << "FAILED: " << errors << "errors\n";
    else
        std::cout << "PASSED\n";

    // free the resources
    for (int i = 0; i < 2; i++) {
        hipFree(A_d[i]);
        hipFree(B_d[i]);
        free(B_h[i]);
    }

    free(A_h);
    free(A_r);

    hipDeviceReset(); // reset the device state, deleting streams, memory, kernel, events
    return errors;
}
