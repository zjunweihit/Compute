#include <iostream>
#include <iomanip> // for cout format output
#include <string>

// hip header after installing ROCm
#include "hip/hip_runtime.h"

#define WIDTH   16
#define NUM (WIDTH * WIDTH)

#define THREAD_PER_BLOCK_X  4
#define THREAD_PER_BLOCK_Y  4

/*
 * Shared memory is accessed more faster than that of global and constant emmory.
 * It could be accessed by all threads in a block.
 *
 * Declare dynamic shared memory in kernel:
 *
 *   HIP_DYNAMIC_SHARED(float, sharedMem);
 *
 * Set size of dynamic shared memory to the kernel function by hipLaunchKernel()
 *
 *   hipLaunchKernel(matrixAdd,
 *                   dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
 *                   dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
 *                   sizeof(float) * WIDTH * WIDTH, // size of dynamic shared memory
 *                   0, // stream
 *                   B_d,
 *                   A_d,
 *                   WIDTH);
 * NOTE:
 *   As every thread in the block can access the shared memory, we have to sync
 *   the data in shared memory before access for each thread.
 *   __syncthreads();
 *
 *   In this simple example, there is no much improvement.
 */

// it must be void
__global__ void matrixAdd(hipLaunchParm lp, float *out, float *in, unsigned int width)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    HIP_DYNAMIC_SHARED(float, sharedMem);

    sharedMem[y * WIDTH + x] = in[y * WIDTH + x];

    __syncthreads();

    out[y * WIDTH + x] = sharedMem[y * WIDTH + x] + 10;
}

void matrixAddByCPU(float *out, float *in, unsigned int width)
{
    for (int i = 0; i < width * width; i++)
        out[i] = in[i] + 10;
}

int main()
{
    float *A_h, *A_d, *A_r;
    float *B_h, *B_d;
    int i;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name: " << devProp.name << std::endl;

    A_h = (float*)malloc(NUM * sizeof(float));
    B_h = (float*)malloc(NUM * sizeof(float));
    A_r = (float*)malloc(NUM * sizeof(float));
    for (i = 0; i < NUM; i++) {
        A_h[i] = (float)i + 1.0;
        B_h[i] = 0;
    }

    hipMalloc((void**)&A_d, NUM * sizeof(float));
    hipMalloc((void**)&B_d, NUM * sizeof(float));

    hipMemcpy(A_d, (const void*)A_h, NUM * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernel(matrixAdd,
                    dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
                    dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
                    sizeof(float) * WIDTH * WIDTH, // size of dynamic shared memory
                    0, // stream
                    B_d,
                    A_d,
                    WIDTH);

    hipMemcpy(B_h, (const void*)B_d, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // verify the results
    int errors = 0;
    matrixAddByCPU(A_r, A_h, WIDTH);
    for (i = 0; i < NUM; i++) {
        if (A_r[i] != B_h[i]) {
            //printf("result %f != %f !\n", B_h[i], A_r[i]);
            std::cout << B_h[i] << " != " << A_r[i] << '\n';
            errors++;
        }
    }

#ifdef _DEBUG
    printf("B_h:\n");
    for (i = 0; i < NUM; i++) {
        if ((i % WIDTH == 0))
            printf("\n");
        std::cout << std::setprecision(2) << std::setiosflags(std::ios::fixed)
                  << std::setfill('0') << std::setw(6) << B_h[i] << ' ';
        //printf("%06.02f ", B_h[i]); // same as above
    }
    std::cout << std::endl;
#endif

    if (errors != 0)
        std::cout << "FAILED: " << errors << "errors\n";
    else
        std::cout << "PASSED\n";

    // free the resources
    hipFree(A_d);
    hipFree(B_d);

    free(A_h);
    free(B_h);
    free(A_r);

    return errors;
}
