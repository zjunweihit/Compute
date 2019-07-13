#include <iostream>
#include <iomanip> // for cout format output

// hip header after installing ROCm
#include "hip/hip_runtime.h"

#define NUM     16
#define THREAD_PER_BLOCK  4
#define _DEBUG

// it must be void
__global__ void VectorAdd(float *a, float *b, float *result)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    result[x] = a[x] + b[x];
}

void vectorAddByCPU(float *a, float *b, float *result)
{
    for (int i = 0; i < NUM; i++)
        result[i] = a[i] + b[i];
}

int main()
{
    float *A_h, *A_d;
    float *B_h, *B_d;
    float *C_h, *C_d, *C_r;
    int i;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name: " << devProp.name << std::endl;

    A_h = (float*)malloc(NUM * sizeof(float));
    B_h = (float*)malloc(NUM * sizeof(float));
    C_h = (float*)malloc(NUM * sizeof(float));
    C_r = (float*)malloc(NUM * sizeof(float)); // CPU result
    for (i = 0; i < NUM; i++) {
        A_h[i] = (float)i + 1.0;
        B_h[i] = (float)1.0;
    }

    hipMalloc((void**)&A_d, NUM * sizeof(float));
    hipMalloc((void**)&B_d, NUM * sizeof(float));
    hipMalloc((void**)&C_d, NUM * sizeof(float));

    hipMemcpy(A_d, (const void*)A_h, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_d, (const void*)B_h, NUM * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(VectorAdd,
                    dim3(NUM),
                    dim3(1),
                    0, // dynamic shared
                    0, // stream
                    A_d,
                    B_d,
                    C_d);

    hipMemcpy(C_h, (const void*)C_d, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // verify the results
    int errors = 0;
    vectorAddByCPU(A_h, B_h, C_r);
    for (i = 0; i < NUM; i++) {
        if (C_r[i] != C_h[i]) {
            //printf("result %f != %f !\n", B_h[i], A_r[i]);
            std::cout << C_h[i] << " != " << C_r[i] << '\n';
            errors++;
        }
    }

#ifdef _DEBUG
    printf("C_h:\n");
    for (i = 0; i < NUM; i++) {
        std::cout << std::setprecision(2) << std::setiosflags(std::ios::fixed)
                  << std::setw(6) << C_h[i] << ' ';
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
    hipFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_r);

    return errors;
}
