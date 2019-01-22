#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include "hipfft.h"

#define N	16
#define M	16
#define TOTAL	(N * M)

int main()
{
    int ver;
    int i, j;
    int size = TOTAL * sizeof(hipfftComplex);
    hipfftComplex inComplexHost[TOTAL];
    hipfftComplex outComplexHost[TOTAL];
    hipfftHandle plan = NULL;

    hipfftGetVersion(&ver);
    printf("hipFFT version: %d\n", ver);

    for (size_t i = 0; i < TOTAL; i++) {
        inComplexHost[i].x = i;
        inComplexHost[i].y = 0;
    }

    // Allocate memory in device
    hipfftComplex *inComplexDev;
    hipMalloc((void**)&inComplexDev, size);

    // Copy data to device
    hipMemcpy(inComplexDev, &inComplexHost[0], size, hipMemcpyHostToDevice);

    // Create 2D plan
    hipfftCreate(&plan);
    hipfftPlan2d(&plan, N, M, HIPFFT_C2C);

    // Execute the plan, generate data in place
    hipfftExecC2C(plan, inComplexDev, inComplexDev, HIPFFT_FORWARD);
    hipDeviceSynchronize();

    // Copy the result back to the host
    hipMemcpy(&outComplexHost[0], inComplexDev, size, hipMemcpyDeviceToHost);

    printf("Output:\n");
    for (i = 0; i < N; i++) {
        printf("Line %d:\n", i);
        for (size_t j = 0; j < M; j++) {
            printf("%f, %fi\n", outComplexHost[i * M + j].x,
                         outComplexHost[i * M + j].y);
        }
        printf("\n");
    }

    // Free resources in device
    hipFree(inComplexDev);
    hipfftDestroy(plan);

    return 0;
}
