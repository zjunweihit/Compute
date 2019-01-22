#include <stdio.h>
#include <fftw3.h>
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
    float err_max = 0;
    float real, imagine, err;
    hipfftComplex inComplexHost[TOTAL];
    hipfftComplex outComplexHost[TOTAL];
    hipfftHandle plan = NULL;

    hipfftGetVersion(&ver);
    printf("hipFFT version: %d\n", ver);

    for (i = 0; i < TOTAL; i++) {
        inComplexHost[i].x = i + (i % 3) - (i % 7);
        inComplexHost[i].y = 0;
    }

    // Compute in CPU
    // ========================================================================
    fftwf_complex *in, *out;
    fftwf_plan p;

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * TOTAL);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * TOTAL);
    for (i = 0; i < TOTAL; i++) {
        in[i][0] = inComplexHost[i].x;
        in[i][1] = inComplexHost[i].y;
    }

    p = fftwf_plan_dft_2d(N, M, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // Compute in GPU
    // ========================================================================

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

    // Calculate error
    printf("Error of each element:\n");
    for (i = 0; i < TOTAL; i++) {
        real = fabs(outComplexHost[i].x - out[i][0]);
        imagine = fabs(outComplexHost[i].y - out[i][1]);
        err = real + imagine;

        if (err > err_max)
            err_max = err;

        printf("[%4d] Real: %12f, Imagine: %12f, Error: %12f\n",
                i, real, imagine, err);
    }
    printf("Max error is %f\n", err_max);

    // Free resources
    hipFree(inComplexDev);
    hipfftDestroy(plan);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return 0;
}
