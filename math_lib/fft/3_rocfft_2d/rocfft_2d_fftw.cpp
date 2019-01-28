#include <vector>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <math.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include "hipfft.h"

int main()
{
    int ver;
    hipfftGetVersion(&ver);
    std::cout << "hipFFT version:" << ver << "\n";

    const size_t N = 16;
    const size_t M = 16;
    const size_t Total = N * M;
    std::vector<float2> inComplexHost(Total);

    for (size_t i = 0; i < Total; i++) {
        inComplexHost[i].x = i + (i % 3) - (i % 7);
        inComplexHost[i].y = 0;
    }

    const size_t SizeOfByte = Total * sizeof(float2);

    // Compute in CPU
    // ========================================================================
    fftwf_complex *in, *out;
    fftwf_plan p;

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Total);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Total);
    for (size_t i = 0; i < Total; i++) {
        in[i][0] = inComplexHost[i].x;
        in[i][1] = inComplexHost[i].y;
    }

    p = fftwf_plan_dft_2d(N, M, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // Compute in GPU
    // ========================================================================
    // Allocate memory in device
    hipfftComplex *inComplexDev;
    hipMalloc((void**)&inComplexDev, SizeOfByte);

    // Copy data to device
    hipMemcpy(inComplexDev, &inComplexHost[0], SizeOfByte, hipMemcpyHostToDevice);

    // Create 2D plan
    hipfftHandle plan = NULL;
    hipfftCreate(&plan);
    hipfftPlan2d(&plan, N, M, HIPFFT_C2C);

    // Execute the plan, generate data in place
    hipfftExecC2C(plan, inComplexDev, inComplexDev, HIPFFT_FORWARD);
    hipDeviceSynchronize();

    // Copy the result back to the host
    std::vector<float2> outComplexHost(Total);
    hipMemcpy(&outComplexHost[0], inComplexDev, SizeOfByte, hipMemcpyDeviceToHost);

    // Calculate error
    // ========================================================================
    float err_max = 0;
    std::cout << "Error of each element:\n";
    for (size_t i = 0; i < Total; i++) {
        float real = fabs(outComplexHost[i].x - out[i][0]);
        float imagine = fabs(outComplexHost[i].y - out[i][1]);
        float err = real + imagine;

        if (err > err_max)
            err_max = err;

        std::cout << "[" << std::setw(4)    << i        << "] " <<
            "Real: "    << std::setw(12)    << real     << ", " <<
            "Imagine: " << std::setw(12)    << imagine  << ", " <<
            "Error: "   << std::setw(12)    << err      <<  "\n";

    }
    std::cout << "Max error is " << err_max << "\n";

    // Free resources
    hipFree(inComplexDev);
    hipfftDestroy(plan);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    return 0;
}
