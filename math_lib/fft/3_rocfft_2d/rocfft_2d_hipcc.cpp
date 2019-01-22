#include <vector>
#include <iostream>
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
        inComplexHost[i].x = i;
        inComplexHost[i].y = 0;
    }

    const size_t SizeOfByte = Total * sizeof(float2);

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

    std::cout << "Output:\n";
    for (size_t i = 0; i < N; i++) {
        std::cout << "Line " << i << ":\n";
        for (size_t j = 0; j < M; j++) {
            std::cout << outComplexHost[i * M + j].x << ", " <<
                         outComplexHost[i * M + j].y << "i\n";
        }
        std::cout << "\n";
    }

    // Free resources in device
    hipFree(inComplexDev);
    hipfftDestroy(plan);

    return 0;
}
