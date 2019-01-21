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

    const size_t Num = 16;
    std::vector<float2> inComplexHost(Num);

    for (size_t i = 0; i < Num; i++) {
        inComplexHost[i].x = i;
        inComplexHost[i].y = 0;
    }

    const size_t SizeOfByte = Num * sizeof(float2);

    // Allocate memory in device
    hipfftComplex *inComplexDev;
    hipMalloc((void**)&inComplexDev, SizeOfByte);

    // Copy data to device
    hipMemcpy(inComplexDev, &inComplexHost[0], SizeOfByte, hipMemcpyHostToDevice);

    // Create 1D plan
    hipfftHandle plan = NULL;
    hipfftCreate(&plan);
    hipfftPlan1d(&plan, Num, HIPFFT_C2C, 1);

    // Execute the plan, generate data in place
    hipfftExecC2C(plan, inComplexDev, inComplexDev, HIPFFT_FORWARD);
    hipDeviceSynchronize();

    // Copy the result back to the host
    std::vector<float2> outComplexHost(Num);
    hipMemcpy(&outComplexHost[0], inComplexDev, SizeOfByte, hipMemcpyDeviceToHost);

    std::cout << "Output:\n";
    for (size_t i = 0; i < Num; i++)
        std::cout << outComplexHost[i].x << ", " << outComplexHost[i].y << "i\n";

    // Free resources in device
    hipFree(inComplexDev);
    hipfftDestroy(plan);

    return 0;
}
