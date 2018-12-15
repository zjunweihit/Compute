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
 * // [1] Create start, stop events
 * hipEvent_t start, stop;
 *
 * hipEventCreate(&start);
 * hipEventCreate(&stop);
 *
 * // [2] Start the timer in a specific steam(NULL: default stream)
 * hipEventRecord(start, NULL);
 *
 * // LaunchKernel or memory operations here...
 *
 * hipEventRecord(stop, NULL);
 *
 * // [3] Synchronize the event, blocking until the event is ready.
 * // Note:
 * //   wait for all previous works in the speified stream at the point of
 * //   hipEventRecord()
 * hipEventSynchronize(stop);
 *
 * // [4] Calculate the time between start and stop events.
 * float eventMs = 1.0f;
 * hipEventElapsedTime(&eventMs, start, stop);
 *
 * Profile HIP API
 * $ HIP_TRACE_API=1 HIP_DB=0x2 ./Event
 * NOTE:
 *   1) need to install rocm-profiler, cxlactivitylogger ??
 *   2) /opt/rocm/bin/rocm-profiler -A -o MT.atp -e HIP_PROFILE_API=1 ./Event ??
 *   3) /opt/rocm/hip/bin/hipdemangleatp MT.atp ??
 */
// it must be void
__global__ void matrixAdd(hipLaunchParm lp, float *out, float *in, unsigned int width)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    out[y * WIDTH + x] = in[y * WIDTH + x] + 10;
}

void matrixAddByCPU(float *out, float *in, unsigned int width)
{
    for (int i = 0; i < width * width; i++)
        out[i] = in[i] + 10;
}

static void printTime(const char *str, float time)
{
    std::cout << std::setprecision(3) << std::setiosflags(std::ios::fixed)
              << std::setfill('0') << std::setw(6)
              << str << " = " << time << "ms\n";
}

int main()
{
    float *A_h, *A_d, *A_r;
    float *B_h, *B_d;
    int i;

    hipEvent_t start, stop;
    float eventMs = 1.0f;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name: " << devProp.name << std::endl;

    hipEventCreate(&start);
    hipEventCreate(&stop);

    A_h = (float*)malloc(NUM * sizeof(float));
    B_h = (float*)malloc(NUM * sizeof(float));
    A_r = (float*)malloc(NUM * sizeof(float));
    for (i = 0; i < NUM; i++) {
        A_h[i] = (float)i + 1.0;
        B_h[i] = 0;
    }

    hipMalloc((void**)&A_d, NUM * sizeof(float));
    hipMalloc((void**)&B_d, NUM * sizeof(float));

    hipEventRecord(start, NULL); // 2.1
    hipMemcpy(A_d, (const void*)A_h, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipEventRecord(stop, NULL); // 2.2
    hipEventSynchronize(stop); // 3
    hipEventElapsedTime(&eventMs, start, stop); // 4
    printTime("hipMemcpyHostToDevice time", eventMs);

    hipEventRecord(start, NULL); // 2.1
    hipLaunchKernel(matrixAdd,
                    dim3(WIDTH / THREAD_PER_BLOCK_X, WIDTH / THREAD_PER_BLOCK_Y),
                    dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
                    0, // dynamic shared
                    0, // stream
                    B_d,
                    A_d,
                    WIDTH);
    hipEventRecord(stop, NULL); // 2.2
    hipEventSynchronize(stop); // 3
    hipEventElapsedTime(&eventMs, start, stop); // 4
    printTime("hipLaunchKernel time", eventMs);

    hipEventRecord(start, NULL); // 2.1
    hipMemcpy(B_h, (const void*)B_d, NUM * sizeof(float), hipMemcpyDeviceToHost);
    hipEventRecord(stop, NULL); // 2.1
    hipEventSynchronize(stop); // 3
    hipEventElapsedTime(&eventMs, start, stop); // 4
    printTime("hipMemcpyDeviceToHost time", eventMs);

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
