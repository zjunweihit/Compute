#include <iostream>
#include <iomanip> // for cout format output

// hip header after installing ROCm
#include "hip/hip_runtime.h"

// A[M x P] * B[P x N] =  C[M x N]
//   i   k      k   j       i   j

#define M           2//1000
#define P           3//1000
#define N           2//1000
#define SZ_A        (M * P)
#define SZ_B        (P * N)
#define SZ_C        (M * N)

float A_h[ M * P ];
float B_h[ P * N ];
float C_h[ M * N ] = { 0.0f };

#define THREAD_PER_BLOCK_X  4
#define THREAD_PER_BLOCK_Y  4

#define _DEBUG

#ifdef _DEBUG
void PrintMatrix(char *name, float *mat, int row, int col)
{
    printf("%s:\n", name);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%10f ", mat[i * col + j]);
        }
        printf("\n");
    }
}
#endif

void InitData(void)
{
    int i, k, j;
    float tmp;

    for (i = 0; i < M; ++i)
        for (k = 0; k < P; ++k)
            A_h[i * P + k] = (float)1;

    for (k = 0; k < P; ++k)
        for (j = 0; j < N; ++j)
            B_h[k * N + j] = (float)1;

    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            tmp = 0.0f;
            for (k = 0; k < P; ++k) {
                tmp += A_h[i * P + k] * B_h[k * N + j];
            }
            C_h[i * N + j] = tmp;
        }
    }

#ifdef _DEBUG
    PrintMatrix((char*)"A_h", A_h, M, P);
    PrintMatrix((char*)"B_h", B_h, P, N);
    PrintMatrix((char*)"C_h", C_h, M, N);
#endif
}

__global__ void matrixMul(
        const int m,
        const int p,
        const int n,
        const float *A,
        const float *B,
        float *C)
{
    int k;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    float tmp = 0.0f;

    if (i >= m || j >= n)
        return;

    for (k = 0; k < p; ++k)
        tmp += A[i * p + k] * B[k * n + j];
    C[i * n + j] = tmp;
}

int VerifyResult(float *mem)
{
    int errors = 0, i;
    float *result = (float*)malloc(SZ_C * sizeof(float));

    hipMemcpy(result, (const void*)mem, SZ_C * sizeof(float), hipMemcpyDeviceToHost);
#ifdef _DEBUG
    PrintMatrix((char*)"result", result, M, N);
#endif

    for (i = 0; i < SZ_C; i++) {
        if (C_h[i] != result[i]) {
            std::cout << C_h[i] << " != " << result[i] << '\n';
            errors++;
        }
    }

    if (errors != 0)
        std::cout << "FAILED: " << errors << "errors\n";
    else
        std::cout << "PASSED\n";

    free(result);

    return errors;
}

int main()
{
    float *A_d, *B_d, *C_d;
    int i, ret = 0;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name: " << devProp.name << std::endl;

    InitData();

    hipMalloc((void**)&A_d, SZ_A * sizeof(float));
    hipMalloc((void**)&B_d, SZ_B * sizeof(float));
    hipMalloc((void**)&C_d, SZ_C * sizeof(float));

    hipMemcpy(A_d, (const void*)A_h, SZ_A * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_d, (const void*)B_h, SZ_B * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(matrixMul,
                    dim3(M, N),
                    dim3(1,1),//NULL,//dim3(1, 1),
                    0, // dynamic shared
                    0, // stream
                    M, P, N,
                    A_d, B_d, C_d);

    ret = VerifyResult(C_d);

    // free the resources
    hipFree(A_d);
    hipFree(B_d);
    hipFree(C_d);

    return ret;
}
