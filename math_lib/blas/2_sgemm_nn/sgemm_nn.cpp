#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <iostream>
#include "hipblas.h"

using namespace std;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error) \
if (error != hipSuccess) { \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
}
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error) \
if (error != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "rocBLAS error: at %s(%d): ", __FILE__, __LINE__); \
    if(error == HIPBLAS_STATUS_NOT_INITIALIZED)fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED"); \
    if(error == HIPBLAS_STATUS_ALLOC_FAILED)fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED"); \
    if(error == HIPBLAS_STATUS_INVALID_VALUE)fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE"); \
    if(error == HIPBLAS_STATUS_MAPPING_ERROR)fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR"); \
    if(error == HIPBLAS_STATUS_EXECUTION_FAILED)fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
    if(error == HIPBLAS_STATUS_INTERNAL_ERROR)fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR"); \
    if(error == HIPBLAS_STATUS_NOT_SUPPORTED)fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED"); \
    fprintf(stderr, "\n"); \
    exit(EXIT_FAILURE); \
}
#endif

void print_matrix(const char *matrix_name, float *matrix, int row, int col)
{
    cout << matrix_name << ":\n";
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << matrix[i * col + j] << " ";
        cout << "\n";
    }
}

int main()
{
    float alpha = 1.0, beta = 0.0;
    int m = 2, n = 3, k = 3;
    int na, nb, nc; // the number of element about A, B, C

    cout << "sgemm example" << endl;

    na = m * k;
    nb = k * n;
    nc = m * n;

    // allocate memory in host
    vector<float> ha(na);
    vector<float> hb(nb);
    vector<float> hc(nc);
    vector<float> hc_res(nc);
    for( int i = 0; i < na; ++i ) { ha[i] = i; }
    for( int i = 0; i < nb; ++i ) { hb[i] = 6 - i; }
    for( int i = 0; i < nc; ++i ) { hc[i] = 0; }
    hc_res = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, na * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, nb * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, nc * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * na, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * nb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * nc, hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    // A (m x k) * B (k x n) = C(m x n)
    // BT(n x k) * AT(k x m) = CT(n x m) in GPU
    CHECK_HIPBLAS_ERROR(hipblasSgemm(handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                n, m, k,
                &alpha,
                db, n,
                da, k,
                &beta,
                dc, n));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * nc, hipMemcpyDeviceToHost));

    // print A: m * k
    print_matrix("A", ha.data(), m, k);
    // print B: k * n
    print_matrix("B", hb.data(), k, n);
    // print C: m * n
    print_matrix("C", hc.data(), m, n);

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

    return EXIT_SUCCESS;
}
