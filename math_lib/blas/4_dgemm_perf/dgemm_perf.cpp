#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <iostream>
#include "hipblas.h"
#include <unistd.h>
#include <ctime>
#include <ctime>

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

enum Trans_t {
    NN = 0,
    NT,
    TN,
    TT
};

template <typename T>
void mat_mult_cpu(T alpha, T beta, int M, int N, int K, T* A, int As1, int As2,
                           T* B, int Bs1, int Bs2, T* C, int Cs1, int Cs2)
{
    for(int i1=0; i1<M; i1++) {
        for(int i2=0; i2<N; i2++) {
            T t = 0.0;
            for(int i3=0; i3<K; i3++) {
                t +=  A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
#ifdef DEBUG_INFO
                printf("A[%d] * B[%d] = %d * %d\n",
                        i1 * As1 + i3 * As2, i3 * Bs1 + i2 * Bs2,
                        (int)A[i1 * As1 + i3 * As2], (int)B[i3 * Bs1 + i2 * Bs2]);
#endif
            }
#ifdef DEBUG_INFO
            printf("\n");
#endif
            C[i1*Cs1 +i2*Cs2] = beta * C[i1*Cs1+i2*Cs2] + alpha * t ;
        }
    }
}

// global setting
static int g_M, g_N, g_K;
static bool g_print = false;
static bool g_true_rand = false;
static bool g_compare_cpu = false;
static bool g_default = false;
static Trans_t g_trans_type = NN;

static void set_def_opt()
{
    g_M = 2;
    g_N = 2;
    g_K = 3;
    g_print = true;
}

static void show_help(void)
{
    cout << "  - p: to print the matrix data" << endl;
    cout << "  - s: set the matrix size m=n=k=<input size>" << endl;
    cout << "  - t: set the matrix transform type" << endl;
    cout << "       0: NN (default)" << endl;
    cout << "       1: NT" << endl;
    cout << "       2: TN" << endl;
    cout << "       3: TT" << endl;
    cout << "  - r: initialize TRUE random data" << endl;
    cout << "  - c: compare data with cpu results" << endl;
    cout << "  - h: this help info" << endl;
}

static int parse_opt(int argc, char **argv)
{
    int opt;
    const char *optstr = "ps:t:rch";
    size_t size = 0;

    while (( opt = getopt(argc, argv, optstr)) != -1) {
        switch (opt) {
        case 'p':
            g_print = true;
            break;
        case 's':
            size = atoi(optarg);
            g_M = size;
            g_N = size;
            g_K = size;
            break;
        case 't':
            g_trans_type = (Trans_t)atoi(optarg);
            break;
        case 'r':
            g_true_rand = true;
            break;
        case 'c':
            g_compare_cpu = true;
            break;
        case 'h':
        default:
            show_help();
            return -1;
        }
    }

    if (g_M == 0 || g_N == 0 || g_K == 0) {
        set_def_opt();
        g_default = true;
    }

    return 0;
}

void print_matrix(const char *matrix_name, double *matrix, int row, int col)
{
    if (!g_print)
        return;

    string seprator(80, '=');
    cout << seprator << endl;
    cout << matrix_name << ":\n";
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << matrix[i * col + j] << " ";
        cout << "\n";
    }
}

int main(int argc, char **argv)
{
    hipblasOperation_t transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N;
    int ret = 0;
    if (parse_opt(argc, argv))
        exit(-1);

    if (g_trans_type == NT) {
        transa = HIPBLAS_OP_N;
        transb = HIPBLAS_OP_T;
    } else if (g_trans_type == TN) {
        transa = HIPBLAS_OP_T;
        transb = HIPBLAS_OP_N;
    } else if (g_trans_type == TT) {
        transa = HIPBLAS_OP_T;
        transb = HIPBLAS_OP_T;
    }
    double alpha = 1.0, beta = 1.0;
    int m = g_M, n = g_N, k = g_K;
    int lda, ldb, ldc;
    int na, nb, nc; // the number of element about A, B, C
    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;

    cout << "dgemm perf test" << endl;
    if (transa == HIPBLAS_OP_N) {
        lda = m;
        na = k * lda;
        a_stride_1 = 1; a_stride_2 = lda;
        cout << "N";
    } else { // HIPBLAS_OP_T
        lda = k;
        na = m * lda;
        a_stride_1 = lda; a_stride_2 = 1;
        cout << "T";
    }

    if (transb == HIPBLAS_OP_N) {
        ldb = k;
        nb = n * ldb;
        b_stride_1 = 1; b_stride_2 = ldb;
        cout << "N: ";
    } else { // HIPBLAS_OP_T
        ldb = n;
        nb = k * ldb;
        b_stride_1 = ldb; b_stride_2 = 1;
        cout << "T: ";
    }

    ldc = m;
    nc = n * ldc;

    // allocate memory in host
    vector<double> ha(na);
    vector<double> hb(nb);
    vector<double> hc(nc);
    vector<double> hc_res(nc);

    srand(g_true_rand ? (unsigned)time(NULL) : 217);


    if (g_default) {
        for( int i = 0; i < na; ++i ) { ha[i] = i; }
        for( int i = 0; i < nb; ++i ) { hb[i] = 6 - i; }
    } else {
        for( int i = 0; i < na; ++i ) { ha[i] = rand(); }
        for( int i = 0; i < nb; ++i ) { hb[i] = rand(); }
    }
    for( int i = 0; i < nc; ++i ) { hc[i] = 0; }
    hc_res = hc;

    // allocate memory on device
    double *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, na * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&db, nb * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dc, nc * sizeof(double)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(double) * na, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(double) * nb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(double) * nc, hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    // A (m x k) * B (k x n) = C(m x n) in GPU
    CHECK_HIPBLAS_ERROR(hipblasDgemm(handle,
                transa, transb,
                m, n, k,
                &alpha,
                da, lda,
                db, ldb,
                &beta,
                dc, ldc));

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(double) * nc, hipMemcpyDeviceToHost));

    cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", "
         << lda << ", " << ldb <<  ", " << ldc << "\n";

    // print A: m * k
    print_matrix("A", ha.data(), m, k);
    // print B: k * n
    print_matrix("B", hb.data(), k, n);
    // print CT: n * m
    print_matrix("C", hc.data(), n, m);

    float t;
    hipEventElapsedTime(&t, start, stop);
    double tflops = (double(m)*n*k*2/1e9)/t;
    double gbps = (double(m*k+k*n+m*n)*sizeof(double)/1e6)/t;
    cout << t << " ms, "
         << tflops << " TFLOPS, "
         << gbps << " GB/s" << endl;

    if (g_compare_cpu) {
        // calculate result by CPU and compare result with GPU output
        mat_mult_cpu<double>(alpha, beta, m, n, k,
                ha.data(), a_stride_1, a_stride_2,
                hb.data(), b_stride_1, b_stride_2,
                hc_res.data(), 1, ldc);

        double max_relative_error = numeric_limits<double>::min();
        for (int i = 0; i < nc; i++) {
            double relative_error = (hc_res[i] - hc[i]) / hc_res[i];
            relative_error = relative_error > 0 ? relative_error : -relative_error;
            max_relative_error = relative_error < max_relative_error ? max_relative_error : relative_error;
        }

        double eps = numeric_limits<double>::epsilon();
        double tolerance = 100;
        if (max_relative_error != max_relative_error || max_relative_error > eps * tolerance) {
            cout << "FAIL: max_relative_error = " << max_relative_error << endl;
        } else {
            cout << "PASS: max_relative_error = " << max_relative_error << endl;
        }
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

    return EXIT_SUCCESS;
}
