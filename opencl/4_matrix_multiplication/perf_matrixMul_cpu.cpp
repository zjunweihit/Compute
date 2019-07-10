#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <time.h>

// A[M x P] * B[P x N] =  C[M x N]
//   i   k      k   j       i   j

#define M           1000
#define P           1000
#define N           1000
#define SZ_A        (M * P)
#define SZ_B        (P * N)
#define SZ_C        (M * N)

#define TEST_CNT    5

float A[ M * P ];
float B[ P * N ];
float C[ M * N ] = { 0.0f };

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

void MatrixMul_ijk(int m, int p, int n, float *A, float *B, float *C)
{
    int i, j, k;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < p; ++k) {
                C[i * n + j] += A[i * p + k] * B[k * p + j];
            }
        }
    }
}

void MatrixMul_ikj(int m, int p, int n, float *A, float *B, float *C)
{
    int i, j, k;
    for (i = 0; i < m; ++i) {
        for (k = 0; k < p; ++k) {
            for (j = 0; j < n; ++j) {
                C[i * n + j] += A[i * p + k] * B[k * p + j];
            }
        }
    }
}

void MatrixMul_kij(int m, int p, int n, float *A, float *B, float *C)
{
    int i, j, k;
    for (k = 0; k < p; ++k) {
        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                C[i * n + j] += A[i * p + k] * B[k * p + j];
            }
        }
    }
}

void MatrixMul_ijk_local(int m, int p, int n, float *A, float *B, float *C)
{
    int i, j, k;
    float tmp;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            tmp = 0.0f;
            for (k = 0; k < p; ++k) {
                tmp += A[i * p + k] * B[k * p + j];
            }
            C[i * n + j] = tmp;
        }
    }
}

void InitData(float *A, float *B)
{
    int i, k, j;
    for (i = 0; i < M; ++i)
        for (k = 0; k < P; ++k)
            A[i * P + k] = (float)1;

    for (k = 0; k < P; ++k)
        for (j = 0; j < N; ++j)
            B[k * N + j] = (float)1;
}

void show_result(char *name,
                 struct timespec *t1,
                 struct timespec *t2,
                 int test_cnt)
{
    double start = t1->tv_sec * 1e9 + t1->tv_nsec;
    double end = t2->tv_sec * 1e9 + t2->tv_nsec;
    double mflops = (M * P * N) / ((end - start) / 1000 / test_cnt);

    printf("%s: %10f MFLOPS\n", name, mflops);
}

void PerfMatrixMul(char *name,
        int test_cnt,
        void (*func)(int, int, int, float*, float*, float*),
        int m, int p, int n, float *A, float *B, float *C)
{
    struct timespec t1, t2;
    int i;

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    for (i = 0; i < test_cnt; ++i)
        (*func)(m, p, n, A, B, C);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
    show_result(name, &t1, &t2, test_cnt);
}

int main()
{
    InitData(A, B);
#ifdef _DEBUG
    PrintMatrix("A", A, M, P);
    PrintMatrix("B", B, P, N);
#endif
    PerfMatrixMul((char*)"[CPU] Matrix Multiplication ijk", TEST_CNT,
                  MatrixMul_ijk, M, P, N, A, B, C);
    PerfMatrixMul((char*)"[CPU] Matrix Multiplication ikj", TEST_CNT,
                  MatrixMul_ikj, M, P, N, A, B, C);
    PerfMatrixMul((char*)"[CPU] Matrix Multiplication kij", TEST_CNT,
                  MatrixMul_kij, M, P, N, A, B, C);
    PerfMatrixMul((char*)"[CPU] Matrix Multiplication ijk(local)", TEST_CNT,
                  MatrixMul_ijk_local, M, P, N, A, B, C);
#ifdef _DEBUG
    PrintMatrix("C", C, M, N);
#endif

    return 0;
}
