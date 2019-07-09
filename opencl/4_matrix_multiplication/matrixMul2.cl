//
// A[M x P]  B[P x N]  C[M x N]
//   i   k     k   j     i   j
__kernel void matrixMul2(
        const int M,
        const int P,
        const int N,
        __global const float *A,
        __global const float *B,
        __global float *C)
{
    int k, j;
    int i = get_global_id(0);
    float ALoc[1000]; // same as A
    float tmp;

    if (i >= M)
        return;

    for (k = 0; k < P; ++k)
        ALoc[k] = A[i * P + k];

    for (j = 0; j < N; ++j) {
        tmp = 0.0f;
        for (k = 0; k < P; ++k) {
            tmp +=  ALoc[k] * B[k * N + j];
        }
        C[i * N + j] = tmp;
    }
}
