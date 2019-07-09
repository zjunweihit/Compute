//
// A[M x P]  B[P x N]  C[M x N]
//   i   k     k   j     i   j
__kernel void matrixMul(
        const int M,
        const int P,
        const int N,
        __global const float *A,
        __global const float *B,
        __global float *C)
{
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0.0f;

    if (i >= M || j >= N)
        return;

    for (k = 0; k < P; ++k) {
        tmp += A[i * P + k] * B[k * N + j];
    }
    C[i * N + j] = tmp;
}
