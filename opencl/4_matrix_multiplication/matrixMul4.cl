//
// A[M x P]  B[P x N]  C[M x N]
//   i   k     k   j     i   j
__kernel void matrixMul4(
        const int M,
        const int P,
        const int N,
        __global const float *A,
        __global const float *B,
        __global float *C,
        __local  float *BLoc)
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
        for (k = 0; k < P; ++k)
            BLoc[k] = B[k * N + j];
        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = 0.0f;
        for (k = 0; k < P; ++k)
            tmp +=  ALoc[k] * BLoc[k];
        C[i * N + j] = tmp;
    }
}
