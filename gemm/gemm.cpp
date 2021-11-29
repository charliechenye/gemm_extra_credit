
// gemm -- general double precision dense matrix-matrix multiplication.
//
// implement: C = alpha * A x B + beta * C, for matrices A, B, C
// Matrix C is M x N  (M rows, N columns)
// Matrix A is M x K
// Matrix B is K x N
//
// Your implementation should make no assumptions about the values contained in any input parameters.

void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha, double beta){

    // blocking implementation with block b_size x b_size
    int b_size = 64;
    // assumes:
    // - n and k are multiples of b_size 
    // - b_size * sizeof(int) is multiple of CACHE_LINE_SIZE

    // A: m x n takes 1 x b_size
    // B: n x k takes b_size x b_size
    // C: m x k yields 1 x b_size
    double sum;
    int i, j, l, l_offset, j_offset;

    for (i = 0; i < m; i ++)
        for (j = 0; j < k; j ++)
            C[i * m + j] = beta * C[i * m + j];


    for (l_offset = 0; l_offset < n; l_offset += b_size)
        for (j_offset = 0; j_offset < k; j_offset += b_size)
            for (i = 0; i < m; i ++)
                for (j = j_offset; j < j_offset + b_size; j ++) {
                    sum = C[i * m + j];
                    for (l = l_offset; l < l_offset + b_size; l ++)
                        sum += A[i * m + l] * B[n * l + j] * alpha;
                    C[i * m + j] = sum;
                }   
}

