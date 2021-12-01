task void gemm_ispc_task(
	uniform const int num_threads_i, uniform const int num_threads_j, 
	uniform const int m, uniform const int n, uniform const int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta, 
	uniform const int i_block_start, uniform const int i_block_end, 
	uniform const int j_block_start, uniform const int j_block_end) 
{
	// YOUR IMPLEMENTATION HERE

	uniform const int thread_i = taskIndex / num_threads_j;
	uniform const int thread_j = taskIndex % num_threads_j;
	// 1 block is splited into num_threads_i x num_threads_j parts
	uniform int i_thread_start = i_block_start + (i_block_end - i_block_start) / num_threads_i * thread_i;
	uniform int i_thread_end = (thread_i < num_threads_i - 1 ? i_thread_start + (i_block_end - i_block_start) / num_threads_i : i_block_end);
	uniform int j_thread_start = j_block_start + (j_block_end - j_block_start) / num_threads_j * thread_j;
	uniform int j_thread_end = (thread_j < num_threads_j - 1 ? j_thread_start + (j_block_end - j_block_start) / num_threads_j : j_block_end);

	// SIMD with 4 ALUS
	for (uniform i = i_thread_start; i < i_thread_end; i ++) {
		foreach (j = j_thread_start ... j_thread_end) {
			double inner_product = 0.0;
			for (uniform int kk = 0; kk < n; kk ++) {
				inner_product += A[i * m + kk] * B[kk * n + j];
			}
			C[i * m + j] = alpha * inner_product + beta * C[i * m + j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) 
{
    // YOUR IMPLEMENTATION HERE
    // A: m x n takes 1 x b_size
    // B: n x k takes b_size x b_size
    // C: m x k yields 1 x b_size
	uniform const int b_size = 64;
	uniform const int num_threads_i = m / b_size;
	uniform const int num_threads_j = k / b_size;
	launch[num_threads_i * num_threads_j] gemm_ispc_task(num_threads_i, num_threads_j,
														m, n, k, A, B, C, alpha, beta, 
														0, m, 0, k);

}
