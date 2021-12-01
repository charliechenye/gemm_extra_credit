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
	foreach (i = i_thread_start ... i_thread_end) {
		for (int j = j_thread_start; j < j_thread_end; j ++) {
			double inner_product = 0.0;
			for (int kk = 0; kk < n; kk ++) {
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
	uniform const int num_threads_i = 16;
	uniform const int num_threads_j = 16;
	uniform const int i_interleaving = 512;
	uniform const int b_size = 512;
	for (uniform int i_block_start = 0; i_block_start < m; i_block_start += i_interleaving) {
		// Save matrix in L3 cache
		uniform int i_block_end = (i_block_start + i_interleaving < m ? i_block_start + i_interleaving : m); // final element in row of each block

		for (uniform int j_block_start = 0; j_block_start < n; j_block_start += b_size) {
			uniform int j_block_end = (j_block_start + b_size < n? j_block_start + b_size : n);
			
			launch[num_threads_i * num_threads_j] gemm_ispc_task(num_threads_i, num_threads_j,
																m, n, k, A, B, C, alpha, beta, 
																i_block_start, i_block_end, j_block_start, j_block_end);
		}
	}
}