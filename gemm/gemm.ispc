task void gemm_ispc_task(
	uniform const int num_threads_i, uniform const int num_threads_j, 
	uniform const int m, uniform const int n, uniform const int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta, 
	uniform const int i_block_start, uniform const int i_block_end, 
	uniform const int j_block_start, uniform const int j_block_end) 
{
	// YOUR IMPLEMENTATION HERE

	// 1 block is splited into 8 parts executed by 8 threads (2 sets of 4 threads)
	uniform int i_thread_start = (taskIndex < 4 ? i_block_start : (i_block_start + i_block_end) / 2);
	uniform int i_thread_end = (taskIndex < 4 ? (i_block_start + i_block_end) / 2 : i_block_end);
	uniform int j_thread_start = j_block_start + (j_block_end - j_block_start) / 4 * (taskIndex % 4); // starting col of each part
	uniform int j_thread_end = (taskIndex % 4 == 3 ? j_block_end : j_thread_start + (j_block_end - j_block_start) / 4); // inal col of each part

	// SIMD with 4 ALUS
	foreach (i = i_thread_start ... i_thread_end) {
		for (uniform int j = j_thread_start; j < j_thread_end; j ++) {
			double inner_prod = 0.0;
			for (uniform int kk = 0; kk < k; kk ++) {
				inner_prod += A[i * m + kk] * B[kk * n + j];
			}
			C[i * m + j] = alpha * inner_prod + beta * C[i * m + j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) 
{
	// YOUR IMPLEMENTATION HERE
	uniform const int num_threads_i = 4;
	uniform const int num_threads_j = 2;
	uniform const int b_size = 64;
	uniform const int i_interleaving = 64;
	for (uniform int i_block_start = 0; i_block_start < m; i_block_start += i_interleaving) {
		// Save matrix in L3 cache
		uniform int i_block_end = (i_block_start + i_interleaving < m ? i_block_start + i_interleaving : m); // final element in row of each block
		for (uniform int i = i_block_start; i < i_block_end; i ++)
			foreach (kk = 0 ... k)
				prefetch_l3(&A[i * m + kk]);
				
		for (uniform int j_block_start = 0; j_block_start < n; j_block_start += b_size) {
			uniform int j_block_end = (j_block_start + b_size < n? j_block_start + b_size : n);
			for (uniform int kk = 0; kk < k; kk ++)
				foreach(j = j_block_start ... j_block_end)
					prefetch_l3(&B[kk * n + j]);

			for (uniform int i = i_block_start; i < i_block_end; i ++){ 
				foreach(j = j_block_start ... j_block_end){ 
					prefetch_l3(&C[i * m + j]);
				}
			}

			launch[num_threads_i * num_threads_j] gemm_ispc_task(num_threads_i, num_threads_j, 
																m, n, k, A, B, C, alpha, beta, 
																i_block_start, i_block_end, j_block_start, j_block_end);
		}
	}
}
