task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int i_block_start, uniform int i_block_end,
	 uniform int j_block_start, uniform int j_block_end) {
	// YOUR IMPLEMENTATION HERE

	// i_interleaving rows are divied into 2 groups
	uniform int i_thread_start = (taskIndex < 4 ? i_block_start : (i_block_start + i_block_end) /2);
	uniform int i_thread_end = (taskIndex < 4 ? (i_block_start + i_block_end) /2 : i_block_end);
	// 1 x block_size block is splited into 4 parts executed by gang of 4 threads
	uniform int j_thread_start = j_block_start + (j_block_end - j_block_start) / 4 * (taskIndex % 4);
	uniform int j_thread_end = j_block_start + (j_block_end - j_block_start) / 4;

	// SIMD with 4 ALUS
	for (uniform int i = i_thread_start; i < i_thread_end; i ++)
		foreach (j = j_thread_start ... j_thread_end) {
			double inner_prod = 0;
			for (uniform int kk = 0; kk < k; kk ++) {
				inner_prod += A[i * k + kk] * B[kk * n + j];
			}
			C[i * n + j] = alpha * inner_prod + beta * C[i * n + j];	
		}
}

task void task_initialize_C(uniform int m, uniform int n, uniform int k,
	uniform double C[], uniform double beta, uniform int i_start) 
{
	uniform int i = i_start + taskIndex;
	foreach (kk = 0 ... k) 
		C[i * m + kk] = beta * C[i * m + kk];
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	// A: m x n takes 1 x b_size
    // B: n x k takes b_size x b_size
    // C: m x k yields 1 x b_size

	
    // assumes:
    // - n and k are multiples of b_size 
    // - b_size * sizeof(int) is multiple of CACHE_LINE_SIZE
	uniform const int num_threads = 8;

	for (uniform int i_start = 0; i_start < m; i_start += num_threads)
		launch[num_threads] task_initialize_C(m, n, k, C, beta, i_start)
	uniform int b_size = 64;
	uniform int i_interleaving = 64;
	for (uniform int i_offset = 0; i_offset < m; i_offset += i_interleaving) {
		// handles i_interleaving rows at a time
		// each row is broken down into blocks of 1 x b_size

		for(uniform int j_offset = 0; j_offset < k; j_offset += b_size) {
			launch[num_threads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, 
												i_offset, i_offset + i_interleaving, j_offset, j_offset + b_size);
		}
		
	}
}
