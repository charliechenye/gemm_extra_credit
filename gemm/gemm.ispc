task void gemm_ispc_task(
	uniform const int num_threads_per_row, uniform const int col_per_thread,
	uniform const int m, uniform const int n, uniform const int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha) 
{
	// YOUR IMPLEMENTATION HERE
	
    // C: m x k yields 1 x b_size
	// num_threads to process C[i_thread, :] 1 row in C
	uniform const int thread_j = taskIndex % num_threads_per_row;
	uniform int i = m / num_threads_per_row;
	uniform int j_thread_start = col_per_thread * (taskIndex % num_threads_per_row);
	uniform int j_thread_end = (thread_j < num_threads_per_row - 1 ? j_thread_start + col_per_thread : k);

	// SIMD with 4 ALUS
	foreach (j = j_thread_start ... j_thread_end) {
		double inner_product = 0.0;
		for (uniform int kk = 0; kk < n; kk ++) {
			inner_product += A[i * m + kk] * B[kk * n + j];
		}
		C[i * m + j] += alpha * inner_product;
	}
}

task void initialization_task(
	uniform const int m, uniform const int k,
	uniform double C[], uniform double beta)
{
	// 1 task = 1 row
	foreach (kk = 0 ... k) 
		C[taskIndex * m + kk] *= beta;
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) 
{
    // YOUR IMPLEMENTATION HERE
    // A: m x n takes 1 x b_size
    // B: n x k takes b_size x b_size
    // C: m x k yields 1 x b_size

	launch[m] initialization_task(m, k, C, beta);
	// 4 ALUS with 2 Hyper-threading
	uniform const int num_threads_per_row = 8;
	uniform const int b_size = (k + num_threads_per_row - 1) / num_threads_per_row;
	launch[m * num_threads_per_row] gemm_ispc_task(num_threads_per_row, b_size,
												m, n, k, A, B, C, alpha);

}
