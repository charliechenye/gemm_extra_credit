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
			C[i * m + j] += alpha * inner_product;
		}
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

task void blocking_task(
	uniform const int block_size, uniform const int l_count, uniform const int j_count,
	uniform const int m, uniform const int n, uniform const int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) 
{
	uniform const int l_offset = taskIndex / j_count * block_size;
	uniform const int j_offset = taskIndex % j_count * block_size;
	uniform const int j_max = (j_offset + block_size < k ? j_offset + block_size : k);
	uniform const int l_max = (l_offset + block_size < n ? l_offset + block_size : n);
	print("Taks ID: %, l range: % to %, j range: % to %\n", taskIndex, l_offset, l_max, j_offset, j_max);
	
	foreach (i = 0 ... m) {
		for (int j = j_offset; j < j_max; j ++) {
			double inner_product = 0.0;
			for (int l = l_offset; l < l_max; l ++) {
			    inner_product += A[i * m + l] * B[l * n + j];
			}
			print("Updated %", i * m + j);
			C[i * m + j] += alpha * inner_product;
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) 
{
    // YOUR IMPLEMENTATION HERE
    // A: m x n takes m x b_size
    // B: n x k takes b_size x b_size
    // C: m x k yields m x b_size

	launch[m] initialization_task(m, k, C, beta);

	uniform const int b_size = 512;
	uniform const int l_count = (n + b_size - 1) / b_size;
	uniform const int j_count = (k + b_size - 1) / b_size;


	// each thread takes a block
	launch[l_count * j_count] blocking_task(b_size, l_count, j_count, m, n, k, A, B, C, alpha, beta);
}