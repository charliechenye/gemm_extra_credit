task inline void blocking_task(
	uniform const int block_size, uniform const int l_count, uniform const int j_count,
	uniform const int m, uniform const int n, uniform const int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta, uniform int launched_task_count, uniform int total_task_count) 
{
	for (uniform int current_task_index = taskIndex; current_task_index < total_task_count; current_task_index+= launched_task_count){
		uniform const int l_offset = current_task_index / (j_count * m) * block_size;
		uniform const int j_offset = (current_task_index % (j_count * m)) / m * block_size;
		uniform const int i = current_task_index % m;
		uniform const int j_max = (j_offset + block_size < k ? j_offset + block_size : k);
		uniform const int l_max = (l_offset + block_size < n ? l_offset + block_size : n);
		// print("Taks ID: %, m: %, l range: % to %, j range: % to %\n", taskIndex, i, l_offset, l_max, j_offset, j_max);

		foreach (j = j_offset ... j_max) {
			double inner_product = 0.0;
			for (uniform int l = l_offset; l < l_max; l ++) {
				inner_product += A[i * m + l] * B[l * n + j];
			}
			// print("Updated %", i * m + j);
			/* given that need to loop through all i, j_offset before move to next block
			 * unlikely to encounter collision. 
			 * atomic add as preventive measures in case number of threads significantly increases
			 */
			atomic_add_local(&C[i * m + j], alpha * inner_product);
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

	// initialize C
	foreach (i = 0 ... m, l = 0 ... k){
		C[i * m + l] *= beta;
	}

	uniform int b_size = 512; 

	uniform const int l_count = (n + b_size - 1) / b_size;
	uniform const int j_count = (k + b_size - 1) / b_size;
	uniform const int total_task_count = l_count * j_count * m;
	uniform const int launched_task_count = min(100000, total_task_count);

	// each thread takes a block
	launch[launched_task_count] blocking_task(b_size, l_count, j_count, m, n, k, A, B, C, alpha, beta, launched_task_count, total_task_count);
}