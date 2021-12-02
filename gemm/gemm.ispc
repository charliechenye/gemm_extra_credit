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
	uniform const int l_offset = taskIndex / (j_count * m) * block_size;
	uniform const int j_offset = (taskIndex % (j_count * m)) / m * block_size;
	uniform const int i = taskIndex % m;
	uniform const int j_max = (j_offset + block_size < k ? j_offset + block_size : k);
	uniform const int l_max = (l_offset + block_size < n ? l_offset + block_size : n);
	// print("Taks ID: %, m: %, l range: % to %, j range: % to %\n", taskIndex, i, l_offset, l_max, j_offset, j_max);
	

	foreach (j = j_offset ... j_max) {
		double inner_product = 0.0;
		for (uniform int l = l_offset; l < l_max; l ++) {
			inner_product += A[i * m + l] * B[l * n + j];
		}
		// print("Updated %", i * m + j);
		C[i * m + j] += alpha * inner_product;
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

	launch[m] initialization_task(m, k, C, beta);

	uniform int b_size = 512;

	// reduce task count
	while ((n + b_size - 1) / b_size * (k + b_size - 1) / b_size * m > 100000) {
		b_size *= 2;
	}
	uniform const int l_count = (n + b_size - 1) / b_size;
	uniform const int j_count = (k + b_size - 1) / b_size;


	// each thread takes a block
	launch[l_count * j_count * m] blocking_task(b_size, l_count, j_count, m, n, k, A, B, C, alpha, beta);
}