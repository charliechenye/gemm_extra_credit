task void gemm_ispc_task( uniform int m, uniform int n, uniform int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int i_block_start, uniform int i_block_end,
	 uniform int j_block_start, uniform int j_block_end) {
	// YOUR IMPLEMENTATION HERE

	// 1 block is splited into 8 parts executed by 8 threads (2 rows x 4 cols)
	uniform int iStart = i_block_start + (int)(taskIndex / 4) * (int)((i_block_end - i_block_start)/2); // starting row of each part
	uniform int iEnd = (iStart == i_block_start? i_block_start + (int)((i_block_end - i_block_start)/2) : i_block_end); // final row of each part
	uniform int jStart = j_block_start + (int)((j_block_end - j_block_start)/4) * (taskIndex % 4); // starting col of each part
	uniform int jEnd = (taskIndex % 4 == 3? j_block_end : jStart + (int)((j_block_end - j_block_start)/4)); // final col of each part

	// SIMD with 4 ALUS
	foreach (i = iStart ... iEnd){
		for(uniform int j=jStart; j<jEnd; j++){
			double inner_prod = 0;
			for(uniform int kk=0; kk<k; kk++){
				inner_prod += A[i*k+kk] * B[kk*n+j];
			}
			C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) 
{
	// YOUR IMPLEMENTATION HERE
	uniform const int b_size = 64;
	uniform const int i_interleaving = 64;
	uniform int nBlock = n / b_size;
	for (uniform int i_block_start = 0; i_block_start < m; i_block_start += i_interleaving) {
		// Save matrix in L3 cache
		uniform int i_block_end = (i_block_start + i_interleaving < m ? i_block_start + i_interleaving : m); // final element in row of each block
		
		for (uniform int j_block_start = 0; j_block_start < n; j_block_start += b_size) {
			uniform int j_block_end = (j_block_start + b_size < n? j_block_start + b_size : n);
			for (uniform int kk = 0; kk < k; kk++){ 
				foreach(j = j_block_start ... j_block_end){
					prefetch_l3(&B[kk*n+j]);
				}
			}

			launch[8] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, i_block_start, i_block_end, j_block_start, j_block_end);
		}
	}
}
