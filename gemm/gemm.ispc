task void gemm_ispc_task( uniform int m, uniform int n, uniform int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int iBlockStart, uniform int iBlockEnd,
	 uniform int jBlockStart, uniform int jBlockEnd) {
	// YOUR IMPLEMENTATION HERE

	// 1 block is splited into 8 parts executed by 8 threads (2 rows x 4 cols)
	uniform int iStart = iBlockStart + (int)(taskIndex / 4) * (int)((iBlockEnd - iBlockStart)/2); // starting row of each part
	uniform int iEnd = (iStart == iBlockStart? iBlockStart + (int)((iBlockEnd - iBlockStart)/2) : iBlockEnd); // final row of each part
	uniform int jStart = jBlockStart + (int)((jBlockEnd - jBlockStart)/4) * (taskIndex % 4); // starting col of each part
	uniform int jEnd = (taskIndex % 4 == 3? jBlockEnd : jStart + (int)((jBlockEnd - jBlockStart)/4)); // final col of each part

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
	uniform int nBlock = m / b_size;
	for (uniform int iBlock = 0; iBlock < nBlock; iBlock++){
		// Save matrix in L3 cache
		uniform int iBlockStart = iBlock * b_size; // starting element in row of each block
		uniform int iBlockEnd = (iBlock < nBlock - 1? iBlockStart + b_size : m); // final element in row of each block
		for(uniform int i = iBlockStart; i < iBlockEnd; i++){ 
			foreach (kk = 0 ... k){ 
				prefetch_l3(&A[i*k+kk]);
			}
		}
		for(uniform int jBlock = 0; jBlock < nBlock; jBlock++){
			uniform int jBlockStart = jBlock * b_size; // starting element in column of each block
			uniform int jBlockEnd = (jBlock < nBlock - 1? jBlockStart + b_size : m); // end element in column of each block

			for (uniform int kk = 0; kk < k; kk++){ 
				foreach(j = jBlockStart ... jBlockEnd){
					prefetch_l3(&B[kk*n+j]);
				}
			}

			for (uniform int i = iBlockStart; i < iBlockEnd; i++){ 
				foreach(j = jBlockStart ... jBlockEnd){ 
					prefetch_l3(&C[i*m+j]);
				}
			}

			launch[8] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, iBlockStart, iBlockEnd, jBlockStart, jBlockEnd);
		}
	}
}
