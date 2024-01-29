#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


#pragma GCC optimize ("peel-loops")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("Ofast")


const char* dgemm_desc = "Lian's simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#define SMALL_BLOCK 64
#endif


#define min(a, b) (((a) < (b)) ? (a) : (b))

void print_matrix(const char* desc, int lda, double* C) {
    printf("%s (First Row):\n", desc);
    for (int j = 0; j < lda; j++) {
        printf("%8.2f ", C[j]);
    }
    printf("\n");
}

static void kernel(int lda, int K, double* A, double* B, double* C)
{
    __m256d aa1, aa2;

    __m256d bb01, bb02, bb03, bb04;
    __m256d bb11, bb12, bb13, bb14;

    __m256d cc01, cc02, cc03, cc04;
    __m256d cc11, cc12, cc13, cc14;

    cc01 = _mm256_loadu_pd(C);
    cc02 = _mm256_loadu_pd(C + lda);
    cc03 = _mm256_loadu_pd(C + 2 * lda);
    cc04 = _mm256_loadu_pd(C + 3 * lda);

    cc11 = _mm256_setzero_pd();
    cc12 = _mm256_setzero_pd();
    cc13 = _mm256_setzero_pd();
    cc14 = _mm256_setzero_pd();

    int KK;
    if(K % 2==0){
        KK = K;
    }else {
        KK = K - 1;
    }
    for (int i = 0; i < KK; i += 2){
        aa1 = _mm256_loadu_pd(A);
        A += 4;

        bb01 = _mm256_broadcast_sd(B++);
        bb02 = _mm256_broadcast_sd(B++);
        bb03 = _mm256_broadcast_sd(B++);
        bb04 = _mm256_broadcast_sd(B++);

        cc01 = _mm256_fmadd_pd(aa1, bb01, cc01);
        cc02 = _mm256_fmadd_pd(aa1, bb02, cc02);
        cc03 = _mm256_fmadd_pd(aa1, bb03, cc03);
        cc04 = _mm256_fmadd_pd(aa1, bb04, cc04);

        aa2 = _mm256_loadu_pd(A);
        A += 4;

        bb11 = _mm256_broadcast_sd(B++);
        bb12 = _mm256_broadcast_sd(B++);
        bb13 = _mm256_broadcast_sd(B++);
        bb14 = _mm256_broadcast_sd(B++);

        cc11 = _mm256_fmadd_pd(aa2, bb11, cc11);
        cc12 = _mm256_fmadd_pd(aa2, bb12, cc12);
        cc13 = _mm256_fmadd_pd(aa2, bb13, cc13);
        cc14 = _mm256_fmadd_pd(aa2, bb14, cc14);
    }

    if(K % 2==1){
        aa1 = _mm256_loadu_pd(A);

        bb01 = _mm256_broadcast_sd(B++);
        bb02 = _mm256_broadcast_sd(B++);
        bb03 = _mm256_broadcast_sd(B++);
        bb04 = _mm256_broadcast_sd(B++);

        cc01 = _mm256_fmadd_pd(aa1, bb01, cc01);
        cc02 = _mm256_fmadd_pd(aa1, bb02, cc02);
        cc03 = _mm256_fmadd_pd(aa1, bb03, cc03);
        cc04 = _mm256_fmadd_pd(aa1, bb04, cc04);
    }

    cc01 = _mm256_add_pd(cc01, cc11);
    cc02 = _mm256_add_pd(cc02, cc12);
    cc03 = _mm256_add_pd(cc03, cc13);
    cc04 = _mm256_add_pd(cc04, cc14);

    _mm256_storeu_pd(C, cc01);
    _mm256_storeu_pd(C+lda, cc02);
    _mm256_storeu_pd(C+2*lda, cc03);
    _mm256_storeu_pd(C+3*lda, cc04);
}


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C){

    double* A_block = (double*)_mm_malloc(M * K * sizeof(double), 64);
    double* B_block = (double*)_mm_malloc(K * N * sizeof(double), 64);
    double *a_ptr, *b_ptr, *c_ptr;
    double cip;

    int M_mod_4 = M % 4;
    int N_mod_4 = N % 4;
    int i=0,j;

    for (j = 0 ; j < N - 3; j += 4){
        b_ptr = &B_block[j * K];

        for (int m = 0; m < K; m++){
            for (int n = 0; n < 4; n++) {
                b_ptr[m * 4 + n] = B[(j + n) * lda + m];
            }
        }

        for (i = 0; i < M-3; i += 4){
            a_ptr = &A_block[i * K];

            if (j == 0){
                double* a_src = A + i;
                double* a_des = a_ptr;
                for (int u = 0; u < K; u++) {
                    for (int o = 0; o < 4; o++) {
                        a_des[o] = a_src[o];
                    }
                    a_des += 4;
                    a_src += lda;
                }
            }

            c_ptr = C + i + j * lda;
            kernel(lda, K, a_ptr, b_ptr, c_ptr);
        }
    }

    if (M_mod_4 != 0){
        for (; i < M; ++i){
            for (int p = 0; p < N; ++p){
                cip = C[i + p * lda];

                for (int k = 0; k < K; ++k){
                    cip += A[i + k * lda] * B[k + p * lda];
                }
                C[i + p * lda] = cip;
            }
        }
    }
    if (N_mod_4 != 0){
        for (; j < N; ++j){
            for (int p = 0; p < M - M_mod_4; ++p){
                cip = C[p + j * lda];

                for (int k = 0; k < K; ++k){
                    cip += A[p + k * lda] * B[k + j * lda];
                }
                C[p + j * lda] = cip;
            }
        }
    }
    _mm_free(A_block);
    _mm_free(B_block);
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C){

    int fixlda=lda;

    // For each block-row of A
    for (int i = 0; i < fixlda; i += BLOCK_SIZE){
        // For each block-row of A
        for (int j = 0; j < fixlda; j += BLOCK_SIZE){
            // For each block-column of B
            for (int k = 0; k < fixlda; k += BLOCK_SIZE){
                int M = min(BLOCK_SIZE, fixlda - i); // Height of the block of A and C
                int N = min(BLOCK_SIZE, fixlda - j); // Width of the block of B and C
                int K = min(BLOCK_SIZE, fixlda - k); // Width of the block of A and height of the block of B
                // Perform individual block dgemm
                for (int f = i; f < i + M; f += SMALL_BLOCK){
                    for (int g = j; g < j + N; g += SMALL_BLOCK) {
                        for (int h = k; h < k + K; h += SMALL_BLOCK){
                            int Q = min(SMALL_BLOCK, i + M - f); // Height of the small block of A and C
                            int W = min(SMALL_BLOCK, j + N - g); // Width of the small block of B and C
                            int E = min(SMALL_BLOCK, k + K - h); // Corresponds to the "inner dimension"
                            // Perform do_block for the small block
                            do_block(fixlda, Q, W, E, A + f + h * fixlda, B + h + g * fixlda, C + f + g * fixlda);
                        }
                    }
                }
            }
        }
}

 //  print_matrix("Result Matrix C", lda, C);
}
