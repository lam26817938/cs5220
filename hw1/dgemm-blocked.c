#include <immintrin.h>
#include <string.h>
#include <stdio.h>


#pragma GCC optimize ("peel-loops")
#pragma GCC optimize("inline")
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

static void inline kernel(int lda, int K, double* restrict A, double* restrict B, double* restrict C)
{
    __m256d aa1, aa2;

    __m256d bb01, bb02, bb03, bb04;
    __m256d bb11, bb12, bb13, bb14;

    __m256d cc01, cc02, cc03, cc04;
    __m256d cc11, cc12, cc13, cc14;

    cc01 = _mm256_load_pd(C);
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
        aa1 = _mm256_load_pd(A);
        A += 4;

        bb01 = _mm256_broadcast_sd(B++);
        bb02 = _mm256_broadcast_sd(B++);
        bb03 = _mm256_broadcast_sd(B++);
        bb04 = _mm256_broadcast_sd(B++);

        cc01 = _mm256_fmadd_pd(aa1, bb01, cc01);
        cc02 = _mm256_fmadd_pd(aa1, bb02, cc02);
        cc03 = _mm256_fmadd_pd(aa1, bb03, cc03);
        cc04 = _mm256_fmadd_pd(aa1, bb04, cc04);

        aa2 = _mm256_load_pd(A);
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
        aa1 = _mm256_load_pd(A);

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

    _mm256_store_pd(C, cc01);
    _mm256_storeu_pd(C+lda, cc02);
    _mm256_storeu_pd(C+2*lda, cc03);
    _mm256_storeu_pd(C+3*lda, cc04);
}


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void inline do_block(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
    double *a_ptr, *b_ptr, *c_ptr;
    int i=0,j;
    double tem;

    int M_mod_4 = M-M % 4;
    int N_mod_4 = N-N % 4;
  //  double A_block[M_mod_4*K],B_block[K*N_mod_4];
    double* A_block = (double*)_mm_malloc(M_mod_4 * K * sizeof(double), 32);
    double* B_block = (double*)_mm_malloc(N_mod_4 * K * sizeof(double), 32);

    for (j = 0 ; j < N - 3; j += 4) {
        b_ptr = &B_block[j * K];

        for (int m = 0; m < K; m++) {
            for (int n = 0; n < 4; n++) {
                b_ptr[m * 4 + n] = B[(j + n) * lda + m];
            }
        }

        for (i = 0; i < M - 3; i += 4) {
            a_ptr = &A_block[i * K];

            if (j == 0) {
                double* a_src = A + i;
                for (int u = 0; u < K; u++) {
                    memcpy(a_ptr + u * 4, a_src, 4 * sizeof(double));
                    a_src += lda;
                }
            }

            c_ptr = C + i + j * lda;
            kernel(lda, K, a_ptr, b_ptr, c_ptr);

        }
    }

    _mm_free(A_block);
    _mm_free(B_block);

    if (M%4 != 0){
        for (; i < M; ++i){
            for (int p = 0; p < N; p++){
                tem = C[i + p * lda];
                for (int k = 0; k < K; k++){
                    tem += A[i + k * lda] * B[k + p * lda];
                }
                C[i + p * lda] = tem;
            }
        }
    }
    if (N%4 != 0){
        for (; j < N; j++){
            for (int p = 0; p < M_mod_4; p++){
                tem = C[p + j * lda];
                for (int k = 0; k < K; k++){
                    tem += A[p + k * lda] * B[k + j * lda];
                }
                C[p + j * lda] = tem;
            }
        }
    }

}

static void inline kernel2(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
    __m256d Aik, Aik_1;
    __m256d Bkj, Bkj_1, Bkj_2, Bkj_3;
    __m256d Cij, Cij_1, Cij_2, Cij_3, Cij_4, Cij_5, Cij_6, Cij_7;

    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 4) {
            Cij = _mm256_load_pd(C+i+j*lda);
            Cij_1 = _mm256_load_pd(C+i+(j+1)*lda);
            Cij_2 = _mm256_load_pd(C+i+(j+2)*lda);
            Cij_3 = _mm256_load_pd(C+i+(j+3)*lda);

            Cij_4 = _mm256_load_pd(C+i+4+j*lda);
            Cij_5 = _mm256_load_pd(C+i+4+(j+1)*lda);
            Cij_6 = _mm256_load_pd(C+i+4+(j+2)*lda);
            Cij_7 = _mm256_load_pd(C+i+4+(j+3)*lda);

            for (int k = 0; k < K; k++) {
                Aik = _mm256_load_pd(A+i+k*lda);
                Aik_1 = _mm256_load_pd(A+i+4+(k)*lda);

                Bkj = _mm256_broadcast_sd(B+k+j*lda);
                Bkj_1 = _mm256_broadcast_sd(B+k+(j+1)*lda);
                Bkj_2 = _mm256_broadcast_sd(B+k+(j+2)*lda);
                Bkj_3 = _mm256_broadcast_sd(B+k+(j+3)*lda);

                Cij = _mm256_fmadd_pd(Aik, Bkj, Cij);
                Cij_1 = _mm256_fmadd_pd(Aik, Bkj_1, Cij_1);
                Cij_2 = _mm256_fmadd_pd(Aik, Bkj_2, Cij_2);
                Cij_3 = _mm256_fmadd_pd(Aik, Bkj_3, Cij_3);

                Cij_4 = _mm256_fmadd_pd(Aik_1, Bkj, Cij_4);
                Cij_5 = _mm256_fmadd_pd(Aik_1, Bkj_1, Cij_5);
                Cij_6 = _mm256_fmadd_pd(Aik_1, Bkj_2, Cij_6);
                Cij_7 = _mm256_fmadd_pd(Aik_1, Bkj_3, Cij_7);
            }
            _mm256_store_pd(C+i+j*lda, Cij);
            _mm256_store_pd(C+i+(j+1)*lda, Cij_1);
            _mm256_store_pd(C+i+(j+2)*lda, Cij_2);
            _mm256_store_pd(C+i+(j+3)*lda, Cij_3);

            _mm256_store_pd(C+i+4+(j)*lda, Cij_4);
            _mm256_store_pd(C+i+4+(j+1)*lda, Cij_5);
            _mm256_store_pd(C+i+4+(j+2)*lda, Cij_6);
            _mm256_store_pd(C+i+4+(j+3)*lda, Cij_7);
        }
    }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* restrict A, double* restrict B, double* restrict C){
    if(lda<32) {
        int fixlda = lda + (8 - lda % 8) % 8;

        double *A_block = (double *) _mm_malloc(fixlda * fixlda * sizeof(double), 32);
        double *B_block = (double *) _mm_malloc(fixlda * fixlda * sizeof(double), 32);
        double *C_block = (double *) _mm_malloc(fixlda * fixlda * sizeof(double), 32);
        memset(C_block, 0, fixlda * fixlda * sizeof(double));
        for (int j = 0; j < fixlda; j++) {
            for (int i = 0; i < fixlda; i++) {
                if (i >= lda || j >= lda) {
                    A_block[i + fixlda * j] = 0;
                    B_block[i + fixlda * j] = 0;
                } else {
                    A_block[i + fixlda * j] = A[i + lda * j];
                    B_block[i + fixlda * j] = B[i + lda * j];
                }
            }
        }

        // For each block-row of A
        for (int i = 0; i < fixlda; i += BLOCK_SIZE) {
            // For each block-row of A
            for (int j = 0; j < fixlda; j += BLOCK_SIZE) {
                // For each block-column of B
                for (int k = 0; k < fixlda; k += BLOCK_SIZE) {
                    int M = min(BLOCK_SIZE, fixlda - i);
                    int N = min(BLOCK_SIZE, fixlda - j);
                    int K = min(BLOCK_SIZE, fixlda - k);
                    //   do_block(fixlda, M, N, K, A_block + i + k * fixlda, B_block + k + j * fixlda, C_block + i + j * fixlda);
                    // Perform individual block dgemm
                    for (int f = i; f < i + M; f += SMALL_BLOCK) {
                        for (int g = j; g < j + N; g += SMALL_BLOCK) {
                            for (int h = k; h < k + K; h += SMALL_BLOCK) {
                                int Q = min(SMALL_BLOCK, i + M - f);
                                int W = min(SMALL_BLOCK, j + N - g);
                                int E = min(SMALL_BLOCK, k + K - h);
                                // Perform do_block for the small block
                                kernel2(fixlda, Q, W, E, A_block + f + h * fixlda, B_block + h + g * fixlda,
                                        C_block + f + g * fixlda);
                            }
                        }
                    }
                }
            }
        }

        for (int j = 0; j < lda; j++) {
            memcpy(C + j * lda, C_block + j * fixlda, lda * sizeof(double));
        }
        //  print_matrix("Result Matrix C", lda, C);
        _mm_free(A_block);
        _mm_free(B_block);
        _mm_free(C_block);

        //   print_matrix("Result Matrix C free", lda, C);

    }else {
        int fixlda = lda;

        // For each block-row of A
        for (int i = 0; i < fixlda; i += BLOCK_SIZE) {
            // For each block-row of A
            for (int j = 0; j < fixlda; j += BLOCK_SIZE) {
                // For each block-column of B
                for (int k = 0; k < fixlda; k += BLOCK_SIZE) {
                    int M = min(BLOCK_SIZE, fixlda - i);
                    int N = min(BLOCK_SIZE, fixlda - j);
                    int K = min(BLOCK_SIZE, fixlda - k);
                    //   do_block(fixlda, M, N, K, A_block + i + k * fixlda, B_block + k + j * fixlda, C_block + i + j * fixlda);
                    // Perform individual block dgemm
                    for (int f = i; f < i + M; f += SMALL_BLOCK) {
                        for (int g = j; g < j + N; g += SMALL_BLOCK) {
                            for (int h = k; h < k + K; h += SMALL_BLOCK) {
                                int Q = min(SMALL_BLOCK, i + M - f);
                                int W = min(SMALL_BLOCK, j + N - g);
                                int E = min(SMALL_BLOCK, k + K - h);
                                // Perform do_block for the small block
                                do_block(fixlda, Q, W, E, A + f + h * fixlda, B + h + g * fixlda,
                                         C + f + g * fixlda);
                            }
                        }
                    }
                }
            }
        }
    }

        //   print_matrix("Result Matrix C", lda, C);
    }

