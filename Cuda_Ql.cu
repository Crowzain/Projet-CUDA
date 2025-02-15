/*
 * eigen_from_file.cu
 *
 * This program reads one or more 5x5 symmetric matrices from a text file.
 * Each matrix is given by 5 lines with 5 doubles per line (row–major).
 * It then computes the eigenvalues of each matrix using a CUDA kernel
 * that replicates your original tridiagonalization and QL algorithm.
 *
 * The device code loads each full matrix (of size 5x5) into dynamic shared memory,
 * converts it to tridiagonal form (via d_tred2) and computes eigenvalues (via d_tqli).
 *
 * Compile with:
 *    nvcc -arch=sm_70 -O3 eigen_from_file.cu -o eigen_from_file
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <sys/time.h>
 #include <cuda_runtime.h>
 
 /* -------------------- Define Constants -------------------- */
 #define MAX_N 32            // Maximum matrix dimension for device routines
 #define SQR(a) ((a)*(a))
 #define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
 #define TOL 1e-12           // Convergence tolerance for QL routine
 
 // For our 5x5 full symmetric matrices, we have:
 const int n = 5;
 
 /* -------------------- Host Utility Functions -------------------- */
 
 double getTimeInMs()
 {
     struct timeval tv;
     gettimeofday(&tv, NULL);
     return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
 }
 
 /* Reads the input file and counts the number of lines.
    Assumes each line is nonempty. */
 int countLines(const char *filename)
 {
     FILE *f = fopen(filename, "r");
     if (!f) {
         perror("fopen");
         exit(EXIT_FAILURE);
     }
     int lines = 0;
     char buffer[256];
     while (fgets(buffer, sizeof(buffer), f) != NULL)
         lines++;
     fclose(f);
     return lines;
 }
 
 /* -------------------- Original CPU Code (unchanged structure) -------------------- */
 
 /* Return sqrt(a^2 + b^2) in a stable way */
 // static double pythag(double a, double b)
 // {
 //     double absa = fabs(a), absb = fabs(b);
 //     if (absa > absb)
 //         return absa * sqrt(1.0 + SQR(absb/absa));
 //     else
 //         return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa/absb)));
 // }
 
 // static int compdouble (void const *a, void const *b)
 // {
 // int ret = 0;
 // double const *pa = (const double *)a;
 // double const *pb = (const double *)b;
 // double diff = *pa - *pb;
 // if (diff > 0){
 //     ret = 1;
 // }
 // else if (diff < 0){
 //     ret = -1;
 // }
 // else{
 //     ret = 0;
 // }
 
 // return ret;
 // }
 
 /* -------------------- CUDA Device Code (optimized) -------------------- */
 
 /* Device version of pythag (same as above) */
 __device__ double d_pythag(double a, double b)
 {
     double absa = fabs(a), absb = fabs(b);
     if (absa > absb)
         return absa * sqrt(1.0 + SQR(absb/absa));
     else
         return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa/absb)));
 }
 
 /* Device version of tred2.
    a is an array of row pointers (each pointing to a row stored in local memory).
 */
 __device__ void d_tred2(double **a, int n, double *d, double *e)
 {
     int l, k, j, i;
     double scale, hh, h, g, f;
     for (i = n - 1; i > 0; i--) {
         l = i - 1;
         h = scale = 0.0;
         if (l > 0) {
             for (k = 0; k <= l; k++)
                 scale += fabs(a[i][k]);
             if (scale == 0.0)
                 e[i] = a[i][l];
             else {
                 for (k = 0; k <= l; k++) {
                     a[i][k] /= scale;
                     h += a[i][k] * a[i][k];
                 }
                 f = a[i][l];
                 g = (f >= 0.0 ? -sqrt(h) : sqrt(h));
                 e[i] = scale * g;
                 h -= f * g;
                 a[i][l] = f - g;
                 f = 0.0;
                 for (j = 0; j <= l; j++) {
                     g = 0.0;
                     for (k = 0; k <= j; k++)
                         g += a[j][k] * a[i][k];
                     for (k = j + 1; k <= l; k++)
                         g += a[k][j] * a[i][k];
                     e[j] = g / h;
                     f += e[j] * a[i][j];
                 }
                 hh = f / (h + h);
                 for (j = 0; j <= l; j++) {
                     f = a[i][j];
                     e[j] = g = e[j] - hh * f;
                     for (k = 0; k <= j; k++)
                         a[j][k] -= (f * e[k] + g * a[i][k]);
                 }
             }
         } else
             e[i] = a[i][l];
         d[i] = h;
     }
     e[0] = 0.0;
     for (i = 0; i < n; i++) {
         d[i] = a[i][i];
     }
 }
 
 /* Device version of tqli.
    Note: A simple bubble–sort is used to sort the eigenvalues.
 */
 __device__ int d_tqli(double *d, double *e, int n, double **z)
 {
     int m, l, iter, i, k;
     double s, r, p, g, f, dd, c, b;
     for (i = 1; i < n; i++)
         e[i - 1] = e[i];
     e[n - 1] = 0.0;
     for (l = 0; l < n; l++) {
         iter = 0;
         do {
             for (m = l; m < n - 1; m++) {
                 dd = fabs(d[m]) + fabs(d[m + 1]);
                 if (fabs(e[m]) <= TOL * dd)
                     break;
             }
             if (m != l) {
                 if (iter++ == 30)
                     return -1;
                 g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                 r = d_pythag(g, 1.0);
                 g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
                 s = c = 1.0;
                 p = 0.0;
                 for (i = m - 1; i >= l; i--) {
                     f = s * e[i];
                     b = c * e[i];
                     e[i + 1] = (r = d_pythag(f, g));
                     if (r == 0.0) {
                         d[i + 1] -= p;
                         e[m] = 0.0;
                         break;
                     }
                     s = f / r;
                     c = g / r;
                     g = d[i + 1] - p;
                     r = (d[i] - g) * s + 2.0 * c * b;
                     d[i + 1] = g + (p = s * r);
                     g = c * r - b;
                 }
                 if (r == 0.0 && i >= l)
                     continue;
                 d[l] -= p;
                 e[l] = g;
                 e[m] = 0.0;
             }
         } while (m != l);
     }
     /* Bubble sort for the eigenvalues */
     for (i = 0; i < n - 1; i++) {
         for (k = i + 1; k < n; k++) {
             if (d[i] > d[k]) {
                 double tmp = d[i];
                 d[i] = d[k];
                 d[k] = tmp;
             }
         }
     }
     return 0;
 }
 
 /* Device wrapper for eigstm (calls d_tqli) */
 __device__ int d_eigstm(double *d, double *e, int n)
 {
     double *z[1];  // dummy pointer for eigenvectors
     return d_tqli(d, e, n, z);
 }
 
 /* -------------------- End of CUDA Device Code -------------------- */
 
 /* -------------------- CUDA Kernel -------------------- */
 /*
  * Each thread processes one matrix.
  * The input matrices are stored contiguously in row–major order.
  * For a 5x5 matrix, each matrix has 25 doubles.
  * Each thread loads its matrix into dynamic shared memory,
  * creates a local array of row pointers, and then calls
  * d_tred2 to convert the full symmetric matrix to tridiagonal form,
  * followed by d_eigstm (which calls d_tqli) to compute the eigenvalues.
  */
 __global__ void eigstm_kernel(const double *d_A, double *d_eigs, int n, int numMatrices)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= numMatrices) return;
 
     int matSize = n * n;  // For 5x5, this is 25
 
     // Declare dynamic shared memory.
     extern __shared__ double s_mat[];
 
     // Pointer to this thread's matrix in shared memory.
     double *a = &s_mat[threadIdx.x * matSize];
 
     // Load matrix from global memory into shared memory.
     for (int i = 0; i < matSize; i++) {
         a[i] = d_A[idx * matSize + i];
     }
     __syncthreads();
 
     // Create an array of row pointers in local memory.
     double *a_ptr[MAX_N];
     for (int i = 0; i < n; i++) {
         a_ptr[i] = &a[i * n];
     }
 
     double d_local[MAX_N];
     double e_local[MAX_N];
 
     // Convert the full symmetric matrix to tridiagonal form.
     d_tred2(a_ptr, n, d_local, e_local);
 
     // Compute eigenvalues.
     d_eigstm(d_local, e_local, n);
 
     // Write the computed eigenvalues to global memory.
     double *out_ptr = d_eigs + idx * n;
     for (int i = 0; i < n; i++) {
         out_ptr[i] = d_local[i];
     }
 }
 
 /* -------------------- Host Main Function -------------------- */
 int main(int argc, char *argv[])
 {
     if (argc < 2) {
         fprintf(stderr, "Usage: %s <matrix_file.txt>\n", argv[0]);
         exit(EXIT_FAILURE);
     }
     const char *filename = argv[1];
 
     /* Count the total number of lines in the file */
     FILE *fin = fopen(filename, "r");
     if (!fin) {
         perror("fopen");
         exit(EXIT_FAILURE);
     }
     int lineCount = 0;
     char buf[256];
     while (fgets(buf, sizeof(buf), fin) != NULL) {
         if (strlen(buf) > 1)  // non-empty
             lineCount++;
     }
     fclose(fin);
     if (lineCount % n != 0) {
         fprintf(stderr, "File format error: number of lines (%d) is not a multiple of %d\n", lineCount, n);
         exit(EXIT_FAILURE);
     }
     int numMatrices = lineCount / n;
     printf("Number of matrices in file: %d\n", numMatrices);
 
     /* Allocate host memory for matrices.
        Each matrix is stored as a contiguous block of 25 doubles (row-major).
     */
     size_t sizeMatrix = n * n * sizeof(double);
     double **h_A_cpu = (double **)malloc(numMatrices * sizeof(double *));
     for (int m = 0; m < numMatrices; m++) {
         h_A_cpu[m] = (double *)malloc(sizeMatrix);
     }
 
     /* Read matrices from file.
        Each matrix is 5 lines of 5 doubles.
     */
     fin = fopen(filename, "r");
     if (!fin) {
         perror("fopen");
         exit(EXIT_FAILURE);
     }
     for (int m = 0; m < numMatrices; m++) {
         for (int i = 0; i < n; i++) {
             for (int j = 0; j < n; j++) {
                 if (fscanf(fin, "%lf", &h_A_cpu[m][i * n + j]) != 1) {
                     fprintf(stderr, "Error reading matrix %d at row %d col %d\n", m, i, j);
                     exit(EXIT_FAILURE);
                 }
             }
         }
     }
     fclose(fin);
 
     /* Pack the matrices into a contiguous host array (SoA layout for the GPU).
        Each matrix is stored in row-major order.
     */
     size_t totalMatrixSize = numMatrices * n * n * sizeof(double);
     double *h_A_gpu = (double *)malloc(totalMatrixSize);
     for (int m = 0; m < numMatrices; m++) {
         memcpy(&h_A_gpu[m * n * n], h_A_cpu[m], n * n * sizeof(double));
     }
 
     /* --------------------- CUDA Version --------------------- */
     double *d_A, *d_eigs;
     cudaMalloc((void **)&d_A, totalMatrixSize);
     cudaMalloc((void **)&d_eigs, numMatrices * n * sizeof(double));
     cudaMemcpy(d_A, h_A_gpu, totalMatrixSize, cudaMemcpyHostToDevice);
 
     /* Choose a block size so that shared memory per block is within limits.
        For a 5x5 matrix, each thread requires 25*8 = 200 bytes.
        With 64 threads per block, that’s 64*200 = 12800 bytes (well within limits).
     */
     int threadsPerBlock = 128;
     int blocks = (numMatrices + threadsPerBlock - 1) / threadsPerBlock;
     size_t sharedBytes = threadsPerBlock * n * n * sizeof(double);
 
     cudaEvent_t startEvent, stopEvent;
     cudaEventCreate(&startEvent);
     cudaEventCreate(&stopEvent);
     cudaEventRecord(startEvent, 0);
 
     eigstm_kernel<<<blocks, threadsPerBlock, sharedBytes>>>(d_A, d_eigs, n, numMatrices);
     cudaDeviceSynchronize();
 
     cudaEventRecord(stopEvent, 0);
     cudaEventSynchronize(stopEvent);
     float gpuTime;
     cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
     printf("GPU kernel execution time for %d matrices (n=%d): %f ms\n", numMatrices, n, gpuTime);
 
     /* Retrieve GPU results */
     double *h_eigs_gpu = (double *)malloc(numMatrices * n * sizeof(double));
     cudaMemcpy(h_eigs_gpu, d_eigs, numMatrices * n * sizeof(double), cudaMemcpyDeviceToHost);
 
     /* Print the eigenvalues for the first few matrices */
     printf("\nFirst few matrices eigenvalues (GPU):\n");
     for (int m = 0; m < (numMatrices < 5 ? numMatrices : 5); m++) {
         printf("Matrix %d eigenvalues:\n", m);
         for (int i = 0; i < n; i++) {
             printf("%f  ", h_eigs_gpu[m * n + i]);
         }
         printf("\n");
     }
     FILE *f_eigen = fopen("Data/valpropres.txt","r");
     if(!f_eigen){
         perror("couldn't open file!\n");
         exit(EXIT_FAILURE);
     }
 
     double error = 0;
     for (int j = 0; j < lineCount; j++) {
             double eigen;
             if (fscanf(f_eigen, "%lf", &eigen) != 1) {
                 fprintf(stderr, "Error reading eigen_value at row %d", j);
                 exit(EXIT_FAILURE);
                 }
             error+= abs(eigen-h_eigs_gpu[j]);
         }
     fclose(f_eigen);
     printf("Error : %.14f\n",error);
     /* Cleanup host memory */
     for (int m = 0; m < numMatrices; m++) {
         free(h_A_cpu[m]);
     }
     free(h_A_cpu);
     free(h_A_gpu);
     free(h_eigs_gpu);
 
     cudaFree(d_A);
     cudaFree(d_eigs);
     cudaEventDestroy(startEvent);
     cudaEventDestroy(stopEvent);
 
     return 0;
 }