/*
 * eigen_from_file_optimized_occupancy.cu
 *
 * This program reads one or more 5x5 symmetric matrices from a text file.
 * Each matrix is given by 5 lines with 5 doubles per line (row–major).
 * It then computes the eigenvalues of each matrix using a CUDA kernel
 * that converts the matrix to tridiagonal form and computes eigenvalues.
 *
 * Optimizations for higher occupancy include:
 *   - Using local arrays sized to the actual matrix dimension (5) instead of MAX_N (32)
 *   - Forcing inlining and unrolling small loops to reduce register pressure.
 *   - Using __restrict__ qualifiers on global pointers.
 *
 * Compile with:
 *    nvcc -arch=sm_70 -O3 eigen_from_file_optimized_occupancy.cu -o eigen_from_file_optimized_occupancy
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Define local array size for our 5x5 matrices.
#define LOCAL_N 5
#define SQR(a) ((a)*(a))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define TOL 1e-12

// Matrix dimension (5x5 matrices)
const int n = 5;

// Number of streams for overlapping transfers and kernel execution.
#define NUM_STREAMS 4

/* -------------------- Host Utility Functions -------------------- */
// Returns current time in milliseconds.
double getTimeInMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Counts the number of lines in a text file.
int countLines(const char *filename) {
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

/* -------------------- CUDA Device Code -------------------- */

// Force inline the helper functions to help reduce register pressure.
__device__ __forceinline__ double d_pythag(double a, double b) {
    double absa = fabs(a), absb = fabs(b);
    if (absa > absb)
        return absa * sqrt(1.0 + SQR(absb/absa));
    else
        return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa/absb)));
}

// Device version of tred2 specialized for 5x5 matrices using LOCAL_N.
__device__ __forceinline__ void d_tred2(double **a, int n, double *d, double *e) {
    int l, k, j, i;
    double scale, hh, h, g, f;
    for (i = n - 1; i > 0; i--) {
        l = i - 1;
        h = scale = 0.0;
#pragma unroll
        for (k = 0; k <= l; k++)
            scale += fabs(a[i][k]);
        if (scale == 0.0)
            e[i] = a[i][l];
        else {
#pragma unroll
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
#pragma unroll
            for (j = 0; j <= l; j++) {
                g = 0.0;
#pragma unroll
                for (k = 0; k <= j; k++)
                    g += a[j][k] * a[i][k];
#pragma unroll
                for (k = j + 1; k <= l; k++)
                    g += a[k][j] * a[i][k];
                e[j] = g / h;
                f += e[j] * a[i][j];
            }
            hh = f / (h + h);
#pragma unroll
            for (j = 0; j <= l; j++) {
                f = a[i][j];
                e[j] = g = e[j] - hh * f;
#pragma unroll
                for (k = 0; k <= j; k++)
                    a[j][k] -= (f * e[k] + g * a[i][k]);
            }
        }
        d[i] = h;
    }
    e[0] = 0.0;
#pragma unroll
    for (i = 0; i < n; i++) {
        d[i] = a[i][i];
    }
}

// Device version of tqli specialized for n=5.
__device__ __forceinline__ int d_tqli(double *d, double *e, int n, double **z) {
    int m, l, iter, i, k;
    double s, r, p, g, f, dd, c, b;
#pragma unroll
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
#pragma unroll
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

__device__ __forceinline__ int d_eigstm(double *d, double *e, int n) {
    double *z[LOCAL_N];  // Use LOCAL_N (which is 5) instead of MAX_N
    return d_tqli(d, e, n, z);
}

// Kernel with __restrict__ qualifiers. Each thread processes one matrix.
__global__ void eigstm_kernel(const double *__restrict__ d_A, double *__restrict__ d_eigs, int n, int numMatrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMatrices) return;

    const int matSize = n * n; // 25 for 5x5 matrices
    extern __shared__ double s_mat[];
    double *a = &s_mat[threadIdx.x * matSize];

#pragma unroll
    for (int i = 0; i < matSize; i++) {
        a[i] = d_A[idx * matSize + i];
    }
    __syncthreads();

    double *a_ptr[LOCAL_N];
#pragma unroll
    for (int i = 0; i < n; i++) {
        a_ptr[i] = &a[i * n];
    }

    double d_local[LOCAL_N]; // array of 5 doubles
    double e_local[LOCAL_N]; // array of 5 doubles

    d_tred2(a_ptr, n, d_local, e_local);
    d_eigstm(d_local, e_local, n);

#pragma unroll
    for (int i = 0; i < n; i++) {
        d_eigs[idx * n + i] = d_local[i];
    }
}

/* -------------------- Host Main Function -------------------- */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.txt>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char *filename = argv[1];

    int lineCount = countLines(filename);
    if (lineCount % n != 0) {
        fprintf(stderr, "File format error: number of lines (%d) is not a multiple of %d\n", lineCount, n);
        exit(EXIT_FAILURE);
    }
    int numMatrices = lineCount / n;
    printf("Number of matrices in file: %d\n", numMatrices);

    size_t totalMatrixSize = numMatrices * n * n * sizeof(double);
    double *h_A_gpu;
    cudaHostAlloc((void **)&h_A_gpu, totalMatrixSize, cudaHostAllocDefault);

    FILE *fin = fopen(filename, "r");
    if (!fin) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    for (int m = 0; m < numMatrices; m++) {
        for (int i = 0; i < n * n; i++) {
            if (fscanf(fin, "%lf", &h_A_gpu[m * n * n + i]) != 1) {
                fprintf(stderr, "Error reading matrix %d element %d\n", m, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fin);

    size_t totalEigSize = numMatrices * n * sizeof(double);
    double *h_eigs_gpu;
    cudaHostAlloc((void **)&h_eigs_gpu, totalEigSize, cudaHostAllocDefault);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int minGridSize, occBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &occBlockSize, eigstm_kernel, 0, 0);
    occBlockSize = (occBlockSize / 64) * 64;
    if (occBlockSize == 0) occBlockSize = 64;

    int maxBlockBySMem = deviceProp.sharedMemPerBlock / (n * n * sizeof(double));
    int optimalBlockSize = occBlockSize;
    if (optimalBlockSize > maxBlockBySMem) {
        optimalBlockSize = 64;
        if (optimalBlockSize == 0) optimalBlockSize = 64;
    }
    printf("Optimal block size (multiple of 64) after adjustments: %d\n", optimalBlockSize);

    size_t sharedBytes = optimalBlockSize * n * n * sizeof(double);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    float occupancyVector[NUM_STREAMS] = {0};
    double startTime = getTimeInMs();

    int chunkSize = (numMatrices + NUM_STREAMS - 1) / NUM_STREAMS;
    for (int s = 0; s < NUM_STREAMS; s++) {
        int startMat = s * chunkSize;
        int endMat = (startMat + chunkSize > numMatrices) ? numMatrices : startMat + chunkSize;
        int numChunk = endMat - startMat;
        if (numChunk <= 0)
            continue;

        size_t chunkMatrixBytes = numChunk * n * n * sizeof(double);
        size_t chunkEigBytes = numChunk * n * sizeof(double);

        double *d_A_chunk, *d_eigs_chunk;
        cudaMalloc((void **)&d_A_chunk, chunkMatrixBytes);
        cudaMalloc((void **)&d_eigs_chunk, chunkEigBytes);

        cudaMemcpyAsync(d_A_chunk, h_A_gpu + startMat * n * n,
                        chunkMatrixBytes, cudaMemcpyHostToDevice, streams[s]);

        int blocks = (numChunk + optimalBlockSize - 1) / optimalBlockSize;
        int activeThreads = blocks * optimalBlockSize;
        int maxThreadsOverall = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
        if (activeThreads > maxThreadsOverall)
            activeThreads = maxThreadsOverall;
        occupancyVector[s] = ((float)activeThreads / (float)maxThreadsOverall) * 100.0f;

        eigstm_kernel<<<blocks, optimalBlockSize, sharedBytes, streams[s]>>>(d_A_chunk, d_eigs_chunk, n, numChunk);

        cudaMemcpyAsync(h_eigs_gpu + startMat * n, d_eigs_chunk,
                        chunkEigBytes, cudaMemcpyDeviceToHost, streams[s]);

        cudaFreeAsync(d_A_chunk, streams[s]);
        cudaFreeAsync(d_eigs_chunk, streams[s]);
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    double endTime = getTimeInMs();
    double totalTimeMs = endTime - startTime;
    printf("Total GPU processing time for %d matrices (n=%d): %f ms\n", numMatrices, n, totalTimeMs);

    double throughput = numMatrices / (totalTimeMs / 1000.0);
    double avgLatency = totalTimeMs / numMatrices;

    int activeBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM, eigstm_kernel, optimalBlockSize, sharedBytes);
    int totalActiveThreads = activeBlocksPerSM * optimalBlockSize * deviceProp.multiProcessorCount;
    int maxThreadsOverall = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
    float overallOccupancy = ((float)totalActiveThreads / (float)maxThreadsOverall) * 100.0f;

    printf("\n--- Performance Metrics ---\n");
    printf("Overall theoretical occupancy: %.2f%%\n", overallOccupancy);
    printf("Per-chunk occupancy (vector):\n");
    for (int s = 0; s < NUM_STREAMS; s++) {
        printf("  Stream %d: %.2f%%\n", s, occupancyVector[s]);
    }
    printf("Throughput: %.2f matrices/second\n", throughput);
    printf("Average latency per matrix: %.4f ms\n", avgLatency);

    printf("\nFirst few matrices eigenvalues (GPU):\n");
    int displayCount = (numMatrices < 5 ? numMatrices : 5);
    for (int m = 0; m < displayCount; m++) {
        printf("Matrix %d eigenvalues:\n", m);
        for (int i = 0; i < n; i++) {
            printf("%f  ", h_eigs_gpu[m * n + i]);
        }
        printf("\n");
    }

    FILE *f_eigen = fopen("Data/valprop1M.txt", "r");
    if (f_eigen) {
        double error = 0.0;
        int totalEig = numMatrices * n;
        for (int j = 0; j < totalEig; j++) {
            double eigen;
            if (fscanf(f_eigen, "%lf", &eigen) != 1) {
                fprintf(stderr, "Error reading eigen_value at index %d\n", j);
                exit(EXIT_FAILURE);
            }
            error += fabs(eigen - h_eigs_gpu[j]);
        }
        fclose(f_eigen);
        printf("Total error compared to reference: %.14f\n", error);
    } else {
        fprintf(stderr, "Reference eigenvalue file (Data/valprop1M.txt) not found.\n");
    }

    cudaFreeHost(h_A_gpu);
    cudaFreeHost(h_eigs_gpu);

    return 0;
}
