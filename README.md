# GPU-Accelerated Eigenvalue Computation for Massive Matrix Batches

**Computing eigenvalues for 100 million matrices in seconds, not hours.**

## Results

| Configuration | Matrices Processed | GPU Time | CPU Time | Speedup |
|---------------|-------------------|----------|----------|---------|
| **64 threads/block** | 10M | ~1s | ~100s | **100x** |
| **128 threads/block** | 10M | ~1.2s | ~100s | **83x** |
| **vs cuSOLVER** | 10M | ~1s | ~10,000s | **10,000x** |

**Performance metrics:**
- **Occupancy:** 99% per chunk (25-31.5% theoretical)
- **Latency:** ~0.0001ms per matrix
- **Throughput ratio (GPU/CPU):** 10⁻² for 10⁷+ matrices

---

## Problem

Eigenvalue computation is fundamental in scientific computing (fluid mechanics, finite differences, quantum chemistry), but existing solutions fail at scale:
- **cuSOLVER optimized for large matrices** → inefficient for millions of small ones
- **CPU sequential processing** → prohibitively slow (hours for 10M+ matrices)
- **Block-based parallelization** → wrong granularity for n ≤ 50

**Challenge:** Process 100 million symmetric 5×5 matrices efficiently.

---

## Solution

**One thread per matrix** parallel strategy using Householder tridiagonalization + implicit-shift QL algorithm.

### Algorithm Pipeline
```
Symmetric Matrix (n×n)
         ↓
Householder Tridiagonalization O(n³)
- Reduces to tridiagonal form
- Preserves eigenvalues
         ↓
Implicit-Shift QL Iteration O(n²)
- Wilkinson shift for quadratic convergence
- Extracts eigenvalues from tridiagonal matrix
         ↓
Eigenvalues (sorted, accurate)
```

### Key Technical Decisions

**1. Thread-per-matrix architecture**  
Each thread processes one complete matrix independently—no cross-thread communication overhead. Ideal for embarrassingly parallel workload.

**2. Shared memory optimization**  
```c
extern __shared__ double s_mat[];
double *a = &s_mat[threadIdx.x * matSize];
```
Each thread allocates 144 bytes (5×5 matrix) in shared memory for 100x faster access vs global memory.

**3. Asynchronous chunked processing**  
```c
cudaMemcpyAsync(d_A, h_A_chunk, size, cudaMemcpyHostToDevice, stream);
eigstm_kernel<<<blocks, threads, sharedMem, stream>>>(d_A, d_eigs, n, numMatrices);
```
Overlaps CPU-GPU transfers with computation using CUDA streams.

**4. Structure-of-Arrays (SoA) layout**  
Matrices stored contiguously for coalesced memory access—critical for cache efficiency on GPU.

---

## Tech Stack

**Core:** C, CUDA, NVCC compiler  
**Hardware:** NVIDIA Quadro P5000 (2560 CUDA cores, 16GB GDDR5, 288 GB/s bandwidth)  
**Compute Capability:** 6.1 (Pascal architecture)  
**Theoretical Peak:** 256 GFLOPS (FP64)

---

## Architecture

### CUDA Execution Model
```
Host (CPU)
- Data preparation
- Kernel launch
- Result collection
         ↓
Device (GPU)
├─ Streaming Multiprocessor 1
│  ├─ Warp 0 (32 threads) → 32 matrices
│  ├─ Warp 1 (32 threads) → 32 matrices
│  └─ Shared Memory (fast)
├─ SM 2, SM 3... (parallel execution)
└─ Global Memory (slow, minimized access)
```

### Memory Hierarchy Exploitation
- **Registers:** Algorithm variables (fastest)
- **Shared Memory:** Matrix data per SM (100x faster than global)
- **L1/L2 Cache:** Automatic caching of frequently accessed data
- **Global Memory:** Input/output only (minimized access)

### Optimization Strategy
1. **Coalesced memory access:** Adjacent threads access adjacent memory
2. **Pinned host memory:** Zero-copy transfers via `cudaHostAlloc`
3. **Stream concurrency:** Multiple kernels execute simultaneously
4. **Occupancy tuning:** `cudaOccupancyMaxPotentialBlockSize` for optimal configuration

---

## Performance Analysis

### vs CPU Sequential
- **100x speedup** for 10M matrices
- **Throughput ratio decreases to 10⁻²** at 10⁷+ matrices (saturation efficiency)

### vs cuSOLVER (state-of-the-art)
- **10,000x improvement** for large batches
- cuSOLVER designed for single large matrices, not batched small ones

### Scalability
- **Linear complexity** with matrix count (embarrassingly parallel)
- **O(n³) per matrix** but constant across threads
- **99% occupancy** proves near-optimal GPU utilization

---

## What We Learned

**1. Parallelism granularity is everything**  
Standard libraries optimize for large single matrices. We inverted this: small matrices, massive parallelism. One thread per matrix eliminated synchronization overhead entirely.

**2. Memory hierarchy mastery**  
Shared memory allocation (`144 bytes × 64 threads = 9KB per block`) was critical. Going beyond 12KB would drop occupancy—measured, tuned, validated with profiling.

**3. Hardware-aware algorithm design**  
Warp size (32 threads) drove block size choices (64, 128). Aligning thread count to warps minimized divergence. Pascal architecture's 2:1 register file constraint informed memory strategy.

**Challenges:**

- **Shared memory constraints:** Limited to 48KB per SM → forced thread-per-matrix approach instead of thread-per-element
- **Occupancy vs throughput trade-off:** Higher occupancy (128 threads/block) didn't always win—64 threads had better cache behavior
- **Numerical stability:** Householder + QL maintains machine precision but required double precision (FP64), halving theoretical throughput

---

## Use Cases

**Scientific Computing:** Quantum chemistry (molecular orbital calculations), structural mechanics (modal analysis)  
**Machine Learning:** PCA on high-dimensional datasets, spectral clustering  
**Financial Modeling:** Covariance matrix eigendecomposition for portfolio risk analysis  
**Signal Processing:** Multi-channel sensor arrays, frequency domain analysis

---

## Complexity Analysis

**Sequential (CPU):**  
- Per matrix: O(n³) tridiagonalization + O(n²) QL  
- M matrices: **O(M × n³)**

**Parallel (GPU):**  
- Per thread: O(n³)  
- With N threads: **O(n³)** wall-clock time (perfect parallelism)  
- **Speedup = M/N** (limited by thread count)

---

## Authors

Yassir Masfour, Pierre Beauvain, Ayoub Achour  
Supervised by: Prof. Jonas Koko  
ISIMA, 2024-2025

---

## Citation

```bibtex
@techreport{gpu_eigenvalues_2025,
  author = {Masfour, Yassir and Beauvain, Pierre and Achour, Ayoub},
  title = {GPU-Accelerated Eigenvalue Computation for Massive Matrix Batches},
  institution = {ISIMA},
  year = {2025},
  type = {Technical Report}
}
```
