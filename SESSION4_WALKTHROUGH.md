# AKT26 Session 4 assignments

Team: **Sayf AI Ciao**, ID **1142**. Branch: `session-4-assignments`.

Source: `Session 4/akt26_gpu_prog.pdf` on the hackathon repository's
`session-4` branch (inspected at `1cdbfef`), slides 29, 31, and 33.
The CUDA starter is commit `7110cb1`. No `AGENTS.md` was found in either
repository or their parent directories. Existing directory names, function
interfaces (apart from the GER error routine's size argument), CPU algorithms,
and the location of both `main` functions are preserved. Exercises 0-2 are untouched.

**Status: implemented, awaiting NVIDIA GPU compilation and validation.**
No GPU timings, GPU error measurements, or speedups have been obtained.
AutoHPC access has not been provided to Team 1142. The team reports that the
organizers explicitly allow Google Colab; the supplied notebook uses that alternative.

## Assignment 1: rank-1 update (GER)

For every row `i` and column `j`, compute
`A(i,j) = A(i,j) + alpha*x[i]*y[j]`. The product of a column vector and a row
vector is a matrix: each column is a scaled copy of `x`.

The supplied CPU code loops over columns and then rows, updating one element
at a time. Column-major storage means all rows of column 0 come first, then
all rows of column 1. In C pointer arithmetic, element `(i,j)` is
`A[(size_t)j*m+i]`; `size_t` keeps the offset calculation wide.

The kernel performs the same update:

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row < m && col < n) {
    A[(size_t)col * m + row] += alpha * x[row] * y[col];
}
```

A block is a group of threads. Here each block contains `32*8 = 256` threads.
The grid is `ceil(m/32)` by `ceil(n/8)` blocks. Each thread owns one element;
extra threads at the matrix edges do nothing. Consecutive x threads access
adjacent rows, so a warp (32 threads executing together) can access neighboring
addresses efficiently. This is called coalesced memory access.

The CPU allocates device arrays, copies `x`, `y`, and the initial `A` into GPU
memory, launches the kernel, and copies `A` back. Each output has exactly one
writer, so there are no races, atomic operations, or block barriers. Device
work and copies use the default stream, which orders these operations.

The starter's cuBLAS comparison calls `cublasSger` with leading dimension `m`
and vector increments 1. The leading dimension is the distance between column
starts in elements. Both GPU paths are compared with the same CPU reference.
The relative Frobenius error is `sqrt(sum((GPU-CPU)^2)/sum(CPU^2))`, accumulated
in double precision. Both paths must have error at most `1e-6`.

The starter times repeated in-place updates after restoring the original A.
For accuracy it restores A again and updates **once**, matching the CPU result;
it does not compare a repeatedly updated matrix with a single update.
The reported operation count follows the starter's approximate `2*m*n`
convention (the literal expression has two multiplies and one addition).

## Assignment 2: batched matrix multiplication (GEMM)

For each batch index `b`, compute `C_b = A_b * B_b`, where each matrix is
`n*n` and `1 <= n <= 32`. This is a sum of products, not elementwise
multiplication:

```cpp
sum += A[base + (size_t)k*n + row] * B[base + (size_t)col*n + k];
```

The CPU reference loops over batches, columns, rows, then `k`. Each dot product
becomes one output entry. `base = (size_t)b*n*n` selects a matrix; consecutive
matrices are packed without padding.

The GPU uses one block per matrix, with `block=(32,8,1)` and
`grid=(batch_size,1,1)`. Threads cooperatively copy both input matrices into
shared memory, a small array accessible by threads in that block. Two arrays
of `32*32` floats occupy 8192 bytes per block. The linear thread number is
`threadIdx.y*32 + threadIdx.x`; it loads indices `tid, tid+256, ... < n*n`.
Every input element is therefore loaded exactly once per block.

```cpp
__syncthreads();
if (row < n) {
    for (int col = threadIdx.y; col < n; col += blockDim.y) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += s_A[k*n + row] * s_B[col*n + k];
        C[base + (size_t)col*n + row] = sum;
    }
}
```

`__syncthreads()` is a barrier: all threads finish loading before any thread
reads the shared arrays. Even threads with `row >= n` reach the barrier.
Afterward the shared arrays are read-only. A thread owns one row and columns
`threadIdx.y, threadIdx.y+8, ...`; thus every output has exactly one writer.
Different blocks write different matrices. No global reduction or atomics
are needed. Shared memory is reused by many output dot products and is never
reloaded or overwritten during computation, so one barrier suffices.

The host allocates A, B, C, copies A/B to the GPU, times the kernel with CUDA
events, and copies C back. C needs no initial data: the kernel overwrites all
entries. It then times `cublasSgemmStridedBatched` using the same buffers:
no transpose, `alpha=1`, `beta=0`, leading dimensions `n`, and strides `n*n`
**elements**, not bytes. Pedantic math mode keeps the comparison in FP32
without reduced-precision products; use CUDA 11 or newer. See the
[NVIDIA cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched).

Both implementations are compared with the CPU result using the starter's
maximum per-matrix forward error `||GPU-CPU||_F / (n*||CPU||_F)`, with tolerance
`1e-6`. Fused multiply-add and different summation orders can produce small
rounding differences; bitwise equality is not required. A zero reference gives
zero error only for an exact zero result; otherwise it gives infinity.
Nonfinite results fail the error checks, including NaNs that the original
batch maximum could accidentally hide. Programs return nonzero on accuracy
failure or CUDA/cuBLAS API failure and reject invalid event timings.

## Validation performed locally

On October 28, 2026, this Apple Silicon Mac (Darwin arm64) had no `nvcc`,
`nvidia-smi`, or NVIDIA GPU. These commands were actually attempted:

```sh
make -C 3-ger
make -C 4-gemm
```

Both printed the compiler command and failed with `make: nvcc: No such file or
directory`. Neither CUDA translation unit has been compiled or run here.

```sh
python3 tests/session4_cpu_checks.py
```

**Passed**, using Apple clang 21.0.0 with
`-std=c++11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined`.
The test extracts the actual CPU and error routines from the assignment files.
It checks a rectangular GER with a hand-computed answer and zero alpha, two
distinct noncommuting GEMM products, left/right identity and zero products for
every `n=1..32` with three batches, known error values, zero norms, NaN, and
infinity. This checks the references and metrics; it does **not** execute the
CUDA kernels, cuBLAS, or any GPU synchronization.

## Run and preserve the Colab results

1. After the branch is pushed, open
   [the notebook in Colab](https://colab.research.google.com/github/mbzuaiciao/akt26_gpu_dev/blob/session-4-assignments/session4_gpu_validation.ipynb).
2. Select **Runtime > Change runtime type > GPU**. Use a fresh runtime, then
   **Runtime > Run all**. All code cells in the committed notebook are unexecuted.
3. The notebook clones the branch and records the exact commit SHA. For an exact
   rerun, set `REVISION` to that SHA in a fresh runtime. The notebook checks for
   `nvidia-smi` and `nvcc` and detects the assigned compute capability.
4. It builds both supplied Makefiles, runs correctness and invalid-input checks,
   and runs Compute Sanitizer memcheck, racecheck, and synccheck if available.
   Absence of Compute Sanitizer is recorded as a skipped check.
5. Benchmarks cover GER `256x256`, `1024x1024`, `4000x2000`, and GEMM
   `(n,batch)=(8,1000),(16,1000),(32,1000),(32,10000)`. Each case runs three
   independent trials with 100 timed repetitions per trial. Correctness also
   covers GER partial blocks and GEMM every supported n with batches 1 and 3.
6. Download the results ZIP in the last cell, including after failures. Also
   **File > Save a copy in Drive** or download the executed `.ipynb` to retain
   outputs. Colab's `/content` disappears when the runtime is recycled.
7. Review `correctness-passed.txt`, `benchmarks-completed.txt`, sanitizer logs,
   `benchmarks.csv`, and every raw output. Add the ZIP/logs to this branch and
   update this walkthrough with actual findings before submission.

The ZIP records GPU name and driver (`nvidia-smi`), CUDA compiler/runtime,
host compiler, cuBLAS version, full build flags/architecture, exact source
commit, launch dimensions, errors, and per-trial timings. CSV timings are in
microseconds and match the printed precision; raw logs retain speedups.
No Python numerical or plotting packages are required.

Equivalent commands on an allocated NVIDIA system, from this repository root:

```sh
nvidia-smi
nvcc --version
c++ --version
# Example for a GPU with compute capability 7.5; use the actual assigned value.
# Set CUDADIR if the toolkit is not at /usr/local/cuda.
make -C 3-ger sm=75
make -C 4-gemm sm=75
./3-ger/ger 33 9 100
./3-ger/ger 4000 2000 100
./4-gemm/gemm 1 1
./4-gemm/gemm 31 3
./4-gemm/gemm 32 1000
compute-sanitizer --tool memcheck --error-exitcode 99 ./3-ger/ger 33 9 2
compute-sanitizer --tool racecheck --error-exitcode 99 ./4-gemm/gemm 31 3
```

Both Makefiles use `-O3 -std=c++11 -lineinfo`, the selected native architecture,
and forward-compatible PTX. CUDA events measure default-stream device work,
excluding allocations and transfers. Warmup is untimed. Divide event milliseconds
by repeats, then multiply by 1000 for microseconds. GFLOP/s is
`FLOPs/(microseconds*1000)`; speedup is `cuBLAS_us/custom_us`.
For tiny jobs, launch gaps can contribute to event timing; record all trials
and compare medians on the same GPU rather than extrapolating across machines.

There are **no measured performance conclusions yet**. Expected behavior to
test: GER has little arithmetic per memory access, while GEMM reuses shared
inputs; small matrices may spend much of their time on launch overhead.
These are hypotheses, not benchmark findings. cuBLAS uses tuned implementations;
this straightforward teaching kernel has no guaranteed speed advantage.

## Requirement and submission checklist

| PDF requirement | Implementation/evidence | Remaining work |
| --- | --- | --- |
| Slide 29: column-major GER kernel and launch | `3-ger/ger.cu`, 32x8 blocks, CPU/SGER checks | Compile and run on NVIDIA GPU |
| Slide 31: packed batched GEMM, n<=32 | `4-gemm/gemm.cu`, one block per matrix | Compile and run on NVIDIA GPU |
| Slide 31: memory, copies, kernel, cuBLAS, timing, accuracy | All starter placeholders filled; cleanup and API checks included | Execute notebook and review actual errors/timings |
| Slide 33: preserve directories and main | Both remain in original locations; exercises 0-2 unchanged | Review PR |
| Slide 33: public fork, code compiles/runs | Fork confirmed public; local CPU checks pass | GPU build/run is mandatory before submission |
| Slide 33: submission procedure TBA | No submission performed | Obtain current organizer instructions |

The PDF has no assignment-specific fixed size sweep, plotting deliverable, or
speed target. Its AXPY performance-graph practice is for the ungraded example,
which is left untouched. Slide 6 says one week after GPU access; the Session 4
introduction says October 30, 2026 at 23:59 Anywhere on Earth. Confirm the
current deadline and submission mechanism with the organizers.

Before hackathon submission: execute the Colab notebook, fix any failures,
preserve and review the real results, update this document, review the PR,
and follow the organizers' current procedure with the public repository URL
and Team 1142 identification. The PDF requires participants to explain every
line, including during a possible demo. Do not treat this unvalidated branch
as submission-ready. The PR is for team review; it must not be merged or
submitted automatically.
