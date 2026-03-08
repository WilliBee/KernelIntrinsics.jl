# KernelIntrinsics.jl

!!! warning "Not affiliated with submodule KernelIntrinsics of KernelAbstractions.jl"
    Despite its name, this package is **not affiliated with or part of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)**. The name collision is a known issue. Currently, KernelIntrinsics.jl is primarily used as a building block for [KernelForge.jl](https://github.com/your-org/KernelForge.jl). A future goal is to upstream its functionality into KernelAbstractions.jl and [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl).

!!! warning "Low-level library — not for end users"
    This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly. KernelIntrinsics.jl is similar in scope to [GPUArraysCore.jl](https://github.com/JuliaGPU/GPUArrays.jl) — a building block for other packages.

!!! danger "Current Limitations"
    - No bounds checking for `vload`/`vstore!` on dense arrays — out-of-bounds access will cause undefined behavior
    - CUDA and ROCm backends supported; other backends planned

## Features

- **Memory Fences**: Explicit memory synchronization with `@fence` macro
- **Ordered Memory Access**: Acquire/release semantics with `@access` macro
- **Warp Operations**: Efficient intra-warp communication and reduction primitives
  - `@shfl`: Warp shuffle operations (Up, Down, Xor, Idx modes)
  - `@warpreduce`: Inclusive scan within a warp
  - `@warpfold`: Warp-wide reduction to a single value
- **Vectorized Memory Operations**: Hardware-accelerated vector loads and stores with `vload`, `vstore!`, `vload_multi`, and `vstore_multi!`. Array views (`SubArray`) are supported and fall back to scalar tuple operations automatically.

## Cross-Architecture Support

KernelIntrinsics.jl currently supports CUDA GPUs via the `CUDABackend` and AMD GPUs via the `ROCBackend`. The CUDA backend leverages PTX instructions for memory fences (`fence.acq_rel.gpu`), ordered memory access (`ld.acquire.gpu`, `st.release.gpu`), warp shuffle operations, and vectorized memory transactions. The ROCm backend provides equivalent primitives using AMDGPU-specific intrinsics.

The macro-based API is designed with portability in mind. Future releases may extend support to additional GPU backends (Intel oneAPI, Apple Metal). Contributions to enable further cross-platform support are welcome.

## Installation
```julia
using Pkg
Pkg.add("KernelIntrinsics")
```

## Quick Start
```julia
using KernelIntrinsics
using KernelAbstractions, CUDA

@kernel function example_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible to all threads
    @access Flag[1] = 1  # Release store
end

X = cu([1])
Flag = cu([0])
example_kernel(CUDABackend())(X, Flag; ndrange=1)
```

## Memory Ordering Semantics

- **@fence**: Full acquire-release fence across all device threads
- **@access Release**: Ensures prior writes are visible before the store
- **@access Acquire**: Ensures subsequent reads see prior writes
- **@access Acquire Device**: Explicitly specifies device scope (default)

## Performance Considerations

- Warp operations are most efficient when all threads in a warp participate
- Vectorized operations can significantly improve memory bandwidth utilization
- Use the minimum required memory ordering (acquire/release over fences when possible)
- Default warp size is 32 for CUDA and 64 for ROCm; operations assume full warp/wavefront participation
- `vload_multi`/`vstore_multi!` have a small runtime overhead for the alignment switch, but this is typically negligible compared to memory latency
- `vload`/`vstore!` on array views fall back to scalar tuple operations; performance-critical code should prefer contiguous arrays for vectorized instructions

## Implementation Notes

### Memory Ordering and Scopes

The implementation of memory fences, orderings, and scopes is inspired by [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl). This package includes tests demonstrating that these primitives work correctly on CUDA and ROCm, generating the expected backend-specific instructions.

### Warp Shuffle Operations

The warp shuffle implementation builds upon [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)'s approach but generalizes it to support **any concrete bitstype struct**, including nested and composite types, as well as `NTuple`s. The backend only needs to implement 32-bit shuffle operations; larger types are automatically decomposed. On ROCm, wavefront-level shuffle operations are used as the equivalent primitive.

### Vectorized Memory Access

Vectorized loads and stores (`vload`, `vstore!`, `vload_multi`, `vstore_multi!`) use LLVM intrinsic functions to generate efficient vector instructions (`ld.global.v4`, `st.global.v4` on CUDA, and their ROCm equivalents) on contiguous arrays. Array views (`SubArray`) are fully supported via an automatic fallback to scalar tuple operations, ensuring correctness at the cost of vectorization.

**Current limitations:**
- No bounds checking for `vload`/`vstore!` on dense arrays — out-of-bounds access will cause undefined behavior
- Future versions may add vectorized support for non-contiguous views

## Requirements

- Julia 1.10+
- KernelAbstractions.jl
- CUDA.jl (for CUDA backend)
- AMDGPU.jl (for ROCm backend)

## Contents
```@contents
Pages = ["api.md", "examples.md"]
Depth = 2
```

## License

MIT License