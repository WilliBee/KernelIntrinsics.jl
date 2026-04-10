# ============================================================================
# Metal Focused Tests: AIR Intrinsics Verification
# ============================================================================
# AIR Intrinsic Signature:
#   air.atomic.fence(i32 flags, i32 order, i32 scope)
#
# Parameters (discovered via preprocessor dump):
#   xcrun metal -x metal -E -dM /dev/null | grep METAL_MEMORY
#     #define __METAL_MEMORY_FLAGS_DEVICE__ 1
#     #define __METAL_MEMORY_FLAGS_NONE__ 0
#     #define __METAL_MEMORY_FLAGS_OBJECT_DATA__ 16
#     #define __METAL_MEMORY_FLAGS_TEXTURE__ 4
#     #define __METAL_MEMORY_FLAGS_THREADGROUP_IMAGEBLOCK__ 8
#     #define __METAL_MEMORY_FLAGS_THREADGROUP__ 2
#     #define __METAL_MEMORY_ORDER_ACQUIRE__ 2
#     #define __METAL_MEMORY_ORDER_ACQ_REL__ 4
#     #define __METAL_MEMORY_ORDER_RELAXED__ 0
#     #define __METAL_MEMORY_ORDER_RELEASE__ 3
#     #define __METAL_MEMORY_ORDER_SEQ_CST__ 5
#     #define __METAL_MEMORY_SCOPE_DEVICE__ 2
#     #define __METAL_MEMORY_SCOPE_SIMDGROUP__ 4
#     #define __METAL_MEMORY_SCOPE_THREADGROUP__ 1
#     #define __METAL_MEMORY_SCOPE_THREAD__ 0
# 
# Examples:
#   air.atomic.fence(i32 1, i32 3, i32 2)  # Device Release, device thread scope
#   air.atomic.fence(i32 2, i32 2, i32 1)  # Workgroup Acquire, threadgroup scope
#   air.atomic.fence(i32 1, i32 4, i32 2)  # Device AcqRel, device thread scope
# ============================================================================

using KernelAbstractions
using KernelIntrinsics
using Metal
using Test

# ── Store Tests ─────────────────────────────────────────────────────────────

@kernel function store_relaxed(data)
    @access Relaxed data[1] = 10
end

@kernel function store_release(data)
    @access Release data[1] = 20
end

@kernel function store_device_release(data)
    @access Device Release data[1] = 30
end

@kernel function store_workgroup_release(data)
    @access Workgroup Release data[1] = 40
end

# ── Load Tests ──────────────────────────────────────────────────────────────

@kernel function load_relaxed(data)
    a = @access Relaxed data[1]
end

@kernel function load_acquire(data)
    a = @access Acquire data[1]
end

@kernel function load_device_acquire(data)
    a = @access Device Acquire data[1]
end

@kernel function load_workgroup_acquire(data)
    a = @access Workgroup Acquire data[1]
end

# ── Fence Tests ─────────────────────────────────────────────────────────────

@kernel function fence_default(data)
    @fence  # Device AcqRel
end

@kernel function fence_device(data)
    @fence Device
end

@kernel function fence_workgroup_acqrel(data)
    @fence Workgroup AcqRel
end

@kernel function fence_acqrel_device(data)
    @fence AcqRel Device
end

# ── Multidimensional Tests ───────────────────────────────────────────────────

@kernel function test_multidim(data)
    @access data[2, 2] = 0x20
end

# ============================================================================
# Run Tests: Capture LLVM IR
# ============================================================================

data = MtlArray{Int32}(zeros(Int32, 16, 16))

# Capture LLVM IR for each kernel
buf1 = IOBuffer()
Metal.@device_code_llvm io = buf1 store_relaxed(MetalBackend())(data; ndrange=256)
asm_store_relaxed = String(take!(buf1))

buf2 = IOBuffer()
Metal.@device_code_llvm io = buf2 store_release(MetalBackend())(data; ndrange=256)
asm_store_release = String(take!(buf2))

buf3 = IOBuffer()
Metal.@device_code_llvm io = buf3 store_device_release(MetalBackend())(data; ndrange=256)
asm_store_device = String(take!(buf3))

buf4 = IOBuffer()
Metal.@device_code_llvm io = buf4 store_workgroup_release(MetalBackend())(data; ndrange=256)
asm_store_workgroup = String(take!(buf4))

buf5 = IOBuffer()
Metal.@device_code_llvm io = buf5 load_relaxed(MetalBackend())(data; ndrange=256)
asm_load_relaxed = String(take!(buf5))

buf6 = IOBuffer()
Metal.@device_code_llvm io = buf6 load_acquire(MetalBackend())(data; ndrange=256)
asm_load_acquire = String(take!(buf6))

buf7 = IOBuffer()
Metal.@device_code_llvm io = buf7 load_device_acquire(MetalBackend())(data; ndrange=256)
asm_load_device = String(take!(buf7))

buf8 = IOBuffer()
Metal.@device_code_llvm io = buf8 load_workgroup_acquire(MetalBackend())(data; ndrange=256)
asm_load_workgroup = String(take!(buf8))

buf9 = IOBuffer()
Metal.@device_code_llvm io = buf9 fence_default(MetalBackend())(data; ndrange=256)
asm_fence_default = String(take!(buf9))

buf10 = IOBuffer()
Metal.@device_code_llvm io = buf10 fence_device(MetalBackend())(data; ndrange=256)
asm_fence_device = String(take!(buf10))

buf11 = IOBuffer()
Metal.@device_code_llvm io = buf11 fence_workgroup_acqrel(MetalBackend())(data; ndrange=256)
asm_fence_workgroup = String(take!(buf11))

buf12 = IOBuffer()
Metal.@device_code_llvm io = buf12 fence_acqrel_device(MetalBackend())(data; ndrange=256)
asm_fence_reversed = String(take!(buf12))

buf13 = IOBuffer()
Metal.@device_code_llvm io = buf13 test_multidim(MetalBackend())(data; ndrange=256)
asm_multidim = String(take!(buf13))

# ============================================================================
# Tests
# ============================================================================

@testset "Metal access fences tests" begin
    # ============================================================================
    # Store
    # ============================================================================
    @testset "Metal Store: Relaxed" begin
        # Relaxed store: no fence, only atomic store
        @test occursin("air.atomic.global.store", asm_store_relaxed)
        @test !occursin(r"air\.atomic\.fence\(", asm_store_relaxed)  # Not a function call
    end

    @testset "Metal Store: Release" begin
        # Release store (default Device): flags=1, order=3 (Release), scope=2 (device thread scope)
        @test occursin("air.atomic.global.store", asm_store_release)
        @test occursin("air.atomic.fence(i32 1, i32 3, i32 2)", asm_store_release)
    end

    @testset "Metal Store: Device Scope" begin
        # Device release: flags=1, order=3 (Release), scope=2 (device thread scope)
        @test occursin("air.atomic.global.store", asm_store_device)
        @test occursin("air.atomic.fence(i32 1, i32 3, i32 2)", asm_store_device)
    end

    @testset "Metal Store: Workgroup Scope" begin
        # Workgroup release: flags=2, order=3 (Release), scope=1 (threadgroup thread scope)
        @test occursin("air.atomic.global.store", asm_store_workgroup)
        @test occursin("air.atomic.fence(i32 2, i32 3, i32 1)", asm_store_workgroup)
    end

    # ============================================================================
    # Load
    # ============================================================================

    @testset "Metal Load: Relaxed" begin
        # Relaxed load: no fence, only atomic load
        @test occursin("air.atomic.global.load", asm_load_relaxed)
        @test !occursin("call void @air.atomic.fence", asm_load_relaxed)  # Not a function call
    end

    @testset "Metal Load: Acquire" begin
        # Acquire load (default Device): flags=1, order=2 (Acquire), scope=2 (device thread scope)
        @test occursin("air.atomic.global.load", asm_load_acquire)
        @test occursin("air.atomic.fence(i32 1, i32 2, i32 2)", asm_load_acquire)
    end

    @testset "Metal Load: Device Scope" begin
        # Device acquire: flags=1, order=2 (Acquire), scope=2 (device thread scope)
        @test occursin("air.atomic.global.load", asm_load_device)
        @test occursin("air.atomic.fence(i32 1, i32 2, i32 2)", asm_load_device)
    end

    @testset "Metal Load: Workgroup Scope" begin
        # Workgroup acquire: flags=2, order=2 (Acquire), scope=1 (threadgroup thread scope)
        @test occursin("air.atomic.global.load", asm_load_workgroup)
        @test occursin("air.atomic.fence(i32 2, i32 2, i32 1)", asm_load_workgroup)
    end

    # ============================================================================
    # Fence
    # ============================================================================

    @testset "Metal Fence: Default" begin
        # Default fence (Device AcqRel): flags=1, order=4 (AcqRel), scope=2 (device thread scope)
        @test occursin("air.atomic.fence(i32 1, i32 4, i32 2)", asm_fence_default)
    end

    @testset "Metal Fence: Device" begin
        # Device fence: flags=1, order=4 (AcqRel), scope=2 (device thread scope)
        # Note: @fence Device defaults to AcqRel ordering
        @test occursin("air.atomic.fence(i32 1, i32 4, i32 2)", asm_fence_device)
    end

    @testset "Metal Fence: Workgroup" begin
        # Workgroup AcqRel fence: flags=2, order=4 (AcqRel), scope=1 (threadgroup scope)
        @test occursin("air.atomic.fence(i32 2, i32 4, i32 1)", asm_fence_workgroup)
    end

    @testset "Metal Fence: Reversed Args" begin
        # Reversed args (AcqRel Device): flags=1, order=4 (AcqRel), scope=2 (device thread scope)
        @test occursin("air.atomic.fence(i32 1, i32 4, i32 2)", asm_fence_reversed)
    end

    # ============================================================================
    # Multidimensional Array
    # ============================================================================

    @testset "Metal: Multidimensional Indexing" begin
        # Multidimensional array access (default @access uses Release for store)
        @test occursin("air.atomic.global.store", asm_multidim)
        @test occursin("air.atomic.fence(i32 1, i32 3, i32 2)", asm_multidim)
    end
end
