# Experiment Log

## 2024-12-21: GPU Tensor System Complete

**Commit:** e844c7b1  
**Branch:** metal-backend

### Summary
Completed layout-aware GPU tensor system with O(1) view operations.

### Key Results

| Metric | Value |
|--------|-------|
| PlainDataType derive overhead | 0% (zero overhead) |
| Nested POD throughput | 13+ GB/s |
| Layout tests passed | 18/18 |
| GPU tests passed | 11/11 |
| GEMM backward | O(1) transpose views working |

### Commands
```bash
lake build GpuTensorTest && .lake/build/bin/GpuTensorTest
lake build StructVsTupleBenchmark && .lake/build/bin/StructVsTupleBenchmark
lake build NestedPodStressTest && .lake/build/bin/NestedPodStressTest
```

### Files Changed
- `SciLean/Data/Tensor/GpuTensor.lean` - Type-safe strided tensor
- `SciLean/FFI/Metal/GpuBufferView.lean` - GPU buffer with layout
- `SciLean/Data/Tensor/Layout.lean` - O(1) view operations
- `SciLean/AD/TensorRevFDeriv.lean` - GEMM backward autodiff
- `SciLean/Data/DataArray/DerivePlainDataTypeFast.lean` - Zero-overhead derive

### Notes
- Renamed Strided* → GpuTensor/GpuBufferView for cleaner API
- All tensors now carry layout metadata from creation
- GEMM backward uses O(1) transpose views (no data copy)

## 2025-12-21: GPU Benchmarks (GpuTensor, GEMM Views, MNIST)

**Timestamp:** 2025-12-21 20:07:20 -0800  
**Commit:** dabf641c5ad93fea2e68d73ff0e2062023ca2ead  
**Branch:** metal-backend  
**Worktree:** dirty (doc comment fixes + benchmark build fixes)

### Commands
```bash
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/GemmViewBenchmark
.lake/build/bin/GpuMNIST
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=2.15104ms, 512=1.19882ms, 1024=2.53737ms
- Chained GEMM: 256=2.80291ms, 512=1.31492ms
- Transfer overhead (upload+download): 256KB=0.03587ms, 1MB=0.09666ms, 4MB=0.71904ms

**GemmViewBenchmark**
- 256: ~0.100 TFLOPs/s (0.030 GFLOPs, baseline 343.6us)
- 512: ~0.130 TFLOPs/s (0.270 GFLOPs, baseline 2.123737ms)
- 1024: ~1.720 TFLOPs/s (2.150 GFLOPs, baseline 1.249760ms)
- 2048: ~3.630 TFLOPs/s (17.180 GFLOPs, baseline 4.734677ms)
- O(1) transpose overhead: 29–37ns; dispatch checks: isContiguous 30ns, isTransposed 29ns

**GpuMNIST (Metal)**
- Dataset: 60,000 train samples, minibatch=256
- Initial accuracy: 13.1%
- Final accuracy (10 epochs): 84.0%
- Epoch times: 567–1391ms (per 234 batches)

### Notes
- GPU available; explicit CPU↔GPU transfers only at start/end for GpuTensor runs
- GEMM view dispatch overhead remains in nanoseconds
