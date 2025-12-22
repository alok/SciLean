# Experiment Log

## 2025-12-21: GPU Benchmarks (current vs prev baseline)

**Timestamp:** 2025-12-21 20:41:03 -0800  
**Commit:** 974019f3  
**Branch:** metal-backend  
**Worktree:** dirty (benchmark metadata update, fmt fix, .gitignore + W&B runs)

### Commands
```bash
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/GemmViewBenchmark
.lake/build/bin/GpuMNIST
```

### Key Results (current)

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=0.29993ms, 512=0.31090ms, 1024=0.64929ms
- Chained GEMM: 256=0.47575ms, 512=0.51175ms
- Transfer overhead (upload+download): 256KB=0.01475ms, 1MB=0.04516ms, 4MB=0.12458ms

**GemmViewBenchmark**
- 256: ~0.160 TFLOPs/s (0.030 GFLOPs, baseline 216.158us)
- 512: ~0.880 TFLOPs/s (0.270 GFLOPs, baseline 304.487us)
- 1024: ~2.570 TFLOPs/s (2.150 GFLOPs, baseline 834.641us)
- 2048: ~5.480 TFLOPs/s (17.180 GFLOPs, baseline 3.133839ms)
- O(1) transpose overhead: 17–19ns; dispatch checks: isContiguous 16ns, isTransposed 17ns

**GpuMNIST (Metal)**
- Dataset: 60,000 train samples, minibatch=256
- Initial accuracy: 12.3%
- Final accuracy (10 epochs): 84.0%
- Epoch times: 225–391ms (per 234 batches)

### Baseline (prev worktree)

**Commit:** dabf641c  
**Worktree:** `/Users/alokbeniwal/SciLean-prev` (local patches to build: disabled `doc.verso`, fmt slice fix, subverso build artifacts copied from current)

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=0.27037ms, 512=0.42254ms, 1024=0.78016ms
- Chained GEMM: 256=0.44688ms, 512=0.59121ms
- Transfer overhead (upload+download): 256KB=0.01379ms, 1MB=0.04370ms, 4MB=0.12225ms

**GemmViewBenchmark**
- 256: ~0.150 TFLOPs/s (0.030 GFLOPs, baseline 228.164us)
- 512: ~0.360 TFLOPs/s (0.270 GFLOPs, baseline 751.162us)
- 1024: ~2.280 TFLOPs/s (2.150 GFLOPs, baseline 941.652us)
- 2048: ~5.010 TFLOPs/s (17.180 GFLOPs, baseline 3.428941ms)
- O(1) transpose overhead: 17ns; dispatch checks: isContiguous 17ns, isTransposed 17ns

**GpuMNIST (Metal)**
- Final accuracy (10 epochs): 83.3%
- Epoch times: 249–404ms (per 234 batches)

### Comparison
- No regression detected; current is faster or similar on most sizes.
- GEMM baseline times improved vs baseline: 256 (−5%), 512 (−60%), 1024 (−11%), 2048 (−9%).
- GpuTensor single GEMM: 256 slightly slower (~+11%), 512/1024 faster (~−26% / −17%).
- GpuTensor chain: 256 slightly slower (~+6%), 512 faster (~−13%).
- MNIST accuracy slightly higher (84.0% vs 83.3%) and epochs are modestly faster.

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
