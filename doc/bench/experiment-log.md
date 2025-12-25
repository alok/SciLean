# Experiment Log

## 2025-12-25: GEMM focus sweep (M4Pro guard vs MPS vs Accelerate)

**Timestamp:** 2025-12-25 01:27:46 -0800  
**Commit:** 5160bf6b  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251225-012746

### Commands
```bash
lake build GEMMFocus
.lake/build/bin/GEMMFocus
```

### Key Results
- 2048×2048: MPS 8.9925ms (1.910 TFLOP/s), Accelerate 12.160ms (1.413 TFLOP/s)
- 4096×4096: MPS 50.2285ms (2.736 TFLOP/s), Accelerate 77.857ms (1.765 TFLOP/s)
- M4Pro raw/guard timings are sub-us; see run log for details

## 2025-12-25: M4 kernel edge-case fix (fallback)

**Timestamp:** 2025-12-25 01:02:04 -0800  
**Commit:** ec77ec9a  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251225-010107

### Commands
```bash
lake build GEMMCorrectness
.lake/build/bin/GEMMCorrectness
```

### Key Results
- M4Pro correctness now matches MPS/Accelerate at 4×4 and 8×8.
- Guarded M4/M4Pro/M4Max kernels now fall back to {name}`gemmAuto` when alignment rules are not met.

## 2025-12-25: Transfer timing fix + M4Pro size guard

**Timestamp:** 2025-12-25 00:36:38 -0800  
**Commit:** 0cd6fa5d  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251225-003553

### Commands
```bash
lake build GpuTensorBenchmark GEMMCorrectness
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/GEMMCorrectness
uv run scripts/bench_regression.py --run-dir doc/bench/runs/20251225-003553
```

### Key Results

**GpuTensor transfer (averaged)**
- 256KB: 0.00373ms upload, 0.00396ms download, 0.00770ms total
- 1MB:   0.01318ms upload, 0.01442ms download, 0.02761ms total
- 4MB:   0.06881ms upload, 0.06261ms download, 0.13143ms total

**GEMMCorrectness**
- M4Pro skipped for 4×4 and 8×8 (requires multiple of 64)
- M4Pro correct at 64×64; MPS and Accelerate correct at all tested sizes

### Regression Check
- `gpu_tensor.transfer_total.512` regression cleared (now −71.4% vs baseline)

## 2025-12-24: Full benchmark sweep (Lean + Python + Graph4)

**Timestamp:** 2025-12-24 23:59:56 -0800  
**Commit:** b000b66d  
**Branch:** metal-backend  
**Worktree:** dirty (bench outputs + many uncommitted files)  
**Run dir:** doc/bench/runs/20251224-234401

### Commands
```bash
lake build BackendBenchmark MetalBenchmark GEMMBenchmark KernelGEMMBenchmark \
  Float32Benchmark RandBenchmark MetalMinimalBenchmark GEMMComparison \
  GEMMFocus GEMMCorrectness GpuTensorBenchmark KernelBenchmark \
  GemmViewBenchmark GpuBatchingBenchmark GpuMNIST AMXBenchmark \
  PodBenchmark StructVsTupleBenchmark LargeGEMM OverheadTest NestedPodStressTest

# Executables (see run dir for full list of outputs)
.lake/build/bin/BackendBenchmark
.lake/build/bin/MetalBenchmark
.lake/build/bin/GEMMBenchmark
.lake/build/bin/KernelGEMMBenchmark
.lake/build/bin/Float32Benchmark
.lake/build/bin/RandBenchmark
.lake/build/bin/MetalMinimalBenchmark
.lake/build/bin/GEMMComparison
.lake/build/bin/GEMMFocus
.lake/build/bin/GEMMCorrectness
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/KernelBenchmark
.lake/build/bin/GemmViewBenchmark
.lake/build/bin/GpuBatchingBenchmark
.lake/build/bin/GpuMNIST
.lake/build/bin/AMXBenchmark
.lake/build/bin/PodBenchmark
.lake/build/bin/StructVsTupleBenchmark
.lake/build/bin/NestedPodStressTest
.lake/build/bin/LargeGEMM
.lake/build/bin/OverheadTest

uv run benchmarks/compare_frameworks.py
uv run benchmarks/mlx_pytorch_comparison.py
uv run benchmarks/conv2d_comparison.py
lake build Conv2DTest && uv run benchmarks/conv2d_comparison.py
uv run scripts/bench_regression.py --run-dir doc/bench/runs/20251224-234401

cd ~/graph4 && ./.lake/build/bin/standalone_bench
```

### Key Results (selected)

**GpuTensorBenchmark**
- Single GEMM: 256=0.358ms, 512=0.348ms, 1024=1.261ms
- Chained GEMM: 256=0.597ms, 512=0.566ms
- Transfer total: 256KB=0.0116ms, 1MB=0.145ms, 4MB=0.4625ms

**GemmViewBenchmark**
- Baseline gemm: 256=0.264ms, 512=0.577ms, 1024=1.351ms, 2048=2.666ms
- O(1) transpose overhead: 18–39ns; dispatch checks 16–22ns

**GpuMNIST (Metal)**
- Final accuracy: 98.3%
- Epoch times: 319–536ms (avg 473.8ms)

**AMXBenchmark**
- Layer1 fwd (256×784 @ 784×128): AMX 917 GFLOP/s vs MPS 126 GFLOP/s
- Layer2 fwd (256×128 @ 128×10): AMX 113 GFLOP/s vs MPS 2.15 GFLOP/s

**Graph4 standalone (1M cycle)**
- CPU SpMV: 43.488s total, 0.00460 GFLOP/s
- GPU CSR SpMV: 73ms total, 2.74 GFLOP/s

**Conv2D (SciLean vs MLX/PyTorch)**
- SciLean fast kernel: 28×28 x32→64 = 0.233ms (124 GFLOP/s)
- SciLean fast kernel: 224×224 x3→64 = 1.586ms (109 GFLOP/s)

**GEMMCorrectness**
- M4Pro failed for 4×4 and 8×8; correct at 64×64

### Regression Check (scripts/bench_regression.py)
- One warning: `gpu_tensor.transfer_total.512` regressed +50.1% (1MB total 0.145ms vs 0.0967ms baseline).
- All other tracked metrics improved or matched baseline.

## 2025-12-24: GpuMNIST (Metal) run

**Timestamp:** 2025-12-24 00:36:40 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251224-003640

### Commands
```bash
lake build GpuMNIST
.lake/build/bin/GpuMNIST
```

### Key Results
- Initial accuracy: 6.2%
- Final accuracy: 97.8%
- Epoch times: 165–320ms (avg 254.9ms)

## 2025-12-23: GpuTensor + GemmView + GpuMNIST (Metal)

**Timestamp:** 2025-12-23 23:51:11 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251223-235111

### Commands
```bash
lake build GpuTensorBenchmark GemmViewBenchmark GpuMNIST
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/GemmViewBenchmark
.lake/build/bin/GpuMNIST
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM: 256=0.297ms, 512=0.308ms, 1024=0.798ms
- Chained GEMM: 256=0.402ms, 512=0.560ms
- Transfer total: 256KB=0.0123ms, 1MB=0.0373ms, 4MB=0.158ms

**GemmViewBenchmark**
- Baseline gemm: 256=0.264ms, 512=0.511ms, 1024=0.599ms, 2048=1.702ms
- Throughput: 256=0.13 TFLOP/s, 512=0.53 TFLOP/s, 1024=3.59 TFLOP/s, 2048=10.09 TFLOP/s

**GpuMNIST (Metal)**
- Initial accuracy: 3.7%
- Output truncated before final accuracy (see run log)

## 2025-12-23: GpuTensor + GemmView (early run)

**Timestamp:** 2025-12-23 23:05:18 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (many local changes)  
**Run dir:** doc/bench/runs/20251223-230518

### Commands
```bash
lake build GpuTensorBenchmark GemmViewBenchmark
.lake/build/bin/GpuTensorBenchmark
.lake/build/bin/GemmViewBenchmark
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM: 256=243.725ms, 512=191.976ms, 1024=125.846ms
- Chained GEMM: 256=316.615ms, 512=57.972ms
- Transfer total: 256KB=0.01387ms, 1MB=0.03562ms, 4MB=0.38537ms

**GemmViewBenchmark**
- Output file empty (`GemmViewBenchmark.txt` is 0 bytes)

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
- `SciLean/Data/Tensor/GpuTensor.lean` - Type-safe layout-aware tensor
- `SciLean/FFI/Metal/GpuBufferView.lean` - GPU buffer with layout
- `SciLean/Data/Tensor/Layout.lean` - O(1) view operations
- `SciLean/AD/TensorRevFDeriv.lean` - GEMM backward autodiff
- `SciLean/Data/DataArray/DerivePlainDataTypeFast.lean` - Zero-overhead derive

### Notes
- Renamed legacy tensor types → GpuTensor/GpuBufferView for cleaner API
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

## 2025-12-23: GPU Benchmarks (GpuTensor only; GemmView/GpuMNIST build stalled)

**Timestamp:** 2025-12-23 23:31:23 -0800  
**Commit:** f907ed01508928d938e8384197204c4b591aaddb  
**Branch:** metal-backend  
**Worktree:** dirty (new bench regression script + metrics/logs)
**Run dir:** doc/bench/runs/20251223-230518

### Commands
```bash
.lake/build/bin/GpuTensorBenchmark | tee doc/bench/runs/20251223-230518/GpuTensorBenchmark.txt
# GemmViewBenchmark + GpuMNIST not run (see notes)
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=243.725ms, 512=191.976ms, 1024=125.846ms
- Chained GEMM: 256=316.615ms, 512=57.9716ms
- Transfer overhead (upload+download): 256KB=0.01387ms, 1MB=0.03562ms, 4MB=0.38537ms

### Comparison vs 2025-12-21 baseline
- GEMM and chained GEMM times regressed by ~48x–159x (see doc/bench/runs/20251223-230518/metrics.json).
- Transfer totals improved by ~46–63%.

### Notes
- Repeated `lake build GemmViewBenchmark` stalled at `SciLean.Analysis.Calculus.Monad.HasRevFDerivMonad` (and earlier at `SciLean.Tactic.DataSynth.Main`); timed out and aborted.
- GpuMNIST not built/run due to the same build stall.
- The GEMM regression is suspiciously large; likely wrong code path or sync behavior. Needs investigation before trusting these numbers.

## 2025-12-23: GPU Benchmarks (GpuTensor after MPS GEMM cache)

**Timestamp:** 2025-12-23 23:51:33 -0800  
**Commit:** f907ed01508928d938e8384197204c4b591aaddb  
**Branch:** metal-backend  
**Worktree:** dirty (MPS GEMM cache + bench artifacts)  
**Run dir:** doc/bench/runs/20251223-235111

### Commands
```bash
.lake/build/bin/GpuTensorBenchmark | tee doc/bench/runs/20251223-235111/GpuTensorBenchmark.txt
scripts/bench_regression.py --run-dir doc/bench/runs/20251223-235111
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=0.29718ms, 512=0.30848ms, 1024=0.79793ms
- Chained GEMM: 256=0.40202ms, 512=0.55973ms
- Transfer overhead (upload+download): 256KB=0.01233ms, 1MB=0.03729ms, 4MB=0.15824ms

### Comparison vs 2025-12-21 baseline
- GEMM and chained GEMM improved by ~57–86%.
- Transfer totals improved by ~61–78%.

### Notes
- Fix: cache MPSMatrixMultiplication objects in `Metal/metal_backend.mm` to avoid per-call kernel creation overhead.
- The previous 2025-12-23 regression run was an artifact of uncached MPS kernel creation.

## 2025-12-23: GPU Benchmarks (GpuTensor + GemmView; GpuMNIST crash)

**Timestamp:** 2025-12-23 23:55:00 -0800  
**Commit:** f907ed01508928d938e8384197204c4b591aaddb  
**Branch:** metal-backend  
**Worktree:** dirty (MPS GEMM cache + bench artifacts)  
**Run dir:** doc/bench/runs/20251223-235111

### Commands
```bash
.lake/build/bin/GpuTensorBenchmark | tee doc/bench/runs/20251223-235111/GpuTensorBenchmark.txt
.lake/build/bin/GemmViewBenchmark | tee doc/bench/runs/20251223-235111/GemmViewBenchmark.txt
stdbuf -oL -eL .lake/build/bin/GpuMNIST 2>&1 | tee doc/bench/runs/20251223-235111/GpuMNIST.txt
scripts/bench_regression.py --run-dir doc/bench/runs/20251223-235111
```

### Key Results

**GpuTensorBenchmark**
- Single GEMM (GpuTensor): 256=0.29718ms, 512=0.30848ms, 1024=0.79793ms
- Chained GEMM: 256=0.40202ms, 512=0.55973ms
- Transfer overhead (upload+download): 256KB=0.01233ms, 1MB=0.03729ms, 4MB=0.15824ms

**GemmViewBenchmark**
- Baseline direct GEMM: 256=263.522us, 512=510.914us, 1024=598.718us, 2048=1.702339ms
- Performance: 256=0.13 TFLOPs/s, 512=0.53 TFLOPs/s, 1024=3.59 TFLOPs/s, 2048=10.09 TFLOPs/s
- O(1) transpose overhead: 15–17ns; dispatch checks: isContiguous 17ns, isTransposed 18ns

**GpuMNIST (Metal)**
- Run aborted after printing `Training (60000 samples, mini-batch=256)...` (no epoch logs; exit code 1).

### Notes
- Fix: cached MPSMatrixMultiplication objects in `Metal/metal_backend.mm` to remove per-call kernel creation overhead.
- GpuMNIST crash appears after training start; no exception message observed. Needs follow-up.

## 2025-12-24: GpuMNIST (Metal) crash fix verification

**Timestamp:** 2025-12-24 00:33:01 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (local changes)

### Commands
```bash
MTL_DEBUG_LAYER=1 METAL_DEVICE_WRAPPER_TYPE=1 stdbuf -oL -eL ./.lake/build/bin/GpuMNIST 2>&1
```

### Key Results

**GpuMNIST (Metal)**
- Dataset: 60,000 train samples, minibatch=256
- Initial accuracy: 6.6%
- Final accuracy (10 epochs): 98.3%
- Epoch times: 263–537ms (per 234 batches)

### Notes
- Fixes: aligned softmax threadgroup memory to 16B; `copyLayout` now reads `Array Nat` safely (prevents huge buffer allocations on sliced batches).

## 2025-12-24: GpuMNIST (Metal) run log (no validation)

**Timestamp:** 2025-12-24 00:37:15 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (local changes)

### Commands
```bash
stdbuf -oL -eL ./.lake/build/bin/GpuMNIST | tee doc/bench/runs/20251224-003640/GpuMNIST.txt
```

### Key Results

**GpuMNIST (Metal)**
- Run log: `doc/bench/runs/20251224-003640/GpuMNIST.txt`
- Initial accuracy: 6.2%
- Final accuracy (10 epochs): 97.8%
- Epoch times: 165–320ms (per 234 batches)

### Notes
- Baseline run for post-fix performance tracking without Metal validation.

## 2025-12-24: GemmViewBenchmark (layout-aware)

**Timestamp:** 2025-12-24 16:30:03 -0800  
**Commit:** f907ed01  
**Branch:** metal-backend  
**Worktree:** dirty (local changes)

### Commands
```bash
./.lake/build/bin/GemmViewBenchmark
```

### Key Results

**GemmViewBenchmark**
- 256: baseline 442.564us, ~0.08 TFLOPs/s
- 512: baseline 502.814us, ~0.53 TFLOPs/s
- 1024: baseline 726.695us, ~2.96 TFLOPs/s
- 2048: baseline 2.352941ms, ~7.30 TFLOPs/s
- O(1) transpose overhead: 26–40ns
- Dispatch checks: isContiguous 25ns, isTransposed 30ns

### Notes
- Layout-aware GEMM path exercised for transpose views.

## 2025-12-24: GpuMNIST (Metal) run

**Timestamp:** 2025-12-24 16:31:46 -0800  
**Commit:** ce85eabd  
**Branch:** metal-backend  
**Worktree:** dirty (local changes)

### Commands
```bash
./.lake/build/bin/GpuMNIST | tee doc/bench/runs/20251224-163053/GpuMNIST.txt
```

### Key Results

**GpuMNIST (Metal)**
- Run log: `doc/bench/runs/20251224-163053/GpuMNIST.txt`
- Dataset: 60,000 train samples, minibatch=256
- Initial accuracy: 6.0%
- Final accuracy (10 epochs): 98.4%
- Epoch times: 308–480ms (per 234 batches)

### Notes
- Loading images took 19.23s; labels 30ms (see run log).

## 2025-12-24 17:50:17 -0800 — Graph4 standalone_bench (CSR SpMV)

**SciLean Commit:** 971855da  
**Graph4 Commit:** b0aefa2  
**SciLean Branch:** metal-backend  
**Graph4 Branch:** main  
**Worktree:** dirty (local changes)

### Commands
```bash
# from /Users/alokbeniwal/graph4
./.lake/build/bin/standalone_bench
```

### Key Results
- CPU SpMV (DataArray, 1M cycle, 100 iters): 40396ms, 0.004951 GFLOPS
- GPU CSR SpMV (Float32, 1M cycle, 100 iters): 61ms, 3.278689 GFLOPS
- GPU final value check: 1.000000
- Run log: `doc/bench/runs/20251224-175017/graph4_standalone_bench.txt`

### Notes
- Metal shaders loaded via Graph4 symlink `Metal -> ../SciLean/Metal` to expose `kmeans.metal` (contains `csr_spmv`).

## 2025-12-25 01:21:14 -0800 — GEMMFocus guard overhead (M4Pro raw vs guarded)

**SciLean Commit:** ed24b457  
**SciLean Branch:** metal-backend  
**Worktree:** dirty (GEMMFocus guard bench + untracked run log)

### Commands
```bash
./.lake/build/bin/GEMMFocus | tee doc/bench/runs/20251225-012114/gemm_focus.txt
```

### Key Results
- Run log: `doc/bench/runs/20251225-012114/gemm_focus.txt`
- Guard overhead (M4Pro vs M4ProRaw):
  - 128: +51.8%
  - 256: -19.0%
  - 512: -18.2%
  - 1024: -13.1%
  - 2048: +0.7%
  - 4096: +17.7%

### Notes
- Overhead swings indicate run-to-run noise; no stable penalty observed. The guard adds a single alignment check before dispatch.

## 2025-12-25 01:31:30 -0800 — GEMMFocus interleaved guard overhead (M4Pro raw vs guarded)

**SciLean Commit:** 8091f0c2  
**SciLean Branch:** metal-backend  
**Worktree:** dirty (new run log files)

### Commands
```bash
./.lake/build/bin/GEMMFocus | tee doc/bench/runs/20251225-013130/gemm_focus.txt
```

### Key Results
- Run log: `doc/bench/runs/20251225-013130/gemm_focus.txt`
- Guard overhead (M4Pro vs M4ProRaw, interleaved chunks, 2× iters, warmup 10):
  - 128: +15.6%
  - 256: +0.9%
  - 512: -0.0%
  - 1024: +2.2%
  - 2048: +4.6%
  - 4096: -0.4%

### Notes
- Interleaving is done by alternating single-iteration chunks to reduce drift.
