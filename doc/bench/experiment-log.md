# Experiment Log

## 2025-12-19 15:25:21 PST

Commit: 343b9e8b

### GEMM view benchmark (Metal)

Command: `./.lake/build/bin/GemmViewBenchmark`

Notes:
- O(1) transpose and dispatch overheads are in the tens of ns.
- Reported performance is based on direct GEMM timing.

Results (avg time):

Size 256x256x256
- Direct: 681.272 us
- Contiguous view: 229.020 us
- gemmTN (A^T @ B): 268.268 us
- gemmNT (A @ B^T): 541.410 us
- A^T view: 313.204 us
- B^T view: 449.847 us
- Backward (dA + dB): 881.810 us
- FLOPs: 0.03 GFLOPs, perf ~0.05 TFLOPs/s

Size 512x512x512
- Direct: 637.066 us
- Contiguous view: 560.635 us
- gemmTN (A^T @ B): 255.331 us
- gemmNT (A @ B^T): 525.914 us
- A^T view: 251.845 us
- B^T view: 414.564 us
- Backward (dA + dB): 844.931 us
- FLOPs: 0.27 GFLOPs, perf ~0.42 TFLOPs/s

Size 1024x1024x1024
- Direct: 804.568 us
- Contiguous view: 1.025277 ms
- gemmTN (A^T @ B): 742.908 us
- gemmNT (A @ B^T): 708.114 us
- A^T view: 711.764 us
- B^T view: 786.183 us
- Backward (dA + dB): 1.288012 ms
- FLOPs: 2.15 GFLOPs, perf ~2.67 TFLOPs/s

Size 2048x2048x2048
- Direct: 2.807731 ms
- Contiguous view: 2.399035 ms
- gemmTN (A^T @ B): 1.788983 ms
- gemmNT (A @ B^T): 2.267847 ms
- A^T view: 2.232402 ms
- B^T view: 2.381424 ms
- Backward (dA + dB): 4.166752 ms
- FLOPs: 17.18 GFLOPs, perf ~6.12 TFLOPs/s

Transpose overhead (metadata only):
- 256x256: 17 ns
- 512x512: 17 ns
- 1024x1024: 19 ns
- 2048x2048: 17 ns
- 4096x4096: 18 ns

Dispatch overhead:
- isContiguous: 18 ns
- isTransposed: 17 ns

### MNIST GPU training (Metal)

Command: `./.lake/build/bin/GpuMNIST`

Notes:
- Dataset: 60000 samples, minibatch 256, 10 epochs.
- Metal GPU available.

Results:
- Initial accuracy: 8.6%
- Epoch 1: 92.0%, 298 ms
- Epoch 2: 95.0%, 237 ms
- Epoch 3: 95.9%, 223 ms
- Epoch 4: 96.6%, 210 ms
- Epoch 5: 96.9%, 231 ms
- Epoch 6: 97.1%, 206 ms
- Epoch 7: 97.5%, 210 ms
- Epoch 8: 97.7%, 217 ms
- Epoch 9: 98.0%, 217 ms
- Epoch 10: 98.1%, 236 ms
- Final accuracy: 98.1%
