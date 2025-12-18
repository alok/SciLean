#!/usr/bin/env python3
"""
MLX MNIST Benchmark for comparison with SciLean GpuMNIST

MLX is Apple's ML framework optimized for Apple Silicon.
Same architecture as GpuMNIST.lean:
- 2-layer MLP: 784 → 128 (GELU) → 10 (softmax)
- SGD optimizer
- Cross-entropy loss

Install: pip install mlx mlx-data
Run: python3 mlx_mnist_benchmark.py
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Try to import mlx-data for MNIST loading
try:
    import mlx.data as dx
    HAS_MLX_DATA = True
except ImportError:
    HAS_MLX_DATA = False

# Configuration (matching GpuMNIST.lean)
NUM_TRAIN = 60000
MINI_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1000
EPOCHS = 10
BASE_LR = 0.5
LR = BASE_LR / MINI_BATCH_SIZE  # ~0.00195


class MLP(nn.Module):
    """2-layer MLP matching GpuMNIST architecture."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = x.reshape(-1, 784)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


def loss_fn(model, x, y):
    """Cross-entropy loss."""
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def load_mnist_raw():
    """Load MNIST from raw files (same as SciLean uses)."""
    import struct
    import os

    # Try multiple paths
    paths = [
        "data",
        "examples/data",
        "../data",
        "data/MNIST/raw",
        "examples/data/MNIST/raw",
    ]

    train_images_path = None
    for base in paths:
        p = os.path.join(base, "train-images-idx3-ubyte")
        if os.path.exists(p):
            train_images_path = p
            train_labels_path = os.path.join(base, "train-labels-idx1-ubyte")
            test_images_path = os.path.join(base, "t10k-images-idx3-ubyte")
            test_labels_path = os.path.join(base, "t10k-labels-idx1-ubyte")
            break

    if train_images_path is None:
        raise FileNotFoundError("MNIST data not found. Run GpuMNIST first to download.")

    def read_images(path):
        with open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows * cols).astype(np.float32) / 255.0

    def read_labels(path):
        with open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

    train_images = mx.array(read_images(train_images_path))
    train_labels = mx.array(read_labels(train_labels_path))
    test_images = mx.array(read_images(test_images_path))
    test_labels = mx.array(read_labels(test_labels_path))

    return train_images, train_labels, test_images, test_labels


def train_epoch(model, optimizer, train_images, train_labels):
    """Train one epoch and return samples/sec."""
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    n = train_images.shape[0]
    indices = mx.array(np.random.permutation(n))

    start = time.time()
    samples = 0

    for i in range(0, n, MINI_BATCH_SIZE):
        batch_indices = indices[i:i + MINI_BATCH_SIZE]
        x = train_images[batch_indices]
        y = train_labels[batch_indices]

        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())  # Force evaluation

        samples += x.shape[0]
        if samples >= NUM_TRAIN:
            break

    elapsed = time.time() - start
    return samples / elapsed


def evaluate(model, test_images, test_labels):
    """Evaluate and return accuracy."""
    logits = model(test_images)
    predictions = mx.argmax(logits, axis=1)
    correct = mx.sum(predictions == test_labels).item()
    return correct / test_labels.shape[0]


def main():
    print("MLX MNIST Benchmark")
    print("=" * 40)
    print(f"MLX backend: {mx.default_device()}")
    print(f"Batch size: {MINI_BATCH_SIZE}")
    print(f"Learning rate: {LR:.6f}")
    print(f"Epochs: {EPOCHS}")

    # Load data
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist_raw()
    print(f"Loaded {train_images.shape[0]} training, {test_images.shape[0]} test samples")

    # Model and optimizer
    model = MLP()
    optimizer = optim.SGD(learning_rate=LR)

    # Count parameters
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            else:
                total += v.size
        return total
    num_params = count_params(model.parameters())
    print(f"Parameters: {num_params:,}")

    # Warmup
    print("\nWarming up...")
    _ = train_epoch(model, optimizer, train_images, train_labels)
    model = MLP()  # Reset
    optimizer = optim.SGD(learning_rate=LR)

    # Training
    print("\nTraining:")
    print("-" * 40)
    total_samples = 0
    total_time = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        samples_per_sec = train_epoch(model, optimizer, train_images, train_labels)
        elapsed = time.time() - start

        total_samples += NUM_TRAIN
        total_time += elapsed

        # Evaluate
        accuracy = evaluate(model, test_images, test_labels)

        print(f"Epoch {epoch:2d}: {samples_per_sec:,.0f} samples/sec, "
              f"accuracy: {accuracy*100:.2f}%, time: {elapsed:.2f}s")

    # Summary
    print("\n" + "=" * 40)
    print("Summary:")
    avg_samples_per_sec = total_samples / total_time
    print(f"  Average throughput: {avg_samples_per_sec:,.0f} samples/sec")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
