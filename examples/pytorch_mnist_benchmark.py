#!/usr/bin/env python3
"""
PyTorch MNIST Benchmark for comparison with SciLean GpuMNIST

Same architecture as GpuMNIST.lean:
- 2-layer MLP: 784 → 128 (GELU) → 10 (softmax)
- SGD optimizer
- Cross-entropy loss
- Same hyperparameters

Run: python3 pytorch_mnist_benchmark.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

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

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train one epoch and return samples/sec."""
    model.train()
    start = time.time()
    samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        samples += data.size(0)

        if samples >= NUM_TRAIN:
            break

    elapsed = time.time() - start
    return samples / elapsed


def evaluate(model, device, test_loader):
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    return correct / total


def main():
    print("PyTorch MNIST Benchmark")
    print("=" * 40)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print(f"Batch size: {MINI_BATCH_SIZE}")
    print(f"Learning rate: {LR:.6f}")
    print(f"Epochs: {EPOCHS}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINI_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    # Model and optimizer
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Warmup
    print("\nWarming up...")
    _ = train_epoch(model, device, train_loader, optimizer, 0)
    model = MLP().to(device)  # Reset
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Training
    print("\nTraining:")
    print("-" * 40)
    total_samples = 0
    total_time = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        samples_per_sec = train_epoch(model, device, train_loader, optimizer, epoch)
        elapsed = time.time() - start

        total_samples += NUM_TRAIN
        total_time += elapsed

        # Evaluate
        accuracy = evaluate(model, device, test_loader)

        print(f"Epoch {epoch:2d}: {samples_per_sec:,.0f} samples/sec, "
              f"accuracy: {accuracy*100:.2f}%, time: {elapsed:.2f}s")

    # Summary
    print("\n" + "=" * 40)
    print("Summary:")
    avg_samples_per_sec = total_samples / total_time
    print(f"  Average throughput: {avg_samples_per_sec:,.0f} samples/sec")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final accuracy: {accuracy*100:.2f}%")

    # Estimate TFLOP/s for GEMM
    # Per sample:
    #   Layer 1: 784 * 128 * 2 = 200,704 FLOPs
    #   Layer 2: 128 * 10 * 2 = 2,560 FLOPs
    #   Backward: ~2x forward
    # Total: ~406,528 * 3 = ~1.2M FLOPs/sample
    flops_per_sample = 784 * 128 * 2 + 128 * 10 * 2  # Forward only
    flops_per_sample *= 3  # Forward + backward + gradients
    total_flops = flops_per_sample * avg_samples_per_sec
    print(f"  Estimated throughput: {total_flops / 1e9:.2f} GFLOPs/s")


if __name__ == "__main__":
    main()
