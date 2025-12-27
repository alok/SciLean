#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def parse_time_to_ms(value: str) -> float:
    m = re.match(r"^\s*([0-9.]+)\s*(ns|us|ms|s)\s*$", value)
    if not m:
        raise ValueError(f"Unrecognized time: {value!r}")
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "ns":
        return num / 1_000_000.0
    if unit == "us":
        return num / 1_000.0
    if unit == "ms":
        return num
    if unit == "s":
        return num * 1000.0
    raise ValueError(f"Unknown unit: {unit}")


def parse_gpu_tensor(text: str) -> dict:
    metrics = {}
    section = None
    for line in text.splitlines():
        if line.startswith("--- Test 1:"):
            section = "gemm"
            continue
        if line.startswith("--- Test 2:"):
            section = "chain"
            continue
        if line.startswith("--- Test 3:"):
            section = "transfer"
            continue

        if section == "gemm":
            if "(no sync)" in line:
                m = re.match(r"^(\d+)x(\d+)\s*\|.*\|\s*([0-9.]+)ms", line)
                if m:
                    size = m.group(1)
                    metrics[f"gpu_tensor.gemm.{size}"] = {
                        "value": float(m.group(3)),
                        "unit": "ms",
                    }
        elif section == "chain":
            m = re.match(r"^(\d+)x(\d+)\s*\|\s*([0-9.]+)ms", line)
            if m:
                size = m.group(1)
                metrics[f"gpu_tensor.chain.{size}"] = {
                    "value": float(m.group(3)),
                    "unit": "ms",
                }
        elif section == "transfer":
            m = re.match(r"^(256KB|1MB|4MB)\s*\([^)]*\)\s*\|\s*([0-9.]+)ms\s*\|\s*([0-9.]+)ms\s*\|\s*([0-9.]+)ms", line)
            if m:
                size_label = m.group(1)
                total_ms = float(m.group(4))
                size_map = {"256KB": "256", "1MB": "512", "4MB": "1024"}
                size = size_map.get(size_label)
                if size:
                    metrics[f"gpu_tensor.transfer_total.{size}"] = {
                        "value": total_ms,
                        "unit": "ms",
                    }

    return metrics


def parse_gemm_view(text: str) -> dict:
    metrics = {}
    current_size = None
    for line in text.splitlines():
        if "GEMM Benchmark:" in line:
            m = re.search(
                r"GEMM Benchmark: (\d+)\u00d7(\d+) @ (\d+)\u00d7(\d+) \u2192 (\d+)\u00d7(\d+)",
                line,
            )
            if m:
                current_size = m.group(1)
            else:
                current_size = None
            continue

        if current_size and "Direct gemm (baseline)" in line:
            parts = line.split("│")
            if len(parts) >= 3:
                time_str = parts[2].strip()
                try:
                    metrics[f"gemm_view.baseline_ms.{current_size}"] = {
                        "value": parse_time_to_ms(time_str),
                        "unit": "ms",
                    }
                except ValueError:
                    pass

        if current_size and "Performance: ~" in line:
            m = re.search(r"Performance: ~([0-9.]+) TFLOPs/s", line)
            if m:
                metrics[f"gemm_view.tflops.{current_size}"] = {
                    "value": float(m.group(1)),
                    "unit": "TFLOPs/s",
                }

    return metrics


def parse_gpu_mnist(text: str) -> dict:
    metrics = {}
    epoch_times = []
    for line in text.splitlines():
        m = re.search(r"Initial accuracy: ([0-9.]+)%", line)
        if m:
            metrics["gpu_mnist.initial_accuracy"] = {
                "value": float(m.group(1)),
                "unit": "percent",
            }

        m = re.search(r"Final accuracy: ([0-9.]+)%", line)
        if m:
            metrics["gpu_mnist.final_accuracy"] = {
                "value": float(m.group(1)),
                "unit": "percent",
            }

        m = re.search(r"Epoch \d+: accuracy = [0-9.]+%, time = ([0-9.]+)ms", line)
        if m:
            epoch_times.append(float(m.group(1)))

    if epoch_times:
        avg = sum(epoch_times) / len(epoch_times)
        metrics["gpu_mnist.epoch_time_ms_avg"] = {
            "value": avg,
            "unit": "ms",
        }

    return metrics


def load_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def compare_metrics(metrics: dict, baseline: dict, max_regress: float, max_drop: float, fail: bool) -> int:
    base_metrics = baseline.get("metrics", {})
    regressions = []

    for key, base in base_metrics.items():
        if key not in metrics:
            continue
        new_val = metrics[key]["value"]
        base_val = base.get("value")
        if base_val in (None, 0):
            continue

        better = base.get("better", "lower")
        warn_pct = base.get("warn_pct")
        threshold = warn_pct if warn_pct is not None else (max_regress if better == "lower" else max_drop)

        if better == "lower":
            delta = (new_val - base_val) / base_val
            status = "OK"
            if delta > threshold:
                status = "REGRESSION"
                regressions.append(key)
        else:
            delta = (base_val - new_val) / base_val
            status = "OK"
            if delta > threshold:
                status = "REGRESSION"
                regressions.append(key)

        unit = metrics[key].get("unit") or base.get("unit", "")
        print(f"{key}: {new_val:.6g}{unit} (baseline {base_val:.6g}{unit}, delta {delta*100:+.1f}%) [{status}]")

    if regressions and fail:
        print("\nRegression(s) detected:")
        for key in regressions:
            print(f"  - {key}")
        return 1

    if regressions:
        print("\nRegression(s) detected (warning only). Use --fail-on-regression to exit non-zero.")

    return 0


def find_latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse benchmark outputs and compare to baseline.")
    parser.add_argument("--baseline", default="doc/bench/baselines/gpu_bench_20251221.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-regress", type=float, default=0.2)
    parser.add_argument("--max-drop", type=float, default=0.2)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(Path("doc/bench/runs"))
    if not run_dir:
        print("No run directory found. Use --run-dir.")
        return 2

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        return 2

    gpu_tensor_text = load_text(run_dir / "GpuTensorBenchmark.txt")
    gemm_view_text = load_text(run_dir / "GemmViewBenchmark.txt")
    gpu_mnist_text = load_text(run_dir / "GpuMNIST.txt")

    metrics = {}
    metrics.update(parse_gpu_tensor(gpu_tensor_text))
    metrics.update(parse_gemm_view(gemm_view_text))
    metrics.update(parse_gpu_mnist(gpu_mnist_text))

    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps({"metrics": metrics}, indent=2))

    baseline = json.loads(baseline_path.read_text())
    return compare_metrics(metrics, baseline, args.max_regress, args.max_drop, args.fail_on_regression)


if __name__ == "__main__":
    raise SystemExit(main())
