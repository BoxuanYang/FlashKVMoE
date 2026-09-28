"""Plot KTransformers MoE latency and fit a linear regression."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRECTORY = Path(__file__).resolve().parent
DATA_PATH = DIRECTORY / "moe_perf.txt"
IMAGE_PATH = DIRECTORY / "moe_linear_regression.png"


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the batch-size and latency columns from moe_perf.txt."""
    batch_sizes = []
    latencies = []

    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) != 2:
            continue

        try:
            batch_sizes.append(int(columns[0]))
            latencies.append(float(columns[1]))
        except ValueError:
            continue

    if len(batch_sizes) < 2:
        raise ValueError(f"Expected at least two measurements in {path}")

    return np.array(batch_sizes), np.array(latencies)


def linear_regression(x: np.ndarray, y: np.ndarray):
    """Return the fitted line, slope, intercept and R-squared."""
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum if total_sum else 1.0
    return slope, intercept, fitted, r_squared


def main():
    batch_sizes, latencies = load_results(DATA_PATH)
    slope, intercept, fitted, r_squared = linear_regression(
        batch_sizes, latencies
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        batch_sizes,
        latencies,
        s=24,
        alpha=0.75,
        color="tab:blue",
        label="Measurements",
    )
    axis.plot(
        batch_sizes,
        fitted,
        linewidth=2,
        color="tab:orange",
        label=f"Linear fit: y = {slope:.6f}x + {intercept:.6f}, R^2 = {r_squared:.6f}",
    )

    axis.set_title("GLM-4.5-Air Layer 8 KTransformers MoE Scaling")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("MoE latency (ms)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(IMAGE_PATH, dpi=200)
    plt.close(figure)

    print(f"Measurements: {len(batch_sizes)}")
    print(f"Slope: {slope:.6f} ms per batch item")
    print(f"Intercept: {intercept:.6f} ms")
    print(f"R^2: {r_squared:.6f}")
    print(f"Image written to {IMAGE_PATH}")


if __name__ == "__main__":
    main()
