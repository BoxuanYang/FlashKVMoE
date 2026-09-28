"""Plot GPU attention latency and its linear regression."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRECTORY = Path(__file__).resolve().parent
DATA_PATH = DIRECTORY / "gpu_perf.txt"
IMAGE_PATH = DIRECTORY / "gpu_attn_linear_regression.png"


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the two numeric columns from gpu_perf.txt."""
    sequence_lengths = []
    gpu_times = []

    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) != 2:
            continue

        try:
            sequence_lengths.append(int(columns[0]))
            gpu_times.append(float(columns[1]))
        except ValueError:
            continue

    if len(sequence_lengths) < 2:
        raise ValueError(f"Expected at least two measurements in {path}")

    return np.array(sequence_lengths), np.array(gpu_times)


def linear_regression(x: np.ndarray, y: np.ndarray):
    """Return the fitted line and coefficient of determination."""
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum
    return slope, intercept, fitted, r_squared


def main():
    sequence_lengths, gpu_times = load_results(DATA_PATH)
    lengths_in_thousands = sequence_lengths / 1000
    slope, intercept, fitted, r_squared = linear_regression(
        lengths_in_thousands, gpu_times
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        lengths_in_thousands,
        gpu_times,
        color="tab:blue",
        s=32,
        alpha=0.8,
        label="Measurements",
    )
    axis.plot(
        lengths_in_thousands,
        fitted,
        color="tab:orange",
        linewidth=2,
        label=f"Linear fit: y = {slope:.6f}x + {intercept:.6f}, R² = {r_squared:.6f}",
    )

    axis.set_title("GLM-4.5-Air GPU Decode Attention Scaling")
    axis.set_xlabel("Sequence length (thousand tokens)")
    axis.set_ylabel("GPU time (ms)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(IMAGE_PATH, dpi=200)
    plt.close(figure)

    print(f"Slope: {slope:.6f} ms per 1K tokens")
    print(f"Intercept: {intercept:.6f} ms")
    print(f"R^2: {r_squared:.6f}")
    print(f"Image written to {IMAGE_PATH}")


if __name__ == "__main__":
    main()
