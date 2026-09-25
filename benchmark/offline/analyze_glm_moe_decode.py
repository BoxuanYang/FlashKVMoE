"""Plot measured batch latency and an ordinary least-squares linear regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fit_results(data):
    import numpy as np

    if not data.get("complete"):
        raise ValueError("Benchmark is incomplete; finish the measurement before fitting")
    rows = sorted(data["results"], key=lambda row: row["batch_size"])
    x = np.array([row["batch_size"] for row in rows], dtype=float)
    y = np.array([row["avg_ms"] for row in rows], dtype=float)
    if (
        len(x) < 2
        or len(set(x)) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (x <= 0).any()
        or (y <= 0).any()
        or (x != np.floor(x)).any()
    ):
        raise ValueError(
            "Need at least two distinct positive integer sizes and finite positive times"
        )
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    predictions = slope * x + intercept
    residuals = y - predictions
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    fit = {
        "method": "ordinary least squares, equal weight per batch-size mean, with intercept",
        "equation": "latency_ms = slope_ms_per_token * batch_size + intercept_ms",
        "slope_ms_per_token": float(slope),
        "intercept_ms": float(intercept),
        "r_squared": 1.0 - sse / sst if sst > 0 else None,
        "rmse_ms": math.sqrt(sse / len(x)),
        "batch_sizes": x.astype(int).tolist(),
        "observed_avg_ms": y.tolist(),
        "predicted_ms": predictions.tolist(),
        "residuals_ms": residuals.tolist(),
    }
    return rows, fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Plot path (default: input name with .png)")
    parser.add_argument(
        "--fit-output", type=Path, help="Regression JSON (default: input stem + _fit.json)"
    )
    args = parser.parse_args(argv)
    plot_path = args.output or args.input.with_suffix(".png")
    fit_path = args.fit_output or args.input.with_name(args.input.stem + "_fit.json")
    if len({path.resolve() for path in (args.input, plot_path, fit_path)}) != 3:
        parser.error("input, plot output, and fit output must be different paths")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows, fit = fit_results(data)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = fit["batch_sizes"], fit["observed_avg_ms"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(x, y, "o-", color="#1764ab", linewidth=2, label="Measured mean")
    if all("std_ms" in row for row in rows):
        ax.errorbar(
            x,
            y,
            yerr=[row["std_ms"] for row in rows],
            fmt="none",
            capsize=3,
            color="#1764ab",
            alpha=0.5,
            label="Sample standard deviation",
        )
    r2 = f"{fit['r_squared']:.4f}" if fit["r_squared"] is not None else "undefined (constant y)"
    equation = f"OLS: y = {fit['slope_ms_per_token']:.5f}x {fit['intercept_ms']:+.5f}; R² = {r2}"
    ax.plot(x, fit["predicted_ms"], "--", color="#d55e00", linewidth=2, label=equation)
    metadata = data.get("metadata", {})
    ax.set_title(
        f"GLM-4.5-Air MoE decode — layer index {metadata.get('layer_index', '?')}\n"
        f"CUDA Graph | KT threads={metadata.get('kt_cpuinfer', '?')} | "
        f"NUMA pools={metadata.get('kt_threadpool_count', '?')}"
    )
    ax.set_xlabel("Batch size (one decode token per request)")
    ax.set_ylabel("MoE latency per batch (ms)")
    # Keep a linear numeric axis: the requested sizes are not evenly spaced.
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.supxlabel(
        f"Inputs: {metadata.get('input_source', 'unknown')} | "
        "Router + routed/shared experts + transfers",
        fontsize=8,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    fit_path.write_text(json.dumps(fit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(equation.replace("²", "^2"))  # Also works in non-UTF-8 Windows terminals.
    print(f"RMSE: {fit['rmse_ms']:.5f} ms\nPlot: {plot_path}\nFit: {fit_path}")


if __name__ == "__main__":
    main()
