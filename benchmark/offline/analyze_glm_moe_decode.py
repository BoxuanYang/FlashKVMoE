"""读取实验 JSON，绘制 routed experts 耗时和线性回归曲线。"""

import json
from pathlib import Path

INPUT = Path("results/glm_moe_decode.json")
PLOT = Path("results/glm_moe_decode.png")
FIT_OUTPUT = Path("results/glm_moe_decode_fit.json")


if __name__ == "__main__":
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    if not data["complete"]:
        raise ValueError("实验尚未完成，不能拟合不完整数据")
    rows = sorted(data["results"], key=lambda row: row["batch_size"])
    x = np.array([row["batch_size"] for row in rows], dtype=float)
    y = np.array([row["avg_ms"] for row in rows], dtype=float)
    if len(x) < 2 or len(set(x)) != len(x) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("至少需要两组不同 batch size 的有效耗时")

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    total_variance = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - np.sum(residual**2) / total_variance) if total_variance > 0 else None
    fit = {
        "slope_ms_per_token": float(slope),
        "intercept_ms": float(intercept),
        "r_squared": r_squared,
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
    }

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x, y, "o-", markersize=3, label="Measured mean")
    ax.plot(x, predicted, "--", label=f"OLS: y = {slope:.5f}x {intercept:+.5f}")
    ax.set_title("GLM-4.5-Air: CPU routed experts only (CUDA Graph)")
    ax.set_xlabel("Decode batch size")
    ax.set_ylabel("Routed-expert compute time per batch (ms)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)
    FIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIT_OUTPUT.write_text(json.dumps(fit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"latency_ms = {slope:.6f} * batch_size + {intercept:.6f}; R^2 = {r_squared}")
    print(f"Plot: {PLOT}\nFit: {FIT_OUTPUT}")
