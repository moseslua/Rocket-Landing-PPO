"""Figures derived only from saved run artifacts, never tuning data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def learning_curves(runs, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        history = run["history"]
        if history:
            ax.plot(
                [row["steps"] for row in history],
                [row["reward"] for row in history],
                label=f"{run['config']['experiment']['name']}-seed-{run['config']['seed']}",
            )
    ax.set(
        xlabel="Environment steps",
        ylabel="Mean rollout reward",
        title="Development learning curves",
    )
    if runs:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def method_comparison(metrics, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in metrics.items():
        rates = [row["summary"]["success_rate"] for row in rows]
        ax.bar(name, np.mean(rates), yerr=np.std(rates), capsize=4)
    ax.set(ylabel="Success rate", ylim=(0, 1), title="Evaluation results")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
