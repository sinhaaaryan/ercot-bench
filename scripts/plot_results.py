"""Benchmark chart for slides/README: accuracy on both test splits + output tokens per answer, per model.

    uv run ercot-bench report                     # refresh results/report/report.csv
    uv run --with matplotlib python scripts/plot_results.py   # -> docs/benchmark.png (+ _dark.png)

Two panels sharing the model axis (never a dual axis): left = pass@1 on the two splits (grouped bars),
right = mean output tokens per answer (one series). Our fine-tuned models are grouped and marked.
Colors: validated categorical slots 1-2 of the dataviz reference palette (blue/orange; CVD dE 24.7).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# (display name, report model id, group)
MODELS = [
    ("Claude Sonnet", "sonnet", "claude"),
    ("Claude Haiku 4.5 (thinking)", "haiku", "claude"),
    ("Claude Haiku 4.5 (no thinking)", "haiku (no thinking)", "claude"),
    ("Qwen3-8B + SFT", "/hackathon/outputs/ercot-sft-8b-lora/export/merged", "ours"),
    ("K2-Horizon-7B + SFT", "/hackathon/outputs/ercot-sft-k2-lora/merged", "ours"),
    ("Qwen3-1.7B + SFT", "/hackathon/outputs/ercot-sft-1p7b/checkpoints/step_80/weights", "ours"),
    ("K2-Horizon-7B base", "IFM/K2-Horizon-7B", "base"),
    ("Qwen3-8B base", "Qwen/Qwen3-8B", "base"),
    ("Qwen3-1.7B base", "Qwen/Qwen3-1.7B", "base"),
]
GROUP_LABEL = {"claude": "Claude (API, via Claude Code)", "ours": "Ours: fine-tuned, self-hosted on 1 GPU",
               "base": "Open models, no fine-tuning"}

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", neutral="#898781", band="#eef3fb"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#898781", axis="#383835",
                 s1="#3987e5", s2="#d95926", neutral="#898781", band="#1f2a38"),
}


def load(csv_path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["group_type"] == "overall":
            out.setdefault(r["model"], {})[r["split"]] = r
    return out


def draw(data, theme: str, path: Path) -> None:
    t = THEMES[theme]
    rows = [(n, m, g) for n, m, g in MODELS if m in data and {"test_in_template", "test_heldout_templates"} <= data[m].keys()]
    n = len(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.62 * n + 2.2), sharey=True,
                                  gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.06})
    fig.patch.set_facecolor(t["surface"])
    y = list(range(n))[::-1]
    bh = 0.30  # each bar; the pair + 2px-ish gap stays well under the row height

    for i, (name, mid, grp) in enumerate(rows):
        yi = y[i]
        if grp == "ours":  # light band behind our rows, both panels
            for a, xmax in ((ax, 100), (ax2, 1e9)):
                a.axhspan(yi - 0.5, yi + 0.5, color=t["band"], zorder=0, lw=0)
        a_in = float(data[mid]["test_in_template"]["pass@1"]) * 100
        a_ho = float(data[mid]["test_heldout_templates"]["pass@1"]) * 100
        for val, off, col in ((a_in, bh / 2 + 0.02, t["s1"]), (a_ho, -bh / 2 - 0.02, t["s2"])):
            ax.barh(yi + off, val, height=bh, color=col, zorder=2)
            ax.text(val + 1, yi + off, f"{val:.1f}%", va="center", ha="left", fontsize=9, color=t["text2"], zorder=3)
        toks = float(data[mid]["test_in_template"]["mean_output_tokens"] or 0)
        ax2.barh(yi, toks, height=bh * 1.2, color=t["neutral"], zorder=2)
        ax2.text(toks * 1.0 + 60, yi, f"{toks:,.0f}", va="center", ha="left", fontsize=9, color=t["text2"], zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    for lbl, (_, _, grp) in zip(ax.get_yticklabels(), rows):
        lbl.set_color(t["text"])
        if grp == "ours":
            lbl.set_fontweight("bold")
    # group separators + group names
    prev = None
    for i, (_, _, grp) in enumerate(rows):
        if grp != prev:
            if prev is not None:
                for a in (ax, ax2):
                    a.axhline(y[i] + 0.5, color=t["axis"], lw=1, zorder=1)
            ax.text(-0.01, y[i] + 0.43, GROUP_LABEL[grp], transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=8.5, color=t["muted"], style="italic")
            prev = grp

    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("pass@1 (mean accuracy over k samples)", color=t["text2"])
    xmax = max(float(data[m]["test_in_template"]["mean_output_tokens"] or 0) for _, m, _ in rows)
    ax2.set_xlim(0, xmax * 1.28)
    ax2.set_xlabel("output tokens per answer (fewer = faster)", color=t["text2"])
    for a in (ax, ax2):
        a.set_facecolor(t["surface"])
        a.set_ylim(-0.6, n - 0.4)
        a.grid(axis="x", color=t["axis"], lw=0.6, alpha=0.6, zorder=0)
        a.tick_params(colors=t["text2"], length=0)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(t["axis"])
    ax.tick_params(axis="y", pad=6)

    # legend (2 series) + titles
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(color=t["s1"], label="Same question types, unseen dates (test_in_template)"),
                             Patch(color=t["s2"], label="Held-out question types (never trained on)")],
                    loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
                    labelcolor=t["text"], handlelength=1.2, borderaxespad=0.3)
    fig.suptitle("ERCOT-Bench: text-to-SQL on Texas grid data", x=0.01, ha="left", y=0.995, fontsize=14,
                 fontweight="bold", color=t["text"])
    fig.text(0.01, 0.955 - 0.004 * (12 - n), "Fine-tuned K2-Horizon-7B beats Claude Haiku 4.5 (no thinking) on both splits with ~1/3 the output; "
             "Claude with thinking still leads on held-out question types.", fontsize=9.5, color=t["text2"],
             ha="left", va="top")
    ax2.set_title("Answer length", loc="left", fontsize=10, color=t["text"], pad=24)
    fig.subplots_adjust(left=0.24, right=0.975, top=1 - 1.25 / (0.62 * n + 2.2), bottom=0.9 / (0.62 * n + 2.2))
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "report" / "report.csv"
    data = load(csv_path)
    draw(data, "light", REPO / "docs" / "benchmark.png")
    draw(data, "dark", REPO / "docs" / "benchmark_dark.png")


if __name__ == "__main__":
    main()
