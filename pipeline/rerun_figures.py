"""
rerun_figures.py  –  Hai hình cho báo cáo chạy lại (đọc {dir}/tables/, ghi {dir}/figures/).

  python pipeline/src/rerun_figures.py --dir pipeline/results/rerun_2026-11-19

  fig_da.png       DA của từng mô hình ở hai cấu hình, so với luật đảo chiều b0rev và mức 50%
  fig_delta.png    ΔDA và Δlog-loss khi thêm On-chain, CI 95% block bootstrap
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a66", "#e4e3df"
C_OHLCV, C_ONCHAIN = "#2a78d6", "#eb6834"          # slot 1, 2 của bảng màu chuẩn (đã kiểm CVD)
ORDER = ["b0maj", "b0pers", "b1", "b1fs", "b1t", "b2", "b2t", "b3", "b3t"]
LEARNED = ["b1", "b1fs", "b1t", "b2", "b2t", "b3", "b3t"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_da(t, out):
    m = t["metrics"].set_index(["model", "config"])["da"] * 100
    rev = m[("b0rev", "ohlcv")]
    models = [x for x in ORDER if (x, "ohlcv") in m.index]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    y = range(len(models))[::-1]
    for yi, name in zip(y, models):
        a, b = m[(name, "ohlcv")], m[(name, "ohlcv_onchain")]
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(a, yi, s=46, color=C_OHLCV, marker="o", edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(b, yi, s=52, color=C_ONCHAIN, marker="D", edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle=":")
    ax.axvline(rev, color=INK, linewidth=1.2, linestyle="--")
    ax.text(rev + 0.1, len(models) - 0.35, f"b0rev {rev:.2f}%".replace(".", ","), color=INK, fontsize=8.5, va="bottom")
    ax.set_yticks(list(y), models, fontfamily="monospace", color=INK)
    ax.set_xlabel("DA (%) trên 2.434 ngày test", color=MUTED, fontsize=9)
    ax.set_xlim(45.5, 54.5)
    ax.set_ylim(-0.7, len(models) - 0.1)
    h = [plt.Line2D([], [], color=C_OHLCV, marker="o", linestyle="", markersize=7, label="Chỉ OHLCV"),
         plt.Line2D([], [], color=C_ONCHAIN, marker="D", linestyle="", markersize=6.5, label="OHLCV + On-chain")]
    ax.legend(handles=h, loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title("DA của từng mô hình, so với luật đảo chiều", color=INK, fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out / "fig_da.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_delta(t, out):
    d = t["delta_onchain"].set_index("model").loc[[x for x in LEARNED if x in set(t["delta_onchain"].model)]]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), facecolor=SURFACE, sharey=True)
    y = list(range(len(d)))[::-1]
    panels = [("d_da", 100, "ΔDA (điểm %)", "> 0: On-chain đúng nhiều hơn"),
              ("d_logloss", 1, "Δ log-loss", "< 0: On-chain tốt hơn")]
    for ax, (col, k, xl, note) in zip(axes, panels):
        style(ax)
        ax.axvline(0, color=MUTED, linewidth=1)
        for yi, (name, r) in zip(y, d.iterrows()):
            lo, hi, v = r[col + "_lo"] * k, r[col + "_hi"] * k, r[col] * k
            ax.plot([lo, hi], [yi, yi], color=C_OHLCV, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter(v, yi, s=40, color=C_OHLCV, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xlabel(f"{xl}   ({note})", color=MUTED, fontsize=8.5)
    axes[0].set_yticks(y, list(d.index), fontfamily="monospace", color=INK)
    axes[0].set_title("Thay đổi khi thêm On-chain, CI 95%", color=INK, fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out / "fig_delta.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    base = ROOT / args.dir
    t = {k: pd.read_csv(base / "tables" / f"{k}.csv") for k in ("metrics", "delta_onchain")}
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_da(t, out)
    fig_delta(t, out)
    print(f"Đã lưu {out / 'fig_da.png'}, {out / 'fig_delta.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
