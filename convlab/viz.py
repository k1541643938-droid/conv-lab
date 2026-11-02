"""绘图模块：所有图都用 matplotlib 画，统一中文字体与配色。"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from .analysis import image_spectrum, radial_response, kernel_spectrum, to_gray
from .ops import normalize01

__all__ = [
    "setup",
    "fig_kernel",
    "fig_process",
    "fig_spectra",
    "fig_bench",
    "fig_conv_vs_corr",
    "fig_report",
    "kernel_rgba",
    "img_for_display",
]

_CJK_CANDIDATES = [
    # Windows 自带（本机跑时优先用，带真正的粗体）
    "Microsoft YaHei",
    "微软雅黑",
    "SimHei",
    "黑体",
    "DengXian",
    # Linux / 云端（项目里带了 Noto Sans SC，见 assets/）
    "Noto Sans SC",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "PingFang SC",
]
_READY = False

# 随项目打包的中文字体（OFL 许可，可随代码分发）。
# Linux 服务器（Streamlit Cloud / Docker）默认没有中文字体，
# 不带上它，图里的中文会全变成方块。
_BUNDLED_FONT = Path(__file__).resolve().parent.parent / "assets" / "NotoSansSC-Regular.otf"
_FONT_REGISTERED = False

# 统一配色
C_MAIN = "#2F6FEB"
C_ACCENT = "#E4572E"
C_GRID = "#D8DEE9"


def _register_bundled_font() -> None:
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    _FONT_REGISTERED = True
    if _BUNDLED_FONT.exists():
        try:
            font_manager.fontManager.addfont(str(_BUNDLED_FONT))
        except Exception:
            pass


def _pick_font() -> str | None:
    _register_bundled_font()
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in have:
            return name
    return None


def setup() -> None:
    global _READY
    if _READY:
        return
    # 服务器上没有粗体字重时 matplotlib 会刷一堆 findfont 警告，这里静音
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    font = _pick_font()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9aa4b2",
            "axes.labelcolor": "#2b3240",
            "axes.grid": False,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    else:  # 没有中文字体时退化为英文，避免出现方块
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    _READY = True


def _as_img(a: np.ndarray) -> np.ndarray:
    """显示用：多通道转 0~1，灰度也转 0~1。"""
    a = np.asarray(a, float)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.clip(normalize01(a), 0, 1)


def img_for_display(a: np.ndarray, signed: bool = False, robust: bool = True) -> np.ndarray:
    """把计算结果转成可 imshow 的 0~1 数组。

    signed=True 时以 0 为中心做对称拉伸（适合拉普拉斯这类有负值的输出）；
    其余情况用 0.5%~99.5% 分位数做稳健拉伸——否则像 Sobel 这种
    "大部分像素接近 0、少数边缘很大" 的结果，直接按 min/max 拉伸会灰成一片。
    """
    a = np.asarray(a, float)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if signed:
        m = float(np.abs(a).max()) or 1.0
        return np.clip(a / (2 * m) + 0.5, 0, 1)
    if robust and a.size > 256:
        lo, hi = np.percentile(a, [0.5, 99.5])
        if hi - lo < 1e-9:
            lo, hi = float(a.min()), float(a.max())
    else:
        lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------------ 卷积核
def fig_kernel(k: np.ndarray, title: str = "卷积核", figsize=None, annotate: bool | None = None):
    setup()
    k = np.asarray(k, float)
    n = max(k.shape)
    if figsize is None:
        side = 3.0 if n <= 3 else 3.4
        figsize = (side, side + 0.55)
    fig, ax = plt.subplots(figsize=figsize)
    m = float(np.abs(k).max())
    if m <= 1e-12 or k.min() >= 0:
        im = ax.imshow(k, cmap="Blues", vmin=min(0, k.min()), vmax=max(1e-9, k.max()))
    else:
        norm = TwoSlopeNorm(vmin=min(k.min(), -1e-9), vcenter=0.0, vmax=max(k.max(), 1e-9))
        im = ax.imshow(k, cmap="RdBu_r", norm=norm)
    ax.set_title(title, pad=8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(length=0, labelsize=8, colors="#6b7280")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    if annotate is None:
        annotate = n <= 5
    if annotate:
        thr = (k.max() + k.min()) / 2 if k.max() > 0 else k.max() / 2
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                v = k[i, j]
                txt = f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.0f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9 if n <= 3 else 6.5,
                    color="white" if abs(v) > abs(thr) else "#1f2937",
                    fontweight="bold",
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def kernel_rgba(k: np.ndarray, size: int = 96) -> np.ndarray:
    """把核渲染成小图（不需要 matplotlib figure，快）。用于核库卡片。"""
    from matplotlib import colormaps
    from PIL import Image

    k = np.asarray(k, float)
    n = max(k.shape)
    m = float(np.abs(k).max()) or 1.0
    if k.min() < 0:
        norm = (k / m + 1) / 2  # 0~1，中心 0.5
        rgba = colormaps["RdBu_r"](norm)
    else:
        rgba = colormaps["Blues"](k / (m + 1e-12))
    img = (rgba[..., :3] * 255).astype(np.uint8)

    # 最近邻放大，保留格子感
    rep = max(1, size // max(1, n))
    img = np.repeat(np.repeat(img, rep, axis=0), rep, axis=1)

    # 画白色网格线
    step = rep
    for i in range(1, n):
        img[i * step - 1 : i * step, :, :] = 255
        img[:, i * step - 1 : i * step, :] = 255

    # 统一缩放到 size×size，保证卡片对齐
    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))
    return img


# ------------------------------------------------------------------ 运算过程
def fig_process(img, k, out, py, px, padding="zero", stride=1, flip=True):
    """滑窗卷积的可视化：原图窗口 → 输入块 × 核 → 逐点乘积 → 求和。"""
    setup()
    img = np.asarray(img, float)
    k = np.asarray(k, float)
    kh, kw = k.shape
    kk = k[::-1, ::-1] if flip else k

    # 输入块（与 padding 一致）
    pad_h, pad_w = kh // 2, kw // 2
    pads = ((pad_h, pad_h), (pad_w, pad_w)) + (((0, 0),) if img.ndim == 3 else ())
    if padding == "zero":
        p = np.pad(img, pads, constant_values=0)
    elif padding == "replicate":
        p = np.pad(img, pads, mode="edge")
    else:
        p = np.pad(img, pads, mode="reflect")

    patch = p[py : py + kh, px : px + kw]
    if patch.ndim == 3:
        patch = to_gray(patch)

    fig = plt.figure(figsize=(11.6, 3.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1.5, 1, 1, 1], wspace=0.28)

    # 1) 原图 + 红框
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_for_display(img), cmap="gray" if img.ndim == 2 else None)
    cy, cx = py, px  # 窗口中心（imshow 的像素中心就在整数坐标上）
    ring = max(11.0, max(kh, kw) + 8.0)  # 定位环：这么小的框在整图上根本看不见
    ax.add_patch(
        plt.Rectangle((cx - ring / 2, cy - ring / 2), ring, ring, fill=False, edgecolor="#FFC53D", lw=1.8)
    )
    ax.add_patch(
        plt.Rectangle((cx - kw / 2 + 0.5, cy - kh / 2 + 0.5), kw, kh, fill=False, edgecolor=C_ACCENT, lw=2.0)
    )
    ax.set_title(f"原图（{img.shape[1]}×{img.shape[0]}）")
    ax.set_xticks([])
    ax.set_yticks([])

    # 2) 输出 + 落点
    ax = fig.add_subplot(gs[0, 1])
    show_out = np.abs(out) if float(np.min(out)) < -1e-9 else out
    ax.imshow(img_for_display(show_out), cmap="gray" if out.ndim == 2 else None)
    oy, ox = py // stride, px // stride
    if 0 <= oy < out.shape[0] and 0 <= ox < out.shape[1]:
        r2 = max(10.0, ring / stride)
        ax.add_patch(
            plt.Rectangle((ox - r2 / 2, oy - r2 / 2), r2, r2, fill=False, edgecolor="#FFC53D", lw=1.6)
        )
        ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, fill=False, edgecolor=C_MAIN, lw=2.0))
    ax.set_title(f"输出（{out.shape[1]}×{out.shape[0]}）")
    ax.set_xticks([])
    ax.set_yticks([])

    def heat(ax, data, title, fmt=".2f", vmin=None, vmax=None):
        data = np.asarray(data, float)
        if vmin is None or vmax is None:
            m = float(np.abs(data).max()) or 1.0
            if data.min() < 0:
                vmin, vmax, cmap = -m, m, "RdBu_r"
            else:
                vmin, vmax, cmap = 0.0, m, "Blues"
        else:
            cmap = "Blues"
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                # 按格子实际颜色选字色，浅底用深字、深底用白字
                r, gg, bb, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * gg + 0.114 * bb
                ax.text(
                    j,
                    i,
                    format(v, fmt),
                    ha="center",
                    va="center",
                    fontsize=10 if kh <= 3 else 6.5,
                    color="#111827" if lum > 0.55 else "white",
                    fontweight="bold",
                )

    prod = patch * kk
    heat(fig.add_subplot(gs[0, 2]), patch, f"输入块 f（{kh}×{kw}）", vmin=0.0, vmax=1.0)
    heat(fig.add_subplot(gs[0, 3]), kk, "核" + ("（已翻转180°）" if flip and not np.allclose(k, kk) else ""))
    heat(fig.add_subplot(gs[0, 4]), prod, "逐点乘积", fmt=".1f" if np.abs(prod).max() >= 10 else ".2f")

    val = float(prod.sum())
    fig.suptitle(
        f"输出 g[{py // stride}, {px // stride}] = Σ 逐点乘积 = {val:.4f}",
        fontsize=11,
        y=0.97,
        color="#111827",
    )
    fig.subplots_adjust(top=0.82, bottom=0.03, left=0.02, right=0.99, wspace=0.25)
    return fig


# ------------------------------------------------------------------ 频域
def fig_spectra(k, img, out):
    setup()
    fig = plt.figure(figsize=(13.2, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    db = kernel_spectrum(k)
    im = ax.imshow(db, cmap="magma", vmin=max(-60, float(db.min())), vmax=0)
    ax.set_title("卷积核的二维频响（dB）")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[0, 1])
    f, g = radial_response(k)
    ax.plot(f, g, color=C_MAIN, lw=2)
    ax.axhline(-3, ls="--", lw=1, color=C_ACCENT)
    ax.text(0.02, -2.2, "-3 dB", color=C_ACCENT, fontsize=8)
    ax.set_xlabel("归一化频率（1 = 奈奎斯特）")
    ax.set_ylabel("增益 (dB)")
    ax.set_title("核的径向频响")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.5)

    ax = fig.add_subplot(gs[0, 2])
    for arr, lab, c in ((img, "原图", "#9aa4b2"), (out, "卷积后", C_MAIN)):
        s = image_spectrum(arr)
        h, w = s.shape
        c0 = np.array([h // 2, w // 2])
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - c0[0], xx - c0[1]) / (h / 2)
        bins = np.linspace(0, 1, 50)
        idx = np.digitize(r.ravel(), bins) - 1
        prof = [s.ravel()[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
        ax.plot((bins[:-1] + bins[1:]) / 2, prof, lw=2, label=lab, color=c)
    ax.set_xlabel("归一化频率")
    ax.set_ylabel("log 幅度")
    ax.set_title("图像功率谱：卷积前后")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.14, wspace=0.28)
    return fig


# ------------------------------------------------------------------ 性能
def fig_bench(rows):
    setup()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    px = [r["像素数"] for r in rows]
    ax.loglog(px, [r["直接卷积(ms)"] for r in rows], "o-", lw=2, color=C_ACCENT, label="直接卷积 O(HW·kh·kw)")
    ax.loglog(px, [r["FFT卷积(ms)"] for r in rows], "s-", lw=2, color=C_MAIN, label="FFT 卷积 O(HW·log HW)")
    ax.set_xlabel("像素数")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("两种卷积实现的速度对比")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_conv_vs_corr():
    """一维小例子：卷积（翻转核）与互相关（不翻转）的差别。"""
    setup()
    f = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], float)
    g = np.array([1, -1], float)
    conv = np.convolve(f, g[::-1], mode="same")
    corr = np.convolve(f, g, mode="same")

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    x = np.arange(len(f))
    ax.plot(x, f, "o-", color="#9aa4b2", lw=2, label="输入信号 f")
    ax.plot(x, conv, "o-", color=C_MAIN, lw=2, label="卷积（核翻转 180°）")
    ax.plot(x, corr, "s--", color=C_ACCENT, lw=2, label="互相关（核不翻转）")
    ax.axhline(0, color="#cbd5e1", lw=1)
    ax.set_xticks(x)
    ax.set_xlabel("位置 n")
    ax.set_ylim(-1.9, 4.4)
    ax.set_title("同一个核 [1, -1]，翻转与不翻转的结果互为镜像")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 报告用图
def fig_report(orig, k, out, title="卷积效果", subtitle="", mode="auto"):
    setup()
    out_arr = np.asarray(out, float)
    signed = float(out_arr.min()) < -1e-9
    if mode == "auto":
        mode = "abs" if signed else "stretch"
    if mode == "abs":
        disp_out, cap = img_for_display(np.abs(out_arr)), "(c) 卷积结果 |g|"
    elif mode == "signed":
        disp_out, cap = img_for_display(out_arr, signed=True), "(c) 卷积结果（0 为灰）"
    else:
        disp_out, cap = img_for_display(out_arr), "(c) 卷积结果"

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 0.8, 1.25, 0.9], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_for_display(orig), cmap="gray" if np.asarray(orig).ndim == 2 else None)
    ax.set_title("(a) 原图", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    kk = np.asarray(k, float)
    m = float(np.abs(kk).max()) or 1.0
    if kk.min() < 0:
        ax.imshow(kk, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m), aspect="equal")
    else:
        ax.imshow(kk, cmap="Blues", aspect="equal")
    ax.set_title("(b) 卷积核", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(disp_out, cmap="gray" if np.asarray(out).ndim == 2 else None)
    ax.set_title(cap, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    from .analysis import mean_gradient, high_freq_ratio

    txt = (
        f"均值    {np.mean(to_gray(out)):.3f}\n"
        f"对比度  {np.std(to_gray(out)):.3f}\n"
        f"清晰度  {mean_gradient(out):.4f}\n"
        f"高频占比 {high_freq_ratio(out):.1%}\n"
        f"核和DC  {kk.sum():.2f}"
    )
    ax.text(0.02, 0.95, txt, va="top", ha="left", fontsize=10, linespacing=1.75)
    ax.set_title("(d) 指标", fontsize=11)

    fig.suptitle((title + "  " + subtitle).strip(), fontsize=12, y=0.965)
    fig.subplots_adjust(top=0.84, bottom=0.02, left=0.01, right=0.99, wspace=0.16)
    return fig
