"""频域分析、定量指标、性能对比——给报告提供数字和图。"""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "to_gray",
    "kernel_spectrum",
    "radial_response",
    "image_spectrum",
    "high_freq_ratio",
    "mean_gradient",
    "metrics_table",
    "psnr",
    "benchmark",
    "fft_convolve",
    "cutoff_freq",
    "insights",
]


def to_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, float)
    if img.ndim == 2:
        return img
    if img.shape[2] >= 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img[..., 0]


# ---------------------------------------------------------------- 频域
def kernel_spectrum(k: np.ndarray, n: int = 256) -> np.ndarray:
    """把核补零到 n×n 后做 FFT，返回中心化的对数幅频（dB，已归一化到 DC）。"""
    k = np.asarray(k, float)
    pad = np.zeros((n, n), float)
    oy, ox = (n - k.shape[0]) // 2, (n - k.shape[1]) // 2
    pad[oy : oy + k.shape[0], ox : ox + k.shape[1]] = k
    H = np.abs(np.fft.fftshift(np.fft.fft2(pad)))
    dc = H[n // 2, n // 2]
    if dc > 1e-12:
        H = H / dc
    else:  # 零和核：用最大值归一，并标记低端为极小
        mx = H.max()
        H = H / mx if mx > 1e-12 else H
    return 20 * np.log10(np.maximum(H, 1e-4))


def radial_response(k: np.ndarray, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """径向平均频响曲线（dB），横轴为归一化频率 0~1（1 = 奈奎斯特）。"""
    db = kernel_spectrum(k, n)
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - c, xx - c) / c  # 0~1.41
    r = np.clip(r, 0, 1.0)
    bins = np.linspace(0, 1.0, 64)
    idx = np.digitize(r.ravel(), bins) - 1
    vals = db.ravel()
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = idx == i
        if m.any():
            out[i] = vals[m].mean()
    centers = (bins[:-1] + bins[1:]) / 2
    ok = ~np.isnan(out)
    return centers[ok], out[ok]


def image_spectrum(img: np.ndarray) -> np.ndarray:
    """图像中心化对数幅度谱（用于展示卷积前后频谱变化）。"""
    g = to_gray(img)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    return np.log1p(F)


def high_freq_ratio(img: np.ndarray, thresh: float = 0.25) -> float:
    """高频能量占比：>thresh×奈奎斯特半径的能量 / 总能量。"""
    g = to_gray(img)
    g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    n = g.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:g.shape[1]]
    r = np.hypot(yy - c, xx - c) / c
    tot = F.sum()
    if tot <= 1e-12:
        return 0.0
    return float(F[r > thresh].sum() / tot)


def mean_gradient(img: np.ndarray) -> float:
    """平均梯度（清晰度指标）：梯度幅值的均值，越大越"锐"。"""
    g = to_gray(img)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float((gx + gy) / 2)


def _match_shape(a: np.ndarray, shape) -> np.ndarray:
    """把 a 最近邻重采样到目标尺寸（stride>1 时结果比原图小，指标要能对齐）。"""
    a = np.asarray(a, float)
    if a.shape[:2] == tuple(shape[:2]):
        return a
    h, w = a.shape[:2]
    H, W = int(shape[0]), int(shape[1])
    yi = (np.arange(H) * h / H).astype(int).clip(0, h - 1)
    xi = (np.arange(W) * w / W).astype(int).clip(0, w - 1)
    return a[yi][:, xi]


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    a, b = to_gray(a), to_gray(b)
    b = _match_shape(b, a.shape)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10 * np.log10(peak**2 / mse))


def metrics_table(orig: np.ndarray, res: np.ndarray) -> list[dict]:
    """原图 vs 结果的定量对比，可直接贴进报告。"""
    return [
        {"指标": "均值（亮度）", "原图": f"{to_gray(orig).mean():.4f}", "结果": f"{to_gray(res).mean():.4f}"},
        {"指标": "标准差（对比度）", "原图": f"{to_gray(orig).std():.4f}", "结果": f"{to_gray(res).std():.4f}"},
        {"指标": "平均梯度（清晰度）", "原图": f"{mean_gradient(orig):.5f}", "结果": f"{mean_gradient(res):.5f}"},
        {"指标": "高频能量占比", "原图": f"{high_freq_ratio(orig):.3%}", "结果": f"{high_freq_ratio(res):.3%}"},
        {"指标": "PSNR（vs 原图）", "原图": "—", "结果": f"{psnr(orig, res):.2f} dB"},
    ]


# ---------------------------------------------------------------- 性能
def fft_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """基于 FFT 的卷积（scipy 实现），用于与直接法对比速度。"""
    from scipy.signal import fftconvolve

    img = np.asarray(img, float)
    k = np.asarray(kernel, float)
    if img.ndim == 2:
        return fftconvolve(img, k, mode="same")
    return np.stack([fftconvolve(img[..., c], k, mode="same") for c in range(img.shape[2])], axis=-1)


def _timeit(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(img: np.ndarray, kernel: np.ndarray, sizes=(128, 256, 512, 768), repeat: int = 3) -> list[dict]:
    """不同图像尺寸下：直接卷积 vs FFT 卷积 的耗时对比。"""
    from .ops import convolve2d

    rows = []
    for s in sizes:
        tile = _tile(img, s)
        t_direct = _timeit(lambda: convolve2d(tile, kernel), repeat)
        t_fft = _timeit(lambda: fft_convolve(tile, kernel), repeat)
        rows.append(
            {
                "尺寸": f"{s}×{s}",
                "像素数": s * s,
                "直接卷积(ms)": round(t_direct * 1000, 2),
                "FFT卷积(ms)": round(t_fft * 1000, 2),
                "直接/FFT 倍数": round(t_direct / t_fft, 2) if t_fft > 0 else None,
            }
        )
    return rows


def _tile(img: np.ndarray, size: int) -> np.ndarray:
    """把图像裁/缩成 size×size 的方块，供基准测试使用。"""
    from .samples import fit_size

    small = fit_size(img, size)
    h, w = small.shape[:2]
    if h >= size and w >= size:
        oy, ox = (h - size) // 2, (w - size) // 2
        return small[oy : oy + size, ox : ox + size]
    pad = [(0, max(0, size - h)), (0, max(0, size - w))] + ([(0, 0)] if small.ndim == 3 else [])
    out = np.pad(small, pad, mode="reflect" if min(h, w) > 1 else "constant")
    return out[:size, :size]


def cutoff_freq(k: np.ndarray) -> float | None:
    """增益首次跌破 -3 dB 的频率（归一化到奈奎斯特），高通核则为 None。"""
    f, g = radial_response(k)
    below = g < -3.0
    if not below.any():
        return 1.0
    if below[0]:  # 低频就被压制 → 不是低通核
        return None
    i = int(np.argmax(below))
    return float(f[i])


def insights(k: np.ndarray, orig: np.ndarray, res: np.ndarray) -> list[str]:
    """根据核的统计量与前后指标，自动生成几条「可以直接抄进报告」的结论。"""
    k = np.asarray(k, float)
    out: list[str] = []
    dc = float(k.sum())
    if abs(dc) < 1e-6:
        out.append(
            "核和 = 0（**零和核**）：输出均值为 0，平坦区域被压成黑，只剩变化剧烈的地方——"
            "这就是所有高通/边缘核的共同特征。"
        )
    elif abs(dc - 1.0) < 1e-6:
        out.append("核和 = 1：**DC 增益为 1**，整体亮度不变，只改变不同频率成分的强弱。")
    else:
        out.append(f"核和 = {dc:.3f}：输出亮度整体被放大/缩小约 {dc:.2f} 倍（卷积也会改变直流分量）。")

    d_hf = high_freq_ratio(res) - high_freq_ratio(orig)
    if d_hf > 0.03:
        out.append(f"高频能量占比 {d_hf:+.1%} → 高频被**增强**，属于高通/锐化类核，边缘变亮、对比变强。")
    elif d_hf < -0.03:
        out.append(f"高频能量占比 {d_hf:+.1%} → 高频被**压制**，属于低通类核，细节被抹平、噪声下降。")
    else:
        out.append(f"高频能量占比 {d_hf:+.1%} → 频谱分布变化不大，这个核主要改变的是相位/方向信息。")

    d_std = to_gray(res).std() - to_gray(orig).std()
    if d_std > 0.02:
        out.append(f"标准差（对比度）{d_std:+.3f} → 对比度提升，视觉上更「硬」。")
    elif d_std < -0.02:
        out.append(f"标准差（对比度）{d_std:+.3f} → 对比度下降，视觉上更「柔」。")

    from .ops import is_separable

    if is_separable(k):
        kh, kw = k.shape
        out.append(
            f"该核**可分离**：可拆成 {kh}×1 与 1×{kw} 两个一维核，乘加次数从 {kh * kw} 次降到 "
            f"{kh + kw} 次，大图上能快数倍。"
        )
    return out
