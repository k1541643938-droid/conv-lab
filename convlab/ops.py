"""
手写卷积算子（纯 NumPy 实现，不依赖任何深度学习框架）

为什么不用 cv2.filter2D / scipy.signal.convolve2d 直接算？
因为作业要求"动手实现卷积过程"。这里刻意使用**卷积的定义式**：

    (f * g)[y, x] = Σ_i Σ_j  k[i, j] · f[y + i - kh//2, x + j - kw//2]

实现上不写三重 for 循环（太慢），而是把它改写成
"先给原图补边，再让核的每一个权重去乘以一个平移后的整幅图，最后累加"：

    out = Σ_{i,j}  k[i, j] · shift(padded, i, j)

这与定义式完全等价，但每一步都是整块 NumPy 数组运算，
一次就能算出 100 万像素，速度与双重循环相差数百倍。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PADDINGS",
    "to_float",
    "normalize01",
    "pad_image",
    "convolve2d",
    "convolve_separable",
    "median_filter",
    "multiply_add_count",
    "is_separable",
    "split_separable",
    "odd",
]

# 三种边界填充策略：中文名 -> numpy 模式
PADDINGS = {
    "zero": "补零（zero）",
    "replicate": "复制边缘（replicate）",
    "reflect": "镜像反射（reflect）",
}


def to_float(img) -> np.ndarray:
    """转成 float64 数组；RGBA 丢掉 alpha 通道。"""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim not in (2, 3):
        raise ValueError(f"只支持 (H,W) 或 (H,W,C) 图像，收到 {a.shape}")
    return a


def normalize01(a) -> np.ndarray:
    """拉伸到 [0,1]，用于显示。"""
    a = np.asarray(a, dtype=np.float64)
    lo = float(np.nanmin(a)) if a.size else 0.0
    hi = float(np.nanmax(a)) if a.size else 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def odd(n: int) -> bool:
    return n > 0 and n % 2 == 1


def pad_image(img: np.ndarray, kh: int, kw: int, padding: str = "zero") -> np.ndarray:
    """按核尺寸在四周补边，保证输出与输入同样大小（'same' 卷积）。"""
    ph, pw = kh // 2, kw // 2
    pads = ((ph, ph), (pw, pw)) if img.ndim == 2 else ((ph, ph), (pw, pw), (0, 0))
    if padding == "zero":
        return np.pad(img, pads, mode="constant", constant_values=0.0)
    if padding == "replicate":
        return np.pad(img, pads, mode="edge")
    if padding == "reflect":
        return np.pad(img, pads, mode="reflect")
    raise ValueError(f"未知 padding: {padding}")


def convolve2d(
    img,
    kernel,
    padding: str = "zero",
    stride: int = 1,
    flip: bool = True,
    bias: float = 0.0,
) -> np.ndarray:
    """二维卷积（支持灰度图与多通道图，逐通道卷积）。

    参数
    ----
    img      : (H,W) 或 (H,W,C)
    kernel   : (kh,kw)，必须是奇数尺寸
    padding  : zero / replicate / reflect
    stride   : 步长
    flip     : True  = 真卷积（核旋转 180°，即 f*g）
               False = 互相关（深度学习里的 "convolution" 其实多是它）
    bias     : 输出加上的常数偏置

    返回
    ----
    与输入同形状（stride=1 时）的结果数组
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or not odd(kernel.shape[0]) or not odd(kernel.shape[1]):
        raise ValueError("卷积核必须是奇数尺寸的二维矩阵")
    if stride < 1:
        raise ValueError("stride 必须 ≥ 1")

    kh, kw = kernel.shape
    k = kernel[::-1, ::-1] if flip else kernel

    padded = pad_image(img, kh, kw, padding)
    h_out = (padded.shape[0] - kh) // stride + 1
    w_out = (padded.shape[1] - kw) // stride + 1
    out = np.full((h_out, w_out) + img.shape[2:], float(bias), dtype=np.float64)

    # 逐权重"平移—相乘—累加"，等价于定义式，但整块向量化
    for i in range(kh):
        for j in range(kw):
            w = k[i, j]
            if w == 0.0:
                continue
            patch = padded[i : i + stride * (h_out - 1) + 1 : stride,
                           j : j + stride * (w_out - 1) + 1 : stride]
            out += w * patch
    return out


def convolve_separable(
    img,
    k_row: np.ndarray,
    k_col: np.ndarray,
    padding: str = "zero",
    bias: float = 0.0,
) -> np.ndarray:
    """可分离卷积：先沿 x 方向做 1×kw，再沿 y 方向做 kh×1。

    数学依据：若 K = k_col ⊗ k_row（秩为 1），则 f * K = (f * k_row) * k_col。
    乘加次数从 kh·kw 降到 kh+kw。

    实现上先按完整的 3×3 边界补一次边，再在补边后的图上做两次"不补边"的一维卷积，
    这样结果与二维卷积逐像素完全一致（不会因为补两次边而在边界上产生差异）。
    """
    img = np.asarray(img, dtype=np.float64)
    k_row = np.asarray(k_row, float).reshape(-1)
    k_col = np.asarray(k_col, float).reshape(-1)
    kh, kw = len(k_col), len(k_row)
    if not odd(kh) or not odd(kw):
        raise ValueError("一维核长度必须是奇数")

    padded = pad_image(img, kh, kw, padding)
    H, W = img.shape[0], img.shape[1]
    tail = img.shape[2:]

    tmp = np.zeros((H + kh - 1, W) + tail, dtype=np.float64)
    kr = k_row[::-1]  # 卷积 = 翻转后相关
    for j in range(kw):
        if kr[j]:
            tmp += kr[j] * padded[:, j : j + W]

    out = np.full((H, W) + tail, float(bias), dtype=np.float64)
    kc = k_col[::-1]
    for i in range(kh):
        if kc[i]:
            out += kc[i] * tmp[i : i + H]
    return out


def median_filter(img, size: int = 3, padding: str = "replicate") -> np.ndarray:
    """中值滤波——注意：它是**非线性**的，严格来说不是卷积。

    放在这里是为了作对照实验，说明"卷积能做什么、不能做什么"。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    img = np.asarray(img, dtype=np.float64)
    if not odd(size) or size < 3:
        raise ValueError("中值滤波窗口必须是 ≥3 的奇数")
    padded = pad_image(img, size, size, padding)
    win = sliding_window_view(padded, (size, size))
    return np.median(win, axis=(-2, -1))


def multiply_add_count(h: int, w: int, kh: int, kw: int) -> int:
    """一次卷积的乘加次数（直接法）。"""
    return int(h) * int(w) * int(kh) * int(kw)


def is_separable(kernel: np.ndarray, tol: float = 1e-9) -> bool:
    """判断二维核是否为秩 1（即可分成两个一维核的乘积）。"""
    k = np.asarray(kernel, float)
    s = np.linalg.svd(k, compute_uv=False)
    if s[0] < tol:
        return False
    return bool(s[1] / s[0] < 1e-6)


def split_separable(kernel: np.ndarray):
    """把秩 1 的核拆成 (k_col, k_row)，使 outer(k_col,k_row) ≈ kernel。"""
    k = np.asarray(kernel, float)
    u, s, vt = np.linalg.svd(k)
    root = float(np.sqrt(s[0]))
    k_col = u[:, 0] * root
    k_row = vt[0, :] * root
    # 只归一化其中一个向量，并把倍数补到另一个上——外积保持不变
    m = float(np.abs(k_col).max())
    if m > 1e-12:
        k_col = k_col / m
        k_row = k_row * m
    return k_col, k_row
