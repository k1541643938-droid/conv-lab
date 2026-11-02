"""内置示例图像：全部用 NumPy 现场生成，不联网、不依赖外部图片文件。

这样做的原因：课堂上/宿舍里都不一定联网，而示例图必须随时可用。
唯一例外是 matplotlib 自带的示例照片（若存在则加载）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SAMPLES", "SAMPLE_NAMES", "load_sample", "fit_size", "to_uint8"]


# --------------------------------------------------------------------------
# 1. 几何测试图：直边、斜边、细线、角点、棋盘格
# --------------------------------------------------------------------------
def geometry(n: int = 512) -> np.ndarray:
    H = W = n
    y, x = np.mgrid[0:H, 0:W].astype(float)
    img = np.full((H, W), 0.45)  # 平坦背景：这样求导类核对"平坦区"的响应才是 0

    # 左上：棋盘格（高频细节，检验低通核的模糊程度）
    y0, y1, x0, x1 = int(0.04 * n), int(0.40 * n), int(0.04 * n), int(0.40 * n)
    cell = max(4, n // 32)
    yy = (np.arange(y0, y1)[:, None] // cell) % 2
    xx = (np.arange(x0, x1)[None, :] // cell) % 2
    img[y0:y1, x0:x1] = 0.92 * (yy ^ xx)

    # 右上：实心圆 + 同心圆环（检验各向同性）
    cy, cx, r = int(0.22 * n), int(0.72 * n), int(0.16 * n)
    d = np.hypot(y - cy, x - cx)
    img[(d < r) & (d > r * 0.7)] = 1.0
    img[(d < r * 0.55) & (d > r * 0.35)] = 1.0
    img[d < r * 0.15] = 1.0

    # 左下：斜边（45°，检验方向核的响应差异）
    m = (x > y) & (y > 0.55 * n) & (x < 0.45 * n)
    img[m] = 0.85

    # 左下角：一条横向灰度斜坡（恒定一阶导，用来解释"导数是常数"）
    ramp = (y > 0.86 * n) & (x < 0.44 * n)
    img[ramp] = 0.15 + 0.7 * (x[ramp] / (0.44 * n))

    # 右下：斜向条纹（方向性纹理）
    reg = (y > 0.55 * n) & (x > 0.55 * n)
    stripes = 0.5 + 0.45 * np.sin((x + y) * 2 * np.pi / (n / 10))
    img[reg] = stripes[reg]

    # 细线十字（1~2 px 宽，检验锐化/模糊对细节的影响）
    img[abs(y - 0.5 * n) < 1] = 1.0
    img[abs(x - 0.5 * n) < 1] = 1.0
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------
# 2. Shepp-Logan 医学体模（标准 CT 测试图，10 个椭圆叠加）
# --------------------------------------------------------------------------
_SL_ELLIPSES = [
    # (振幅 A, 长轴 a, 短轴 b, 中心 x0, 中心 y0, 旋转角 φ/度)
    (1.00, 0.6900, 0.9200, 0.000, 0.000, 0),
    (-0.80, 0.6624, 0.8740, 0.000, -0.0184, 0),
    (-0.20, 0.1100, 0.3100, 0.220, 0.000, -18),
    (-0.20, 0.1600, 0.4100, -0.220, 0.000, 18),
    (0.10, 0.2100, 0.2500, 0.000, 0.350, 0),
    (0.10, 0.0460, 0.0460, 0.000, 0.100, 0),
    (0.10, 0.0460, 0.0460, 0.000, -0.100, 0),
    (0.10, 0.0460, 0.0230, -0.080, -0.605, 0),
    (0.10, 0.0230, 0.0230, 0.000, -0.606, 0),
    (0.10, 0.0230, 0.0460, 0.060, -0.605, 0),
]


def shepp_logan(n: int = 512) -> np.ndarray:
    H = W = n
    y, x = np.mgrid[0:H, 0:W].astype(float)
    x = (x / (W - 1) - 0.5) * 1.45
    y = (y / (H - 1) - 0.5) * 1.45
    img = np.zeros((H, W), float)
    for A, a, b, x0, y0, phi in _SL_ELLIPSES:
        t = np.deg2rad(phi)
        xr = (x - x0) * np.cos(t) + (y - y0) * np.sin(t)
        yr = -(x - x0) * np.sin(t) + (y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += A
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------
# 3. 彩色测试图：三通道独立卷积的直观证据
# --------------------------------------------------------------------------
def color_chart(n: int = 384) -> np.ndarray:
    H = W = n
    y, x = np.mgrid[0:H, 0:W].astype(float)
    img = np.zeros((H, W, 3), float)
    # 背景：三通道相位不同的渐变（形成对角线色带）
    img[..., 0] = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y) / (2.2 * n))
    img[..., 1] = 0.5 + 0.5 * np.sin(2 * np.pi * (x - y) / (2.6 * n) + 1.2)
    img[..., 2] = x / W
    # 中央三个纯色圆
    d = np.hypot(y - n / 2, x - n / 2)
    for i, (r, col) in enumerate(
        [(0.30 * n, (1.0, 0.25, 0.2)), (0.20 * n, (0.2, 0.9, 0.35)), (0.10 * n, (0.25, 0.45, 1.0))]
    ):
        mask = (d < r) & (d > r - 3)
        img[mask] = col
    img[d < 0.08 * n] = 1.0
    # 左侧白色方块（检验锐化的光晕）
    img[int(0.08 * n) : int(0.28 * n), int(0.06 * n) : int(0.26 * n)] = 1.0
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------
# 4. 椒盐噪声图：均值 vs 中值 的对照实验素材
# --------------------------------------------------------------------------
def salt_pepper(n: int = 384, ratio: float = 0.05, seed: int = 7) -> np.ndarray:
    base = geometry(n)
    rng = np.random.default_rng(seed)
    out = base.copy()
    m = rng.random((n, n))
    out[m < ratio / 2] = 0.0
    out[(m >= ratio / 2) & (m < ratio)] = 1.0
    return out


# --------------------------------------------------------------------------
# 5. 真实照片（matplotlib 内置示例，离线可用；不可用则跳过）
# --------------------------------------------------------------------------
def _photo() -> np.ndarray:
    from matplotlib import cbook
    from PIL import Image

    path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    im = Image.open(path).convert("RGB")
    im.thumbnail((640, 640), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0


SAMPLES = {
    "几何测试图": ("直边/斜边/细线/棋盘格，最能看出各个核的差异", geometry),
    "Shepp-Logan 医学体模": ("标准 CT 测试图，用于模拟医学影像处理", shepp_logan),
    "彩色色带图": ("RGB 三通道，观察逐通道卷积", color_chart),
    "椒盐噪声图": ("含椒盐噪声，用于对照 均值核 vs 中值滤波", salt_pepper),
}
try:  # 有就用，没有不影响
    _p = _photo()
    SAMPLES["真实照片（Grace Hopper）"] = ("自然图像，彩色细节丰富", _photo)
except Exception:  # pragma: no cover
    _p = None

SAMPLE_NAMES = list(SAMPLES.keys())


def load_sample(name: str) -> np.ndarray:
    fn = SAMPLES[name][1]
    return np.asarray(fn(), float)


def fit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """把图像缩放到最长边不超过 max_side（保持长宽比）。"""
    img = np.asarray(img, float)
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    yi = (np.arange(nh) / scale).astype(int).clip(0, h - 1)
    xi = (np.arange(nw) / scale).astype(int).clip(0, w - 1)
    return img[yi][:, xi]


def to_uint8(a: np.ndarray) -> np.ndarray:
    """把任意范围的数组拉伸到 0~255（仅用于显示）。"""
    a = np.asarray(a, float)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
