"""
预设卷积核库（11 个核，覆盖低通 / 高通 / 方向 / 带通 / 非线性对照）

每个核除了数值矩阵，还带上四条「可以直接写进报告」的分析：
    kind  —— 类型（低通 / 高通 / 带通 / 方向性 / 非线性）
    freq  —— 频域本质（它放行哪些频率、压制哪些频率）
    use   —— 适用场景
    idea  —— 设计思路（为什么长这样）
"""

from __future__ import annotations

import numpy as np

from .ops import is_separable


def _box(n: int) -> np.ndarray:
    return np.ones((n, n), float) / (n * n)


def _gaussian(n: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian(n: int = 5, sigma: float = 1.0) -> np.ndarray:
    """公开版高斯核生成器（基准测试里用来造不同尺寸的核）。"""
    return _gaussian(n, sigma if sigma > 0 else max(1.0, n / 5.0))


def _log(n: int = 5, sigma: float = 1.2) -> np.ndarray:
    """高斯拉普拉斯（Laplacian of Gaussian）：先平滑再求二阶导，带通。"""
    ax = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    g = (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    g -= g.mean()  # 零和，保证不改变整体亮度
    return g


def _motion(n: int = 9, angle_deg: float = 45.0) -> np.ndarray:
    """沿指定方向的匀速运动模糊核：一条线，线上权重相等。"""
    k = np.zeros((n, n), float)
    c = (n - 1) / 2.0
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-c, c, 4 * n):
        y = int(round(c + t * np.sin(a)))
        x = int(round(c + t * np.cos(a)))
        if 0 <= y < n and 0 <= x < n:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s else k


_RAW = [
    {
        "key": "identity",
        "name": "00 恒等核（对照）",
        "matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "kind": "恒等",
        "freq": "频响恒为 1，全部频率原样通过。",
        "use": "验证卷积实现是否正确，对比卷积与互相关（核翻转）的差异。",
        "idea": "只有中心为 1，等于「什么都不做」。但它正好用来检查边界处理、核翻转、步长有没有写错。",
    },
    {
        "key": "mean3",
        "name": "01 均值滤波 3×3",
        "matrix": _box(3).tolist(),
        "kind": "低通",
        "freq": "理想低通的粗糙近似：通带内缓慢衰减，在 f=1/3、2/3（cycles/px）处出现零点，"
                "旁瓣会带来轻微振铃。",
        "use": "抑制高斯噪声、快速模糊、降采样前的抗混叠。",
        "idea": "把 3×3 邻域等权平均，相当于取局部直流分量。权重全为 1/9 是为了让 DC 增益 = 1，"
                "整体亮度不变——归一化是设计低通核的第一原则。",
    },
    {
        "key": "gauss5",
        "name": "02 高斯滤波 5×5 (σ=1)",
        "matrix": _gaussian(5, 1.0).tolist(),
        "kind": "低通",
        "freq": "高斯的傅里叶变换仍是高斯：没有旁瓣、没有振铃，通带平滑单调衰减。σ 越大截止频率越低。",
        "use": "平滑去噪、抗锯齿、CNN 预处理、多尺度金字塔的基础。",
        "idea": "权重随距中心距离按 exp(-r²/2σ²) 衰减——离得越远越不重要。相比均值核，它不会"
                "产生方形的块状边界伪影，是各向同性的最优平滑核。",
    },
    {
        "key": "sobel_x",
        "name": "03 Sobel-X（竖直边缘）",
        "matrix": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "kind": "方向高通",
        "freq": "沿 x 方向是一阶差分（近似 ∂/∂x），频响在低频随频率线性上升——高通；"
                "沿 y 方向是 [1,2,1] 平滑，所以对水平方向的高频不敏感。",
        "use": "提取竖直边缘；与 Sobel-Y 合成梯度幅值/方向，是 Canny 的第一步。",
        "idea": "拆开看 = [1,2,1]ᵀ（竖直平滑）⊗ [-1,0,1]（水平差分）。差分抓边缘、平滑压噪声，"
                "两者相乘得到抗噪的一阶微分。核和 = 0（零和核），平坦区域输出 0。",
    },
    {
        "key": "sobel_y",
        "name": "04 Sobel-Y（水平边缘）",
        "matrix": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "kind": "方向高通",
        "freq": "与 Sobel-X 相差 90° 旋转：沿 y 差分、沿 x 平滑，对水平边缘响应最强。",
        "use": "提取水平边缘；与 Sobel-X 一起求梯度方向 θ = atan2(Gy, Gx)。",
        "idea": "把 Sobel-X 转置得到。想检测哪个方向的边缘，就让差分方向垂直于边缘走向——"
                "这是设计方向核的通用法则。",
    },
    {
        "key": "prewitt_x",
        "name": "05 Prewitt-X",
        "matrix": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "kind": "方向高通",
        "freq": "同样是一阶差分，但平滑部分用等权均值，旁瓣比 Sobel 大，对噪声更敏感。",
        "use": "教学对照：用来解释平滑权重如何影响抗噪性。",
        "idea": "把 Sobel 中间的 [1,2,1] 换成 [1,1,1]。中心加权（2）相当于给正对的像素更高话语权，"
                "让响应更集中在边缘法向上——这一点差异就是 Sobel 更常用的原因。",
    },
    {
        "key": "laplace4",
        "name": "06 拉普拉斯 4 邻域",
        "matrix": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        "kind": "各向同性高通",
        "freq": "二阶微分：频响 ∝ -(4 - 2cos2πu - 2cos2πv)，对旋转各向同性，低频≈0、高频放大。",
        "use": "边缘检测、图像增强、锐化的核心算子（与高斯组合成 LoG / DoG）。",
        "idea": "离散二阶导 = 四邻域和 - 4×中心，核和 = 0（零和）。零和意味着输出均值为 0、"
                "会有负值——所以显示时必须拉伸或取绝对值，否则看起来一片黑。二阶导比一阶导更怕噪声。",
    },
    {
        "key": "sharpen",
        "name": "07 锐化（非锐化掩蔽）",
        "matrix": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "kind": "高通增强",
        "freq": "DC 增益 = 1（亮度不变），中低频≈1，高频被放大到 5 倍以上——典型的高频提升滤波器。",
        "use": "照片锐化、医学影像细节增强、翻拍文档增强。",
        "idea": "I + 4×拉普拉斯 = 原图 + 高频。本质是让图像加上自己的边缘，使边缘两侧对比更强。"
                "系数过大（如 9）会产生明显光晕（halo），这是锐化过度的典型症状。",
    },
    {
        "key": "emboss",
        "name": "08 浮雕（方向差分）",
        "matrix": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "kind": "方向差分 + 直流偏移",
        "freq": "沿 45° 方向的一阶差分，同时核和 = 1 保留了直流分量。",
        "use": "艺术滤镜、纹理增强、石雕或金属表面质感提取。",
        "idea": "方向导数只能得到边缘，看起来发黑；再加上一个直流偏置（让核和 = 1），输出就变成"
                "中灰底色加亮暗浮雕，像从左上角打光。核和 = 1 就是偏置的实现方式。",
    },
    {
        "key": "motion9",
        "name": "09 运动模糊 9×9（45°）",
        "matrix": _motion(9, 45).tolist(),
        "kind": "方向低通",
        "freq": "沿运动方向的频率被强烈压制，垂直于运动方向的频率基本保留 → 各向异性低通。",
        "use": "模拟运动模糊、去运动模糊（反卷积）的对照实验、艺术效果。",
        "idea": "把一条线段上的像素等权累加。线段越长模糊越重，方向决定拖影往哪走。"
                "它也是图像复原里最有名的退化模型（模糊核估计）。",
    },
    {
        "key": "log5",
        "name": "10 高斯拉普拉斯 LoG 5×5",
        "matrix": _log(5, 1.2).tolist(),
        "kind": "带通",
        "freq": "先高斯低通、再拉普拉斯高通 → 带通。中心频率由 σ 决定，σ 越大越偏向粗尺度结构。",
        "use": "斑点检测（blob detection）、多尺度边缘、纹理分析。",
        "idea": "直接对图像求二阶导会被噪声淹没，所以先用高斯平滑再求导（微分与卷积可交换）。"
                "因为零和，输出只保留尺度接近 σ 的结构，平坦区和极细纹理都被压掉。",
    },
    {
        "key": "median3",
        "name": "11 中值滤波 3×3（非线性）",
        "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "kind": "非线性（不是卷积）",
        "freq": "没有频率响应可言——它不满足叠加原理，不能写成加权求和。",
        "use": "椒盐（脉冲）噪声去除，同时比均值滤波更好地保住边缘。",
        "idea": "取 9 个数的中位数而不是加权和。中位数对极端值免疫，所以孤立白点/黑点会被直接丢掉。"
                "放进平台是为了做对照：卷积核解决不了的问题需要非线性算子——这正是 CNN 里"
                "要加 ReLU 和池化的原因。",
    },
]

KERNELS: list[dict] = []
for _k in _RAW:
    m = np.asarray(_k["matrix"], float)
    KERNELS.append(
        {
            **{kk: vv for kk, vv in _k.items() if kk != "matrix"},
            "matrix": m,
            "size": m.shape[0],
            "linear": _k["key"] != "median3",
            "separable": bool(is_separable(m)) if _k["key"] != "median3" else False,
            "dc_gain": float(m.sum()),
        }
    )

BY_KEY = {k["key"]: k for k in KERNELS}
NAMES = [k["name"] for k in KERNELS]


def get(key: str) -> dict:
    return BY_KEY[key]


def load_kernel(name: str) -> dict:
    for k in KERNELS:
        if k["name"] == name:
            return k
    raise KeyError(name)


def resize_kernel(k: np.ndarray, n: int) -> np.ndarray:
    """把核缩放/裁剪到 n×n（自定义编辑器切换尺寸时用）。"""
    k = np.asarray(k, float)
    if k.shape == (n, n):
        return k.copy()
    out = np.zeros((n, n), float)
    if k.shape[0] <= n:
        o = (n - k.shape[0]) // 2
        out[o : o + k.shape[0], o : o + k.shape[1]] = k
        return out
    o = (k.shape[0] - n) // 2
    return k[o : o + n, o : o + n].copy()
