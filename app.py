"""卷积核实验室 · 图像卷积交互平台（选题 3：卷出无限效果）

运行：
    streamlit run app.py
或双击 一键启动.bat
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from convlab import analysis as A
from convlab import kernels as KL
from convlab import samples as S
from convlab import viz as V
from convlab.ops import PADDINGS, convolve2d, is_separable, median_filter, split_separable

# ---------------------------------------------------------------- 页面设置
st.set_page_config(
    page_title="卷积核实验室 · 卷出无限效果",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
V.setup()

# 关掉浏览器的"自动翻译"。
# 原因：谷歌/Edge 翻译会在 DOM 里插入 <font> 标签，React 卸载元素时会报
# "Failed to execute 'removeChild' on 'Node'"，整个界面就乱掉了。
components.html(
    """
<script>
  try {
    const d = window.parent.document;
    d.documentElement.setAttribute('translate', 'no');
    d.documentElement.classList.add('notranslate');
    if (!d.querySelector('meta[name="google"][content="notranslate"]')) {
      const m = d.createElement('meta');
      m.name = 'google';
      m.content = 'notranslate';
      d.head.appendChild(m);
    }
  } catch (e) {}
</script>
""",
    height=0,
)

st.markdown(
    """
<style>
  .block-container {padding-top: 2.0rem; padding-bottom: 3.0rem; max-width: 1500px;}
  .hero {background: linear-gradient(120deg,#eef3ff 0%,#f7f3ff 100%);
         border:1px solid #e3e9f7; border-radius:18px; padding:18px 26px; margin-bottom:16px;}
  .hero h1 {font-size:1.55rem; margin:0 0 6px 0; color:#14203a; letter-spacing:.5px;}
  .hero p  {margin:0; color:#4b5563; font-size:.9rem; line-height:1.6;}
  .chip {background:#f7f9fd; border:1px solid #e4eaf5; border-radius:14px;
         padding:10px 14px; height:100%;}
  .chip .l {font-size:.72rem; color:#6b7280; letter-spacing:.6px;}
  .chip .v {font-size:1.15rem; font-weight:600; color:#1f2b45; margin-top:2px;}
  .chip .s {font-size:.72rem; color:#8a93a5; margin-top:2px;}
  .note {background:#fff8ec; border-left:4px solid #e8a33d; padding:10px 14px;
         border-radius:0 10px 10px 0; font-size:.88rem; color:#5c4a26;}
  div[data-testid="stMetricValue"] {font-size:1.2rem;}
  section[data-testid="stSidebar"] {min-width: 330px;}
  /* 注意：stExpandSidebarButton（展开侧栏的小箭头）就住在 stToolbar 里面，
     所以绝对不能整体隐藏 stToolbar，否则侧栏一旦折叠就再也打不开了。
     这里反过来做：锁死折叠按钮（演示时误点会让人以为侧栏消失了），
     同时确保展开箭头一定可见。 */
  [data-testid="stSidebarCollapseButton"] {display: none !important;}
  [data-testid="stExpandSidebarButton"] {display: flex !important; visibility: visible !important;}
  [data-testid="stSkillsNudge"] {display: none !important;}
  [data-testid="stDecoration"] {display: none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def chip(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="chip"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render(fig) -> None:
    """统一渲染 matplotlib 图并释放内存。"""
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------- 数据与计算
@st.cache_data(show_spinner=False, max_entries=12)
def get_sample(name: str, size: int) -> np.ndarray:
    return S.fit_size(S.load_sample(name), size)


@st.cache_data(show_spinner=False, max_entries=12)
def read_upload(raw: bytes, size: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "P", "LA"):
        im = im.convert("RGB")
    a = np.asarray(im, dtype=float)
    if a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    a = a / 255.0 if a.max() > 1.5 else a
    return S.fit_size(a, size)


@st.cache_data(show_spinner=False, max_entries=48)
def run_op(img, kernel, mode, msize, padding, stride, flip, bias):
    if mode == "median":
        return median_filter(img, int(msize), padding=padding)
    return convolve2d(img, kernel, padding=padding, stride=stride, flip=flip, bias=bias)


def png_bytes(fig, dpi=300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def disp_uint8(a) -> np.ndarray:
    """显示用：彩色图按 0~1 直接映射，灰度图做自适应拉伸。"""
    a = np.asarray(a, float)
    if a.ndim == 3:
        if a.max() > 1.5:  # 0~255 的输入
            a = a / 255.0
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return S.to_uint8(a)


def arr_png(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(disp_uint8(a)).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.markdown("### ① 选择图像")
    src = st.radio("图像来源", ["内置示例", "上传图片"], horizontal=True, label_visibility="collapsed")
    upload = None
    if src == "内置示例":
        sample_name = st.selectbox("示例图", S.SAMPLE_NAMES, label_visibility="collapsed")
        st.caption(S.SAMPLES[sample_name][0])
    else:
        sample_name = S.SAMPLE_NAMES[0]
        upload = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "bmp", "webp"], label_visibility="collapsed")
        if upload is None:
            st.caption("未上传时使用内置示例图")

    max_side = st.select_slider("处理分辨率（最长边）", options=[256, 384, 512, 640, 768], value=512)

    st.markdown("### ② 选择卷积核")
    _default_kernel = next(i for i, k in enumerate(KL.KERNELS) if k["key"] == "sobel_x")
    pick = st.selectbox(
        "预设卷积核", KL.NAMES, index=_default_kernel, key="kernel_pick", label_visibility="collapsed"
    )
    custom = st.toggle("自定义卷积核（右侧编辑）", key="custom_on", value=False)

    st.markdown("### ③ 运算参数")
    padding = st.selectbox(
        "边界填充 padding", list(PADDINGS.keys()), index=0, format_func=lambda k: PADDINGS[k]
    )
    stride = st.number_input("步长 stride", min_value=1, max_value=4, value=1, step=1)
    op_type = st.radio("运算类型", ["卷积（核翻转 180°）", "互相关（不翻转）"], horizontal=False)
    bias = st.number_input("输出偏置 bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
    flip = op_type.startswith("卷积")

    st.markdown("---")
    st.caption("提示：`边界填充` 影响图像四周的结果；`步长` >1 会让输出变小，相当于下采样。")

# ---------------------------------------------------------------- 取图
size = int(max_side)
img = read_upload(upload.getvalue(), size) if upload is not None else get_sample(sample_name, size)
img = np.asarray(img, float)

preset = KL.load_kernel(pick)
kernel = preset["matrix"]
mode = "median" if preset["key"] == "median3" else "conv"
msize = 3

# ---------------------------------------------------------------- 页头
st.markdown(
    """
<div class="hero">
  <h1>🔬 卷积核实验室 · 卷出无限效果</h1>
  <p>选一张图 → 选一个卷积核 → 立刻看到卷积结果、运算过程和频域解释。
  不用装任何环境，改一个数字就能看到图像变化——这就是卷积的“所见即所得”。</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- 自定义核
if custom:
    with st.expander("✏️ 自定义卷积核：直接改数字，改完立刻看效果", expanded=True):
        c0, c1, c2 = st.columns([1, 2.4, 2.4])
        with c0:
            n = st.select_slider("核尺寸", options=[3, 5, 7], value=3, key="custom_n")
            st.caption("切换尺寸会重置为当前预设核。")
        seed = preset["matrix"] if preset["matrix"].shape[0] == n else KL.resize_kernel(preset["matrix"], n)
        init = pd.DataFrame(seed, columns=[f"c{j + 1}" for j in range(n)])
        with c1:
            edited = st.data_editor(
                init,
                key=f"editor_{n}",
                hide_index=True,
                num_rows="fixed",
                width="stretch",
                column_config={
                    f"c{j + 1}": st.column_config.NumberColumn(f"c{j + 1}", step=0.5, format="%.2f")
                    for j in range(n)
                },
            )
            kernel = np.asarray(edited.values, dtype=float)
            mode = "conv"
            st.caption(
                f"核和（DC 增益）= **{kernel.sum():.3f}**　"
                f"可分性检测：{'可分离' if is_separable(kernel) else '不可分离'}"
            )

        def _reset_editor():
            for k in (3, 5, 7):
                st.session_state.pop(f"editor_{k}", None)

        with c2:
            st.button("↺ 把当前预设载入编辑器", on_click=_reset_editor)
            st.caption("数据可直接从 Excel / 论文里复制粘贴进来。")
else:
    if mode == "median":
        msize = st.sidebar.select_slider("中值窗口大小", options=[3, 5, 7], value=3)

# ---------------------------------------------------------------- 计算
res = run_op(img, kernel, mode, msize, padding, int(stride), flip, float(bias))
h, w = img.shape[:2]
oh, ow = res.shape[:2]
mults = oh * ow * int(kernel.shape[0]) * int(kernel.shape[1]) if mode == "conv" else oh * ow * msize * msize

c1, c2, c3, c4 = st.columns(4)
chip(c1, "输入尺寸", f"{w} × {h}", "（高 × 宽，像素）")
chip(c2, "卷积核", f"{kernel.shape[0]} × {kernel.shape[1]}", preset["kind"] if not custom else "自定义")
chip(c3, "输出尺寸", f"{ow} × {oh}", "stride = " + str(int(stride)))
chip(c4, "乘加次数", f"{mults:,}", "输出像素数 × 核元素数")

st.write("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🎛 实验台", "🔎 运算过程", "📈 频域分析", "⚡ 性能对比", "📚 卷积核库", "📝 原理与报告"]
)

# ================================================================ Tab 1
with tab1:
    left, mid, right = st.columns([1.15, 0.85, 1.15], gap="medium")
    with left:
        st.image(disp_uint8(img), caption="原图", width="stretch")
    with mid:
        if mode == "median":
            st.markdown(
                f'<div class="note">中值滤波是<b>非线性</b>算子，没有卷积核，也谈不上翻转与步长。<br>'
                f"这里固定用 {msize}×{msize} 邻域取中位数。</div>",
                unsafe_allow_html=True,
            )
            st.image(V.kernel_rgba(np.zeros((msize, msize))), width=110)
            st.caption("（图上这个方块只是为了占位，没有数值意义）")
        else:
            render(V.fig_kernel(kernel, f"卷积核 · {preset['name'] if not custom else '自定义'}", figsize=(3.1, 3.5)))
            st.caption(f"核和（DC 增益）= {kernel.sum():.3f}")
    with right:
        disp = st.radio(
            "显示方式",
            ["自动拉伸到 0~255", "以 0 为中心（保留正负）", "取绝对值"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if disp == "以 0 为中心（保留正负）":
            show = V.img_for_display(res, signed=True)
            cap = "卷积结果（0 显示为灰）"
        elif disp == "取绝对值":
            show = V.img_for_display(np.abs(res))
            cap = "卷积结果 |g|"
        elif float(np.min(res)) < -1e-9:
            # 有负值的结果若直接线性拉伸，0 会落在中间灰，边缘几乎看不见
            show = V.img_for_display(np.abs(res))
            cap = "卷积结果 |g|（结果含负值，自动取绝对值显示）"
        else:
            show = V.img_for_display(res)
            cap = "卷积结果"
        st.image(disp_uint8(show), caption=cap, width="stretch")

    st.divider()
    m1, m2 = st.columns([1, 1.15], gap="large")
    with m1:
        st.markdown("##### 定量指标")
        st.dataframe(
            pd.DataFrame(A.metrics_table(img, res)),
            hide_index=True,
            width="stretch",
        )
        if mode == "conv":
            st.caption(
                f"理论乘加次数 = {ow}×{oh}×{kernel.shape[0]}×{kernel.shape[1]} = **{mults:,}** 次"
            )
    with m2:
        st.markdown("##### 自动解读")
        if mode == "median":
            st.markdown(
                "- 中值滤波对**椒盐噪声**特别有效：孤立的白点/黑点因为不是中位数而被直接丢弃。\n"
                "- 它不是卷积，**不能**用频域方法分析，也不满足线性叠加。\n"
                "- 换成「均值滤波」再对比一次：同一张噪声图，均值会把噪声“摊平”而不是“去掉”。"
            )
        else:
            for line in A.insights(kernel, img, res):
                st.markdown("- " + line)

    st.divider()
    st.markdown("##### 导出")
    e1, e2, e3 = st.columns(3)
    rep_fig = V.fig_report(
        img,
        kernel,
        res,
        title=f"卷积效果 · {preset['name'] if not custom else '自定义核'}",
        subtitle=f"padding={padding}, stride={stride}",
        mode={"自动拉伸到 0~255": "stretch", "以 0 为中心（保留正负）": "signed", "取绝对值": "abs"}[disp],
    )
    e1.download_button(
        "⬇️ 报告图（PNG, 300dpi）",
        data=png_bytes(rep_fig),
        file_name="卷积效果对比.png",
        mime="image/png",
        width="stretch",
    )
    plt.close(rep_fig)
    e2.download_button(
        "⬇️ 卷积结果图（PNG）",
        data=arr_png(show),
        file_name="卷积结果.png",
        mime="image/png",
        width="stretch",
    )
    csv = pd.DataFrame(A.metrics_table(img, res)).to_csv(index=False).encode("utf-8-sig")
    e3.download_button(
        "⬇️ 指标表（CSV）", data=csv, file_name="指标.csv", mime="text/csv", width="stretch"
    )

# ================================================================ Tab 2
with tab2:
    st.markdown("##### 卷积是「加权求和」：把核扣在图像上，逐格相乘再相加")
    st.markdown(
        "下面这张图会跟着滑块移动。左边红框是当前扣住的位置，右边展示这一步的完整算术："
        "**输入块 × 卷积核 → 逐点乘积 → 求和 = 输出的一个像素**。"
    )
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    max_oy, max_ox = max(oh - 1, 0), max(ow - 1, 0)
    for kk, mm in (("proc_oy", max_oy), ("proc_ox", max_ox)):
        st.session_state.setdefault(kk, mm // 2)
        st.session_state[kk] = int(min(max(int(st.session_state[kk]), 0), mm))

    oy = cc1.slider("输出行 y", 0, max_oy, key="proc_oy")
    ox = cc2.slider("输出列 x", 0, max_ox, key="proc_ox")

    def _next_px():
        st.session_state["proc_ox"] = (int(st.session_state.get("proc_ox", 0)) + 1) % (max_ox + 1)

    cc3.button("▶ 下一像素", on_click=_next_px, width="stretch")
    cc3.caption(f"当前输出像素 g[{oy}, {ox}]，对应输入中心 f[{oy * int(stride)}, {ox * int(stride)}]")

    if mode == "conv":
        render(V.fig_process(img, kernel, res, oy * int(stride), ox * int(stride), padding, int(stride), flip))
    else:
        st.info("中值滤波没有“逐点乘积”的过程，请在上面选择卷积类核后再看这一页。")

    with st.expander("🤔 为什么卷积要把核翻转 180°，而深度学习里的 conv 其实不翻转？"):
        st.markdown(
            "数学上的卷积定义是 $g[n]=\\sum_k f[n-k]\\,h[k]$，滑动时核是**反着**扣上去的；"
            "而深度学习框架（PyTorch/TensorFlow）实现的是互相关 $\\sum_k f[n+k]\\,h[k]$，不翻转。\n\n"
            "两者只差一次翻转，对“滤波效果”（比如边缘检测）几乎没影响，所以工程上常混着叫。"
            "这里保留 `卷积 / 互相关` 开关，就是为了让你亲手看到这个差别。"
        )
        render(V.fig_conv_vs_corr())

# ================================================================ Tab 3
with tab3:
    st.markdown("##### 卷积核的本质，是一个「频率选择器」")
    st.markdown(
        "卷积在频域等于**相乘**：$G(u,v)=F(u,v)\\cdot H(u,v)$。所以卷积核 = 对图像频谱做加权的乘法器。"
        "左图是核的二维频响（亮 = 放行，暗 = 压制）；中图是它的径向平均；右图是图像卷积前后的功率谱。"
    )
    if mode == "conv":
        render(V.fig_spectra(kernel, img, res))
        fc = A.cutoff_freq(kernel)
        a1, a2, a3 = st.columns(3)
        chip(a1, "-3 dB 截止频率", "无（高通）" if fc is None else f"{fc:.2f}", "归一化到奈奎斯特")
        chip(a2, "直流增益", f"{kernel.sum():.3f}", "= 核元素之和")
        chip(a3, "高频能量变化", f"{A.high_freq_ratio(res) - A.high_freq_ratio(img):+.1%}", "卷积后 - 卷积前")
        st.write("")
        st.markdown(
            f"**怎么看这组图**：{preset['kind']}类核的频响"
            + ("在低频处接近 1、高频迅速滚降——它放行平坦区域、压制细节。" if preset["kind"].startswith("低通") else "")
            + ("在低频处接近 0、高频抬升——它几乎只保留边缘，平坦区域被压成黑。" if "高通" in preset["kind"] else "")
            + ("的形状决定了它只对某个尺度/方向的纹理敏感。" if preset["kind"] in ("带通", "方向低通", "方向差分 + 直流偏移") else "")
        )
    else:
        st.info("中值滤波是非线性算子，没有频率响应曲线。")

# ================================================================ Tab 4
with tab4:
    st.markdown("##### 同一个卷积，三种算法：算得快不快，决定了能不能实时")
    st.markdown(
        "直接卷积的复杂度是 $O(HW\\cdot k_h k_w)$，和核大小成正比；"
        "FFT 卷积把它变成 $O(HW\\log HW)$，与核大小**无关**。"
        "所以核越大、图越大，FFT 越占优；小核小图反而直接法更快（没有变换开销）。"
    )
    cc1, cc2 = st.columns([1, 1.2])
    with cc1:
        bench_choice = st.radio(
            "用多大的核测速",
            ["当前核", "高斯 9×9", "高斯 21×21"],
            horizontal=True,
            help="3×3 的核太小，FFT 的变换开销盖过收益；把核放大到 21×21 就能看到 FFT 反超。",
        )
    with cc2:
        run_bench = st.button("▶ 运行速度测试（约 3~20 秒）", type="primary")

    bench_k = {
        "当前核": kernel,
        "高斯 9×9": KL.gaussian(9, 2.0),
        "高斯 21×21": KL.gaussian(21, 4.0),
    }[bench_choice]
    bench_key = f"bench_{bench_choice}_{int(bench_k.shape[0])}_{img.shape}"

    if run_bench:
        with st.spinner("正在计时……"):
            rows = A.benchmark(img, bench_k, sizes=(128, 256, 512, 768))
        st.session_state["bench_rows"] = rows
        st.session_state["bench_key"] = bench_key
        st.session_state["bench_desc"] = f"{bench_choice}（{bench_k.shape[0]}×{bench_k.shape[1]}）"

    rows = st.session_state.get("bench_rows") if st.session_state.get("bench_key") == bench_key else None
    if rows:
        b1, b2 = st.columns([1, 1.1], gap="large")
        with b1:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                f"测试对象：{st.session_state['bench_desc']}，"
                f"{'多通道' if img.ndim == 3 else '灰度'}图，每档取 3 次最快值。"
                "「直接/FFT 倍数」> 1 表示 FFT 更快。"
            )
        with b2:
            render(V.fig_bench(rows))

        st.markdown("##### 可分离卷积：把 9 次乘法变成 6 次")
        if is_separable(bench_k):
            kc, kr = split_separable(bench_k)
            hh, ww = img.shape[:2]
            if max(bench_k.shape) <= 5:
                detail = f"这个核**可以拆开**：`{np.round(kc, 2).tolist()}` ⊗ `{np.round(kr, 2).tolist()}`。\n\n"
            else:
                detail = (
                    f"这个核**可以拆开**成 {bench_k.shape[0]}×1 和 1×{bench_k.shape[1]} 两个一维核"
                    f"（秩为 1，所有高斯核都满足）。\n\n"
                )
            st.markdown(
                detail
                + f"直接法需要 {hh * ww * bench_k.shape[0] * bench_k.shape[1]:,} 次乘加，"
                f"拆成两次一维卷积只要 {hh * ww * (bench_k.shape[0] + bench_k.shape[1]):,} 次，"
                f"约为原来的 **{(bench_k.shape[0] * bench_k.shape[1]) / (bench_k.shape[0] + bench_k.shape[1]):.1f} 分之一**。"
            )
        else:
            st.markdown(
                "当前的核**不可分离**（秩 > 1）——它不能被拆成两个一维核的乘积。"
                "像 Sobel、高斯、均值这些常见核都可以，但拉普拉斯、浮雕、运动模糊不行。"
            )
    else:
        st.info("点上面的按钮开始计时。")

# ================================================================ Tab 5
with tab5:
    st.markdown("##### 12 个预设核：点「载入」直接换到实验台")


    def _load(name: str) -> None:
        st.session_state["kernel_pick"] = name
        st.session_state["custom_on"] = False

    cols = st.columns(4, gap="medium")
    for i, kern in enumerate(KL.KERNELS):
        with cols[i % 4]:
            if kern["key"] == "median3":
                st.markdown(
                    '<div style="width:88px;height:88px;border-radius:10px;background:#eef2f9;'
                    "display:flex;align-items:center;justify-content:center;color:#5b6b85;"
                    'font-size:12px;line-height:1.35;text-align:center">非线性<br>取中位数</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.image(V.kernel_rgba(kern["matrix"], 88), width=88)
            st.markdown(f"**{kern['name']}**")
            st.markdown(
                f'<div style="min-height:38px;color:#6b7280;font-size:0.78rem;line-height:1.45">'
                f"{kern['kind']}<br>{'可分离' if kern['separable'] else '不可分离'} · 核和 {kern['dc_gain']:.2f}</div>",
                unsafe_allow_html=True,
            )
            st.button("载入", key=f"ld_{kern['key']}", on_click=_load, args=(kern["name"],), width="stretch")

    st.divider()
    st.markdown(f"##### 当前核（{preset['name']}）的分析")
    st.markdown(f"**类型**：{preset['kind']}")
    st.markdown(f"**频域本质**：{preset['freq']}")
    st.markdown(f"**典型用途**：{preset['use']}")
    st.markdown(f"**设计思路**：{preset['idea']}")

# ================================================================ Tab 6
with tab6:
    st.markdown("##### 一、卷积到底在做什么")
    st.latex(r"g[y,x]=\sum_{i}\sum_{j} f[y+i-\lfloor k_h/2\rfloor,\; x+j-\lfloor k_w/2\rfloor]\cdot k[i,j]")
    st.markdown(
        "把一个小矩阵（卷积核）在图像上滑动，每滑到一个位置就把「覆盖住的像素」和「核的权重」"
        "逐点相乘再求和，得到输出的一个像素。所以卷积核 = **一个局部加权规则**。\n\n"
        "三条容易忽略的推论：\n"
        "1. **核和决定亮度**：核和为 1 → 亮度不变；为 0 → 输出均值为 0，只剩边缘；大于 1 → 整体变亮。\n"
        "2. **核的形状决定频率**：权重平滑变化 → 低通；权重正负交替 → 高通；"
        "沿某方向差分 → 只对该方向的边缘敏感。\n"
        "3. **卷积 = 频域相乘**：$G=F\\cdot H$。设计卷积核，本质上是在设计一个频率响应函数。"
    )
    st.divider()

    st.markdown("##### 二、卷积核的选择，到底和什么有关？")
    st.markdown(
        "| 影响因素 | 怎么影响核的设计 | 例子 |\n"
        "|---|---|---|\n"
        "| 目标特征的空间尺度 | 核大小 ≈ 想抓的特征尺寸 | 细纹理用 3×3，粗结构用 9×9 或增大 σ |\n"
        "| 特征的方向性 | 沿垂直方向差分，检该方向的边缘 | Sobel-X 检竖直边缘 |\n"
        "| 噪声类型 | 高斯噪声→线性低通；椒盐噪声→中值 | 均值核 vs 中值滤波 |\n"
        "| 直流分量要求 | 归一化保亮度，零和提边缘 | 均值核 ÷9；Sobel 核和为 0 |\n"
        "| 计算预算 | 可分离核能把 k² 降到 2k | 高斯、Sobel 都可分离 |\n"
        "| 输出显示范围 | 有负值的核要截断/取绝对值 | 拉普拉斯输出必须拉伸 |"
    )
    st.divider()

    st.markdown("##### 三、设计一个卷积核的四步")
    st.markdown(
        "1. **先问要什么**：去噪（低通）、找边缘（高通）、找斑点（带通）、做特效（方向差分）。\n"
        "2. **定频率**：想保留的频段权重给 1，想压制的给 0，中间的做平滑过渡（避免振铃）。\n"
        "3. **定归一化**：需要保持亮度就除以核和；需要提取变化就让它零和。\n"
        "4. **验证**：用本平台看频响曲线 + 高频占比 + 对比度，和预期是否一致。"
    )
    st.divider()

    st.markdown("##### 四、报告可以怎么写（把这里的截图和数字直接用上）")
    st.markdown(
        "1. **问题与方法**：说明用卷积实现图像处理，5 个以上核，说明卷积的定义式与边界处理方式。\n"
        "2. **每个核一节**：贴上「原图 | 核 | 结果」三联图（实验台可一键导出 300dpi），"
        "配一张频响曲线，然后写：它放行什么频率、结果图哪里变了、为什么。\n"
        "3. **横向对比表**：把 10 个核的 对比度 / 平均梯度 / 高频占比 / PSNR 列成一张表"
        "（逐个导出 CSV 汇总即可）。\n"
        "4. **卷积核选择规律**：回答“和什么有关”（见上面第二节），用你自己的实验数据支撑。\n"
        "5. **设计思路**：从“差分=求导”“平滑=积分”出发，解释为什么 Sobel = 平滑⊗差分、"
        "为什么锐化 = 原图 + 高频。\n"
        "6. **计算代价**：附性能对比图，说明可分离性与 FFT 的意义。"
    )
    st.divider()
    st.caption(
        "本平台为课程作业「卷出无限效果」设计：卷积、padding、stride、频域分析全部用 NumPy 手写实现，"
        "未调用任何深度学习框架的卷积函数。"
    )
