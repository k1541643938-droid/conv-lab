"""把 录屏讲解脚本.md 转成排版好的 Word 文档（python-docx）。

用法（在项目根目录执行）：
    python tools/md2docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "录屏讲解脚本.md"
DST = ROOT / "录屏讲解脚本.docx"

CN_FONT = "微软雅黑"
MONO_FONT = "Consolas"
ACCENT = RGBColor(0x2F, 0x6F, 0xEB)
GRAY = RGBColor(0x5B, 0x65, 0x73)

TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def set_run_font(run, name=CN_FONT, size=None, bold=None, color=None, mono=False):
    font = run.font
    font.name = MONO_FONT if mono else name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), MONO_FONT if mono else name)
    rfonts.set(qn("w:hAnsi"), MONO_FONT if mono else name)
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if color is not None:
        font.color.rgb = color


def add_inline(par, text, size=10.5, color=None, bold_all=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = par.add_run(part[2:-2])
            set_run_font(r, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = par.add_run(part[1:-1])
            set_run_font(r, size=size - 0.5, color=RGBColor(0xB4, 0x3B, 0x2E), mono=True)
        else:
            r = par.add_run(part)
            set_run_font(r, size=size, bold=bold_all, color=color)


def shade(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def cell_text(cell, text, bold=False, color=None, size=9.5):
    cell.text = ""
    par = cell.paragraphs[0]
    par.paragraph_format.space_before = Pt(2)
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.line_spacing = 1.15
    add_inline(par, text, size=size, color=color, bold_all=bold)


def main():
    lines = open(SRC, encoding="utf-8").read().splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.3)
    sec.right_margin = Cm(2.3)

    normal = doc.styles["Normal"]
    normal.font.name = CN_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    # 页脚页码
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._element.append(fld)
    set_run_font(run, size=9, color=GRAY)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        # 标题
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, stripped[2:], size=20, color=RGBColor(0x14, 0x20, 0x3A), bold_all=True)
            i += 1
            continue

        if stripped.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, stripped[3:], size=14, color=RGBColor(0x14, 0x20, 0x3A), bold_all=True)
            # 标题下加一条细线
            pPr = p._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "4")
            bottom.set(qn("w:color"), "D8DEE9")
            pbdr.append(bottom)
            pPr.append(pbdr)
            i += 1
            continue

        if stripped.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            add_inline(p, stripped[4:], size=12, color=ACCENT, bold_all=True)
            i += 1
            continue

        # 表格
        if stripped.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncol)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False

            # 版心宽 16.4cm，按列数给一个舒服的宽度分配
            total = 16.4
            ratios = {1: [1.0], 2: [0.36, 0.64], 3: [0.18, 0.30, 0.52]}.get(ncol, [1 / ncol] * ncol)
            widths = [Cm(total * r) for r in ratios]

            for ri, row in enumerate(rows):
                trow = table.rows[ri]
                tr_pr = trow._tr.get_or_add_trPr()
                tr_pr.append(OxmlElement("w:cantSplit"))  # 一行不跨页断开
                if ri == 0:
                    tr_pr.append(OxmlElement("w:tblHeader"))  # 表头跨页重复
                for ci in range(ncol):
                    txt = row[ci] if ci < len(row) else ""
                    cell = table.cell(ri, ci)
                    cell.width = widths[ci]
                    if ri == 0:
                        shade(cell, "EEF3FF")
                        cell_text(cell, txt, bold=True, color=RGBColor(0x1F, 0x2B, 0x45))
                    else:
                        cell_text(cell, txt)
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_before = Pt(0)
            spacer.paragraph_format.space_after = Pt(0)
            spacer.paragraph_format.line_spacing = 1.0
            r = spacer.add_run("")
            r.font.size = Pt(4)
            continue

        # 引用块
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            add_inline(p, stripped.lstrip("> ").strip(), size=9.5, color=GRAY)
            i += 1
            continue

        # 无序列表
        if re.match(r"^[-*]\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.2
            add_inline(p, re.sub(r"^[-*]\s+", "", stripped))
            i += 1
            continue

        # 有序列表：保留原编号，用悬挂缩进，避免 Word 把多段列表串成连续编号
        if re.match(r"^\d+\.\s+", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.25
            add_inline(p, stripped)
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.35
        add_inline(p, stripped)
        i += 1

    doc.save(DST)
    print("saved", DST)


if __name__ == "__main__":
    main()
