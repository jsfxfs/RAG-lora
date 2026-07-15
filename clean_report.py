# -*- coding: utf-8 -*-
"""年报语料清洗与结构化模块(新增阶段, 落在 02_文本抽取 内)

解决 pypdf 直抽的三个真问题:
  1) 数字被抽乱(-54.55% 类)  → 第5步 pdfplumber 抽结构化表格
  2) 页眉页脚污染            → 第2步 跨页检测 + 删除
  3) 章节顺序错乱(财务报告窜前) → 第6步 按"第X节"重排

对外接口:
  extract_pages(path)        -> list[str]  每页原始文本(pypdf)
  clean_pages(pages)         -> str        清洗后纯文本(步骤1-4)
  extract_tables(path)       -> list[dict] 结构化表格(Markdown + 标签)
  reorder_sections(text)     -> str        按章节号重排(步骤6前半)
  build_chunks(text, tables) -> list[str]  文本块 + 表格块(供下游切块)
  dedup(chunks)              -> list[str]  去重(步骤6后半)
"""
import os, re, json, unicodedata
from collections import Counter
from pypdf import PdfReader
import pdfplumber

# ---------- 章节锚点 ----------
CN_NUM = "零一二三四五六七八九十百"
SECTION_PAT = re.compile(
    r'(第\s*[' + CN_NUM + r']+\s*节)|(^[' + CN_NUM + r']+\s*[、.])', re.M)

# 财报三大表 + 主要会计数据关键词(用于给表格打标题)
STMT_KW = ["合并资产负债表", "母公司资产负债表", "资产负债表",
           "合并利润表", "母公司利润表", "利润表",
           "合并现金流量表", "母公司现金流量表", "现金流量表",
           "合并所有者权益变动表", "所有者权益变动表",
           "主要会计数据", "主要财务指标"]


# ============ 步骤 1-4: 文本清洗 ============
def extract_pages(path):
    """按页抽取原始文本(保留页边界, 便于跨页检测页眉页脚)。"""
    r = PdfReader(path)
    return [(pg.extract_text() or "") for pg in r.pages]


def _norm_line(s):
    return unicodedata.normalize("NFKC", s).strip()


def clean_pages(pages):
    # 1. 逐页: NFKC 归一 + 删纯数字行(页码)
    cleaned = []
    for pg in pages:
        lines = [_norm_line(l) for l in pg.split("\n")]
        lines = [l for l in lines if not re.fullmatch(r"\d+", l)]
        cleaned.append(lines)

    # 2. 跨页检测页眉/页脚: 在 >=3 页重复出现的首/尾行即判定为运行页眉页脚
    firsts = [ls[0] for ls in cleaned if ls]
    lasts = [ls[-1] for ls in cleaned if ls]
    fcount, lcount = Counter(firsts), Counter(lasts)
    header_lines = {t for t, c in fcount.items() if c >= 3 and len(t) > 2}
    footer_lines = {t for t, c in lcount.items() if c >= 3 and len(t) > 2}

    body_pages = []
    for ls in cleaned:
        if ls and ls[0] in header_lines:
            ls = ls[1:]
        if ls and ls[-1] in footer_lines:
            ls = ls[:-1]
        body_pages.append(ls)

    # 3. 行断裂修复: 强制换行被截断的句子合并回段落; 遇句末标点/章节标题/短行则断段
    paras = []
    buf = ""
    for ls in body_pages:
        for l in ls:
            if not l:
                if buf:
                    paras.append(buf)
                    buf = ""
                continue
            buf = (buf + l) if buf else l
            if (re.search(r"[。！？；：”]", l[-1:]) or SECTION_PAT.match(l)
                    or len(l) <= 6):
                paras.append(buf)
                buf = ""
    if buf:
        paras.append(buf)

    text = "\n".join(paras)

    # 4. 目录/套话清理: 只删明显像目录条目的行(点线引导或"标题  数字")
    text = _remove_toc(text)
    return text


def _remove_toc(text):
    out = []
    for ln in text.split("\n"):
        # 点线引导 / 省略号 / 横线引导 -> 目录条目
        if re.search(r"\.{3,}|…|\u2500{3,}", ln):
            if not re.search(r"[。！？]", ln):
                continue
        # "任意标题 + 末尾页码" 且无句末标点 -> 目录条目
        if re.search(r"^\S{1,30}\s+\d+\s*$", ln) and not re.search(r"[。！？]", ln):
            continue
        out.append(ln)
    return "\n".join(out)


# ============ 步骤 5: 表格结构化 ============
def extract_tables(path):
    results = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages):
            ptext = page.extract_text() or ""
            for t in page.extract_tables():
                rows = [[(c or "").strip() for c in row] for row in t]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2:
                    continue
                md = _to_markdown(rows)
                results.append({
                    "page": idx + 1,
                    "title": _find_stmt_title(ptext),
                    "markdown": md,
                    "rows": rows,
                })
    # 只保留能可靠识别为财报三大表/主要会计数据的表格(标题命中 STMT_KW);
    # pdfplumber 易把栏式正文误判为表, 未命中的"表格"多为噪声 —— 丢弃以保质量、控额度
    results = [r for r in results if r["title"] != "表格"]
    return results


def _to_markdown(rows):
    header = rows[0]
    body = rows[1:]
    fmt = lambda r: "| " + " | ".join(c.replace("|", "/") for c in r) + " |"
    lines = [fmt(header), "| " + " | ".join("---" for _ in header) + " |"]
    for r in body:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        lines.append(fmt(r[:len(header)]))
    return "\n".join(lines)


def _find_stmt_title(ptext):
    for kw in STMT_KW:
        if kw in ptext:
            return kw
    return "表格"


# ============ 步骤 6: 章节重排 + 去重 ============
def _cn_to_int(s):
    """把 '一二三...' / '十' / '十二' 转成 int。"""
    s = s.replace(" ", "")
    if not s:
        return 999
    d = {c: i for i, c in enumerate("零一二三四五六七八九")}
    if s == "十":
        return 10
    if "十" in s:
        a, b = s.split("十", 1)
        tens = d.get(a, 1) if a else 1
        ones = d.get(b, 0) if b else 0
        return tens * 10 + ones
    return d.get(s, 999)


def _section_rank(header):
    m = re.search(r"第\s*([零一二三四五六七八九十百]+)\s*节", header)
    if m:
        return _cn_to_int(m.group(1))
    m = re.search(r"^([一二三四五六七八九十百]+)\s*[、.]", header)
    if m:
        return _cn_to_int(m.group(1))
    return 999


def _split_sections(text):
    matches = list(SECTION_PAT.finditer(text))
    if len(matches) < 2:
        return None
    secs = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        secs.append((m.group().strip(), text[m.end():end].strip()))
    return secs


def reorder_sections(text):
    """按'第X节'序号把被 pypdf 抽乱的章节排回 1→N 顺序。"""
    secs = _split_sections(text)
    if not secs:
        return text
    secs_sorted = sorted(secs, key=lambda x: _section_rank(x[0]))
    return "\n".join((h + "\n" + b) for h, b in secs_sorted)


def dedup(chunks):
    seen, out = set(), []
    for c in chunks:
        key = re.sub(r"\s+", "", c)[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ============ 切块(章节切 + 表格块) ============
def _sliding(text, size=1000, step=1000):
    """长章节内部细切: 不重叠, 仅丢极短碎片, 不过滤数字。"""
    pieces, i = [], 0
    while i < len(text):
        seg = text[i:i + size].strip()
        if len(seg) > 50:
            pieces.append(seg)
        i += step
    return pieces


def chunk_text(text, size=1000, step=1000):
    secs = _split_sections(text)
    if secs:
        # _split_sections 返回 (header, body) 元组列表; 合并成带标题的文本再做切块
        raw = [s if isinstance(s, str) else (s[0] + "\n" + s[1]) for s in secs]
    else:
        raw = _sliding(text, size, step)
    out = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > size:
            out.extend(_sliding(piece, size, step))
        else:
            out.append(piece)
    # 仅丢极短碎片; 数字密集段(财务报表)一律保留——它们是财报问答核心
    return [p for p in out if len(p) > 50]


def build_chunks(text, tables, size=1000, step=1000):
    """清洗文本重排后章节切 + 结构化表格作为特殊块插入。"""
    ordered = reorder_sections(text)
    chunks = chunk_text(ordered, size, step)
    for tb in tables:
        chunks.append(f"【{tb['title']} 第{tb['page']}页】\n{tb['markdown']}")
    return dedup(chunks)
