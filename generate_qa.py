#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文档→QA 生成器(金融年报语料) —— 已接入 clean_report(清洗+表格结构化+按章节切)
流程: 读 PDF → clean_report 抽页/清洗/结构化表格/章节重排切块 → 调强模型(Anthropic 兼容端点)生成问答对
      → 4 阶段目录: 01_年报PDF/(原始PDF) · 02_文本抽取/(清洗文本+表格+切块) · 03_LoRA数据/(alpaca+sharegpt) · 04_RAG数据/(chunks)
用法:
  python generate_qa.py --pilot            # 仅第 1 份年报, 跑满(不再限量8块), 验证清洗+表格质量
  python generate_qa.py                 # 全量 50 份(带断点续存, 已完成的自动跳过)
  python generate_qa.py --limit 10         # 仅前 10 份(显式控制子集)
  python generate_qa.py --max-chunks 20    # 每份最多 20 块
   python generate_qa.py --workers 2        # API 并发数(默认 2; 山大共享代理省着用, 可调大)
   python generate_qa.py --merge-chars 8000  # 相邻非表格块合并发, 单次调用封顶字数(默认 8000; 0=不合并)
   python generate_qa.py --qa-per 6          # 每次调用让模型生成的问答对数量(默认 6)
   python generate_qa.py --fresh            # 忽略已有产出, 从头重跑(目标子集内)
断点续存: 每跑完一份即落盘; 重跑时载入已有 qa_alpaca/qa_sharegpt/rag_chunks, 按 source 跳过已完成 PDF。
"""
import os, sys, json, re, urllib.request, ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
# 4 阶段流水线: 每阶段一个文件夹
PDFS = os.path.join(HERE, "01_年报PDF")           # ① 下载: 原始 PDF
EXTRACT_DIR = os.path.join(HERE, "02_文本抽取")    # ② 抽取: 每份 .txt 纯文本 + _chunks.json
LORA_DIR = os.path.join(HERE, "03_LoRA数据")       # ③ LoRA: alpaca + sharegpt 两种格式
RAG_DIR = os.path.join(HERE, "04_RAG数据")         # ④ RAG: 检索 chunks
os.makedirs(PDFS, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
OUT_ALPACA = os.path.join(LORA_DIR, "qa_alpaca.json")
OUT_SHARE = os.path.join(LORA_DIR, "qa_sharegpt.json")
OUT_RAG = os.path.join(RAG_DIR, "rag_chunks.json")
ENVF = os.path.join(HERE, ".env")
def _arg(name, default=None):
    """读命令行参数, 兼容 `--name=value` 与 `--name value` 两种写法。"""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default

PILOT = "--pilot" in sys.argv
_mc = _arg("--max-chunks")
MAX_CHUNKS = int(_mc) if _mc is not None else None
_lim = _arg("--limit")
LIMIT = int(_lim) if _lim is not None else None      # 仅跑前 N 份(显式子集)
_w = _arg("--workers")
WORKERS = int(_w) if _w is not None else 2           # API 并发数(默认 2, 山大共享代理省着用)
_mc2 = _arg("--merge-chars")
MERGE_CHARS = int(_mc2) if _mc2 is not None else 8000  # 相邻非表格块合并封顶字数(0=不合并)
_qp = _arg("--qa-per")
QA_PER = int(_qp) if _qp is not None else 6            # 单次调用生成的问答对数量
FRESH = "--fresh" in sys.argv                        # 忽略已有产出, 从头重跑

# ---- 读 .env ----
env = {}
with open(ENVF, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
BASE = env["ANTHROPIC_BASE_URL"].rstrip("/")
TOKEN = env["ANTHROPIC_AUTH_TOKEN"]
MODEL = env["ANTHROPIC_DEFAULT_HAIKU_MODEL"]
CTX = ssl.create_default_context()

import clean_report as cr   # 清洗 + 表格结构化 + 按章节切块(见 clean_report.py)


def digit_ratio(seg):
    return sum(c.isdigit() for c in seg) / max(1, len(seg))


def build_prompt():
    lo, hi = max(2, QA_PER - 1), QA_PER + 1
    return (
        "你是金融财报分析助手。下面是一份上市公司年度报告的若干片段(可能来自相邻章节, 用【块N】分隔)。\n"
        f"请基于【片段原文】尽可能多地生成中文问答对(建议 {lo}~{hi} 个, 片段多/信息丰富时可更多), 要求:\n"
        "1) 问题要自然(像真实用户会问的), 回答严格基于片段、不编造;\n"
        "2) 优先事实/定义/业务/定性分析类; 若片段含财务数字, 仅在原文明确给出时使用, 不要推算;\n"
        "3) 只输出 JSON 数组, 不要解释, 格式:[{\"instruction\":\"...\",\"output\":\"...\"}]\n\n"
        "【片段原文】\n"
    )


def call_model(prefix, seg):
    body = json.dumps({
        "model": MODEL, "max_tokens": 2048,
        "messages": [{"role": "user", "content": prefix + seg}],
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/messages", data=body, headers={
            "x-api-key": TOKEN, "anthropic-version": "2023-06-01",
            "content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        d = json.loads(r.read().decode())
    return "".join(c.get("text", "") for c in d.get("content", []) if isinstance(c, dict))


def parse_qa(txt):
    # 去掉可能的 ```json 围栏, 截取首个 [ ... ] 数组
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    s, e = txt.find("["), txt.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        arr = json.loads(txt[s:e + 1])
        return [{"instruction": x.get("instruction", "").strip(),
                 "output": x.get("output", "").strip()}
                for x in arr if x.get("instruction") and x.get("output")]
    except Exception:
        return []


def _load_json(path, default):
    """断点续存: 载入已有产出; --fresh 时忽略, 文件不存在/损坏时回退默认。"""
    if FRESH:
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(alpaca, sharegpt, rag):
    """每处理完一份报告即落盘(断点续存), 中途崩溃不丢已完成的产出。"""
    with open(OUT_ALPACA, "w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=2)
    with open(OUT_SHARE, "w", encoding="utf-8") as f:
        json.dump(sharegpt, f, ensure_ascii=False, indent=2)
    with open(OUT_RAG, "w", encoding="utf-8") as f:
        json.dump(rag, f, ensure_ascii=False, indent=2)


def main():
    files = sorted(f for f in os.listdir(PDFS) if f.endswith(".pdf"))
    if PILOT:
        files = files[:1]          # --pilot: 仅第 1 份
    elif LIMIT is not None:
        files = files[:LIMIT]      # --limit N: 仅前 N 份
    # 断点续存: 载入已有产出, 跳过已完成的 PDF(按 source 去重)
    alpaca = _load_json(OUT_ALPACA, [])
    sharegpt = _load_json(OUT_SHARE, [])
    rag = _load_json(OUT_RAG, [])
    done = set() if FRESH else {e.get("source") for e in rag if isinstance(e, dict)}
    skipped = 0
    for fi, fn in enumerate(files, 1):
        if fn in done:
            skipped += 1
            print(f"[{fi}/{len(files)}] {fn}  已完成, 跳过(断点续存)")
            continue
        path = os.path.join(PDFS, fn)
        try:
            pages = cr.extract_pages(path)
            clean = cr.clean_pages(pages)
            tables = cr.extract_tables(path)
            chunks = cr.build_chunks(clean, tables)
        except Exception as e:
            print(f"[{fi}] 清洗/结构化失败 {fn}: {e}")
            continue
        if MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
        print(f"[{fi}/{len(files)}] {fn}  清洗文本≈{len(clean)}字, 表格={len(tables)}张, 切块≈{len(chunks)}")
        # ② 中间产物: 每份报告保存清洗文本 + 结构化表格 + 切块(便于回看/复现)
        base = re.sub(r"\.pdf$", "", fn)
        with open(os.path.join(EXTRACT_DIR, base + "_clean.txt"), "w", encoding="utf-8") as f:
            f.write(clean)
        with open(os.path.join(EXTRACT_DIR, base + "_tables.json"), "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        with open(os.path.join(EXTRACT_DIR, base + "_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        # 预构建 rag 列表(所有块, 含跳过的纯数字块); 仅非表格纯数字密集块跳过 QA
        # QA 组: 把相邻同类块(非表格 / 表格)按字符预算合并, 减少 API 调用次数; 跨类型不合并以保上下文干净
        groups = []  # [(group_id, text)]
        buf, buf_chars, buf_type, gid = [], 0, None, 0
        for ci, seg in enumerate(chunks, 1):
            is_table = bool(re.match(r"^【.+第\d+页】", seg))
            rag.append({"source": fn, "chunk_id": ci, "text": seg, "is_table": is_table})
            if (not is_table) and digit_ratio(seg) >= 0.5:
                print(f"   块{ci}: 纯数字密集(非表格), 跳过QA(仍进RAG)")
                continue
            cur = "table" if is_table else "text"
            # 类型切换 或 超预算 -> 封口; MERGE_CHARS<=0 时每块独立成组(不合并)
            if buf and (not MERGE_CHARS or cur != buf_type or buf_chars + len(seg) > MERGE_CHARS):
                gid += 1
                groups.append((gid, "\n\n---\n\n".join(buf)))
                buf, buf_chars, buf_type = [], 0, None
            buf.append(f"【块{ci}】\n{seg}")
            buf_chars += len(seg)
            buf_type = cur
        if buf:
            gid += 1
            groups.append((gid, "\n\n---\n\n".join(buf)))
        # {workers} 并发生成 QA(各组独立调用, 不依赖返回顺序)
        qa_by_gid = {}
        if groups:
            prefix = build_prompt()
            def _gen(gid, text):
                try:
                    return gid, parse_qa(call_model(prefix, text))
                except Exception as e:
                    print(f"   组{gid}: 调用失败 {e}")
                    return gid, []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_gen, gid, text) for gid, text in groups]
                for f in as_completed(futs):
                    gid, qas = f.result()
                    qa_by_gid[gid] = qas
                    print(f"   组{gid}: 生成 {len(qas)} 对 (累计 {sum(len(v) for v in qa_by_gid.values())})")
        for gid in sorted(qa_by_gid):
            for q in qa_by_gid[gid]:
                alpaca.append(q)
                sharegpt.append({"conversations": [
                    {"role": "user", "content": q["instruction"]},
                    {"role": "assistant", "content": q["output"]}]})
        # 每完成一份即落盘(断点续存): 后续崩溃不丢本份及之前已完成的产出
        _save_json(alpaca, sharegpt, rag)
        done.add(fn)
    _save_json(alpaca, sharegpt, rag)
    print(f"\n完成: QA对={len(alpaca)}  进 RAG 块={len(rag)}" + (f"  (跳过已完成 {skipped} 份)" if skipped else ""))
    print(f"  -> {os.path.basename(OUT_ALPACA)} / {os.path.basename(OUT_SHARE)} / {os.path.basename(OUT_RAG)}")


if __name__ == "__main__":
    main()
