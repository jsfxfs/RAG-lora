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


PROMPT = (
    "你是金融财报分析助手。下面是一份上市公司年度报告的一个片段。\n"
    "请基于【片段原文】生成 2~3 个中文问答对, 要求:\n"
    "1) 问题要自然(像真实用户会问的), 回答严格基于片段、不编造;\n"
    "2) 优先事实/定义/业务/定性分析类; 若片段含财务数字, 仅在原文明确给出时使用, 不要推算;\n"
    "3) 只输出 JSON 数组, 不要解释, 格式:[{\"instruction\":\"...\",\"output\":\"...\"}]\n\n"
    "【片段原文】\n"
)


def call_model(seg):
    body = json.dumps({
        "model": MODEL, "max_tokens": 1024,
        "messages": [{"role": "user", "content": PROMPT + seg}],
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
        seg_by_idx = {}
        for ci, seg in enumerate(chunks, 1):
            is_table = bool(re.match(r"^【.+第\d+页】", seg))
            rag.append({"source": fn, "chunk_id": ci, "text": seg, "is_table": is_table})
            if (not is_table) and digit_ratio(seg) >= 0.5:
                print(f"   块{ci}: 纯数字密集(非表格), 跳过QA(仍进RAG)")
                continue
            seg_by_idx[ci] = seg
        # {workers} 并发生成 QA(按块顺序组装结果, 不依赖返回顺序)
        qa_by_idx = {}
        if seg_by_idx:
            def _gen(ci, seg):
                try:
                    return ci, parse_qa(call_model(seg))
                except Exception as e:
                    print(f"   块{ci}: 调用失败 {e}")
                    return ci, []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_gen, ci, seg) for ci, seg in seg_by_idx.items()]
                for f in as_completed(futs):
                    ci, qas = f.result()
                    qa_by_idx[ci] = qas
                    print(f"   块{ci}: 生成 {len(qas)} 对 (累计 {sum(len(v) for v in qa_by_idx.values())})")
        for ci in seg_by_idx:
            for q in qa_by_idx.get(ci, []):
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
