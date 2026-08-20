# -*- coding: utf-8 -*-
"""held-out 评测集生成器(仅产出, 绝不污染 03/04 训练数据)

用途: 从 01_年报PDF/ 里挑出"训练集(03/04 已有的 10 家)之外"的未见公司,
      为它们生成 切块(RAG 知识库) + QA(评测题), 全部写到 heldout_B/,
      与训练集 03_LoRA数据/ 04_RAG数据/ 物理隔离 —— 保证 FT 从没见过 B。

实验设计(详见 heldout_README.md):
  训练集 A = 03/04 已有的 10 家  -> FT 已训(外部完成, 本脚本不碰)
  评测集 B = 本脚本生成的未见 ~15 家 -> FT 零样本答(它从没见过 B)
                                  -> RAG 的 KB 只放 B 的 chunks 再答
  同题 A/B: X%(FT零样本) vs Y%(RAG) -> "FT 只会背 A, RAG 能答新公司"

用法(需山大代理额度, 由用户确认后跑):
  python gen_heldout.py --dry                 # 预览将挑哪 15 家(不调 API)
  python gen_heldout.py --manifest heldout_B/manifest.json  # 用手动清单(已挑好的 15 家)
  python gen_heldout.py --pick 15             # 自动跨板型挑 15 家并生成
  python gen_heldout.py --merge-chars 8000    # 相邻同型块合并发, 封顶字数(默认 8000; 0=不合并)
  python gen_heldout.py --qa-per 6            # 每次调用生成问答对数(默认 6)
  python gen_heldout.py --workers 4           # API 并发(默认 4)
  python gen_heldout.py --fresh               # 忽略已有 chunks, 从头重跑
注意: 已写出 chunks 的 PDF 会自动跳过(断点续存); --fresh 可强制重跑。

2026-07-26 方案一改造(修复 QA 归属缺失, 见 RAG进展总结 §6.4):
  - 每个问题强制带公司简称(不再用"公司"指代), 每条 QA 记 source(来源 PDF)与 company;
  - QA 断点续存改为按 qa_done.json 进度表(与 chunks 是否存在解耦, chunks 已有则直接复用不重抽 PDF);
  - 每完成 10 组即落盘一次(分段落盘), 单组失败自动重试 3 次。"""
import os, sys, json, re, time, urllib.request, ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(HERE, "01_年报PDF")
TRAIN_RAG = os.path.join(HERE, "04_RAG数据", "rag_chunks.json")  # 训练集来源(用于排除已见公司)
HELDOUT = os.path.join(HERE, "heldout_B")
CHUNK_DIR = os.path.join(HELDOUT, "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)
OUT_ALP = os.path.join(HELDOUT, "qa_alpaca.json")
OUT_SHA = os.path.join(HELDOUT, "qa_sharegpt.json")

ENVF = os.path.join(HERE, ".env")
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

import clean_report as cr


def digit_ratio(seg):
    return sum(c.isdigit() for c in seg) / max(1, len(seg))


def company_of(fn):
    """从文件名解析公司简称: NN_代码_简称_日期.pdf -> 简称(去内部空格, 兼容"五 粮 液")。"""
    parts = re.sub(r"\.pdf$", "", os.path.basename(fn)).split("_")
    return parts[2].replace(" ", "") if len(parts) >= 3 else parts[0]


def build_prompt(qa_per, company):
    lo, hi = max(2, qa_per - 1), qa_per + 1
    return (
        f"你是金融财报分析助手。下面是上市公司「{company}」年度报告的若干片段(可能来自相邻章节, 用【块N】分隔)。\n"
        f"请基于【片段原文】尽可能多地生成中文问答对(建议 {lo}~{hi} 个, 片段多/信息丰富时可更多), 要求:\n"
        f"1) 问题要自然(像真实用户会问的), 且每个问题必须明确写出公司名「{company}」, 禁止用\"公司\"\"该公司\"\"贵公司\"等指代;\n"
        "2) 回答严格基于片段、不编造; 优先事实/定义/业务/定性分析类; 若片段含财务数字, 仅在原文明确给出时使用, 不要推算;\n"
        "3) 只输出 JSON 数组, 不要解释, 格式:[{\"instruction\":\"...\",\"output\":\"...\"}]\n\n"
        "【片段原文】\n"
    )


def call_model(prefix, seg, tries=3):
    body = json.dumps({
        "model": MODEL, "max_tokens": 2048,
        "messages": [{"role": "user", "content": prefix + seg}],
    }).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                f"{BASE}/v1/messages", data=body, headers={
                    "x-api-key": TOKEN, "anthropic-version": "2023-06-01",
                    "content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
                d = json.loads(r.read().decode())
            return "".join(c.get("text", "") for c in d.get("content", []) if isinstance(c, dict))
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def parse_qa(txt):
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


def _arg(n, d=None):
    for i, a in enumerate(sys.argv):
        if a == n and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(n + "="):
            return a.split("=", 1)[1]
    return d


def board_type(fn):
    b = os.path.basename(fn)
    if "科创" in b or re.search(r"688", b):
        return "科创板"
    if "创业" in b or re.search(r"30\d{4}", b):
        return "创业板"
    if b.startswith("ST") or "ST" in b:
        return "ST风险"
    return "主板"


def discover_candidates():
    """排除训练集(03/04 已有的来源)后的未见 PDF。"""
    seen = set()
    if os.path.exists(TRAIN_RAG):
        for c in json.load(open(TRAIN_RAG, encoding="utf-8")):
            if isinstance(c, dict) and c.get("source"):
                seen.add(c["source"])
    return [f for f in sorted(os.listdir(PDFS)) if f.endswith(".pdf") and f not in seen]


def auto_pick(cands, n):
    """跨板型尽量均匀挑 n 家, 保证行业/难度多样。"""
    by = {}
    for f in cands:
        by.setdefault(board_type(f), []).append(f)
    picked = []
    while len(picked) < n:
        progress = False
        for k in sorted(by):
            if by[k] and len(picked) < n:
                picked.append(by[k].pop(0))
                progress = True
        if not progress:
            break
    return picked


def main():
    dry = "--dry" in sys.argv
    fresh = "--fresh" in sys.argv
    manifest = _arg("--manifest")
    pick = _arg("--pick")
    n = int(pick) if pick else 15
    merge = _arg("--merge-chars")
    MERGE_CHARS = int(merge) if merge is not None else 8000
    qp = _arg("--qa-per")
    QA_PER = int(qp) if qp is not None else 6
    w = _arg("--workers")
    WORKERS = int(w) if w is not None else 4
    if manifest:
        picks = json.load(open(manifest, encoding="utf-8"))
    else:
        cands = discover_candidates()
        picks = auto_pick(cands, n) if pick else cands
    print(f"候选未见公司: {len(discover_candidates())} 家; 本次将处理: {len(picks)} 家")
    for f in picks:
        print("  -", f, f"[{board_type(f)}]")
    if dry:
        print("\n[--dry] 仅预览, 未调 API、未写文件。")
        return
    # 断点续存(方案一改造): QA 进度以 qa_done.json 为准(整家跑完才记), 与 chunks 解耦;
    # chunks 已存在则直接复用(不重抽 PDF), 未完成家的残留 QA 重跑前先按 source 清掉防重复。
    DONEF = os.path.join(HELDOUT, "qa_done.json")
    done = set()
    if not fresh and os.path.exists(DONEF):
        try:
            done = set(json.load(open(DONEF, encoding="utf-8")))
        except Exception:
            pass
    # 续存时把磁盘上已有的 QA 载入, 否则被跳过 PDF 的 QA 会丢失(只保留新生成的)
    alpaca, sharegpt = [], []
    if not fresh:
        for p, lst in ((OUT_ALP, alpaca), (OUT_SHA, sharegpt)):
            if os.path.exists(p):
                try:
                    lst.extend(json.load(open(p, encoding="utf-8")))
                except Exception:
                    pass

    def _flush():
        json.dump(alpaca, open(OUT_ALP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(sharegpt, open(OUT_SHA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for fi, fn in enumerate(picks, 1):
        base = re.sub(r"\.pdf$", "", fn)
        if fn in done:
            print(f"[{fi}/{len(picks)}] {fn}  QA 已完成, 跳过")
            continue
        company = company_of(fn)
        chunk_f = os.path.join(CHUNK_DIR, base + "_chunks.json")
        if not fresh and os.path.exists(chunk_f):
            chunks = json.load(open(chunk_f, encoding="utf-8"))  # 复用已有切块, 不重抽 PDF
        else:
            path = os.path.join(PDFS, fn)
            try:
                pages = cr.extract_pages(path)
                clean = cr.clean_pages(pages)
                tables = cr.extract_tables(path)
                chunks = cr.build_chunks(clean, tables)
            except Exception as e:
                print(f"[{fi}] 清洗/结构化失败 {fn}: {e}")
                continue
            # RAG 知识库(B 的 chunks, 本地, 免费) -> heldout_B/chunks/ (切块保持原样, 不合并)
            json.dump(chunks, open(chunk_f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        # 本家上次中断的残留 QA 先清掉, 避免重复
        alpaca[:] = [q for q in alpaca if q.get("source") != fn]
        sharegpt[:] = [q for q in sharegpt if q.get("source") != fn]
        prefix = build_prompt(QA_PER, company)
        # QA 组: 相邻同类块(非表格/表格)按字符预算合并, 减少 API 调用; 跨类型不合并保上下文
        groups = []
        buf, buf_chars, buf_type = [], 0, None
        for ci, seg in enumerate(chunks, 1):
            is_table = bool(re.match(r"^【.+第\d+页】", seg))
            if (not is_table) and digit_ratio(seg) >= 0.5:
                continue
            cur = "table" if is_table else "text"
            if buf and (not MERGE_CHARS or cur != buf_type or buf_chars + len(seg) > MERGE_CHARS):
                groups.append("\n\n---\n\n".join(buf))
                buf, buf_chars, buf_type = [], 0, None
            buf.append(f"【块{ci}】\n{seg}")
            buf_chars += len(seg)
            buf_type = cur
        if buf:
            groups.append("\n\n---\n\n".join(buf))

        def _gen(text):
            try:
                return parse_qa(call_model(prefix, text))
            except Exception as e:
                print(f"   组调用失败 {e}")
                return []

        if groups:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_gen, g) for g in groups]
                for gi, f in enumerate(as_completed(futs), 1):
                    for q in f.result():
                        q["source"], q["company"] = fn, company
                        alpaca.append(q)
                        sharegpt.append({"conversations": [
                            {"role": "user", "content": q["instruction"]},
                            {"role": "assistant", "content": q["output"]}],
                            "source": fn, "company": company})
                    # 分段落盘: 每 10 组存一次, 中途挂掉也只重跑本家
                    if gi % 10 == 0:
                        _flush()
                        print(f"   [{company}] 组 {gi}/{len(groups)} 已落盘", flush=True)
        _flush()
        done.add(fn)
        json.dump(sorted(done), open(DONEF, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[{fi}/{len(picks)}] {fn}  切块={len(chunks)}  组={len(groups)}  QA累计={len(alpaca)}")
    print(f"\n完成: heldout_B/qa_alpaca={len(alpaca)} 条; chunks 已写入 heldout_B/chunks/ (RAG KB)")


if __name__ == "__main__":
    main()
