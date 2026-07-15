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
  python gen_heldout.py --pick 15            # 自动跨板型挑 15 家并生成
  python gen_heldout.py --manifest heldout_B/manifest.json  # 手动指定清单
"""
import os, sys, json, re, urllib.request, ssl
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
    manifest = _arg("--manifest")
    pick = _arg("--pick")
    n = int(pick) if pick else 15
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
    alpaca, sharegpt = [], []
    for fi, fn in enumerate(picks, 1):
        path = os.path.join(PDFS, fn)
        try:
            pages = cr.extract_pages(path)
            clean = cr.clean_pages(pages)
            tables = cr.extract_tables(path)
            chunks = cr.build_chunks(clean, tables)
        except Exception as e:
            print(f"[{fi}] 清洗/结构化失败 {fn}: {e}")
            continue
        base = re.sub(r"\.pdf$", "", fn)
        # RAG 知识库(B 的 chunks, 本地, 免费) -> heldout_B/chunks/
        json.dump(chunks, open(os.path.join(CHUNK_DIR, base + "_chunks.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        seg_by_idx = {}
        for ci, seg in enumerate(chunks, 1):
            is_table = bool(re.match(r"^【.+第\d+页】", seg))
            if (not is_table) and digit_ratio(seg) >= 0.5:
                continue
            seg_by_idx[ci] = seg

        def _gen(ci, seg):
            try:
                return ci, parse_qa(call_model(seg))
            except Exception as e:
                print(f"   块{ci}: 调用失败 {e}")
                return ci, []

        if seg_by_idx:
            with ThreadPoolExecutor(max_workers=2) as ex:
                for f in as_completed(ex.submit(_gen, ci, s) for ci, s in seg_by_idx.items()):
                    ci, qas = f.result()
                    for q in qas:
                        alpaca.append(q)
                        sharegpt.append({"conversations": [
                            {"role": "user", "content": q["instruction"]},
                            {"role": "assistant", "content": q["output"]}]})
        json.dump(alpaca, open(OUT_ALP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(sharegpt, open(OUT_SHA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[{fi}/{len(picks)}] {fn}  切块={len(chunks)}  QA={len(alpaca)}")
    print(f"\n完成: heldout_B/qa_alpaca={len(alpaca)} 条; chunks 已写入 heldout_B/chunks/ (RAG KB)")


if __name__ == "__main__":
    main()
