# -*- coding: utf-8 -*-
"""recall@k 直算：四档检索配置的检索命中率（纯本地零 API，见进展总结 §9.1 #4）。

evidence 覆盖率只是代理指标；新考卷 QA 100% 带 source 字段、答案带【块N】标注，
检索命中率可以免 API 精确计算，给消融曲线（acc: vec 50%→hr_route 85%）补检索侧注脚。

配置与 eval_retrieval.py 完全同名同义（vec / hybrid / hr / hr_route），
抽样也用同一 seed+limit，保证与已有 acc/evidence 数字同一批 100 题、可直接并表。

命中判定（两级）：
  doc    文档级：top-k 内出现来自 gold source PDF 的块。
         对齐：QA["source"]="xx.pdf" vs 索引 record["file"]="xx_chunks.json"，比词干。
  chunk  块级：参考答案里的【块N】↔ 索引 chunk_idx。
         对齐已核实：gen_heldout.py 用 enumerate(chunks,1) 编号（含被过滤块），
         build_index.py 用 enumerate(data) 0-based（含空块占位跳过但序号保留），
         故 gold chunk_idx = N-1，不存在错位。答案无【块N】的题不计入该级分母。

k 的取值：@6（精排后送生成器的量）与 @20（粗排候选量）。
注意 hr 的 @20 与 hybrid 的 @20 恒等（rerank 不改变候选集合），照算以便直接读表。

用法（llf 环境，RAG-lora/ 下单行前台运行，CPU 约十几分钟）：
  python eval_validity/01_recall/eval_recall.py
断点续存：跑完一档立即落盘，重跑自动跳过已完成档（--fresh 强制全部重跑）。
结果存 eval_validity/01_recall/recall_results.json（含逐题明细）。
"""
import os
import sys
import json
import re
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RAG-lora/
sys.path.insert(0, ROOT)

from rag.retrieve import FAISSRetriever, HybridRetriever  # noqa: E402

CONFIGS = ["vec", "hybrid", "hr", "hr_route"]
_CHUNK_PAT = re.compile(r"【块(\d+)】")


def stem_of(name):
    """'xx.pdf' / 'xx_chunks.json' -> 'xx'，两边对齐用。"""
    return re.sub(r"(_chunks\.json|\.pdf|\.json)$", "", name)


def gold_of(x):
    """QA 条目 -> (source 词干, gold chunk_idx 集合)。答案无【块N】时集合为空。"""
    stem = stem_of(x["source"])
    idxs = {int(n) - 1 for n in _CHUNK_PAT.findall(x["output"])}
    return stem, idxs


def retrieve_ranked(cfg, retr, reranker, question, candidates):
    """返回 (排好序的 top-candidates 命中列表, 路由公司或 None)。

    vec/hybrid 直接取 top-candidates（top-6 即其前缀）；
    hr/hr_route 对 candidates 全量精排（rerank 只重排不换集合）。
    """
    if cfg == "vec":
        return FAISSRetriever.search(retr, question, k=candidates), None
    if cfg == "hybrid":
        return retr.search(question, k=candidates, candidates=candidates), None
    company = retr.detect_company(question) if cfg == "hr_route" else None
    hits = retr.search(question, k=candidates, candidates=candidates, company=company)
    hits = reranker.rerank(question, hits, top_k=candidates)
    return hits, company


def _pct(v):
    return f"{v:.1%}" if v is not None else "-"


def judge(hits, stem, gold_idxs, k):
    """top-k 的两级命中：(doc_hit, chunk_hit)；无 gold 块时 chunk_hit=None。"""
    top = hits[:k]
    doc = any(stem_of(h["file"]) == stem for h in top)
    if not gold_idxs:
        return doc, None
    chunk = any(stem_of(h["file"]) == stem and h["chunk_idx"] in gold_idxs for h in top)
    return doc, chunk


def main():
    ap = argparse.ArgumentParser(description="四档检索配置 recall@k 直算（零 API）")
    ap.add_argument("--limit", type=int, default=100, help="抽样题数（同 eval_retrieval）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--ks", default="6,20", help="逗号分隔的 k 值")
    ap.add_argument("--candidates", type=int, default=20, help="粗排候选数=最大 k")
    ap.add_argument("--qa", default=os.path.join(ROOT, "heldout_B", "qa_alpaca.json"))
    ap.add_argument("--index", default="index", help="索引基名（默认 index=heldout_B）")
    ap.add_argument("--out", default=os.path.join(HERE, "recall_results.json"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="bge/reranker 设备（无空卡默认 cpu）")
    ap.add_argument("--fresh", action="store_true", help="忽略已有结果从头跑")
    args = ap.parse_args()
    configs = [c for c in args.configs.split(",") if c]
    ks = sorted(int(k) for k in args.ks.split(","))
    assert max(ks) <= args.candidates, "k 不能超过粗排候选数"

    # 断点续存：已完成配置直接跳过
    report, details = {}, {}
    if os.path.exists(args.out) and not args.fresh:
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            report = {c: v for c, v in old.get("report", {}).items() if not v.get("partial")}
            details = {c: old["details"][c] for c in report if c in old.get("details", {})}
            if report:
                print(f"[续存] 已完成配置：{', '.join(report)}（跳过，--fresh 可重跑）")
        except Exception:
            report, details = {}, {}
    todo = [c for c in configs if c not in report]

    # 与 eval_retrieval.py 逐字节一致的抽样，保证同一批题
    qa = json.load(open(args.qa, encoding="utf-8"))
    sample = random.Random(args.seed).sample(qa, min(args.limit, len(qa)))
    golds = [gold_of(x) for x in sample]
    n_chunk = sum(1 for _, g in golds if g)
    print(f"[数据] {os.path.basename(args.qa)} 共 {len(qa)} 题，抽样 {len(sample)} "
          f"(seed={args.seed})；含【块N】标注 {n_chunk}/{len(sample)} 题（chunk 级分母）")

    idx = os.path.join(ROOT, "rag", args.index + ".faiss")
    meta = os.path.join(ROOT, "rag", args.index + ".json")
    print(f"[检索] 加载 bge-large-zh + BM25 ({args.device}) …")
    retr = HybridRetriever(index_path=idx, meta_path=meta, device=args.device)
    reranker = None
    if any(c in ("hr", "hr_route") for c in todo):
        from rag.rerank import Reranker
        print(f"[精排] 加载 bge-reranker-large ({args.device}) …")
        reranker = Reranker(model_cache_dir=os.path.join(
            os.path.dirname(ROOT), "models"), device=args.device)

    def _save():
        json.dump({"args": vars(args), "report": report, "details": details},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for cfg in todo:
        rows = []
        for i, (x, (stem, gidx)) in enumerate(zip(sample, golds), 1):
            hits, company = retrieve_ranked(cfg, retr, reranker,
                                            x["instruction"], args.candidates)
            row = {"q": x["instruction"], "source": x["source"],
                   "gold_chunks": sorted(gidx), "routed": company}
            for k in ks:
                row[f"doc@{k}"], row[f"chunk@{k}"] = judge(hits, stem, gidx, k)
            rows.append(row)
            if i % 20 == 0 or i == len(sample):
                d = sum(r[f"doc@{ks[0]}"] for r in rows)
                print(f"  [{cfg}] {i}/{len(sample)}  实时doc@{ks[0]}={d / i:.1%}", flush=True)
                if i < len(sample):     # 配置内分段落盘（partial 标记，续跑时重跑本档）
                    report[cfg] = {"partial": True, "n": i}
                    details[cfg] = rows
                    _save()

        rep = {"n": len(rows), "routed": sum(1 for r in rows if r["routed"])}
        for k in ks:
            rep[f"doc_recall@{k}"] = sum(r[f"doc@{k}"] for r in rows) / len(rows)
            cs = [r[f"chunk@{k}"] for r in rows if r[f"chunk@{k}"] is not None]
            rep[f"chunk_recall@{k}"] = (sum(cs) / len(cs)) if cs else None
            rep["chunk_n"] = len(cs)
        report[cfg], details[cfg] = rep, rows
        _save()                          # 每档跑完立即落盘
        print(f"[{cfg}] " + "  ".join(
            f"doc@{k}={rep[f'doc_recall@{k}']:.1%} chunk@{k}={_pct(rep[f'chunk_recall@{k}'])}"
            for k in ks) + f"  routed={rep['routed']}\n", flush=True)

    # 汇总表
    hdr = f"{'配置':<10}" + "".join(f" {'doc@' + str(k):>8} {'chunk@' + str(k):>9}" for k in ks)
    print(hdr + f" {'routed':>7}")
    for cfg in configs:
        r = report.get(cfg)
        if not r or r.get("partial"):
            continue
        cells = "".join(f" {r[f'doc_recall@{k}']:>8.1%} {_pct(r[f'chunk_recall@{k}']):>9}"
                        for k in ks)
        print(f"{cfg:<10}{cells} {r['routed']:>7}")
    print(f"\n已存 {args.out}")


if __name__ == "__main__":
    main()
