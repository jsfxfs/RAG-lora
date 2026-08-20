# -*- coding: utf-8 -*-
"""口语化改写评测：40 题改写版跑 hr_route 完整链路，对比原问法 82.5%（§9.1 #1）。

考卷 rewritten_40.json：Kimi-K2.5（异家族）改写 + 人工复核（#36 手修），
公司名/考点/参考答案与 original_40.json 一致，仅问法口语化。
检索/生成/裁判与主评测 eval_retrieval.py 的 hr_route 档完全同口径
（k=6、candidates=20、DeepSeek 生成+裁判、同 JUDGE_TMPL），保证 acc 可直接对比：
掉 ≤5pp 判皮实；掉多了则说明检索对问法敏感，是上 query 增强（HyDE 等）的依据。

CPU 约定：当前无空卡，检索全程 --device cpu（bge/reranker CPU 推理，40 题可承受）。

用法（llf 环境，RAG-lora/ 下单行前台运行）：
  /data1/jiajun/.conda/envs/llf/bin/python eval_validity/03_colloquial/eval_colloquial.py 2>&1 | tee eval_validity/03_colloquial/run.log
断点续存：每 10 题落盘；重跑跳过已判完的题（--fresh 强制重跑）。
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RAG-lora/
sys.path.insert(0, ROOT)

# 预导入 transformers 类：懒加载在子线程首次触发会失败（同 eval_retrieval）
from transformers import AutoModel, AutoTokenizer  # noqa: E402, F401
from rag import llm  # noqa: E402
from rag.retrieve import HybridRetriever  # noqa: E402

# 与 eval_retrieval.py 完全同款，保证口径可比
PROMPT_TMPL = "根据以下年报片段回答问题，严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}"
JUDGE_TMPL = (
    "对照参考答案判断待评回答是否正确：关键事实/数值/结论一致记 1（表述不同没关系），"
    "事实错误、数值不符、拒答或答非所问记 0。只回一个数字。\n"
    "问题: {question}\n参考答案: {ref}\n待评回答: {answer}"
)


def chat_retry(client, prompt, tries=3, timeout=120):
    """带重试的 API 调用（同 eval_retrieval）：偶发读超时退避重试不炸全场。"""
    for t in range(tries):
        try:
            return client.chat(prompt, timeout=timeout)
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def main():
    ap = argparse.ArgumentParser(description="口语化改写评测（hr_route + API 生成/裁判）")
    ap.add_argument("--questions", default=os.path.join(HERE, "rewritten_40.json"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8, help="API 并发数")
    ap.add_argument("--index", default="index", help="索引基名（默认 index=heldout_B）")
    ap.add_argument("--out", default=os.path.join(HERE, "colloquial_results.json"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="bge/reranker 设备（当前约定 cpu）")
    ap.add_argument("--fresh", action="store_true", help="忽略已有结果从头跑")
    args = ap.parse_args()

    qs = json.load(open(args.questions, encoding="utf-8"))
    orig_acc = sum(1 for x in qs if x.get("orig_judge") == 1) / len(qs)
    print(f"[数据] {os.path.basename(args.questions)} 共 {len(qs)} 题  "
          f"原问法 hr_route acc={orig_acc:.1%}")

    data = {}
    if os.path.exists(args.out):
        try:
            data = json.load(open(args.out, encoding="utf-8"))
        except Exception:
            data = {}
    old = {} if args.fresh else {d["idx"]: d for d in data.get("details", [])
                                 if d.get("judge") is not None}
    todo = [x for x in qs if x["idx"] not in old]
    if old:
        print(f"[续存] 已判完 {len(old)} 题（跳过，--fresh 可重跑）")

    items = []
    if todo:
        idx = os.path.join(ROOT, "rag", args.index + ".faiss")
        meta = os.path.join(ROOT, "rag", args.index + ".json")
        print(f"[检索] 加载 bge-large-zh + BM25 ({args.device}) …")
        retr = HybridRetriever(index_path=idx, meta_path=meta, device=args.device)
        from rag.rerank import Reranker
        print(f"[精排] 加载 bge-reranker-large ({args.device}) …")
        reranker = Reranker(model_cache_dir=os.path.join(llm.PROJ, "models"),
                            device=args.device)
        for i, x in enumerate(todo, 1):
            q = x["rewritten"]                              # 用改写后的口语问法检索
            company = retr.detect_company(q)                # hr_route 同款路由
            hits = retr.search(q, k=args.candidates,
                               candidates=args.candidates, company=company)
            hits = reranker.rerank(q, hits, top_k=args.k)
            ctx = "\n".join(f"[{j + 1}] {h['text']}" for j, h in enumerate(hits))
            items.append({"idx": x["idx"], "company": x["company"],
                          "original": x["question"], "q": q, "ref": x["output"],
                          "orig_judge": x.get("orig_judge"), "routed": company,
                          "hit_companies": sorted({h["company"] for h in hits}),
                          "ctx": ctx})
            if i % 10 == 0 or i == len(todo):
                print(f"  [检索] {i}/{len(todo)}", flush=True)

        # 检索完毕释放（CPU 下无显存问题，保持与既有脚本一致的清理习惯）
        import gc
        retr = reranker = None
        gc.collect()

    client = llm.APIClient()
    print(f"[生成/裁判] API: {client.model}\n")

    def _save():
        rows = sorted(list(old.values())
                      + [it for it in items if it.get("judge") is not None],
                      key=lambda d: d["idx"])
        n = len(rows)
        acc = sum(r["judge"] for r in rows) / n if n else None
        # 逐题翻转统计：原对现错=regressed，原错现对=improved
        reg = [r["idx"] for r in rows if r.get("orig_judge") == 1 and r["judge"] == 0]
        imp = [r["idx"] for r in rows if r.get("orig_judge") == 0 and r["judge"] == 1]
        rep = {"n": n, "acc": acc, "orig_acc": orig_acc,
               "delta_pp": (acc - orig_acc) * 100 if acc is not None else None,
               "regressed_idx": reg, "improved_idx": imp}
        data.update({"args": vars(args), "report": rep, "details": rows})
        json.dump(data, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return rep

    def _proc(it):
        try:
            it["answer"] = chat_retry(client, PROMPT_TMPL.format(ctx=it["ctx"],
                                                                 question=it["q"]))
            out = chat_retry(client, JUDGE_TMPL.format(question=it["q"], ref=it["ref"],
                                                       answer=it["answer"]))
            it["judge"] = 1 if out.strip().startswith("1") else 0
        except Exception as e:
            it.setdefault("answer", f"[ERROR] {e}")
            it["judge"], it["error"] = 0, True
        return it

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_proc, items):
            done += 1
            if done % 10 == 0 or done == len(items):
                rep = _save()
                print(f"  [评测] {done}/{len(items)}  当前 acc={rep['acc']:.1%}",
                      flush=True)

    rep = _save()
    print(f"\n原问法 acc={rep['orig_acc']:.1%}  口语化 acc={rep['acc']:.1%}  "
          f"差值={rep['delta_pp']:+.1f}pp（掉 ≤5pp 判皮实）")
    print(f"翻转明细：原对现错 {rep['regressed_idx']}  原错现对 {rep['improved_idx']}")
    print(f"\n已存 {args.out}（details 含检索上下文与答案供人工复核）")


if __name__ == "__main__":
    main()
