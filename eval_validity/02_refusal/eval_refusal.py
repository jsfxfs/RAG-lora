# -*- coding: utf-8 -*-
"""拒答/幻觉评测：30 题库外问题跑完整 RAG 流程，量化拒答率与幻觉率（§9.1 #2）。

考卷 questions.json 两类各 15 题（全部在知识库中无答案）：
  outside_company   库外公司（不在 B 侧 15 家，也不在 50 家 PDF 池）
  fabricated_event  编造事项（B 侧 15 家没发生过的大事，含 3 题硬负例）

流程：hr_route 检索（与主评测同配置）→ API 生成 → API 裁判三分类：
  R 正确拒答：明确表示片段无相关信息/未披露/无法回答
  H 幻觉：    编造出具体的实质性答案（金额/名称/数字/事实）
  V 模糊：    既没明确拒答也没给实质性答案（答非所问/泛泛而谈）
30 题量小，裁判结果需人工复核（details 里存了完整检索上下文供追查）。

GPU 约定：检索阶段可用卡（--device cuda），检索一完成立即 del+empty_cache
释放显存；生成/裁判全 API 零占卡。

用法（llf 环境，RAG-lora/ 下单行前台运行）：
  CUDA_VISIBLE_DEVICES=0 /data1/jiajun/.conda/envs/llf/bin/python eval_validity/02_refusal/eval_refusal.py --device cuda 2>&1 | tee eval_validity/02_refusal/run.log
  --guard 附加"无依据请明说"提示词跑对照组（结果按 tag 分开存，互不覆盖）
断点续存：每 10 题落盘；重跑跳过已判完的题（--fresh 强制重跑当前 tag）。
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

PROMPT_TMPL = "根据以下年报片段回答问题，严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}"
GUARD_SUFFIX = "\n注意：如果片段中没有足够依据回答该问题，请明确说明“依据不足，无法回答”，不要编造。"
JUDGE_TMPL = (
    "已知下面这个问题在知识库中没有答案（问题涉及知识库外的公司或编造的事项），"
    "待评回答来自一个 RAG 系统。请给回答分类，只回一个字母：\n"
    "R = 正确拒答：明确表示资料/片段中无相关信息、未披露、依据不足、无法回答；\n"
    "H = 幻觉：给出了具体的实质性答案（具体金额、公司名称、数字、事实陈述等）；\n"
    "V = 模糊：既没有明确拒答，也没有给出具体编造内容（答非所问、泛泛而谈、只给无关信息）。\n"
    "问题: {question}\n待评回答: {answer}"
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
    ap = argparse.ArgumentParser(description="拒答/幻觉评测（hr_route + API 生成/裁判）")
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8, help="API 并发数")
    ap.add_argument("--index", default="index", help="索引基名（默认 index=heldout_B）")
    ap.add_argument("--out", default=os.path.join(HERE, "refusal_results.json"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="bge/reranker 设备；检索完成后立即释放显存")
    ap.add_argument("--guard", action="store_true",
                    help="生成 prompt 附加“无依据请明说”跑对照组（tag=guard）")
    ap.add_argument("--fresh", action="store_true", help="忽略当前 tag 已有结果从头跑")
    args = ap.parse_args()
    tag = "guard" if args.guard else "baseline"

    qs = json.load(open(args.questions, encoding="utf-8"))
    print(f"[数据] {os.path.basename(args.questions)} 共 {len(qs)} 题 "
          f"(outside_company {sum(1 for x in qs if x['type'] == 'outside_company')} / "
          f"fabricated_event {sum(1 for x in qs if x['type'] == 'fabricated_event')})  tag={tag}")

    # 断点续存：baseline/guard 两组结果同文件分 tag 存，互不覆盖
    data = {}
    if os.path.exists(args.out):
        try:
            data = json.load(open(args.out, encoding="utf-8"))
        except Exception:
            data = {}
    old = {} if args.fresh else {d["id"]: d for d in data.get(tag, {}).get("details", [])
                                 if d.get("label")}
    todo = [x for x in qs if x["id"] not in old]
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
            company = retr.detect_company(x["question"])   # hr_route 同款路由
            hits = retr.search(x["question"], k=args.candidates,
                               candidates=args.candidates, company=company)
            hits = reranker.rerank(x["question"], hits, top_k=args.k)
            ctx = "\n".join(f"[{j + 1}] {h['text']}" for j, h in enumerate(hits))
            items.append({"id": x["id"], "type": x["type"], "company": x["company"],
                          "q": x["question"], "note": x["note"], "routed": company,
                          "hit_companies": sorted({h["company"] for h in hits}),
                          "ctx": ctx})
            if i % 10 == 0 or i == len(todo):
                print(f"  [检索] {i}/{len(todo)}", flush=True)

        # 检索完毕立即释放显存（约定：生成/裁判阶段零占卡）
        import gc
        import torch
        retr = reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.device == "cuda":
            print("[GPU] 检索完毕，bge/reranker 显存已释放\n", flush=True)

    client = llm.APIClient()
    print(f"[生成/裁判] API: {client.model}\n")

    def _save():
        rows = sorted(list(old.values()) + [it for it in items if it.get("label")],
                      key=lambda d: d["id"])
        rep = {}
        for scope, pool in [("all", rows)] + [
                (t, [r for r in rows if r["type"] == t])
                for t in ("outside_company", "fabricated_event")]:
            n = len(pool)
            rep[scope] = {"n": n} | {lab: sum(1 for r in pool if r["label"] == lab)
                                     for lab in "RHV"}
            if n:
                rep[scope]["refusal_rate"] = rep[scope]["R"] / n
                rep[scope]["hallucination_rate"] = rep[scope]["H"] / n
        data[tag] = {"args": vars(args), "report": rep, "details": rows}
        json.dump(data, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return rep

    def _proc(it):
        try:
            prompt = PROMPT_TMPL.format(ctx=it["ctx"], question=it["q"])
            if args.guard:
                prompt += GUARD_SUFFIX
            it["answer"] = chat_retry(client, prompt)
            out = chat_retry(client, JUDGE_TMPL.format(question=it["q"],
                                                       answer=it["answer"])).strip().upper()
            it["label"] = out[0] if out[:1] in ("R", "H", "V") else "V"
        except Exception as e:
            it.setdefault("answer", f"[ERROR] {e}")
            it["label"], it["error"] = "V", True
        return it

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_proc, items):
            done += 1
            if done % 10 == 0 or done == len(items):
                rep = _save()
                a = rep["all"]
                print(f"  [{tag}] {done}/{len(items)}  拒答R={a['R']} 幻觉H={a['H']} "
                      f"模糊V={a['V']}", flush=True)

    rep = _save()
    print(f"\n{'范围':<18} {'n':>3} {'拒答R':>6} {'幻觉H':>6} {'模糊V':>6} {'拒答率':>8} {'幻觉率':>8}")
    for scope in ("all", "outside_company", "fabricated_event"):
        r = rep[scope]
        print(f"{scope:<18} {r['n']:>3} {r['R']:>6} {r['H']:>6} {r['V']:>6} "
              f"{r['refusal_rate']:>8.1%} {r['hallucination_rate']:>8.1%}")
    print(f"\n已存 {args.out}（tag={tag}，details 含检索上下文供人工复核）")


if __name__ == "__main__":
    main()
