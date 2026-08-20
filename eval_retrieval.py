# -*- coding: utf-8 -*-
"""检索增强量化评测：对比多种检索配置的端到端答案正确率（生成/裁判全走 API）。

配置（同一份抽样 QA、同一生成/裁判模型，只变检索）：
  vec        纯向量 top-k（基线，与 eval_heldout 的 RAG 侧一致）
  hybrid     BM25+向量混合召回（RRF）top-k
  hr         混合召回 20 候选 -> bge-reranker 精排 top-k
  hr_route   hr + 公司路由（问题里识别到代码/简称时只搜该公司块）

指标：
  acc       LLM-as-Judge 对照参考答案打 1/0（2 次 API/题：生成+裁判）
  evidence  证据命中率（免费代理指标）：参考答案里 >=4 位的数字是否出现在
            检索上下文中；参考答案无数字的题不计入该指标分母
  routed    路由触发数（仅 hr_route 有意义）

用法（llf 环境）：
  python eval_retrieval.py --limit 100            # 默认：生成/裁判全 API，CPU 检索
  python eval_retrieval.py --configs vec,hr       # 只跑指定配置
  # 本地生成（裁判仍走 API 保证可比）：LoRA / 纯 base 对照，检索也上 GPU
  CUDA_VISIBLE_DEVICES=3 python eval_retrieval.py --gen-backend local \
      --retrieval-device cuda --configs hr,hr_route --out eval_local/xx.json
结果存 eval_retrieval_results.json（含逐题明细，可追查 bad case）。
断点续存：已完成的配置直接跳过（结果文件里有就不重跑，--fresh 强制重跑）；
配置内每 20 题落盘一次部分结果（partial 标记，重跑时丢弃未完成配置的部分）。
"""
import os
import sys
import json
import re
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 预导入 transformers 类：懒加载在子线程首次触发会失败（同 eval_heldout）
from transformers import AutoModel, AutoTokenizer  # noqa: E402, F401
from rag import llm  # noqa: E402
from rag.retrieve import FAISSRetriever, HybridRetriever  # noqa: E402

PROMPT_TMPL = "根据以下年报片段回答问题，严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}"
JUDGE_TMPL = (
    "对照参考答案判断待评回答是否正确：关键事实/数值/结论一致记 1（表述不同没关系），"
    "事实错误、数值不符、拒答或答非所问记 0。只回一个数字。\n"
    "问题: {question}\n参考答案: {ref}\n待评回答: {answer}"
)
CONFIGS = ["vec", "hybrid", "hr", "hr_route"]


def chat_retry(client, prompt, tries=3, timeout=120):
    """带重试的 API 调用：代理偶发读超时/断连时退避重试，而非炸掉整场评测。"""
    for t in range(tries):
        try:
            return client.chat(prompt, timeout=timeout)
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def evidence_hit(ref, ctx):
    """参考答案中 >=4 位数字是否出现在上下文（去千分位逗号后比对）。无数字返回 None。"""
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", ref)
            if len(re.sub(r"\D", "", n)) >= 4]
    if not nums:
        return None
    ctx_norm = ctx.replace(",", "")
    return any(n in ctx_norm for n in nums)


def retrieve(cfg, retr, reranker, question, k, candidates):
    """按配置检索，返回 (hits, 路由命中的公司或 None)。"""
    if cfg == "vec":                      # 基类的纯向量检索
        return FAISSRetriever.search(retr, question, k=k), None
    if cfg == "hybrid":
        return retr.search(question, k=k), None
    company = retr.detect_company(question) if cfg == "hr_route" else None
    hits = retr.search(question, k=candidates, company=company)
    hits = reranker.rerank(question, hits, top_k=k)
    return hits, company


def main():
    ap = argparse.ArgumentParser(description="检索配置量化对比（纯 API）")
    ap.add_argument("--limit", type=int, default=100, help="抽样题数（默认 100）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help=f"逗号分隔（默认全部：{','.join(CONFIGS)}）")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8, help="API 并发数")
    ap.add_argument("--qa", default=os.path.join(HERE, "heldout_B", "qa_alpaca.json"))
    ap.add_argument("--index", default="index", help="索引基名（默认 index=heldout_B）")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_retrieval_results.json"))
    ap.add_argument("--fresh", action="store_true", help="忽略已有结果文件从头跑")
    ap.add_argument("--gen-backend", default="api", choices=["api", "local"],
                    help="生成后端：api=山大代理（默认），local=本地 Qwen3-4B+LoRA（需 GPU；裁判仍走 API）")
    ap.add_argument("--no-adapter", action="store_true",
                    help="local 后端不加载 LoRA，用纯 base 模型对照")
    ap.add_argument("--retrieval-device", default="cpu", choices=["cpu", "cuda"],
                    help="bge 编码器/reranker 设备（默认 cpu）")
    ap.add_argument("--stage", default="all", choices=["all", "gen", "judge"],
                    help="all=一次跑完；gen=只检索+本地生成存答案（零API，占卡）；"
                         "judge=只读已存答案统一API打分（零GPU）")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    configs = [c for c in args.configs.split(",") if c]
    if args.stage == "gen" and args.gen_backend != "local":
        sys.exit("--stage gen 仅配合 --gen-backend local 使用（API 生成无需分阶段）")

    # 断点续存：载入已有结果，跑完的配置（非 partial）直接跳过；
    # gens 为分阶段中转站（gen 阶段存答案，judge 阶段读），不受 --fresh 影响
    report, details, gens = {}, {}, {}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            gens = old.get("gens", {})
            if not args.fresh:
                report = {k: v for k, v in old.get("report", {}).items() if not v.get("partial")}
                details = {k: old["details"][k] for k in report if k in old.get("details", {})}
                if report:
                    print(f"[续存] 已完成配置：{', '.join(report)}（跳过，--fresh 可重跑）")
        except Exception:
            report, details, gens = {}, {}, {}
    todo = [c for c in configs if c not in report]
    if args.stage == "gen":
        todo = [c for c in todo
                if not (c in gens and all(g.get("answer") for g in gens[c])
                        and len(gens[c]) >= args.limit)]
        if set(configs) - set(todo):
            print(f"[续存] 已生成完答案：{', '.join(set(configs) - set(todo))}（跳过）")
    if args.stage == "judge":
        no_gen = [c for c in todo if c not in gens]
        if no_gen:
            sys.exit(f"[错误] 以下配置尚无已存答案，请先跑 --stage gen：{', '.join(no_gen)}")

    qa = json.load(open(args.qa, encoding="utf-8"))
    sample = random.Random(args.seed).sample(qa, min(args.limit, len(qa)))
    print(f"[数据] {os.path.basename(args.qa)} 共 {len(qa)} 题，抽样 {len(sample)} "
          f"(seed={args.seed})")

    idx = os.path.join(HERE, "rag", args.index + ".faiss")
    meta = os.path.join(HERE, "rag", args.index + ".json")
    rdev = args.retrieval_device
    retr = reranker = local = client = None
    if args.stage != "judge":            # judge 阶段不需要检索/生成模型
        print(f"[检索] 加载 bge-large-zh + BM25 ({rdev}) …")
        retr = HybridRetriever(index_path=idx, meta_path=meta, device=rdev)
        if any(c in ("hr", "hr_route") for c in todo):
            from rag.rerank import Reranker
            print(f"[精排] 加载 bge-reranker-large ({rdev}) …")
            reranker = Reranker(model_cache_dir=os.path.join(llm.PROJ, "models"), device=rdev)
        if args.gen_backend == "local":
            adapter = None if args.no_adapter else llm.DEFAULT_ADAPTER
            print("[生成] 本地 Qwen3-4B"
                  + (f" + LoRA({os.path.basename(adapter)})" if adapter else "（纯 base）")
                  + " 加载中…")
            local = llm.LocalLLM(adapter=adapter)
    if args.stage != "gen":              # gen 阶段零 API 调用（裁判始终走 API，可比）
        client = llm.APIClient()
        print(f"[裁判] API: {client.model}\n")

    def _save():
        json.dump({"args": vars(args), "report": report, "details": details, "gens": gens},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    GEN_KEYS = ("q", "ref", "answer", "routed", "evidence", "error")

    # 第一遍A：全部配置先检索——bge/reranker 用完立刻释放，把显存全部让给生成模型
    prepared = {}
    if args.stage != "judge":
        for cfg in todo:
            items = []
            for i, x in enumerate(sample, 1):
                hits, company = retrieve(cfg, retr, reranker, x["instruction"],
                                         args.k, args.candidates)
                ctx = "\n".join(f"[{j + 1}] {h['text']}" for j, h in enumerate(hits))
                items.append({"q": x["instruction"], "ref": x["output"], "ctx": ctx,
                              "routed": company, "evidence": evidence_hit(x["output"], ctx)})
                if i % 20 == 0 or i == len(sample):
                    print(f"  [{cfg}] 检索 {i}/{len(sample)}", flush=True)
            prepared[cfg] = items

        import gc
        import torch
        retr = reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rdev == "cuda":
            print("[GPU] 检索完毕，bge/reranker 显存已释放\n", flush=True)

        # 第一遍B（仅 local）：串行 GPU 生成；已存答案的题直接复用（中断重跑不白算）
        if local is not None:
            for cfg in todo:
                items = prepared[cfg]
                cached = {g["q"]: g["answer"] for g in gens.get(cfg, [])
                          if g.get("answer") and not g.get("error")}
                for i, it in enumerate(items, 1):
                    if it["q"] in cached:
                        it["answer"] = cached[it["q"]]
                    else:
                        try:
                            it["answer"] = local.chat(PROMPT_TMPL.format(ctx=it["ctx"],
                                                                         question=it["q"]))
                        except Exception as e:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()   # OOM 后清碎片，避免连环炸
                            it["answer"], it["judge"], it["error"] = f"[ERROR] {e}", 0, True
                    if i % 10 == 0 or i == len(items):
                        print(f"  [{cfg}] 本地生成 {i}/{len(items)}", flush=True)
                        # 生成答案增量落盘（不含 ctx，体积小）
                        gens[cfg] = [{k: t.get(k) for k in GEN_KEYS}
                                     for t in items[:i]]
                        _save()
                print(f"[{cfg}] 生成完成 {len(items)} 题\n", flush=True)

            local = None                 # 生成全部完成，释放显卡：打分阶段不占卡
            gc.collect()
            torch.cuda.empty_cache()
            print("[GPU] 全部生成完毕，显存已释放\n", flush=True)

    if args.stage == "gen":
        print(f"已存 {args.out}（答案就绪，后续 --stage judge 统一打分）")
        return

    # 第二遍：统一并行 API 打分（零 GPU）
    for cfg in todo:
        # 本轮已检索/生成的直接用；judge 阶段则读 gen 存好的答案（含 evidence/routed）
        items = prepared.get(cfg) or [dict(g) for g in gens[cfg]]

        # 阶段2：并行 API 生成（local 已预填 answer 则只裁判）；单题失败计 0 不中断
        def _proc(it):
            try:
                if it.get("error"):
                    it["judge"] = 0             # 本地生成已失败，不再裁判
                    return 0
                if "answer" not in it:
                    it["answer"] = chat_retry(client, PROMPT_TMPL.format(ctx=it["ctx"],
                                                                         question=it["q"]))
                out = chat_retry(client, JUDGE_TMPL.format(question=it["q"], ref=it["ref"],
                                                           answer=it["answer"]))
                it["judge"] = 1 if out.strip().startswith("1") else 0
            except Exception as e:
                # 裁判/生成失败：保留已有答案（local 已生成的不白算），只记错误
                it.setdefault("answer", f"[ERROR] {e}")
                it["judge"] = 0
                it["error"] = True
            return it["judge"]

        done, scores = 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for s in ex.map(_proc, items):
                scores.append(s)
                done += 1
                if done % 20 == 0 or done == len(items):
                    print(f"  [{cfg}] 评测 {done}/{len(items)} 实时acc={sum(scores)/done:.1%}",
                          flush=True)
                    # 配置内分段落盘：标 partial，中途挂掉可追查已完成部分（续存时重跑本配置）
                    if done < len(items):
                        report[cfg] = {"partial": True, "n": done,
                                       "acc": sum(scores) / done}
                        details[cfg] = [{k: it.get(k) for k in ("q", "ref", "answer", "judge",
                                                                "routed", "evidence")}
                                        for it in items if "judge" in it]
                        _save()

        ev = [it["evidence"] for it in items if it["evidence"] is not None]
        report[cfg] = {
            "n": len(items),
            "acc": sum(it["judge"] for it in items) / len(items),
            "evidence_hit": (sum(ev) / len(ev)) if ev else None,
            "evidence_n": len(ev),
            "routed": sum(1 for it in items if it["routed"]),
            "errors": sum(1 for it in items if it.get("error")),
        }
        # 明细不存 ctx（太大），保留答案与判分供 bad case 追查
        details[cfg] = [{k: it.get(k) for k in ("q", "ref", "answer", "judge",
                                                "routed", "evidence")} for it in items]
        # 每个配置跑完立即落盘，中途挂掉也不丢已完成的结果
        _save()
        r = report[cfg]
        ev = f"{r['evidence_hit']:.1%}" if r["evidence_hit"] is not None else "-"
        print(f"[{cfg}] acc={r['acc']:.1%}  evidence={ev}"
              f"({r['evidence_n']}题)  routed={r['routed']}  errors={r['errors']}\n",
              flush=True)

    print(f"{'配置':<10} {'acc':>7} {'evidence':>9} {'routed':>7}")
    for cfg in configs:
        r = report.get(cfg)
        if not r or r.get("partial"):
            continue
        ev = f"{r['evidence_hit']:.1%}" if r["evidence_hit"] is not None else "-"
        print(f"{cfg:<10} {r['acc']:>7.1%} {ev:>9} {r['routed']:>7}")
    print(f"\n已存 {args.out}")


if __name__ == "__main__":
    main()
