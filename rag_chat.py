# -*- coding: utf-8 -*-
"""RAG 问答入口：向量检索 top-k 年报片段 -> 拼 prompt -> 生成答案。

生成后端二选一（见 rag/llm.py）：
  --backend api    山大代理（DeepSeek-V4-Flash，从 .env 读配置，默认）
  --backend local  本地 Qwen3-4B + LoRA（sft_rank16），需 GPU；
                   --no-adapter 可切纯 base 模型对比

用法（llf 环境）
----------------
  PY=/data1/jiajun/.conda/envs/llf/bin/python
  # 交互问答（API 生成，KB=heldout_B 15 家）
  $PY rag_chat.py
  # 单问单答
  $PY rag_chat.py -q "威力传动 2025 年营业收入是多少？"
  # 本地 LoRA 模型生成（用物理卡 5）
  CUDA_VISIBLE_DEVICES=5 $PY rag_chat.py --backend local -q "..."
  # 检索增强：BM25+向量混合召回 / bge-reranker 精排（均 CPU 小模型，可叠加）
  $PY rag_chat.py --hybrid --rerank -q "..."
  # 换 A 侧索引（04_RAG数据，需先 build_index.py --index-name indexA）
  $PY rag_chat.py --index indexA

前置：rag/index.faiss 已由 rag/build_index.py 生成。
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROMPT_TMPL = "根据以下年报片段回答问题，严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}"


def build_prompt(question, hits):
    ctx = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    return PROMPT_TMPL.format(ctx=ctx, question=question)


def show_hits(hits):
    for i, h in enumerate(hits, 1):
        preview = h["text"][:60].replace("\n", " ")
        extra = ""
        if "vec_rank" in h:                      # 混合召回的双路排名
            extra += f" vec#{h['vec_rank']} bm25#{h['bm25_rank']}"
        if "rerank_score" in h:
            extra += f" rr={h['rerank_score']:.2f}"
        print(f"  [{i}] {h['company']:<6} score={h['score']:.3f}{extra}  {preview}…")


def make_backend(args):
    """返回 chat(prompt)->str 的生成函数。"""
    from rag import llm
    if args.backend == "api":
        client = llm.APIClient()
        print(f"[后端] API: {client.model}")
        return client.chat
    adapter = None if args.no_adapter else llm.DEFAULT_ADAPTER
    print(f"[后端] 本地 Qwen3-4B"
          + (f" + LoRA({os.path.basename(adapter)})" if adapter else "（无 adapter）")
          + " 加载中…")
    model = llm.LocalLLM(adapter=adapter)
    return model.chat


def main():
    ap = argparse.ArgumentParser(description="金融年报 RAG 问答")
    ap.add_argument("-q", "--question", default=None,
                    help="单问单答；不传则进入交互模式")
    ap.add_argument("--backend", default="api", choices=["api", "local"],
                    help="生成后端：api=山大代理（默认），local=本地 Qwen3-4B+LoRA")
    ap.add_argument("--no-adapter", action="store_true",
                    help="local 后端不加载 LoRA，用纯 base 模型")
    ap.add_argument("--index", default="index",
                    help="索引基名（默认 index=heldout_B；indexA=04_RAG数据）")
    ap.add_argument("-k", type=int, default=6, help="召回片段数（默认 6）")
    ap.add_argument("--hybrid", action="store_true",
                    help="BM25+向量混合召回（RRF 融合）")
    ap.add_argument("--rerank", action="store_true",
                    help="bge-reranker 精排（先召回 --candidates 条再重排取 top-k）")
    ap.add_argument("--candidates", type=int, default=20,
                    help="粗排候选数，供混合融合/精排用（默认 20）")
    ap.add_argument("--no-route", action="store_true",
                    help="关闭公司路由（默认从问题识别代码/简称后只在该公司块内检索）")
    ap.add_argument("--show-hits", action="store_true", help="打印召回片段明细")
    ap.add_argument("--retrieval-device", default="cpu", choices=["cpu", "cuda"],
                    help="bge 编码器/reranker 的设备（默认 cpu；显存宽裕时用 cuda 加速）")
    args = ap.parse_args()

    idx = os.path.join(HERE, "rag", args.index + ".faiss")
    meta = os.path.join(HERE, "rag", args.index + ".json")
    if not (os.path.exists(idx) and os.path.exists(meta)):
        sys.exit(f"缺少索引 {idx}，请先运行:\n"
                 f"  python rag/build_index.py"
                 + ("" if args.index == "index"
                    else f" --chunks-dir 04_RAG数据 --out-dir rag --index-name {args.index}"))

    from rag.retrieve import FAISSRetriever, HybridRetriever
    # 检索编码器(bge)默认 CPU：local 后端时不与生成模型争显存；
    # 显存宽裕时 --retrieval-device cuda 可提速（bge+reranker 共约 3.5G）
    rdev = args.retrieval_device
    cls = HybridRetriever if args.hybrid else FAISSRetriever
    print(f"[检索] 加载 {args.index}.faiss + bge-large-zh ({rdev})"
          + (" + BM25混合" if args.hybrid else "") + " …")
    retriever = cls(index_path=idx, meta_path=meta, device=rdev)
    reranker = None
    if args.rerank:
        from rag.rerank import Reranker
        from rag.llm import PROJ
        print(f"[精排] 加载 bge-reranker-large ({rdev}) …")
        reranker = Reranker(model_cache_dir=os.path.join(PROJ, "models"), device=rdev)
    chat = make_backend(args)

    def ask(question):
        t0 = time.perf_counter()
        # 公司路由：识别到代码/简称则只在该公司块内检索，避免跨公司串扰
        company = None if args.no_route else retriever.detect_company(question)
        if company:
            print(f"[路由] {company}（限定 {len(retriever._by_company[company])} 块）")
        t_route = time.perf_counter()
        # 开精排时粗排多取候选再重排取 top-k，否则直接取 top-k
        n = args.candidates if reranker else args.k
        hits = retriever.search(question, k=n, company=company)
        t_search = time.perf_counter()
        if reranker:
            hits = reranker.rerank(question, hits, top_k=args.k)
        t_rerank = time.perf_counter()
        if args.show_hits:
            show_hits(hits)
        answer = chat(build_prompt(question, hits))
        t_gen = time.perf_counter()
        stages = [("路由", t_route - t0), ("检索", t_search - t_route)]
        if reranker:
            stages.append(("精排", t_rerank - t_search))
        stages.append(("生成", t_gen - t_rerank))
        print("[耗时] " + " | ".join(f"{name} {sec:.2f}s" for name, sec in stages)
              + f" | 总计 {t_gen - t0:.2f}s")
        return answer

    if args.question:
        print(ask(args.question))
        return
    print("交互模式，输入问题回车（exit/quit/空行 退出）")
    while True:
        try:
            q = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        print(ask(q))


if __name__ == "__main__":
    main()
