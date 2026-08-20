# -*- coding: utf-8 -*-
"""检索耗时基准：四档配置在 CPU 上的单题检索用时（与 API 后端评测同环境）。"""
import os
import sys
import json
import time
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rag.retrieve import FAISSRetriever, HybridRetriever  # noqa: E402
from rag.rerank import Reranker  # noqa: E402
from rag import llm  # noqa: E402
from eval_retrieval import retrieve  # noqa: E402

N, K, CAND = 20, 6, 20
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qa = json.load(open(os.path.join(HERE, "heldout_B", "qa_alpaca.json")))
random.seed(42)
qs = [x["instruction"] for x in random.sample(qa, N)]

t0 = time.time()
retr = HybridRetriever(index_path=os.path.join(HERE, "rag", "index.faiss"),
                       meta_path=os.path.join(HERE, "rag", "index.json"), device=DEV)
t_load_retr = time.time() - t0
t0 = time.time()
rk = Reranker(model_cache_dir=os.path.join(llm.PROJ, "models"), device=DEV)
t_load_rk = time.time() - t0
print(f"[{DEV}] 模型加载: 检索器 {t_load_retr:.1f}s, reranker {t_load_rk:.1f}s\n")

print(f"{'配置':<10}{'平均':>9}{'中位':>9}{'最快':>9}{'最慢':>9}")
for cfg in ("vec", "hybrid", "hr", "hr_route"):
    retrieve(cfg, retr, rk, qs[0], K, CAND)          # 预热，排除首次开销
    ts = []
    for q in qs:
        t0 = time.time()
        retrieve(cfg, retr, rk, q, K, CAND)
        ts.append(time.time() - t0)
    print(f"{cfg:<10}{statistics.mean(ts):>8.2f}s{statistics.median(ts):>8.2f}s"
          f"{min(ts):>8.2f}s{max(ts):>8.2f}s")
