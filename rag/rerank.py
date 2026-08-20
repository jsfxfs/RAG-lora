# -*- coding: utf-8 -*-
"""bge-reranker 精排模块（交叉编码器，CPU 可跑）。

粗排（向量/混合召回）拿 top-N 候选后，用 BAAI/bge-reranker-large 对
(query, passage) 逐对打分重排，取 top-k。交叉编码器逐 token 交互，
比双塔向量的相关性判别准得多，代价是只能作用于少量候选（N≈20）。

注意：这是 ~0.3B 的判别式小模型，跑在 CPU 上（默认），不是 LLM 生成推理。

用法
----
  from rag.rerank import Reranker
  rr = Reranker()                       # 首次自动经 ModelScope 下载 (~1.3GB)
  hits = rr.rerank(query, hits, top_k=6)   # hits 追加 rerank_score 并重排
"""
import torch

from .retrieve import resolve_model_path

DEFAULT_RERANKER = "BAAI/bge-reranker-large"


class Reranker:
    def __init__(self, model=DEFAULT_RERANKER, model_source="modelscope",
                 model_cache_dir=None, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_path = resolve_model_path(model, model_source, model_cache_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 同 retrieve.py：关 low_cpu_mem_usage 规避 .bin 权重 meta tensor 问题
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, low_cpu_mem_usage=False).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query, texts, batch=16):
        """返回 query 与每个 text 的相关性分（越大越相关，未归一化 logit）。"""
        out = []
        for i in range(0, len(texts), batch):
            pairs = [(query, t) for t in texts[i:i + batch]]
            enc = self.tokenizer(pairs, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt").to(self.device)
            out.extend(self.model(**enc).logits.view(-1).float().cpu().tolist())
        return out

    def rerank(self, query, hits, top_k=6):
        """对粗排 hits 重排：追加 rerank_score，按其降序取 top_k。"""
        if not hits:
            return hits
        scores = self.score(query, [h["text"] for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = s
        return sorted(hits, key=lambda h: -h["rerank_score"])[:top_k]
