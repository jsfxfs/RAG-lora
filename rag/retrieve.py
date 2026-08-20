# -*- coding: utf-8 -*-
"""FAISS 向量检索模块（配合 build_index.py 产出的索引）。

用法
----
  from rag.retrieve import FAISSRetriever
  retr = FAISSRetriever(index_path="rag/index.faiss", meta_path="rag/meta.json")
  hits = retr.search("威力传动 2025 年营业收入是多少？", k=6)
  for h in hits:
      print(h["company"], h["score"], h["text"][:80])

hits 每项含: company / file / chunk_idx / text / score(余弦相似度)。

HybridRetriever 在此基础上叠加 BM25 关键词召回（jieba 分词），用 RRF
（倒数排名融合）合并两路结果：财报里“营业收入”“300904”这类精确术语/
代码 BM25 更准，语义改写问法则靠向量，两者互补。hits 额外含 vec_rank /
bm25_rank（未被该路召回时为 None），score 为 RRF 分。

公司路由：detect_company() 从问题识别股票代码/公司简称（映射表来自索引
meta 的 company/file 字段），search(company=...) 只在该公司块内检索，
避免多公司共库时的跨公司串扰。识别不到时传 None，行为与全库检索一致。
"""
import os
import re
import json
import numpy as np
import torch

DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_MODEL_SOURCE = "modelscope"   # 或 "hf"；默认走镜像规避 HF 不可达
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
# 6 位股票代码（前后不能再接数字，避免从长数字串里误截）
_CODE_PAT = re.compile(r"(?<!\d)\d{6}(?!\d)")


def resolve_model_path(model, source=DEFAULT_MODEL_SOURCE, cache_dir=None):
    """返回可用的本地模型目录（与 build_index.py 同一套逻辑，复用 ModelScope 缓存）。"""
    if os.path.isdir(model):
        return model
    if source == "hf":
        return model
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise RuntimeError(
            "缺少 modelscope，请先安装：pip install modelscope\n"
            "或初始化时传入 model_source='hf' 直连 HuggingFace。")
    return snapshot_download(model, cache_dir=cache_dir)


class FAISSRetriever:
    def __init__(self, index_path, meta_path, model=DEFAULT_MODEL,
                 model_source=DEFAULT_MODEL_SOURCE, model_cache_dir=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        try:
            import faiss
        except ImportError:
            raise RuntimeError("缺少 faiss，请先安装：pip install faiss-cpu")
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        # 优先复用建索引时记录的确切模型路径，避免重复下载
        if isinstance(self.meta, dict) and self.meta.get("model_path"):
            model_path = self.meta["model_path"]
        else:
            model_path = resolve_model_path(model, model_source, model_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # torch 2.6 + transformers 4.57 默认走 meta device 初始化，
        # 对 .bin 权重会报 "Cannot copy out of meta tensor"。
        # 关掉 low_cpu_mem_usage 让权重直接落到 CPU，再 .to(device)。
        self.model = AutoModel.from_pretrained(
            model_path, low_cpu_mem_usage=False).to(device)
        self.model.eval()
        self.index = faiss.read_index(index_path)
        # records 取 chunk 列表（兼容老格式纯 list 与新格式 dict）
        self.records = self.meta["records"] if isinstance(self.meta, dict) else self.meta
        # 公司路由表：company -> 记录id列表；股票代码(取自文件名) -> company
        self._by_company = {}
        self._code2company = {}
        for i, r in enumerate(self.records):
            self._by_company.setdefault(r["company"], []).append(i)
            for p in r.get("file", "").split("_"):
                if len(p) == 6 and p.isdigit():
                    self._code2company.setdefault(p, r["company"])
                    break
        self._company_vecs = {}   # company -> 向量矩阵懒缓存（从 faiss reconstruct）

    def detect_company(self, query):
        """从问题识别公司：优先 6 位股票代码，其次公司简称子串；识别不出返回 None。"""
        for code in _CODE_PAT.findall(query):
            if code in self._code2company:
                return self._code2company[code]
        q = query.replace(" ", "")
        for name in self._by_company:
            if name.replace(" ", "") in q:   # “五 粮 液”这类文件名带空格，去空格匹配
                return name
        return None

    def _vector_ids(self, v, n, company=None):
        """向量路取 top-n 候选 [(id, 相似度)]。company 命中路由表时只搜该公司的块。"""
        if company in self._by_company:
            ids = self._by_company[company]
            if company not in self._company_vecs:
                # IndexFlat 存的就是原始向量，reconstruct 取子集后 numpy 精确计算
                self._company_vecs[company] = np.vstack(
                    [self.index.reconstruct(int(i)) for i in ids])
            sims = self._company_vecs[company] @ v[0]
            order = np.argsort(sims)[::-1][:n]
            return [(ids[j], float(sims[j])) for j in order]
        D, I = self.index.search(v, n)
        return [(int(i), float(s)) for s, i in zip(D[0], I[0]) if i >= 0]

    @torch.no_grad()
    def _embed_query(self, q):
        enc = self.tokenizer(QUERY_INSTRUCTION + q, return_tensors="pt",
                             truncation=True, max_length=512).to(self.device)
        h = self.model(**enc).last_hidden_state[:, 0]        # CLS pooling
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        return h.cpu().numpy().astype("float32")

    def search(self, query, k=6, company=None):
        v = self._embed_query(query)
        hits = []
        for i, score in self._vector_ids(v, k, company):
            m = dict(self.records[i])
            m["score"] = score
            hits.append(m)
        return hits


def _tokenize(text):
    """jieba 搜索引擎模式分词，丢弃空白与单字符标点。"""
    import jieba
    return [t for t in jieba.lcut_for_search(text)
            if t.strip() and not (len(t) == 1 and not t.isalnum())]


class HybridRetriever(FAISSRetriever):
    """向量 + BM25 双路召回，RRF 融合。

    初始化时对全部 chunk 分词建 BM25（纯内存，秒级）；search() 各取
    candidates 条后按 RRF（score = Σ 1/(rrf_k + rank)）合并取 top-k。
    """

    def __init__(self, *args, rrf_k=60, **kw):
        super().__init__(*args, **kw)
        from rank_bm25 import BM25Okapi
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([_tokenize(r["text"]) for r in self.records])

    def search(self, query, k=6, candidates=20, company=None):
        # 向量路（company 命中时已限定在该公司块内）
        v = self._embed_query(query)
        vec_ids = [i for i, _ in self._vector_ids(v, candidates, company)]
        # BM25 路：在路由限定池（或全库）内打分取 top-candidates
        scores = self.bm25.get_scores(_tokenize(query))
        pool = (self._by_company[company] if company in self._by_company
                else range(len(self.records)))
        bm25_ids = sorted(pool, key=lambda i: -scores[i])[:candidates]
        # RRF 融合（rank 从 1 起）
        fused = {}
        for rank, i in enumerate(vec_ids, 1):
            fused.setdefault(i, {"vec_rank": None, "bm25_rank": None, "score": 0.0})
            fused[i]["vec_rank"] = rank
            fused[i]["score"] += 1.0 / (self.rrf_k + rank)
        for rank, i in enumerate(bm25_ids, 1):
            i = int(i)
            if scores[i] <= 0:          # 无关键词重叠的不计入
                continue
            fused.setdefault(i, {"vec_rank": None, "bm25_rank": None, "score": 0.0})
            fused[i]["bm25_rank"] = rank
            fused[i]["score"] += 1.0 / (self.rrf_k + rank)
        top = sorted(fused.items(), key=lambda x: -x[1]["score"])[:k]
        hits = []
        for i, f in top:
            m = dict(self.records[i])
            m.update(f)
            hits.append(m)
        return hits
