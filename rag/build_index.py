# -*- coding: utf-8 -*-
"""为 RAG 知识库构建向量索引（本地 bge-large-zh-v1.5 + FAISS）。

设计
----
评测集 B 的 KB = heldout_B/chunks/*_chunks.json（纯文本字符串数组）。
把每个 chunk 文本用 bge 编码成 1024 维向量，L2 归一化后写入 FAISS
IndexFlatIP（内积 = 余弦相似度）。同时导出 meta.json 保存每条向量的
文本 / 公司 / 来源文件 / 块序号，供检索时回取原文。

bge 用法要点
------------
- query 端加指令前缀 "为这个句子生成表示以用于检索相关文章："；passage 端不加。
- 取 [CLS] 位置（last_hidden_state[:,0]）作句向量，再 L2 归一化。

用法（用 llf 环境）
------------------
  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py
  # 指定模型落盘根目录（最终为 <dir>/BAAI/bge-large-zh-v1.5）
  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py \
      --model-cache-dir /data2/Data/jiajun/3dgs/BrandingWall/llmfactory/models
  # 对 A 侧 KB 也建一份索引（可选，供 RAG-on-A 用）
  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py \
      --chunks-dir 04_RAG数据 --out-dir rag --index-name indexA

依赖
----
  torch, transformers（环境已装）；faiss（pip install faiss-cpu）。
模型
----
  默认通过 ModelScope 镜像下载 BAAI/bge-large-zh-v1.5（~1.3GB，缓存复用），
  规避 HuggingFace 不可达问题。可用 --model-source hf 改回 HuggingFace 直连，
  或 --model 指向本地已下载的快照目录。
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # RAG-lora/
DEFAULT_CHUNKS = os.path.join(ROOT, "heldout_B", "chunks")
DEFAULT_OUT = HERE                                 # rag/
MODEL_SOURCE = "modelscope"                        # 或 "hf"；默认走镜像规避 HF 不可达
MODEL_REPO = "BAAI/bge-large-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："   # bge 官方 query 前缀


def resolve_model_path(model, source=MODEL_SOURCE, cache_dir=None):
    """返回可用的本地模型目录。

    - model 是本地已有目录 -> 直接用；
    - 否则按 source 下载/读取：modelscope 走镜像（cache_dir 指定落盘根目录，
      最终路径为 <cache_dir>/<model_id>），hf 交给 transformers 直连。
    """
    if os.path.isdir(model):
        return model
    if source == "hf":
        return model                                # 交给 transformers.from_pretrained 直连 HF
    try:
        from modelscope import snapshot_download
    except ImportError:
        sys.exit("缺少 modelscope，请先安装：\n"
                 "  /data1/jiajun/.conda/envs/llf/bin/python -m pip install modelscope\n"
                 "或改用 --model-source hf 直连 HuggingFace。")
    print(f"[模型] 通过 ModelScope 镜像下载/复用: {model}"
          + (f" (cache_dir={cache_dir})" if cache_dir else ""))
    return snapshot_download(model, cache_dir=cache_dir)


def company_from_filename(fn):
    """11_300904_威力传动_2026-06-18_chunks.json / 01_..._2026-07-02.pdf -> 公司名"""
    base = os.path.splitext(fn)[0]
    if base.endswith("_chunks"):
        base = base[:-len("_chunks")]
    parts = base.split("_")
    return parts[2] if len(parts) >= 3 else base


def load_chunks(chunks_dir):
    """读取目录下所有 *_chunks.json，返回 [{company,file,chunk_idx,text}]。"""
    records = []
    for fn in sorted(os.listdir(chunks_dir)):
        if not fn.endswith(".json"):
            continue
        data = json.load(open(os.path.join(chunks_dir, fn), encoding="utf-8"))
        if not isinstance(data, list):
            print(f"[warn] 跳过非列表文件: {fn}")
            continue
        company = company_from_filename(fn)
        for i, c in enumerate(data):
            text = c if isinstance(c, str) else c.get("text", "")
            if not (text and text.strip()):
                continue
            # 合并式 chunks（如 04_RAG数据/rag_chunks.json）每条自带 source，
            # 优先从 source 解析公司，避免整库归到合并文件名下
            src = c.get("source") if isinstance(c, dict) else None
            records.append({"company": company_from_filename(src) if src else company,
                            "file": src or fn, "chunk_idx": i, "text": text})
    return records


@torch.no_grad()
def embed(texts, model, tokenizer, device, is_query, batch=32):
    prefix = QUERY_INSTRUCTION if is_query else ""
    feats = []
    for i in range(0, len(texts), batch):
        batch_texts = [prefix + t for t in texts[i:i + batch]]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state[:, 0]          # CLS pooling (bge 官方)
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        feats.append(h.cpu().numpy())
    return np.vstack(feats).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default=DEFAULT_CHUNKS,
                    help="含 *_chunks.json 的目录（默认 heldout_B/chunks）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="索引与 meta 输出目录（默认 rag/）")
    ap.add_argument("--index-name", default="index",
                    help="索引基名，输出 <name>.faiss / <name>.json（默认 index）")
    ap.add_argument("--model", default=MODEL_REPO)
    ap.add_argument("--model-source", default=MODEL_SOURCE, choices=["modelscope", "hf"],
                    help="模型下载源：modelscope（默认，镜像）或 hf（HuggingFace 直连）")
    ap.add_argument("--model-cache-dir", default=None,
                    help="ModelScope 模型落盘根目录，最终为 <dir>/BAAI/bge-large-zh-v1.5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    try:
        import faiss
    except ImportError:
        sys.exit("缺少 faiss，请先安装：\n"
                 "  /data1/jiajun/.conda/envs/llf/bin/python -m pip install faiss-cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    model_path = resolve_model_path(args.model, args.model_source, args.model_cache_dir)
    print(f"[1/3] 加载模型 {model_path} -> {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(args.device)
    model.eval()

    print(f"[2/3] 读取 chunks: {args.chunks_dir}")
    records = load_chunks(args.chunks_dir)
    print(f"      共 {len(records)} 个 chunk")
    if not records:
        sys.exit("无 chunk，退出")

    emb = embed([r["text"] for r in records], model, tokenizer,
                args.device, is_query=False)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    index_path = os.path.join(args.out_dir, args.index_name + ".faiss")
    meta_path = os.path.join(args.out_dir, args.index_name + ".json")
    faiss.write_index(index, index_path)
    # meta 存为 dict：model_path 供检索端直接复用同一模型，records 为 chunk 列表
    meta = {"model_path": model_path, "records": records}
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[3/3] 已写出:\n      {index_path}\n      {meta_path}")
    print(f"      模型路径已记录: {model_path}")
    print("完成。评测时检索端用 rag/retrieve.py 的 FAISSRetriever 加载上述索引。")


if __name__ == "__main__":
    main()
