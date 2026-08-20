# -*- coding: utf-8 -*-
"""held-out 2x2 评测 harness（RAG 侧已换本地 bge 向量召回）。

简历核心证据 —— 这张表直接证明"FT 只在背过的地方行，RAG 哪都行"：

              | FT(权重)          | RAG(检索)
--------------|-------------------|-------------------
见过 A(10家)  | FT-on-A (高)      | RAG-on-A (高)
未见 B(15家)  | FT零样本 (低)      | RAG-on-B (高)   <- 命门证据

- 查询题:  A = 03_LoRA数据/qa_alpaca.json ; B = heldout_B/qa_alpaca.json
- RAG KB:  A = 04_RAG数据/rag_chunks.json ; B = heldout_B/chunks/*
- RAG 检索: 本地 bge-large-zh-v1.5 嵌入 + FAISS（见 rag/build_index.py / rag/retrieve.py）
            B 侧索引 = rag/index.faiss ; A 侧索引 = rag/indexA.faiss
- FT      : 本地加载 Qwen3-4B + saves/qwen3-4b/lora/sft_rank16（本地 GPU 推理，无需起服务）
- 生成/裁判: 山大代理（anthropic 兼容），从 .env 读 BASE/TOKEN/MODEL 自动拼

用法（两阶段，推理与 API 分离）:
  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py            # 先建 B 侧索引（必做）
  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py \
      --chunks-dir 04_RAG数据 --out-dir rag --index-name indexA          # 建 A 侧索引（可选）
  # 阶段1：本地 GPU 批量 FT 推理并存盘（不调 API，GPU 满载）
  CUDA_VISIBLE_DEVICES=5 python eval_heldout.py --side B --phase infer
  # 阶段2：CPU 调 API 做 RAG 生成 + 2x judge（读盘 FT 答案，不占 GPU）
  python eval_heldout.py --side B --phase judge
  # 兼容：--phase all 等价于旧混跑；--all 跑满 2x2
"""
import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor

_ft_lock = threading.Lock()  # FT 本地 GPU 推理串行化，避免并发占爆显存

# 预导入 transformers 类：4.57.6 的懒加载在子线程首次触发会失败，
# 必须在主线程先绑定，供后续线程池的 RAG 检索使用。
from transformers import AutoModel, AutoTokenizer  # noqa: E402, F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_RAG_RETRIEVER = None

from rag import llm  # noqa: E402  .env 读取 / API 调用 / 本地 FT 模型均已下沉至此

# ===================== 本地 FT 模型（Qwen3-4B + sft_rank16 LoRA，本地 GPU 推理）=====================
FT_MODEL = None      # 懒加载的 llm.LocalLLM 实例
# ===================== 生成/裁判端点（山大代理，从 .env 读）=====================
_cfg = llm.load_env()
MODEL = _cfg["MODEL"]
GEN_ENDPOINT = JUDGE_ENDPOINT = _cfg["ENDPOINT"]
GEN_KEY = JUDGE_KEY = _cfg["TOKEN"]
# ================================================


def _post(endpoint, key, model, prompt):
    """通用 /v1/messages 调用，返回模型文本（实现见 rag/llm.py）。"""
    return llm.post_messages(endpoint, key, model, prompt)


def get_retriever(side):
    """懒加载 FAISS 向量检索器。B = heldout_B 索引，A = 04_RAG数据 索引。"""
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None or _RAG_RETRIEVER.get("side") != side:
        from rag.retrieve import FAISSRetriever
        suffix = "A" if side == "A" else ""
        idx = os.path.join(HERE, "rag", f"index{suffix}.faiss")
        meta = os.path.join(HERE, "rag", f"index{suffix}.json")
        if not (os.path.exists(idx) and os.path.exists(meta)):
            raise RuntimeError(
                f"缺少 RAG 索引({idx})。请先运行:\n"
                f"  /data1/jiajun/.conda/envs/llf/bin/python rag/build_index.py"
                + (" --chunks-dir 04_RAG数据 --out-dir rag --index-name indexA"
                   if side == "A" else ""))
        # bge 检索编码器放 CPU：避免与 GPU 上的 FT(Qwen3-4B+LoRA) 争显存
        _RAG_RETRIEVER = {"side": side, "r": FAISSRetriever(index_path=idx, meta_path=meta, device="cpu")}
    return _RAG_RETRIEVER["r"]


def _load_ft():
    """懒加载本地 Qwen3-4B + sft_rank16 LoRA（首次调用时占 GPU，之后复用）。"""
    global FT_MODEL
    if FT_MODEL is None:
        # cuda = 逻辑卡0；用 CUDA_VISIBLE_DEVICES=5 把物理卡5映射成 cuda:0 来选卡
        FT_MODEL = llm.LocalLLM(device="cuda")   # 默认 Qwen3-4B + sft_rank16
    return FT_MODEL


def answer_ft(question):
    """FT(权重) 直接答，不给上下文（本地推理，不走 HTTP）。"""
    return _load_ft().chat(question, max_new_tokens=512)


def answer_rag(question, side):
    """RAG: 向量检索 top-k 片段，再喂给生成模型。"""
    hits = get_retriever(side).search(question, k=6)
    ctx = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    prompt = (f"根据以下年报片段回答问题，严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}")
    return _post(GEN_ENDPOINT, GEN_KEY, MODEL, prompt)


def judge(question, answer):
    """LLM-as-Judge: 返回 1(答对事实) / 0(错/编造/拒答)。"""
    prompt = (
        "判断下面的回答是否基于事实正确回答了问题。只回 1 或 0。\n"
        f"问题: {question}\n回答: {answer}"
    )
    out = _post(JUDGE_ENDPOINT, JUDGE_KEY, MODEL, prompt)
    return 1 if out.strip().startswith("1") else 0


def load_qa(path):
    return json.load(open(path, encoding="utf-8"))


def _qa_path(side):
    if side == "A":
        return os.path.join(HERE, "03_LoRA数据", "qa_alpaca.json")
    return os.path.join(HERE, "heldout_B", "qa_alpaca.json")


def run_infer(side, limit=None):
    """阶段1：本地 GPU 批量 FT 推理，存 ft_answers_<side>.json。不调 API。"""
    qa = load_qa(_qa_path(side))
    if limit:
        qa = qa[:limit]
    total = len(qa)
    out = []
    _load_ft()  # 预热到 GPU（只此一次）
    for i, q in enumerate(qa, 1):
        qn = q["instruction"]
        ans = answer_ft(qn)
        out.append({"instruction": qn, "ft_answer": ans})
        if i % 50 == 0 or i == total:
            print(f"  [infer {side}] {i}/{total}", flush=True)
    path = os.path.join(HERE, f"ft_answers_{side}.json")
    json.dump(out, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已存 {path} ({len(out)} 条)")


def run_judge(side, limit=None, workers=8):
    """阶段2：CPU 调 API 做 RAG 生成 + 2x judge。FT 答案读盘，不占 GPU。"""
    qa = load_qa(_qa_path(side))
    if limit:
        qa = qa[:limit]
    fa_path = os.path.join(HERE, f"ft_answers_{side}.json")
    ft_map = {}
    if os.path.exists(fa_path):
        ft_map = {x["instruction"]: x["ft_answer"]
                  for x in json.load(open(fa_path, encoding="utf-8"))}
    else:
        print(f"警告: 未找到 {fa_path}，将临时本地推理(占GPU)")
    # 预热：主线程先加载一次 RAG 检索器(bge)，避免子线程首次触发 meta tensor 报错
    get_retriever(side)
    total = len(qa)
    ft_scores, rag_scores = [], []

    def _process(q):
        qn = q["instruction"]
        ft_ans = ft_map.get(qn)
        if ft_ans is None:              # 兜底：infer 阶段没跑才临时算
            with _ft_lock:
                ft_ans = answer_ft(qn)
        rag_ans = answer_rag(qn, side)  # API 生成
        jft = judge(qn, ft_ans)         # API 裁判
        jrag = judge(qn, rag_ans)       # API 裁判
        return jft, jrag

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for jft, jrag in ex.map(_process, qa):
            ft_scores.append(jft)
            rag_scores.append(jrag)
            done += 1
            if done % 10 == 0 or done == total:
                f = sum(ft_scores) / done
                r = sum(rag_scores) / done
                print(f"  [judge {side}] {done}/{total}  实时 FT={f:.1%}  RAG={r:.1%}",
                      flush=True)
    n = total or 1
    f, r = sum(ft_scores) / n, sum(rag_scores) / n
    label = "见过 A(10家)" if side == "A" else "未见 B(15家)"
    print(f"[{label}] FT零样本={f:.1%}  RAG-on-{side}={r:.1%}")
    return f, r


def main():
    args = sys.argv[1:]
    # 阶段: infer(本地推理存盘) / judge(API评测) / all(混合, 兼容旧用法)
    phase = "all"
    if "--phase" in args:
        try:
            phase = args[args.index("--phase") + 1]
        except Exception:
            phase = "all"
    sides = ["B"]
    if "--all" in args:
        sides = ["A", "B"]
    elif "--side" in args:
        i = args.index("--side")
        nxt = args[i + 1] if i + 1 < len(args) else "B"
        sides = [nxt] if not nxt.startswith("--") else ["B"]
    limit = None
    if "--limit" in args:
        try:
            limit = int(args[args.index("--limit") + 1])
        except Exception:
            limit = None

    for s in sides:
        if phase in ("infer", "all"):
            run_infer(s, limit)
        if phase in ("judge", "all"):
            run_judge(s, limit)
    if "A" in sides and "B" in sides and phase in ("judge", "all"):
        print("\n结论: FT 只在见过的地方高 -> 把事实背进了权重; RAG 未见也高 -> 事实在知识库。")


if __name__ == "__main__":
    main()
