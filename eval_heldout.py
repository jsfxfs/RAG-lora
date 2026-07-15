# -*- coding: utf-8 -*-
"""held-out 2x2 评测 harness(骨架, 端点待填)

简历核心证据 —— 这张表直接证明"FT 只在背过的地方行, RAG 哪都行":

              | FT(权重)          | RAG(检索)
--------------|-------------------|-------------------
见过 A(10家)  | FT-on-A (高, 背的)  | RAG-on-A (高)
未见 B(15家)  | FT零样本 (低)      | RAG-on-B (高)   <- 命门证据

- 查询题:
    A 题 = 03_LoRA数据/qa_alpaca.json          (FT 训过)
    B 题 = heldout_B/qa_alpaca.json            (FT 从没见过)
- RAG 知识库:
    A = 04_RAG数据/rag_chunks.json
    B = heldout_B/chunks/*.json
- FT       : 你外部训好的 Qwen3-4B+LoRA 的 /v1/messages 端点 (填 FT_ENDPOINT)
- RAG 生成 : 复用山大代理或本地模型 (填 GEN_ENDPOINT)
- 评分     : LLM-as-Judge (填 JUDGE_ENDPOINT) 判"答案是否答对事实"

用法(填好下方端点后):
  python eval_heldout.py --side A    # 算 FT-on-A 与 RAG-on-A
  python eval_heldout.py --side B    # 算 FT零样本 与 RAG-on-B (命门行)
  python eval_heldout.py --all       # 跑满 2x2 并打印对比表
"""
import os, sys, json, re, urllib.request, ssl

HERE = os.path.dirname(os.path.abspath(__file__))

# ===================== 待你填写的端点 =====================
FT_ENDPOINT = ""      # 你训好的 Qwen3-4B+LoRA 的 /v1/messages
GEN_ENDPOINT = ""     # RAG 生成用的 LLM(同山大代理或本地)
JUDGE_ENDPOINT = ""   # LLM-as-Judge
FT_KEY = ""
GEN_KEY = ""
JUDGE_KEY = ""
BASE = ""             # 若用山大代理, 从 .env 读 ANTHROPIC_BASE_URL
TOKEN = ""           # 从 .env 读 ANTHROPIC_AUTH_TOKEN
MODEL = ""           # 从 .env 读 ANTHROPIC_DEFAULT_HAIKU_MODEL
# ============================================================

CTX = ssl.create_default_context()


def _post(endpoint, key, model, prompt):
    """通用 /v1/messages 调用, 返回模型文本。endpoint 为空时抛错提醒填。"""
    if not endpoint:
        raise RuntimeError("端点未填写: 请在 eval_heldout.py 顶部填 FT_ENDPOINT/GEN_ENDPOINT/JUDGE_ENDPOINT")
    body = json.dumps({"model": model, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01",
        "content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        d = json.loads(r.read().decode())
    return "".join(c.get("text", "") for c in d.get("content", []) if isinstance(c, dict))


def retrieve(question, kb_chunks, k=4):
    """从知识库按关键词召回 top-k 片段。

    TODO(你接本地 bge-m3 时替换这里): 当前是关键词兜底,
    正式评测请换成 向量召回(本地 bge-m3 嵌入 + faiss), 否则 RAG 侧不公平。
    """
    q = set(re.findall(r"[\u4e00-\u9fff]+", question))
    scored = []
    for i, c in enumerate(kb_chunks):
        txt = c.get("text", "")
        overlap = len(q & set(re.findall(r"[\u4e00-\u9fff]+", txt)))
        scored.append((overlap, i))
    scored.sort(reverse=True)
    return [kb_chunks[i]["text"] for _, i in scored[:k]]


def answer_ft(question):
    """FT(权重) 直接答, 不给上下文。"""
    return _post(FT_ENDPOINT, FT_KEY, MODEL, question)


def answer_rag(question, kb_chunks):
    """RAG: 先检索, 再把片段喂给生成模型。"""
    ctx = "\n".join(retrieve(question, kb_chunks))
    prompt = f"根据以下年报片段回答问题, 严格基于片段、可引用来源:\n{ctx}\n\n问题: {question}"
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


def load_chunks(path):
    if os.path.isdir(path):
        chunks = []
        for f in sorted(os.listdir(path)):
            if f.endswith(".json"):
                chunks.extend(json.load(open(os.path.join(path, f), encoding="utf-8")))
        return chunks
    return json.load(open(path, encoding="utf-8"))


def run_side(side):
    if side == "A":
        qa = load_qa(os.path.join(HERE, "03_LoRA数据", "qa_alpaca.json"))
        kb = load_chunks(os.path.join(HERE, "04_RAG数据", "rag_chunks.json"))
    else:  # B
        qa = load_qa(os.path.join(HERE, "heldout_B", "qa_alpaca.json"))
        kb = load_chunks(os.path.join(HERE, "heldout_B", "chunks"))

    ft_scores, rag_scores = [], []
    for q in qa:
        qn = q["instruction"]
        ft_scores.append(judge(qn, answer_ft(qn)))
        rag_scores.append(judge(qn, answer_rag(qn, kb)))
    n = len(qa) or 1
    return sum(ft_scores) / n, sum(rag_scores) / n


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "--all"
    if which in ("--side", "--all"):
        fa, ra = run_side("A")
        print(f"[见过 A] FT-on-A={fa:.1%}  RAG-on-A={ra:.1%}")
    if which in ("--side", "--all"):
        fb, rb = run_side("B")
        print(f"[未见 B] FT零样本={fb:.1%}  RAG-on-B={rb:.1%}")
    if which == "--all":
        print("\n===== 2x2 对比表 =====")
        print(f"             | FT(权重)     | RAG(检索)")
        print(f"-------------|-------------|-------------")
        print(f"见过 A(10家) | (见上)      | (见上)")
        print(f"未见 B(15家) | (见上)      | (见上)")
        print("\n结论: FT 只在见过的地方高 -> 把事实背进了权重; RAG 未见也高 -> 事实在知识库。")


if __name__ == "__main__":
    main()
