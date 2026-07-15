#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金融年报语料 数据质控脚本
用法:
  python qc_data.py                 # 默认: 格式校验 + 重复率 + 抽 10 条
  python qc_data.py --sample 20     # 抽 20 条打印
  python qc_data.py --no-sample     # 只出统计, 不打印样本
校验: 03_LoRA数据/qa_alpaca.json + qa_sharegpt.json + 04_RAG数据/rag_chunks.json
"""
import os, sys, json, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ALPACA = os.path.join(HERE, "03_LoRA数据", "qa_alpaca.json")
SHARE = os.path.join(HERE, "03_LoRA数据", "qa_sharegpt.json")
RAG = os.path.join(HERE, "04_RAG数据", "rag_chunks.json")

SAMPLE = 10
NO_SAMPLE = "--no-sample" in sys.argv
for i, a in enumerate(sys.argv):
    if a.startswith("--sample"):
        SAMPLE = int(a.split("=")[1] if "=" in a else sys.argv[i + 1])

def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[错误] 读取 {os.path.basename(p)} 失败: {e}")
        return None

def norm(s):
    return "".join(s.split()).lower()

print("=" * 60)
print("1) 文件加载 & 基本量")
print("=" * 60)
alp = load(ALPACA)
sha = load(SHARE)
rag = load(RAG)
if alp is None or sha is None or rag is None:
    sys.exit(1)
print(f"  qa_alpaca   : {len(alp)} 条")
print(f"  qa_sharegpt : {len(sha)} 条")
print(f"  rag_chunks  : {len(rag)} 条")
print(f"  rag 含表格块: {sum(1 for c in rag if c.get('is_table'))} 个")
srcs = sorted({c.get("source") for c in rag})
print(f"  覆盖报告数  : {len(srcs)} 份")

print("\n" + "=" * 60)
print("2) 格式校验")
print("=" * 60)
err_alp = 0
for x in alp:
    if not isinstance(x, dict) or not x.get("instruction") or not x.get("output"):
        err_alp += 1
err_sha = 0
for x in sha:
    conv = x.get("conversations") if isinstance(x, dict) else None
    if not isinstance(conv, list) or len(conv) < 2:
        err_sha += 1
        continue
    roles = [m.get("role") for m in conv]
    if roles != ["user", "assistant"] * (len(conv) // 2) and not (
        roles[0] == "user" and roles[-1] == "assistant"
    ):
        err_sha += 1
err_rag = 0
for c in rag:
    if not all(k in c for k in ("source", "text", "is_table", "chunk_id")):
        err_rag += 1
print(f"  alpaca 异常条数 : {err_alp}")
print(f"  sharegpt 异常条数: {err_sha}")
print(f"  rag 异常条数    : {err_rag}")
print(f"  alpaca/sharegpt 条数一致: {len(alp) == len(sha)}")

print("\n" + "=" * 60)
print("3) 重复率 & 空值")
print("=" * 60)
ins_all = [x["instruction"] for x in alp]
ins_empty = sum(1 for x in alp if not x["instruction"].strip())
out_empty = sum(1 for x in alp if not x["output"].strip())
distinct_ins = len({norm(i) for i in ins_all})
distinct_io = len({(norm(x["instruction"]), norm(x["output"])) for x in alp})
print(f"  指令为空      : {ins_empty}")
print(f"  回答为空      : {out_empty}")
print(f"  不同指令数    : {distinct_ins} / {len(ins_all)}")
print(f"  指令重复率    : {1 - distinct_ins / len(ins_all):.1%}")
print(f"  不同(指令,答) : {distinct_io} / {len(ins_all)}")
print(f"  (指令,答)重复率: {1 - distinct_io / len(ins_all):.1%}")

print("\n" + "=" * 60)
print("4) 长度分布 (alpaca)")
print("=" * 60)
il = [len(x["instruction"]) for x in alp]
ol = [len(x["output"]) for x in alp]
def stat(name, xs):
    xs = sorted(xs)
    print(f"  {name}: 最小 {xs[0]}  中位 {xs[len(xs)//2]}  最大 {xs[-1]}  均值 {sum(xs)//len(xs)}")
stat("指令长度", il)
stat("回答长度", ol)
short_out = sum(1 for x in ol if x < 15)
long_out = sum(1 for x in ol if x > 800)
print(f"  回答<15字(偏短, 多为数字/名称/否定, 一般非废答): {short_out}")
print(f"  回答>800字(偏长)        : {long_out}")
# 导出短回答供人工回看(非废答判定, 仅列示)
short_list = [x for x in alp if len(x["output"]) < 15]
short_path = os.path.join(HERE, "03_LoRA数据", "_short_answers.json")
json.dump(short_list, open(short_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"  已导出短回答 {len(short_list)} 条 -> {os.path.basename(short_path)}")

print("\n" + "=" * 60)
print(f"5) 随机抽 {SAMPLE} 条 (供人工肉眼检查)")
print("=" * 60)
if not NO_SAMPLE and alp:
    random.seed(42)
    for x in random.sample(alp, min(SAMPLE, len(alp))):
        print("-" * 50)
        print("Q:", x["instruction"])
        print("A:", x["output"])
print("\n完成。异常/重复较多时, 建议先清洗再进 LLaMA-Factory / 向量库。")
