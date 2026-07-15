#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清洗 LoRA 问答数据中的 RAG 痕迹短语
问题: 约 19% 的回答带 '根据片段原文，'/'根据财报片段，'/'根据年报片段，' 等检索语境残留,
      既出现在句首, 也散落句中('需注意。根据片段，应付利息…')。脱离检索单独用 LoRA 时,
      模型会张口 '根据财报片段' 却没片段可引, 显得破绽。
做法: 全局正则删除 '根据(文档类词)，' 短语(句首/句中都清), 保留后续真实内容;
      原始文件备份为 *.raw.json(已存在则不再覆盖)。
注意: 仅删 '根据+文档类词', 保留 '根据企业会计准则/公司章程/《公司法》' 等合法用法。
用法:
  python clean_qa.py            # 清洗并覆盖 qa_alpaca/qa_sharegpt, 备份原文件
  python clean_qa.py --dry      # 只统计/预览, 不写盘
"""
import os, sys, json, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ALPACA = os.path.join(HERE, "03_LoRA数据", "qa_alpaca.json")
SHARE = os.path.join(HERE, "03_LoRA数据", "qa_sharegpt.json")
DRY = "--dry" in sys.argv

# 文档类词(长词在前, 避免 '报告' 抢匹配 '报告原文'); 不含 企业/公司/《 等合法前缀
TRACE = re.compile(
    r"根据(财报片段|年报片段|报告原文|财报原文|年报原文|片段原文|报告|财报|年报|片段|原文|上述)"
    r"[，,：:、\s]*"
)
# 统计用(更宽, 含句中/句首各类痕迹, 用于对比清洗前后)
STAT = re.compile(r"根据.{0,6}(原文|片段|片段原文)|片段原文|根据上述")

def clean(text):
    new = TRACE.sub("", text)
    # 剥离后过短/只剩标点, 视为无法安全清洗, 退回原值
    if len(new.strip()) <= 1:
        return text
    return new

def stat_count(items, get_text):
    return sum(1 for it in items if STAT.search(get_text(it)))

print("加载:", os.path.basename(ALPACA), os.path.basename(SHARE))
alp = json.load(open(ALPACA, encoding="utf-8"))
sha = json.load(open(SHARE, encoding="utf-8"))

before_a = stat_count(alp, lambda x: x["output"])
before_s = stat_count(sha, lambda x: x["conversations"][-1]["content"])

n_a = n_s = 0
examples = []
for x in alp:
    o = x["output"]
    s = clean(o)
    if s != o:
        x["output"] = s
        n_a += 1
        if len(examples) < 6:
            examples.append((o, s))
for x in sha:
    o = x["conversations"][-1]["content"]
    s = clean(o)
    if s != o:
        x["conversations"][-1]["content"] = s
        n_s += 1

after_a = stat_count(alp, lambda x: x["output"])
after_s = stat_count(sha, lambda x: x["conversations"][-1]["content"])

print(f"\n清洗前 RAG 痕迹(宽口径): alpaca={before_a}  sharegpt={before_s}")
print(f"本次删除短语条数:         alpaca={n_a}  sharegpt={n_s}")
print(f"清洗后 RAG 痕迹(宽口径): alpaca={after_a}  sharegpt={after_s}")
print("\n预览(前6条 前->后):")
for o, s in examples:
    print("  -", o[:36], "->", s[:36])

if DRY:
    print("\n[--dry] 未写盘。")
    sys.exit(0)

for path in (ALPACA, SHARE):
    raw = path[:-5] + ".raw.json"
    if not os.path.exists(raw):
        shutil.copy(path, raw)
        print("备份:", os.path.basename(raw))
json.dump(alp, open(ALPACA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(sha, open(SHARE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n已写回:", os.path.basename(ALPACA), os.path.basename(SHARE))
