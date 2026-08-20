# -*- coding: utf-8 -*-
"""口语化改写：用异家族模型 Kimi-K2.5 改写 40 题，切断 DeepSeek 近亲链（§9.1 #1）。

原题来自 original_40.json（Random(42) 取 100 → Random(7) 取 40，原 hr_route 82.5%）。
出题/答题/裁判均为 DeepSeek 家族，改写者换成 Ali-dashscope/Kimi-K2.5（山大代理
同端点，仅换模型名），保证"考卷不是被考者自己出的措辞"。

改写约束（写死在 prompt 里）：
  1. 公司名必须原样保留（路由靠它，丢了就不是检索皮实性测试了）；
  2. 考点不变：问的还是同一个事实，参考答案不用改；
  3. 去年报腔换成散户/口语问法，允许语序颠倒、口语垫词。

产物 rewritten_40.json：每题 {idx, company, original, rewritten, ...}，
改写后自动检查公司名是否保留（company_kept 字段），丢名的题需人工修。

用法（llf 环境，RAG-lora/ 下单行前台运行，纯 API 零占卡）：
  /data1/jiajun/.conda/envs/llf/bin/python eval_validity/03_colloquial/rewrite_40.py 2>&1 | tee eval_validity/03_colloquial/rewrite.log
断点续存：每 10 题落盘；重跑跳过已改完的题（--fresh 强制全部重改）。
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RAG-lora/
sys.path.insert(0, ROOT)

from rag import llm  # noqa: E402

REWRITE_MODEL = "Ali-dashscope/Kimi-K2.5"

REWRITE_TMPL = (
    "把下面这个正式的年报问答题改写成普通散户的口语问法。要求：\n"
    "1. 公司名「{company}」必须一字不差地保留在改写后的问题里；\n"
    "2. 问的事实/考点必须完全不变（同一年份、同一指标、同一事项），答案不能因改写而变；\n"
    "3. 去掉书面腔，用日常聊天口吻，可以颠倒语序、加口语垫词（如\"想问下\"\"到底\"），"
    "但不要添加原题没有的信息，也不要把具体指标名换成含糊说法；\n"
    "4. 只输出改写后的问题一句话，不要任何解释、引号或前缀。\n"
    "原题: {question}"
)


def chat_retry(client, prompt, tries=3, timeout=120):
    """带重试的 API 调用（同 eval_retrieval）：偶发读超时退避重试不炸全场。"""
    for t in range(tries):
        try:
            return client.chat(prompt, timeout=timeout)
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def main():
    ap = argparse.ArgumentParser(description="口语化改写 40 题（Kimi-K2.5，纯 API）")
    ap.add_argument("--src", default=os.path.join(HERE, "original_40.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "rewritten_40.json"))
    ap.add_argument("--workers", type=int, default=8, help="API 并发数")
    ap.add_argument("--fresh", action="store_true", help="忽略已有结果全部重改")
    args = ap.parse_args()

    rows = json.load(open(args.src, encoding="utf-8"))
    old = {}
    if not args.fresh and os.path.exists(args.out):
        try:
            old = {d["idx"]: d for d in json.load(open(args.out, encoding="utf-8"))
                   if d.get("rewritten")}
        except Exception:
            old = {}
    todo = [x for x in rows if x["idx"] not in old]
    print(f"[数据] {len(rows)} 题，已改 {len(old)}，待改 {len(todo)}  模型={REWRITE_MODEL}")

    client = llm.APIClient()
    client.model = REWRITE_MODEL          # 端点/key 复用 .env，仅换模型名

    def _save():
        done = sorted(list(old.values()) + [x for x in todo if x.get("rewritten")],
                      key=lambda d: d["idx"])
        json.dump(done, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return done

    def _proc(x):
        try:
            out = chat_retry(client, REWRITE_TMPL.format(
                company=x["company"], question=x["question"])).strip()
            x["rewritten"] = out.splitlines()[0].strip() if out else ""
            x["company_kept"] = x["company"] in x["rewritten"]
        except Exception as e:
            x["rewritten"], x["error"] = "", str(e)
        return x

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_proc, todo):
            n += 1
            if n % 10 == 0 or n == len(todo):
                _save()
                print(f"  [改写] {n}/{len(todo)}", flush=True)

    done = _save()
    lost = [d for d in done if not d.get("company_kept")]
    fail = [d for d in done if not d.get("rewritten")]
    print(f"\n[完成] {len(done)}/{len(rows)} 题已存 {args.out}")
    if fail:
        print(f"[警告] {len(fail)} 题改写失败: idx={[d['idx'] for d in fail]}")
    if lost:
        print(f"[警告] {len(lost)} 题公司名丢失需人工修: idx={[d['idx'] for d in lost]}")
    if not fail and not lost:
        print("[自检] 40 题公司名全部保留 ✓")


if __name__ == "__main__":
    main()
