# -*- coding: utf-8 -*-
"""生成后端统一封装：山大代理 API + 本地 Qwen3-4B(+LoRA)。

被 rag_chat.py（问答入口）与 eval_heldout.py（评测）共用，避免两处各写一套
.env 读取 / HTTP 调用 / 本地模型加载。

- load_env()      : 读 RAG-lora/.env，返回 {BASE, TOKEN, MODEL, ENDPOINT}
- post_messages() : 通用 anthropic 兼容 /v1/messages 调用，返回模型文本
- APIClient       : 上面两者的便捷组合（从 .env 自动初始化）
- LocalLLM        : 本地 Qwen3-4B + 可选 LoRA adapter，chat(prompt) 出答案
"""
import os
import json
import ssl
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # RAG-lora/
PROJ = os.path.dirname(ROOT)                       # llmfactory/
DEFAULT_BASE_MODEL = os.path.join(PROJ, "models", "Qwen3-4B")
DEFAULT_ADAPTER = os.path.join(PROJ, "saves", "qwen3-4b", "lora", "sft_rank16")

_SSL_CTX = ssl.create_default_context()


def load_env(path=None):
    """读 .env（默认 RAG-lora/.env），返回配置 dict。

    键：BASE / TOKEN / MODEL / ENDPOINT（ENDPOINT = BASE + /v1/messages）。
    """
    env = {}
    cand = path or os.path.join(ROOT, ".env")
    try:
        for line in open(cand, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    base = env.get("ANTHROPIC_BASE_URL", "")
    return {
        "BASE": base,
        "TOKEN": env.get("ANTHROPIC_AUTH_TOKEN", ""),
        "MODEL": env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
        "ENDPOINT": base.rstrip("/") + "/v1/messages" if base else "",
    }


def post_messages(endpoint, key, model, prompt,
                  max_tokens=1024, temperature=0, timeout=60):
    """通用 /v1/messages 调用，返回模型文本。endpoint 为空时抛错提醒填。"""
    if not endpoint:
        raise RuntimeError("端点未填写: 请检查 RAG-lora/.env 的 ANTHROPIC_BASE_URL")
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "thinking": {"type": "disabled"},   # 关思考：加速、省 token
                       "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01",
        "content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return "".join(c.get("text", "") for c in d.get("content", []) if isinstance(c, dict))


class APIClient:
    """从 .env 自动初始化的山大代理客户端。chat(prompt) -> 文本。"""

    def __init__(self, env_path=None):
        cfg = load_env(env_path)
        self.endpoint = cfg["ENDPOINT"]
        self.key = cfg["TOKEN"]
        self.model = cfg["MODEL"]

    def chat(self, prompt, **kw):
        return post_messages(self.endpoint, self.key, self.model, prompt, **kw)


class LocalLLM:
    """本地 Qwen3-4B（可选叠加 LoRA adapter），chat(prompt) -> 文本。

    adapter=None 时用纯 base 模型；device 用 "cuda" 时可配合
    CUDA_VISIBLE_DEVICES=N 选卡（物理卡 N 映射为 cuda:0）。
    """

    def __init__(self, base_model=DEFAULT_BASE_MODEL, adapter=DEFAULT_ADAPTER,
                 device="cuda"):
        import torch  # noqa: F401  延迟到实例化时才要求 torch/GPU
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype="auto").to(device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    def chat(self, prompt, max_new_tokens=512, max_input_tokens=6144):
        import torch
        tok = self.tokenizer
        # 超长输入截断（保头保尾：尾部是问题，不能丢），避免长表格块把 KV cache 撑爆
        ids = tok(prompt)["input_ids"]
        if len(ids) > max_input_tokens:
            keep = max_input_tokens // 2
            prompt = (tok.decode(ids[:keep], skip_special_tokens=True)
                      + "\n…(中间已截断)…\n"
                      + tok.decode(ids[-keep:], skip_special_tokens=True))
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False).to(self.model.device)  # 关思考模式：省显存/不被512token截断
        attn = inputs != tok.pad_token_id if tok.pad_token_id is not None else None
        with torch.no_grad():
            out = self.model.generate(inputs, attention_mask=attn,
                                      max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
