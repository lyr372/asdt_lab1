import argparse
import yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from src.models.slm_wrapper import SLMWrapper
from src.models.llm_wrapper import LLMWrapper
from src.engine.cost_meter import CostMeter
from src.utils.logging_utils import init_logging
from src.data.gsm8k_loader import load_gsm8k_subset

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_tokenizer(cfg_models):
    tok_name = cfg_models.get("tokenizer_name", cfg_models.get("llm_name"))
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_model(wrapper_cls, name, tokenizer, cfg_models):
    device = cfg_models.get("device", "auto")
    dtype = cfg_models.get("dtype", "float16")
    use_4bit = bool(cfg_models.get("use_4bit", False))
    use_8bit = bool(cfg_models.get("use_8bit", False))
    m = wrapper_cls(name, tokenizer, device=device, dtype=dtype, use_4bit=use_4bit, use_8bit=use_8bit)
    m.load()
    return m

def generate_baseline(model, tokenizer, prompt, max_new_tokens=64, temperature=0.7, top_p=0.9, eos_token_id=None):
    ctx = tokenizer.encode(prompt, add_special_tokens=True)
    base_len = len(ctx)
    ns = 0
    produced = 0
    while produced < max_new_tokens:
        logprobs = model.forward_logprobs(ctx)
        ns += 1
        pos = len(ctx) - 1
        probs = np.exp(logprobs[pos])
        token = model.sample_from_probs(probs, temperature=temperature, top_p=top_p)
        ctx.append(token)
        produced += 1
        if eos_token_id is not None and token == eos_token_id:
            break
    text = tokenizer.decode(ctx[base_len:])
    return {"text": text, "ns": ns, "nl": 0, "tokens": produced}

def run(mode, prompts, config):
    logger = init_logging()
    cfg_models = config.get("models", {})
    cfg_engine = config.get("engine", {})
    cfg_cost = config.get("cost", {})
    tokenizer = build_tokenizer(cfg_models)
    eos_id = tokenizer.eos_token_id
    c_s = float(cfg_cost.get("c_s", 1.0))
    c_l = float(cfg_cost.get("c_l", 10.0))
    cm = CostMeter(c_s=c_s, c_l=c_l)
    if mode == "slm":
        slm = build_model(SLMWrapper, cfg_models.get("slm_name"), tokenizer, cfg_models)
        for p in tqdm(prompts):
            out = generate_baseline(slm.lm, tokenizer, p, max_new_tokens=cfg_engine.get("max_new_tokens", 64), temperature=cfg_engine.get("temperature", 0.7), top_p=cfg_engine.get("top_p", 0.9), eos_token_id=eos_id)
            cm.update_counts(ns=out["ns"], nl=0, t=out["tokens"]) 
            logger.info(f"SLM text: {out['text']}")
    elif mode == "llm":
        llm = build_model(LLMWrapper, cfg_models.get("llm_name"), tokenizer, cfg_models)
        for p in tqdm(prompts):
            out = generate_baseline(llm.lm, tokenizer, p, max_new_tokens=cfg_engine.get("max_new_tokens", 64), temperature=cfg_engine.get("temperature", 0.7), top_p=cfg_engine.get("top_p", 0.9), eos_token_id=eos_id)
            cm.update_counts(ns=0, nl=out["ns"], t=out["tokens"]) 
            logger.info(f"LLM text: {out['text']}")
    else:
        raise ValueError("mode must be 'slm' or 'llm'")
    stats = cm.compute_average()
    logger.info(f"Avg cost: {stats['avg_cost']:.4f}, Ns/T: {stats['avg_ns_per_t']:.4f}, Nl/T: {stats['avg_nl_per_t']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["slm", "llm"])
    ap.add_argument("--config", type=str, default="src/config/default_config.yaml")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--use_gsm8k", action="store_true")
    ap.add_argument("--num_samples", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pr = args.prompts if args.prompts else []
    if args.use_gsm8k and len(pr) == 0:
        pr = load_gsm8k_subset(args.num_samples)
    if len(pr) == 0:
        pr = ["What is 12 + 35?", "Write a short poem about winter."][:args.num_samples]
    run(args.mode, pr, cfg)