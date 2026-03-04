import argparse
import yaml
from src.run_baseline import run as run_baseline
from src.run_hybrid import run as run_hybrid_main
from src.data.gsm8k_loader import load_gsm8k_subset

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["slm", "llm", "hybrid"])
    ap.add_argument("--config", type=str, default="src/config/default_config.yaml")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--use_gsm8k", action="store_true")
    ap.add_argument("--num_samples", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prompts = args.prompts if args.prompts else []
    if args.use_gsm8k and len(prompts) == 0:
        prompts = load_gsm8k_subset(args.num_samples)
    if len(prompts) == 0:
        prompts = ["What is 12 + 35?", "Write a short poem about winter."][:args.num_samples]
    if args.mode in ("slm", "llm"):
        run_baseline(args.mode, prompts, cfg)
    else:
        run_hybrid_main(prompts, cfg)