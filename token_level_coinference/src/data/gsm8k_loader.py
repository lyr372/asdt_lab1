from typing import List

def load_gsm8k_subset(n: int = 10) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception:
        return [
            "Solve: 12 + 35",
            "If Tom has 5 apples and buys 7 more, how many?",
            "Compute: 23 + 57",
            "A rectangle has sides 3 and 4. Area?",
            "What is 100 - 37?",
        ][:n]
    ds = load_dataset("gsm8k", "main", split="train")
    prompts = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"].strip()
        prompts.append(q)
    return prompts