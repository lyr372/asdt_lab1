import numpy as np
import torch
from transformers import AutoModelForCausalLM

class BaseLM:
    def __init__(self, model_name, tokenizer, device="auto", dtype="float16", use_4bit=False, use_8bit=False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        self.model = None

    def load(self):
        kwargs = {}
        kwargs["torch_dtype"] = self.torch_dtype
        if self.use_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        elif self.use_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not (self.use_4bit or self.use_8bit):
            self.model.to(self.device)
        self.model.eval()

    def forward_logits(self, input_ids):
        ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids)
        logits = out.logits[0].detach().cpu().numpy()
        return logits

    def forward_logprobs(self, input_ids):
        ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids)
            logits = out.logits[0]
            logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs.detach().cpu().numpy()

    def sample_from_probs(self, probs, temperature=1.0, top_p=1.0):
        p = np.array(probs, dtype=np.float64)
        if temperature > 0 and temperature != 1.0:
            p = p ** (1.0 / temperature)
        p = p / (p.sum() + 1e-12)
        if top_p < 1.0:
            idx = np.argsort(p)[::-1]
            sorted_p = p[idx]
            cumsum = np.cumsum(sorted_p)
            keep = cumsum <= top_p
            if not keep.any():
                keep[0] = True
            mask = np.zeros_like(p)
            mask[idx[keep]] = 1.0
            p = p * mask
            s = p.sum()
            if s <= 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / s
        r = np.random.random()
        c = np.cumsum(p)
        j = np.searchsorted(c, r, side="right")
        if j >= len(p):
            j = len(p) - 1
        return int(j)

    def draft_generate_k(self, input_ids, k, eos_token_id, temperature=1.0, top_p=1.0):
        produced = []
        steps = 0
        ctx = list(input_ids)
        for _ in range(k):
            logits = self.forward_logprobs(ctx)
            steps += 1
            pos = len(ctx) - 1
            probs = np.exp(logits[pos])
            token = self.sample_from_probs(probs, temperature=temperature, top_p=top_p)
            produced.append(token)
            ctx.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
        return produced, steps