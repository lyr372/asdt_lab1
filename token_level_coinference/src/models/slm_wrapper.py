from .base_lm import BaseLM

class SLMWrapper:
    def __init__(self, model_name, tokenizer, device="auto", dtype="float16", use_4bit=False, use_8bit=False):
        self.lm = BaseLM(model_name=model_name, tokenizer=tokenizer, device=device, dtype=dtype, use_4bit=use_4bit, use_8bit=use_8bit)

    def load(self):
        self.lm.load()

    def forward_logprobs(self, input_ids):
        return self.lm.forward_logprobs(input_ids)

    def draft_generate_k(self, input_ids, k, eos_token_id, temperature=1.0, top_p=1.0):
        return self.lm.draft_generate_k(input_ids, k, eos_token_id, temperature=temperature, top_p=top_p)

    def sample_from_probs(self, probs, temperature=1.0, top_p=1.0):
        return self.lm.sample_from_probs(probs, temperature=temperature, top_p=top_p)