import math

class HybridInferenceEngine:
    def __init__(self, slm, llm, tokenizer, block_size=4, p_t=0.2, max_new_tokens=64, temperature=0.7, top_p=0.9, eos_token_id=None, cost_meter=None):
        self.slm = slm
        self.llm = llm
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.p_t = float(p_t)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.cost_meter = cost_meter

    def generate(self, prompt):
        ctx = self.tokenizer.encode(prompt, add_special_tokens=True)
        base_len = len(ctx)
        ns = 0
        nl = 0
        produced = 0
        while produced < self.max_new_tokens:
            draft, steps = self.slm.draft_generate_k(ctx, self.block_size, self.eos_token_id, temperature=self.temperature, top_p=self.top_p)
            ns += steps
            if len(draft) == 0:
                logprobs = self.llm.forward_logprobs(ctx)
                nl += 1
                pos = len(ctx) - 1
                row = logprobs[pos]
                probs = [math.exp(v) for v in row]
                token = self.llm.sample_from_probs(probs, temperature=self.temperature, top_p=self.top_p)
                ctx.append(token)
                produced += 1
                if self.eos_token_id is not None and token == self.eos_token_id:
                    break
                continue
            extended = ctx + draft
            logprobs = self.llm.forward_logprobs(extended)
            nl += 1
            p = len(ctx)
            tau = 0
            for i in range(len(draft)):
                j = p + i - 1
                prob = math.exp(logprobs[j][draft[i]])
                if prob <= self.p_t:
                    break
                tau += 1
            accepted = draft[:tau]
            j_free = p + tau - 1
            probs_free = [math.exp(v) for v in logprobs[j_free]]
            free_token = self.llm.sample_from_probs(probs_free, temperature=self.temperature, top_p=self.top_p)
            ctx.extend(accepted + [free_token])
            produced += len(accepted) + 1
            if self.eos_token_id is not None:
                if any(t == self.eos_token_id for t in accepted) or free_token == self.eos_token_id:
                    break
            if produced >= self.max_new_tokens:
                break
        text = self.tokenizer.decode(ctx[base_len:])
        if self.cost_meter is not None:
            self.cost_meter.update_counts(ns=ns, nl=nl, t=produced)
        return {"text": text, "ns": ns, "nl": nl, "tokens": produced}