import unittest
from src.engine.hybrid_inference import HybridInferenceEngine

class ToyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
    def encode(self, s, add_special_tokens=True):
        return [1, 2]
    def decode(self, ids):
        return " ".join(str(i) for i in ids)

class ToySLM:
    def __init__(self):
        pass
    def draft_generate_k(self, input_ids, k, eos_token_id, temperature=1.0, top_p=1.0):
        return [3, 3, 4][:k], k

class ToyLLM:
    def __init__(self, vocab_size=6):
        self.vocab_size = vocab_size
    def forward_logprobs(self, input_ids):
        n = len(input_ids)
        L = [[-10.0 for _ in range(self.vocab_size)] for _ in range(n)]
        for i in range(n):
            if i <= 2:
                L[i][3] = -0.1
            else:
                L[i][4] = -0.05
        return L
    def sample_from_probs(self, probs, temperature=1.0, top_p=1.0):
        m = max(probs)
        for i, v in enumerate(probs):
            if v == m:
                return i
        return 0

class TestHybridEngine(unittest.TestCase):
    def test_accept_then_free(self):
        tok = ToyTokenizer()
        slm = ToySLM()
        llm = ToyLLM()
        engine = HybridInferenceEngine(slm=slm, llm=llm, tokenizer=tok, block_size=3, p_t=0.2, max_new_tokens=3, temperature=0.0, top_p=1.0, eos_token_id=tok.eos_token_id, cost_meter=None)
        out = engine.generate("x")
        self.assertIn("text", out)
        self.assertGreater(out["tokens"], 0)
        self.assertGreaterEqual(out["nl"], 1)

    def test_tau_zero(self):
        class LLMLow(ToyLLM):
            def forward_logprobs(self, input_ids):
                n = len(input_ids)
                L = [[-10.0 for _ in range(self.vocab_size)] for _ in range(n)]
                for i in range(n):
                    L[i][5] = -0.01
                return L
        tok = ToyTokenizer()
        slm = ToySLM()
        llm = LLMLow()
        engine = HybridInferenceEngine(slm=slm, llm=llm, tokenizer=tok, block_size=2, p_t=0.99, max_new_tokens=2, temperature=0.0, top_p=1.0, eos_token_id=tok.eos_token_id, cost_meter=None)
        out = engine.generate("x")
        self.assertEqual(out["nl"], 1)
        self.assertGreaterEqual(out["tokens"], 1)

if __name__ == "__main__":
    unittest.main()