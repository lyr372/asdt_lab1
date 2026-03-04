class CostMeter:
    def __init__(self, c_s=1.0, c_l=10.0):
        self.c_s = c_s
        self.c_l = c_l
        self.ns_total = 0
        self.nl_total = 0
        self.t_total = 0
        self.num_samples = 0

    def update_counts(self, ns, nl, t):
        self.ns_total += ns
        self.nl_total += nl
        self.t_total += t
        self.num_samples += 1

    def compute_average(self):
        if self.num_samples == 0:
            return {"avg_cost": 0.0, "avg_ns_per_t": 0.0, "avg_nl_per_t": 0.0}
        t = self.t_total / self.num_samples
        ns_per_t = self.ns_total / (self.t_total + 1e-12)
        nl_per_t = self.nl_total / (self.t_total + 1e-12)
        cost = self.c_s * ns_per_t + self.c_l * nl_per_t
        return {"avg_cost": cost, "avg_ns_per_t": ns_per_t, "avg_nl_per_t": nl_per_t, "avg_T": t}