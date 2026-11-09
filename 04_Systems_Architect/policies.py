"""Decision policies for the deliberation loop. A policy is a callable  policy(observation) -> decision  where
decision = {"type": "tool", "call": {"tool": ..., "arguments": {...}}}   or   {"type": "final", "answer": "..."}.
No LLM is used here: these scripted and random policies stand in for AURA's model so the control loop can be tested."""
import random


def tool(name, **arguments):
    return {"type": "tool", "call": {"tool": name, "arguments": arguments}}


def final(answer="Warp engaged."):
    return {"type": "final", "answer": answer}


GOOD_SCRIPT = [tool("run_diagnostic", system="warp_core"), tool("run_diagnostic", system="nacelles"),
               tool("set_warp_factor", factor=6.0), tool("engage_warp")]

SLOPPY_SCRIPT = [tool("set_warp_factor", factor=12),          # out of range -> invalid params
                 {"type": "tool", "call": {"tool": "hyperdrive", "arguments": {}}},     # unknown tool
                 tool("engage_warp")] + GOOD_SCRIPT           # engage too early -> tool exception, then recovery


class ScriptedPolicy:
    def __init__(self, script, answer="Warp engaged."):
        self.script, self.answer, self.i = list(script), answer, 0

    def __call__(self, obs):
        if self.i < len(self.script):
            self.i += 1
            return self.script[self.i - 1]
        return final(self.answer)


class LoopPolicy:
    """Ship Anomaly 06 generator: repeats one action forever and never claims to be finished."""
    def __init__(self, decision=None):
        self.decision = decision or tool("run_diagnostic", system="warp_core")

    def __call__(self, obs):
        return self.decision


class FalseClaimPolicy:
    def __call__(self, obs):
        return final("Warp engaged.")                      # claims success without doing anything


class RandomPolicy:
    def __init__(self, seed=0, p_final=0.08, p_invalid=0.10):
        self.rng, self.p_final, self.p_invalid = random.Random(seed), p_final, p_invalid

    def __call__(self, obs):
        u = self.rng.random()
        if u < self.p_final:
            return final("Warp engaged.")
        if u < self.p_final + self.p_invalid:
            return tool("set_warp_factor", factor=self.rng.choice([-3, 15, 99]))
        choice = self.rng.choice(["status", "diag", "set", "engage", "disengage"])
        if choice == "status":
            return tool("get_warp_status")
        if choice == "diag":
            return tool("run_diagnostic", system=self.rng.choice(["warp_core", "nacelles", "shields"]))
        if choice == "set":
            return tool("set_warp_factor", factor=self.rng.choice([0.0, 2.0, 5.0, 6.0, 9.0]))
        return tool("engage_warp") if choice == "engage" else tool("disengage_warp")
