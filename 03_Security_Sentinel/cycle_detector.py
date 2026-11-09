"""Floyd cycle detection and halting guardrails for the deliberation loop (standard library only).
Station: Security Sentinel.   Ship Anomaly 06 (our working definition): a runaway deliberation loop in warp control, where the
agent repeats the same action in the same ship state without making progress, burning steps and energy.

Floyd's tortoise-and-hare finds a cycle in the sequence x0, f(x0), f(f(x0)), ... using O(1) extra memory:
  phase 1: tortoise moves 1 step, hare 2 steps, until they meet (a cycle exists) or the hare falls off the end (none);
  phase 2: restart the tortoise at x0, move both 1 step at a time; they meet at the cycle start index mu;
  phase 3: walk the hare around once to measure the cycle length lambda.
Requirement: f must be a FUNCTION (each node has exactly one successor). An agent trace only satisfies this when the policy and the
environment are deterministic, so detect_in_trace() checks that and otherwise falls back to a tail-repetition test."""
import time
from dataclasses import dataclass
from typing import Callable, Hashable, List, Optional


@dataclass
class CycleInfo:
    has_cycle: bool
    mu: int = -1            # index where the cycle starts
    lam: int = 0            # cycle length
    method: str = "floyd"   # 'floyd' or 'tail-period'


def floyd(f: Callable[[Hashable], Optional[Hashable]], x0, max_steps=1_000_000) -> CycleInfo:
    """f(x) returns the successor of x, or None when the sequence ends. Returns CycleInfo."""
    tortoise = f(x0)                         # index 1
    if tortoise is None:
        return CycleInfo(False)
    hare = f(tortoise)                       # index 2
    if hare is None:
        return CycleInfo(False)
    steps = 0
    while tortoise != hare:
        tortoise = f(tortoise)               # the tortoise is behind the hare, so this cannot be None
        hare = f(hare)
        if hare is None:
            return CycleInfo(False)
        hare = f(hare)
        if hare is None:
            return CycleInfo(False)
        steps += 1
        if steps > max_steps:
            return CycleInfo(False)
    mu, tortoise = 0, x0
    while tortoise != hare:
        tortoise, hare, mu = f(tortoise), f(hare), mu + 1
    lam, hare = 1, f(tortoise)
    while tortoise != hare:
        hare, lam = f(hare), lam + 1
    return CycleInfo(True, mu, lam)


def tail_period(nodes: List[Hashable], repeats=2) -> CycleInfo:
    """Fallback for non-deterministic traces: does the trace end with some block of length p repeated `repeats` times?"""
    n = len(nodes)
    for p in range(1, n // repeats + 1):
        block = nodes[n - p:]
        if all(nodes[n - p * k: n - p * (k - 1)] == block for k in range(2, repeats + 1)):
            return CycleInfo(True, n - p * repeats, p, "tail-period")
    return CycleInfo(False, method="tail-period")


def detect_in_trace(nodes: List[Hashable], fallback_repeats=2) -> CycleInfo:
    """Build the observed successor map; if it is a function, run Floyd from the first node, else use the tail test
    (which needs `fallback_repeats` full repetitions before it reports a cycle)."""
    succ = {}
    for a, b in zip(nodes, nodes[1:]):
        if succ.setdefault(a, b) != b:
            return tail_period(nodes, fallback_repeats)  # same node led to different successors: not a function
    if not nodes:
        return CycleInfo(False)
    return floyd(succ.get, nodes[0])


# ---------------- halting guardrails ----------------
CYCLE, MAX_STEPS, ERRORS, TIME = "SHIP_ANOMALY_06_CYCLE", "MAX_STEPS", "CONSECUTIVE_ERRORS", "TIME_BUDGET"


@dataclass
class Verdict:
    halt: bool
    reason: Optional[str] = None
    detail: str = ""


class HaltingGuard:
    """Call observe() after every tool execution. It halts the loop on: a confirmed cycle (seen `min_cycle_repeats` times),
    too many steps, too many consecutive tool errors, or an exhausted time budget."""

    def __init__(self, max_steps=40, max_consecutive_errors=4, min_cycle_repeats=3, max_seconds=None, enable_cycle=True):
        """min_cycle_repeats: how many times the looping block must occur before the guard halts. A trace that merely LOOKS
        deterministic can come from a sampling policy, where one or two repeats prove nothing, so the default is 3.
        With a known-deterministic policy and environment, 2 is sound and halts one step earlier."""
        self.max_steps, self.max_err, self.reps = max_steps, max_consecutive_errors, min_cycle_repeats
        self.max_seconds, self.enable_cycle = max_seconds, enable_cycle
        self.reset()

    def reset(self):
        self.nodes, self.steps, self.err_run, self.t0 = [], 0, 0, time.monotonic()

    def observe(self, state_key, action_key, ok: bool) -> Verdict:
        self.steps += 1
        self.nodes.append((state_key, action_key))
        self.err_run = 0 if ok else self.err_run + 1
        if self.enable_cycle:
            info = detect_in_trace(self.nodes, self.reps)
            if info.has_cycle and len(self.nodes) >= info.mu + info.lam * self.reps:
                return Verdict(True, CYCLE, f"cycle start={info.mu}, length={info.lam}, method={info.method}, after {self.steps} steps")
        if self.err_run >= self.max_err:
            return Verdict(True, ERRORS, f"{self.err_run} consecutive tool errors")
        if self.steps >= self.max_steps:
            return Verdict(True, MAX_STEPS, f"step budget {self.max_steps} reached")
        if self.max_seconds is not None and time.monotonic() - self.t0 > self.max_seconds:
            return Verdict(True, TIME, f"time budget {self.max_seconds}s exceeded")
        return Verdict(False)
