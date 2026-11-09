"""MDP trajectory logger and empirical discounted-return evaluator (standard library only).  Station: Data Specialist.
An episode is a trajectory of transitions (s_t, a_t, r_t, s_{t+1}, done). The discounted return from step t is
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...  = r_t + gamma * G_{t+1},     G_T = 0
Evaluation here is EMPIRICAL (Monte Carlo): average G_0 over logged episodes, and first-visit state values V(s) = mean G_t
over the first visit of s in each episode."""
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Transition:
    step: int
    state: str
    action: str
    reward: float
    next_state: str
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    def __init__(self):
        self.episodes: List[List[Transition]] = []
        self._current: List[Transition] = []

    def start_episode(self):
        if self._current:
            self.end_episode()
        self._current = []

    def log(self, state, action, reward, next_state, done=False, **info):
        self._current.append(Transition(len(self._current), str(state), str(action), float(reward), str(next_state), bool(done), info))

    def end_episode(self):
        if self._current:
            self._current[-1].done = True
            self.episodes.append(self._current)
        self._current = []

    # persistence: one JSON object per transition, with an episode index
    def save_jsonl(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for ei, ep in enumerate(self.episodes):
                for t in ep:
                    f.write(json.dumps({"episode": ei, **asdict(t)}) + "\n")

    @classmethod
    def load_jsonl(cls, path):
        lg, eps = cls(), {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            ei = d.pop("episode")
            eps.setdefault(ei, []).append(Transition(**d))
        lg.episodes = [eps[k] for k in sorted(eps)]
        return lg


def discounted_return(rewards, gamma):
    """G_0 = sum_t gamma^t r_t (Horner form, no powers)."""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def returns_to_go(rewards, gamma):
    """[G_0, G_1, ..., G_{T-1}] computed backwards with G_t = r_t + gamma G_{t+1}."""
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def evaluate(episodes, gamma):
    """Empirical statistics of G_0 over episodes."""
    rets = [discounted_return([t.reward for t in ep], gamma) for ep in episodes]
    n = len(rets)
    if n == 0:
        return {"episodes": 0}
    mean = sum(rets) / n
    var = sum((x - mean) ** 2 for x in rets) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return {"episodes": n, "gamma": gamma, "mean_return": mean, "std": sd, "stderr": sd / math.sqrt(n) if n > 1 else 0.0,
            "min": min(rets), "max": max(rets), "mean_length": sum(len(ep) for ep in episodes) / n}


def gamma_sweep(episodes, gammas=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)):
    return [evaluate(episodes, g) for g in gammas]


def mc_state_values(episodes, gamma):
    """First-visit Monte Carlo estimate V(s) = mean of G_t at the first visit of s. Returns {state: (value, visits)}."""
    sums, counts = {}, {}
    for ep in episodes:
        rtg, seen = returns_to_go([t.reward for t in ep], gamma), set()
        for t, g in zip(ep, rtg):
            if t.state not in seen:
                seen.add(t.state)
                sums[t.state] = sums.get(t.state, 0.0) + g
                counts[t.state] = counts.get(t.state, 0) + 1
    return {s: (sums[s] / counts[s], counts[s]) for s in sums}
