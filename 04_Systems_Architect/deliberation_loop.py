"""The deliberation loop: observe -> decide -> act -> observe, controlled by AgentStateMachine, with every tool call going
through the Protocol Specialist's dispatcher, every step logged by the Data Specialist's logger, and every step checked by the
Security Sentinel's halting guard.   Station: Systems Architect."""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
for d in ("01_Data_Specialist", "02_Protocol_Specialist", "03_Security_Sentinel", "04_Systems_Architect"):
    sys.path.insert(0, str(ROOT / d))
from agent_state_machine import AgentStateMachine           # noqa: E402
from tool_protocol import Dispatcher, METHOD_NOT_FOUND, INVALID_PARAMS, INVALID_REQUEST, PARSE_ERROR   # noqa: E402
from mdp_logger import TrajectoryLogger                      # noqa: E402
from cycle_detector import HaltingGuard                      # noqa: E402
from ship_sim import ShipSim, build_registry                 # noqa: E402

REWARD = {"step": -1.0, "progress": 2.0, "invalid_call": -5.0, "tool_exception": -3.0, "goal": 50.0, "false_claim": -10.0, "halt": -20.0}
INVALID_CODES = {METHOD_NOT_FOUND, INVALID_PARAMS, INVALID_REQUEST, PARSE_ERROR}


def action_key(call):
    try:
        return f"{call['tool']}({json.dumps(call.get('arguments', {}), sort_keys=True)})"
    except Exception:
        return "<malformed>"


@dataclass
class EpisodeResult:
    outcome: str                      # SUCCESS | FALSE_CLAIM | HALTED | FAILED
    halt_reason: Optional[str]
    tool_calls: int
    errors: int
    total_reward: float
    final_state: str
    machine_path: List[str] = field(default_factory=list)


class DeliberationLoop:
    def __init__(self, policy, sim=None, logger=None, guard=None, hard_cap=200):
        self.sim = sim or ShipSim()
        self.dispatcher = Dispatcher(build_registry(self.sim))
        self.policy, self.logger, self.guard, self.hard_cap = policy, logger, guard, hard_cap

    def _log(self, *a, **k):
        if self.logger:
            self.logger.log(*a, **k)

    def run(self) -> EpisodeResult:
        sim, sm, last = self.sim, AgentStateMachine(), None
        sim.reset()
        if self.guard:
            self.guard.reset()
        if self.logger:
            self.logger.start_episode()
        sm.fire("start")
        calls = errors = 0
        total, outcome, reason = 0.0, "FAILED", None
        for n in range(self.hard_cap):
            state = sim.state_key()
            try:
                decision = self.policy({"state": state, "last_result": last, "step": n})
                kind = decision["type"]
            except Exception as exc:                                  # a broken policy must not crash the loop
                sm.fire("fail")
                self._log(state, "<policy error>", -10.0, state, True, error=str(exc))
                total -= 10.0
                break
            if kind == "final":
                sm.fire("decide_final")
                ok = sim.goal_reached()
                r = REWARD["goal"] if ok else REWARD["false_claim"]
                total += r
                self._log(state, "FINAL", r, state, True, answer=decision.get("answer"))
                outcome = "SUCCESS" if ok else "FALSE_CLAIM"
                break
            sm.fire("decide_tool")
            call = decision.get("call")
            res = self.dispatcher.dispatch(call)
            calls += 1
            sm.fire("tool_ok" if res.ok else "tool_error")
            nxt = sim.state_key()
            r = REWARD["step"]
            if res.ok and nxt != state:
                r += REWARD["progress"]
            if not res.ok:
                errors += 1
                r += REWARD["invalid_call"] if res.error["code"] in INVALID_CODES else REWARD["tool_exception"]
            akey = action_key(call)
            verdict = self.guard.observe(state, akey, res.ok) if self.guard else None
            if verdict and verdict.halt:
                r += REWARD["halt"]
                total += r
                self._log(state, akey, r, nxt, True, ok=res.ok, halt=verdict.reason, detail=verdict.detail)
                sm.fire("halt")
                outcome, reason = "HALTED", verdict.reason
                break
            total += r
            self._log(state, akey, r, nxt, False, ok=res.ok)
            sm.fire("continue")
            last = res.to_dict()
        else:                                                         # hard cap reached without any guard verdict
            r = REWARD["halt"]
            total += r
            self._log(sim.state_key(), "<hard cap>", r, sim.state_key(), True)
            sm.fire("halt")
            outcome, reason = "HALTED", "HARD_CAP"
        if self.logger:
            self.logger.end_episode()
        return EpisodeResult(outcome, reason, calls, errors, total, sim.state_key(), [h[2].value for h in sm.history])
