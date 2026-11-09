"""Simulated warp-control environment and its tool registry (standard library only).  Station: Systems Architect.
The ship is deterministic, so the same action in the same state always gives the same result (needed for Floyd cycle detection)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_Protocol_Specialist"))
from tool_protocol import ToolRegistry, ToolSpec   # noqa: E402


class ShipSim:
    SYSTEMS = ("warp_core", "nacelles", "shields")

    def __init__(self):
        self.reset()

    def reset(self):
        self.factor, self.engaged = 0.0, False
        self.diag = {s: False for s in self.SYSTEMS}

    def state_key(self):
        d = "".join(str(int(self.diag[s])) for s in self.SYSTEMS)
        return f"f={self.factor:.1f}|diag={d}|eng={int(self.engaged)}"

    def goal_reached(self):
        return self.engaged and self.factor >= 5.0

    # ---- tools ----
    def get_warp_status(self):
        return {"warp_factor": self.factor, "engaged": self.engaged, "diagnostics_passed": [s for s, ok in self.diag.items() if ok]}

    def run_diagnostic(self, system):
        self.diag[system] = True
        return f"{system}: PASS"

    def set_warp_factor(self, factor):
        self.factor = round(float(factor), 1)
        return {"warp_factor": self.factor}

    def engage_warp(self):
        if not (self.diag["warp_core"] and self.diag["nacelles"]):
            raise RuntimeError("engage refused: warp_core and nacelles diagnostics are incomplete")
        if self.factor <= 0:
            raise RuntimeError("engage refused: warp factor is not set")
        self.engaged = True
        return {"engaged": True, "warp_factor": self.factor}

    def disengage_warp(self):
        self.engaged = False
        return {"engaged": False}


def build_registry(sim: ShipSim) -> ToolRegistry:
    r = ToolRegistry()
    r.register(ToolSpec("get_warp_status", "Read the current warp status.", {}, sim.get_warp_status))
    r.register(ToolSpec("run_diagnostic", "Run a diagnostic on one ship system.",
                        {"system": {"type": "string", "enum": list(ShipSim.SYSTEMS)}}, sim.run_diagnostic))
    r.register(ToolSpec("set_warp_factor", "Set the warp factor (0 to 9.9).",
                        {"factor": {"type": "number", "min": 0, "max": 9.9}}, sim.set_warp_factor))
    r.register(ToolSpec("engage_warp", "Engage warp drive (needs warp_core and nacelles diagnostics and a factor above 0).", {}, sim.engage_warp))
    r.register(ToolSpec("disengage_warp", "Disengage warp drive.", {}, sim.disengage_warp))
    return r
