"""Ship Anomaly 06 demo: the same runaway loop with and without the halting guard, plus Floyd versus a visited-set method."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "04_Systems_Architect"))
from run_mission import run_one                    # noqa: E402
from policies import LoopPolicy                    # noqa: E402
from cycle_detector import floyd                   # noqa: E402

print("Anomaly 06: AURA repeats run_diagnostic(warp_core) forever without progress.\n")
for label, guard in (("guard OFF (stops only at hard cap 200)", False), ("guard ON", True)):
    res, lg = run_one(LoopPolicy(), guard=guard, hard_cap=200)
    print(f"{label:<40} tool calls={res.tool_calls:>3}  outcome={res.outcome}  reason={res.halt_reason}  total reward={res.total_reward:.0f}")
res, lg = run_one(LoopPolicy())
print("Guard verdict:", lg.episodes[0][-1].info["detail"])

print("\nFloyd (2 pointers, O(1) memory) vs visited-set (O(n) memory) on a tail of 1,000,000 nodes + cycle of 1,000:")
T, C = 1_000_000, 1_000
f = lambda x: x + 1 if x < T + C - 1 else T
info = floyd(f, 0, max_steps=10_000_000)
print(f"  Floyd found: cycle start mu={info.mu}, length lambda={info.lam}  (expected {T} and {C})")
seen, x = set(), 0
while x not in seen:
    seen.add(x); x = f(x)
print(f"  visited-set method stored {len(seen):,} nodes to find the same cycle; Floyd stored 2 pointers.")
