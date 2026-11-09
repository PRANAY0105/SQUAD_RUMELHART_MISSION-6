"""End-to-end mission run: five scenarios plus a 200-episode random-policy evaluation (standard library only).
    python run_mission.py"""
import sys
from collections import Counter
from pathlib import Path

from deliberation_loop import DeliberationLoop, ROOT
from policies import ScriptedPolicy, LoopPolicy, FalseClaimPolicy, RandomPolicy, GOOD_SCRIPT, SLOPPY_SCRIPT
from cycle_detector import HaltingGuard
from mdp_logger import TrajectoryLogger, evaluate, gamma_sweep, mc_state_values, discounted_return

GAMMA = 0.95


def run_one(policy, guard=True, hard_cap=200, **gk):
    lg = TrajectoryLogger()
    res = DeliberationLoop(policy, logger=lg, guard=HaltingGuard(**gk) if guard else None, hard_cap=hard_cap).run()
    return res, lg


def main():
    print(f"{'scenario':<34} {'outcome':<11} {'halt reason':<24} {'calls':>5} {'errs':>4} {'reward':>7} {'G(0.95)':>8}")
    scenarios = [("A good plan", ScriptedPolicy(GOOD_SCRIPT), True), ("B sloppy plan, recovers", ScriptedPolicy(SLOPPY_SCRIPT), True),
                 ("C Ship Anomaly 06 loop (guard on)", LoopPolicy(), True), ("D same loop, guard OFF (cap 60)", LoopPolicy(), False),
                 ("E false success claim", FalseClaimPolicy(), True)]
    for name, pol, g in scenarios:
        res, lg = run_one(pol, guard=g, hard_cap=60)
        G = discounted_return([t.reward for t in lg.episodes[0]], GAMMA)
        print(f"{name:<34} {res.outcome:<11} {str(res.halt_reason):<24} {res.tool_calls:>5} {res.errors:>4} {res.total_reward:>7.1f} {G:>8.2f}")

    print("\nState path of scenario C (loop, guard on):")
    res, lg = run_one(LoopPolicy())
    for t in lg.episodes[0]:
        print(f"  step {t.step}: {t.state} --{t.action}--> {t.next_state}  reward={t.reward}  {t.info.get('detail', '')}")

    N = 200
    print(f"\nRandom policy, {N} episodes (seeds 0..{N - 1}), gamma={GAMMA}")
    out = {}
    for label, guard in (("guard ON  (max 40 steps)", True), ("guard OFF (hard cap 200)", False)):
        lg, results = TrajectoryLogger(), []
        for seed in range(N):
            res = DeliberationLoop(RandomPolicy(seed), logger=lg, guard=HaltingGuard(max_steps=40) if guard else None, hard_cap=200).run()
            results.append(res)
        e = evaluate(lg.episodes, GAMMA)
        oc = Counter(r.outcome for r in results)
        rs = Counter(r.halt_reason for r in results if r.halt_reason)
        out[label] = (lg, results)
        print(f"  {label}: success {oc['SUCCESS']}/{N}, false claim {oc['FALSE_CLAIM']}, halted {oc['HALTED']}; mean length {e['mean_length']:.1f}; "
              f"mean G0 = {e['mean_return']:.2f} +/- {e['stderr']:.2f} (sd {e['std']:.1f}); halt reasons {dict(rs)}")

    lg = out["guard ON  (max 40 steps)"][0]
    print("\nGamma sweep (random policy, guard ON):")
    print(f"  {'gamma':>6} {'mean G0':>9} {'stderr':>7}")
    for e in gamma_sweep(lg.episodes):
        print(f"  {e['gamma']:>6} {e['mean_return']:>9.2f} {e['stderr']:>7.2f}")
    vals = mc_state_values(lg.episodes, GAMMA)
    print("\nFirst-visit Monte Carlo values V(s) of the start state and the goal-adjacent states (gamma=0.95):")
    for s in ("f=0.0|diag=000|eng=0", "f=6.0|diag=110|eng=0"):
        if s in vals:
            print(f"  {s}: V = {vals[s][0]:.2f} from {vals[s][1]} first visits")
    sample = Path(ROOT / "01_Data_Specialist" / "sample_trajectories.jsonl")
    small = TrajectoryLogger(); small.episodes = lg.episodes[:5]
    small.save_jsonl(sample)
    print(f"\nSaved the first 5 random-policy episodes to {sample.name}")


if __name__ == "__main__":
    main()
