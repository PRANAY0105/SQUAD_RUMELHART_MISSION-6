"""Hand-checkable demonstration of discounted returns and Monte Carlo evaluation (no ship needed)."""
from mdp_logger import TrajectoryLogger, discounted_return, returns_to_go, evaluate, gamma_sweep, mc_state_values

rewards = [-1, -1, 10]
print("Rewards:", rewards)
for g in (0.0, 0.5, 0.9, 1.0):
    print(f"  gamma={g}: G0 = {discounted_return(rewards, g):.3f}")
print("\nBy hand for gamma=0.9: -1 + 0.9*(-1) + 0.81*10 = 6.2   | returns-to-go:", [round(x, 3) for x in returns_to_go(rewards, 0.9)])

lg = TrajectoryLogger()
for rs in ([-1, -1, 10], [-1, 10], [-1, -1, -1, -1, 10]):          # three episodes of different length
    lg.start_episode()
    for i, r in enumerate(rs):
        lg.log(f"s{i}", f"a{i}", r, f"s{i + 1}")
    lg.end_episode()
print("\nThree episodes, gamma=0.9:", {k: round(v, 3) if isinstance(v, float) else v for k, v in evaluate(lg.episodes, 0.9).items()})
print("Sweep over gamma (mean G0):", [(e["gamma"], round(e["mean_return"], 2)) for e in gamma_sweep(lg.episodes)])
print("First-visit V(s0), gamma=0.9:", {s: (round(v, 3), n) for s, (v, n) in mc_state_values(lg.episodes, 0.9).items() if s == "s0"})
print("\nReading: a small gamma makes the agent short-sighted (the +10 goal barely counts); gamma near 1 makes long episodes pay for every step cost.")
