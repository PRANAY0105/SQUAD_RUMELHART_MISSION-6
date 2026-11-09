import math
import os
import tempfile
import unittest
from mdp_logger import TrajectoryLogger, discounted_return, returns_to_go, evaluate, gamma_sweep, mc_state_values


def episode(lg, rewards, states=None):
    lg.start_episode()
    states = states or [f"s{i}" for i in range(len(rewards) + 1)]
    for i, r in enumerate(rewards):
        lg.log(states[i], f"a{i}", r, states[i + 1])
    lg.end_episode()


class TestReturns(unittest.TestCase):
    def test_hand_computed_return(self):
        self.assertAlmostEqual(discounted_return([-1, -1, 10], 0.9), -1 - 0.9 + 0.81 * 10)      # 6.2

    def test_gamma_zero_is_first_reward_and_gamma_one_is_sum(self):
        self.assertEqual(discounted_return([3, 4, 5], 0.0), 3)
        self.assertEqual(discounted_return([3, 4, 5], 1.0), 12)

    def test_empty(self):
        self.assertEqual(discounted_return([], 0.9), 0.0)

    def test_returns_to_go_recurrence(self):
        r, g = [1, 2, 3, 4], 0.8
        rtg = returns_to_go(r, g)
        self.assertAlmostEqual(rtg[0], discounted_return(r, g))
        for t in range(len(r) - 1):
            self.assertAlmostEqual(rtg[t], r[t] + g * rtg[t + 1])
        self.assertEqual(rtg[-1], r[-1])

    def test_matches_explicit_power_sum(self):
        r, g = [2, -1, 0.5, 7], 0.95
        self.assertAlmostEqual(discounted_return(r, g), sum(x * g ** i for i, x in enumerate(r)))


class TestLoggerAndEval(unittest.TestCase):
    def test_done_flag_set_on_last_transition(self):
        lg = TrajectoryLogger(); episode(lg, [1, 2, 3])
        self.assertEqual([t.done for t in lg.episodes[0]], [False, False, True])

    def test_evaluate_statistics(self):
        lg = TrajectoryLogger(); episode(lg, [1]); episode(lg, [3])
        e = evaluate(lg.episodes, 0.9)
        self.assertEqual((e["episodes"], e["mean_return"], e["min"], e["max"]), (2, 2.0, 1.0, 3.0))
        self.assertAlmostEqual(e["std"], math.sqrt(2))
        self.assertAlmostEqual(e["stderr"], 1.0)

    def test_evaluate_no_episodes(self):
        self.assertEqual(evaluate([], 0.9), {"episodes": 0})

    def test_gamma_sweep_length(self):
        lg = TrajectoryLogger(); episode(lg, [1, 1])
        self.assertEqual(len(gamma_sweep(lg.episodes)), 6)

    def test_first_visit_values(self):
        lg = TrajectoryLogger()
        episode(lg, [1, 1], states=["A", "B", "A"])       # A visited twice: only the first visit counts
        v = mc_state_values(lg.episodes, 1.0)
        self.assertEqual(v["A"], (2.0, 1)); self.assertEqual(v["B"], (1.0, 1))

    def test_jsonl_round_trip(self):
        lg = TrajectoryLogger(); episode(lg, [1, 2]); episode(lg, [5])
        p = os.path.join(tempfile.mkdtemp(), "t.jsonl"); lg.save_jsonl(p)
        lg2 = TrajectoryLogger.load_jsonl(p)
        self.assertEqual([[t.reward for t in ep] for ep in lg2.episodes], [[1.0, 2.0], [5.0]])


if __name__ == "__main__":
    unittest.main()
