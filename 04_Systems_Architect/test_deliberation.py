import unittest
from agent_state_machine import State
from deliberation_loop import DeliberationLoop
from policies import ScriptedPolicy, LoopPolicy, FalseClaimPolicy, RandomPolicy, GOOD_SCRIPT, SLOPPY_SCRIPT
from ship_sim import ShipSim
from cycle_detector import HaltingGuard, CYCLE, ERRORS
from mdp_logger import TrajectoryLogger, discounted_return


def run(policy, guard=True, **kw):
    lg = TrajectoryLogger()
    res = DeliberationLoop(policy, logger=lg, guard=HaltingGuard(**kw) if guard else None).run()
    return res, lg


class TestLoop(unittest.TestCase):
    def test_good_policy_succeeds(self):
        res, lg = run(ScriptedPolicy(GOOD_SCRIPT))
        self.assertEqual((res.outcome, res.tool_calls, res.errors), ("SUCCESS", 4, 0))
        self.assertEqual(res.machine_path[-1], State.DONE.value)
        self.assertAlmostEqual(res.total_reward, 4 * (-1 + 2) + 50)          # four progress steps and the goal bonus

    def test_looping_policy_is_halted_by_floyd(self):
        res, lg = run(LoopPolicy())
        self.assertEqual((res.outcome, res.halt_reason, res.tool_calls), ("HALTED", CYCLE, 4))
        self.assertEqual(res.machine_path[-1], State.HALTED.value)

    def test_without_guard_the_loop_only_stops_at_the_hard_cap(self):
        lg = TrajectoryLogger()
        res = DeliberationLoop(LoopPolicy(), logger=lg, guard=None, hard_cap=60).run()
        self.assertEqual((res.outcome, res.halt_reason, res.tool_calls), ("HALTED", "HARD_CAP", 60))

    def test_sloppy_policy_recovers_from_three_errors(self):
        res, _ = run(ScriptedPolicy(SLOPPY_SCRIPT))
        self.assertEqual((res.outcome, res.errors), ("SUCCESS", 3))

    def test_false_claim_is_penalised(self):
        res, _ = run(FalseClaimPolicy())
        self.assertEqual((res.outcome, res.total_reward), ("FALSE_CLAIM", -10.0))

    def test_error_storm_halts(self):
        bad = {"type": "tool", "call": {"tool": "hyperdrive", "arguments": {}}}
        calls = [dict(bad, call={"tool": f"x{i}", "arguments": {}}) for i in range(10)]     # different calls: no cycle
        res, _ = run(ScriptedPolicy(calls), max_consecutive_errors=4)
        self.assertEqual((res.outcome, res.halt_reason), ("HALTED", ERRORS))

    def test_broken_policy_does_not_crash_loop(self):
        def broken(obs):
            raise ValueError("model crashed")
        res, _ = run(broken)
        self.assertEqual(res.outcome, "FAILED")

    def test_logger_records_return_consistent_with_total_reward(self):
        res, lg = run(ScriptedPolicy(GOOD_SCRIPT))
        rewards = [t.reward for t in lg.episodes[0]]
        self.assertAlmostEqual(sum(rewards), res.total_reward)
        self.assertAlmostEqual(discounted_return(rewards, 1.0), res.total_reward)

    def test_random_policy_is_reproducible(self):
        a = [run(RandomPolicy(seed=s))[0].total_reward for s in range(5)]
        b = [run(RandomPolicy(seed=s))[0].total_reward for s in range(5)]
        self.assertEqual(a, b)

    def test_ship_engage_preconditions(self):
        s = ShipSim()
        with self.assertRaises(RuntimeError):
            s.engage_warp()
        s.run_diagnostic("warp_core"); s.run_diagnostic("nacelles"); s.set_warp_factor(6)
        s.engage_warp()
        self.assertTrue(s.goal_reached())


if __name__ == "__main__":
    unittest.main()
