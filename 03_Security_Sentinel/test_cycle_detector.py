import random
import unittest
from cycle_detector import floyd, detect_in_trace, tail_period, HaltingGuard, CYCLE, MAX_STEPS, ERRORS, TIME


def brute(f, x0, limit=10_000):
    seen, x, i = {}, x0, 0
    while x is not None and i < limit:
        if x in seen:
            return True, seen[x], i - seen[x]
        seen[x] = i
        x, i = f(x), i + 1
    return False, -1, 0


class TestFloyd(unittest.TestCase):
    def test_textbook_tail_and_cycle(self):
        g = {0: 1, 1: 2, 2: 3, 3: 4, 4: 2}               # tail 0,1 then cycle 2,3,4
        r = floyd(g.get, 0)
        self.assertEqual((r.has_cycle, r.mu, r.lam), (True, 2, 3))

    def test_self_loop(self):
        r = floyd({"a": "a"}.get, "a")
        self.assertEqual((r.has_cycle, r.mu, r.lam), (True, 0, 1))

    def test_no_cycle(self):
        self.assertFalse(floyd({0: 1, 1: 2}.get, 0).has_cycle)

    def test_single_node_end(self):
        self.assertFalse(floyd({}.get, 0).has_cycle)

    def test_pure_cycle_from_start(self):
        r = floyd({0: 1, 1: 2, 2: 0}.get, 0)
        self.assertEqual((r.mu, r.lam), (0, 3))

    def test_matches_brute_force_on_random_functions(self):
        rng = random.Random(7)
        for _ in range(500):
            n = rng.randint(2, 40)
            g = {i: (rng.randrange(n) if rng.random() > 0.15 else None) for i in range(n)}
            x0 = rng.randrange(n)
            has, mu, lam = brute(g.get, x0)
            r = floyd(g.get, x0)
            self.assertEqual((r.has_cycle, r.mu if has else -1, r.lam if has else 0), (has, mu, lam))


class TestTrace(unittest.TestCase):
    def test_deterministic_trace_uses_floyd(self):
        r = detect_in_trace(["a", "b", "c", "b", "c", "b"])
        self.assertEqual((r.has_cycle, r.mu, r.lam, r.method), (True, 1, 2, "floyd"))

    def test_nondeterministic_trace_falls_back_to_tail_period(self):
        r = detect_in_trace(["a", "b", "a", "c", "x", "y", "x", "y"])   # 'a' has two successors
        self.assertEqual((r.has_cycle, r.method), (True, "tail-period"))

    def test_nondeterministic_without_repetition(self):
        self.assertFalse(detect_in_trace(["a", "b", "a", "c", "d"]).has_cycle)

    def test_tail_period_needs_full_repeats(self):
        self.assertFalse(tail_period(["x", "y", "x"], 2).has_cycle)


class TestGuard(unittest.TestCase):
    def test_cycle_halts_when_seen_three_times_by_default(self):
        g = HaltingGuard(max_steps=100)
        v = [g.observe("s0", "diag", True)] + [g.observe("s1", "diag", True) for _ in range(3)]
        self.assertEqual([x.halt for x in v], [False, False, False, True])
        self.assertEqual(v[3].reason, CYCLE)

    def test_two_repeats_suffice_when_configured_for_deterministic_policies(self):
        g = HaltingGuard(max_steps=100, min_cycle_repeats=2)
        v = [g.observe("s0", "diag", True), g.observe("s1", "diag", True), g.observe("s1", "diag", True)]
        self.assertTrue(v[2].halt)

    def test_two_identical_steps_are_not_a_loop_by_default(self):
        g = HaltingGuard(max_steps=100)
        g.observe("s0", "a", True)
        self.assertFalse(g.observe("s1", "x", True).halt)
        self.assertFalse(g.observe("s1", "x", True).halt)

    def test_progress_is_not_flagged(self):
        g = HaltingGuard()
        for i in range(10):
            self.assertFalse(g.observe(f"s{i}", "act", True).halt)

    def test_max_steps(self):
        g = HaltingGuard(max_steps=5, enable_cycle=False)
        verdicts = [g.observe(f"s{i}", "a", True) for i in range(5)]
        self.assertEqual(verdicts[-1].reason, MAX_STEPS)

    def test_consecutive_errors(self):
        g = HaltingGuard(max_consecutive_errors=3, enable_cycle=False)
        out = [g.observe(f"s{i}", "a", False) for i in range(3)]
        self.assertEqual(out[-1].reason, ERRORS)

    def test_success_resets_error_run(self):
        g = HaltingGuard(max_consecutive_errors=3, enable_cycle=False)
        for ok in (False, False, True, False, False):
            v = g.observe("s", "a", ok)
        self.assertFalse(v.halt)

    def test_time_budget(self):
        g = HaltingGuard(max_seconds=-1, enable_cycle=False)
        self.assertEqual(g.observe("s", "a", True).reason, TIME)


if __name__ == "__main__":
    unittest.main()
