import unittest
from agent_state_machine import AgentStateMachine, State, InvalidTransition, TRANSITIONS, TERMINAL, EVENTS


class TestStateMachine(unittest.TestCase):
    def test_happy_path_with_one_tool(self):
        m = AgentStateMachine()
        for e in ("start", "decide_tool", "tool_ok", "continue", "decide_final"):
            m.fire(e)
        self.assertEqual(m.state, State.DONE)
        self.assertEqual(len(m.history), 5)

    def test_tool_error_returns_to_observation_not_failure(self):
        m = AgentStateMachine()
        for e in ("start", "decide_tool", "tool_error"):
            m.fire(e)
        self.assertEqual(m.state, State.OBSERVING)

    def test_illegal_event_raises_and_state_is_unchanged(self):
        m = AgentStateMachine()
        with self.assertRaises(InvalidTransition):
            m.fire("decide_tool")                    # cannot act before starting
        self.assertEqual(m.state, State.IDLE)
        self.assertEqual(m.history, [])

    def test_terminal_states_only_accept_reset(self):
        for s in TERMINAL:
            m = AgentStateMachine(); m.state = s
            self.assertEqual(m.available_events(), ["reset"])

    def test_halt_and_fail_from_every_non_terminal_state(self):
        for s in State:
            if s in TERMINAL:
                continue
            for ev, target in (("halt", State.HALTED), ("fail", State.FAILED)):
                m = AgentStateMachine(); m.state = s
                self.assertEqual(m.fire(ev), target)

    def test_exhaustive_every_state_event_pair(self):
        for s in State:
            for e in EVENTS:
                m = AgentStateMachine(); m.state = s
                if (s, e) in TRANSITIONS:
                    self.assertEqual(m.fire(e), TRANSITIONS[(s, e)])
                else:
                    with self.assertRaises(InvalidTransition):
                        m.fire(e)

    def test_callback_and_reset(self):
        seen = []
        m = AgentStateMachine(on_transition=lambda a, e, b, i: seen.append((a, e, b)))
        m.fire("start"); m.fire("halt"); m.fire("reset")
        self.assertEqual(len(seen), 3)
        self.assertEqual(m.state, State.IDLE)

    def test_dot_export_mentions_main_edges(self):
        d = AgentStateMachine.to_dot()
        self.assertIn("DELIBERATING -> ACTING", d)
        self.assertIn("OBSERVING -> DELIBERATING", d)


if __name__ == "__main__":
    unittest.main()
