"""AgentStateMachine: the explicit control core of the deliberation loop (native Python, standard library only).
Station: Systems Architect.

States:  IDLE -> DELIBERATING -> ACTING -> OBSERVING -> DELIBERATING ...   terminal: DONE, FAILED, HALTED
Events:  start, decide_tool, decide_final, tool_ok, tool_error, continue, halt, fail, reset
Every legal (state, event) pair lives in one TRANSITIONS table; anything else raises InvalidTransition and leaves the
machine unchanged. Terminal states accept only 'reset'. 'halt' and 'fail' are accepted from every non-terminal state."""
from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    DELIBERATING = "DELIBERATING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    DONE = "DONE"
    FAILED = "FAILED"
    HALTED = "HALTED"


TERMINAL = {State.DONE, State.FAILED, State.HALTED}


class InvalidTransition(Exception):
    pass


def _build_table():
    t = {
        (State.IDLE, "start"): State.DELIBERATING,
        (State.DELIBERATING, "decide_tool"): State.ACTING,
        (State.DELIBERATING, "decide_final"): State.DONE,
        (State.ACTING, "tool_ok"): State.OBSERVING,
        (State.ACTING, "tool_error"): State.OBSERVING,      # error boundary: failures come back as observations
        (State.OBSERVING, "continue"): State.DELIBERATING,
    }
    for s in State:
        if s in TERMINAL:
            t[(s, "reset")] = State.IDLE
        else:
            t[(s, "halt")] = State.HALTED
            t[(s, "fail")] = State.FAILED
    return t


TRANSITIONS = _build_table()
EVENTS = sorted({e for _, e in TRANSITIONS})


class AgentStateMachine:
    def __init__(self, on_transition=None):
        self.state = State.IDLE
        self.history = []                    # [(from_state, event, to_state)]
        self.on_transition = on_transition   # optional callback(from_state, event, to_state, info)

    def can_fire(self, event):
        return (self.state, event) in TRANSITIONS

    def available_events(self):
        return sorted(e for (s, e) in TRANSITIONS if s == self.state)

    def fire(self, event, **info):
        key = (self.state, event)
        if key not in TRANSITIONS:
            raise InvalidTransition(f"event '{event}' is not allowed in state {self.state.value}; allowed: {self.available_events()}")
        old, new = self.state, TRANSITIONS[key]
        self.state = new
        self.history.append((old, event, new))
        if self.on_transition:
            self.on_transition(old, event, new, info)
        return new

    @property
    def is_terminal(self):
        return self.state in TERMINAL

    def reset(self):
        self.state, self.history = State.IDLE, []

    @staticmethod
    def to_dot():
        lines = ["digraph AgentStateMachine {", "  rankdir=LR;", "  start [shape=point];", "  start -> IDLE;"]
        for (s, e), n in sorted(TRANSITIONS.items(), key=lambda kv: (kv[0][0].value, kv[0][1])):
            if e in ("halt", "fail", "reset"):
                continue
            lines.append(f'  {s.value} -> {n.value} [label="{e}"];')
        lines += ['  "any non-terminal" -> HALTED [label="halt", style=dashed];', '  "any non-terminal" -> FAILED [label="fail", style=dashed];',
                  '  "terminal" -> IDLE [label="reset", style=dashed];', "}"]
        return "\n".join(lines)
