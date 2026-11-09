"""Tool Protocol: schema validator and error-boundary dispatcher (standard library only).  Station: Protocol Specialist.

Call envelope:   {"tool": "<name>", "arguments": {...}}        Result: ToolResult(ok, output | error{code, message, data})
Error codes follow JSON-RPC 2.0:  -32700 parse error, -32600 invalid request, -32601 unknown tool,
                                  -32602 invalid arguments, -32603 tool raised an exception.
The dispatcher is an ERROR BOUNDARY: dispatch() never raises for ordinary failures; every failure comes back as a
structured ToolResult so the agent loop can treat it as an observation and recover."""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603
TYPE_NAMES = ("string", "integer", "number", "boolean")


def _type_ok(value, tname):
    if tname == "string":
        return isinstance(value, str)
    if tname == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):                       # bool is an int subclass in Python: reject it for numbers
        return False
    return isinstance(value, int) if tname == "integer" else isinstance(value, (int, float))


def validate_args(params: Dict[str, dict], args: Any) -> List[str]:
    """params: {name: {type, required?, enum?, min?, max?, max_length?}}. Returns a list of error strings (empty = valid)."""
    if not isinstance(args, dict):
        return [f"arguments must be an object, got {type(args).__name__}"]
    errors = []
    for k in args:
        if k not in params:
            errors.append(f"unexpected argument '{k}'")
    for name, rule in params.items():
        if name not in args:
            if rule.get("required", True):
                errors.append(f"missing required argument '{name}'")
            continue
        v, t = args[name], rule["type"]
        if t not in TYPE_NAMES:
            errors.append(f"schema error: unknown type '{t}' for '{name}'")
            continue
        if not _type_ok(v, t):
            errors.append(f"'{name}' must be {t}, got {type(v).__name__}")
            continue
        if "enum" in rule and v not in rule["enum"]:
            errors.append(f"'{name}' must be one of {rule['enum']}, got {v!r}")
        if "min" in rule and v < rule["min"]:
            errors.append(f"'{name}' must be >= {rule['min']}, got {v}")
        if "max" in rule and v > rule["max"]:
            errors.append(f"'{name}' must be <= {rule['max']}, got {v}")
        if "max_length" in rule and len(v) > rule["max_length"]:
            errors.append(f"'{name}' is longer than {rule['max_length']} characters")
    return errors


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, dict]
    func: Callable[..., Any]


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: Optional[dict] = None
    truncated: bool = False

    def to_dict(self):
        d = {"ok": self.ok}
        d.update({"output": self.output} if self.ok else {"error": self.error})
        if self.truncated:
            d["truncated"] = True
        return d


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' already registered")
        self._tools[spec.name] = spec
        return spec

    def get(self, name):
        return self._tools.get(name)

    def names(self):
        return sorted(self._tools)

    def describe(self):
        return [{"name": s.name, "description": s.description, "parameters": s.params} for s in self._tools.values()]


class Dispatcher:
    def __init__(self, registry: ToolRegistry, max_output_chars=2000):
        self.registry, self.max_output_chars = registry, max_output_chars
        self.stats: Dict[Any, int] = {"ok": 0}

    def _fail(self, code, message, data=None):
        self.stats[code] = self.stats.get(code, 0) + 1
        return ToolResult(False, error={"code": code, "message": message, **({"data": data} if data else {})})

    def dispatch(self, call: Any) -> ToolResult:
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
            return self._fail(INVALID_REQUEST, "call must be an object with a string field 'tool'")
        spec = self.registry.get(call["tool"])
        if spec is None:
            return self._fail(METHOD_NOT_FOUND, f"unknown tool '{call['tool']}'", {"available": self.registry.names()})
        args = call.get("arguments", {})
        errors = validate_args(spec.params, args)
        if errors:
            return self._fail(INVALID_PARAMS, "; ".join(errors), {"tool": spec.name})
        try:
            output = spec.func(**args)
        except Exception as exc:                       # ERROR BOUNDARY: a tool bug or refusal must not crash the loop
            return self._fail(INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", {"tool": spec.name})
        self.stats["ok"] += 1
        text = json.dumps(output, default=str) if not isinstance(output, str) else output
        if len(text) > self.max_output_chars:
            return ToolResult(True, text[: self.max_output_chars], truncated=True)
        return ToolResult(True, output)

    def dispatch_json(self, text: str) -> ToolResult:
        try:
            call = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            return self._fail(PARSE_ERROR, f"invalid JSON: {exc}")
        return self.dispatch(call)
