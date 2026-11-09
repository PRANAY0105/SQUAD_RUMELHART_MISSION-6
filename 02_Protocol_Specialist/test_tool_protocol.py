import unittest
from tool_protocol import (ToolRegistry, ToolSpec, Dispatcher, validate_args, PARSE_ERROR, INVALID_REQUEST,
                           METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR)

PARAMS = {"factor": {"type": "number", "min": 0, "max": 9.9},
          "system": {"type": "string", "enum": ["warp_core", "nacelles"]},
          "note": {"type": "string", "required": False, "max_length": 5},
          "count": {"type": "integer", "required": False}}


def make():
    r = ToolRegistry()
    r.register(ToolSpec("set", "set factor", {"factor": PARAMS["factor"]}, lambda factor: {"factor": factor}))
    r.register(ToolSpec("diag", "diagnose", {"system": PARAMS["system"]}, lambda system: "PASS"))
    r.register(ToolSpec("boom", "always fails", {}, lambda: 1 / 0))
    r.register(ToolSpec("big", "long output", {}, lambda: "x" * 5000))
    return Dispatcher(r, max_output_chars=100)


class TestValidator(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_args(PARAMS, {"factor": 5, "system": "nacelles"}), [])

    def test_missing_required_and_unexpected(self):
        e = validate_args(PARAMS, {"extra": 1})
        self.assertTrue(any("missing required argument 'factor'" in x for x in e))
        self.assertTrue(any("unexpected argument 'extra'" in x for x in e))

    def test_type_range_enum_length(self):
        self.assertTrue(validate_args(PARAMS, {"factor": "fast", "system": "warp_core"}))
        self.assertTrue(validate_args(PARAMS, {"factor": 12, "system": "warp_core"}))
        self.assertTrue(validate_args(PARAMS, {"factor": 1, "system": "engine"}))
        self.assertTrue(validate_args(PARAMS, {"factor": 1, "system": "warp_core", "note": "toolong"}))

    def test_bool_is_not_a_number_or_integer(self):
        self.assertTrue(validate_args(PARAMS, {"factor": True, "system": "warp_core"}))
        self.assertTrue(validate_args(PARAMS, {"factor": 1, "system": "warp_core", "count": True}))

    def test_integer_rejects_float(self):
        self.assertTrue(validate_args(PARAMS, {"factor": 1, "system": "warp_core", "count": 2.5}))

    def test_arguments_must_be_object(self):
        self.assertTrue(validate_args(PARAMS, [1, 2]))


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.d = make()

    def test_success(self):
        r = self.d.dispatch({"tool": "set", "arguments": {"factor": 6}})
        self.assertTrue(r.ok); self.assertEqual(r.output, {"factor": 6})

    def test_unknown_tool_code(self):
        r = self.d.dispatch({"tool": "hyperdrive", "arguments": {}})
        self.assertEqual(r.error["code"], METHOD_NOT_FOUND)

    def test_invalid_params_code(self):
        r = self.d.dispatch({"tool": "set", "arguments": {"factor": 99}})
        self.assertEqual(r.error["code"], INVALID_PARAMS)

    def test_tool_exception_is_contained(self):
        r = self.d.dispatch({"tool": "boom"})            # must not raise
        self.assertEqual(r.error["code"], INTERNAL_ERROR); self.assertIn("ZeroDivisionError", r.error["message"])

    def test_invalid_request_shapes(self):
        for bad in (None, [], {"arguments": {}}, {"tool": 5}):
            self.assertEqual(self.d.dispatch(bad).error["code"], INVALID_REQUEST)

    def test_parse_error(self):
        self.assertEqual(self.d.dispatch_json("{not json").error["code"], PARSE_ERROR)
        self.assertTrue(self.d.dispatch_json('{"tool": "diag", "arguments": {"system": "warp_core"}}').ok)

    def test_output_truncation(self):
        r = self.d.dispatch({"tool": "big"})
        self.assertTrue(r.ok and r.truncated and len(r.output) == 100)

    def test_stats_count_each_outcome(self):
        self.d.dispatch({"tool": "set", "arguments": {"factor": 1}})
        self.d.dispatch({"tool": "nope"}); self.d.dispatch({"tool": "boom"})
        self.assertEqual((self.d.stats["ok"], self.d.stats[METHOD_NOT_FOUND], self.d.stats[INTERNAL_ERROR]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
