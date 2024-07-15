import numbers
import os
import unittest

import modules.flags
from modules import extra_utils


class TestUtils(unittest.TestCase):
    def test_try_eval_env_var(self):
        test_cases = [
            {
                "input": ("foo", str),
                "output": "foo"
            },
            {
                "input": ("1", int),
                "output": 1
            },
            {
                "input": ("1.0", float),
                "output": 1.0
            },
            {
                "input": ("1", numbers.Number),
                "output": 1
            },
            {
                "input": ("1.0", numbers.Number),
                "output": 1.0
            },
            {
                "input": ("true", bool),
                "output": True
            },
            {
                "input": ("True", bool),
                "output": True
            },
            {
                "input": ("false", bool),
                "output": False
            },
            {
                "input": ("False", bool),
                "output": False
            },
            {
                "input": ("True", str),
                "output": "True"
            },
            {
                "input": ("False", str),
                "output": "False"
            },
            {
                "input": ("['a', 'b', 'c']", list),
                "output": ['a', 'b', 'c']
            },
            {
                "input": ("{'a':1}", dict),
                "output": {'a': 1}
            },
            {
                "input": ("('foo', 1)", tuple),
                "output": ('foo', 1)
            }
        ]

        for test in test_cases:
            value, expected_type = test["input"]
            expected = test["output"]
            actual = extra_utils.try_eval_env_var(value, expected_type)
            self.assertEqual(expected, actual)
