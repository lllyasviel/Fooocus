import unittest
from modules import util


class TestUtils(unittest.TestCase):
    def test_can_parse_tokens_with_lora(self):

        test_cases = [
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4>,  cool   <lora:you-lora:0.2>", [], 5),
                "output": [("hey-lora.safetensors", 0.4), ("you-lora.safetensors", 0.2)],
            },
            # Test can not exceed limit
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4>,  cool   <lora:you-lora:0.2>", [], 1),
                "output": [("hey-lora.safetensors", 0.4)],
            },
            # test Loras from UI take precedence over prompt
            {
                "input": (
                    "some prompt, very cool, <lora:l1:0.4>, <lora:l2:-0.2>, <lora:l3:0.3>, <lora:l4:0.5>, <lora:l6:0.24>, <lora:l7:0.1>",
                    [("hey-lora.safetensors", 0.4)],
                    5,
                ),
                "output": [
                    ("hey-lora.safetensors", 0.4),
                    ("l1.safetensors", 0.4),
                    ("l2.safetensors", -0.2),
                    ("l3.safetensors", 0.3),
                    ("l4.safetensors", 0.5),
                ],
            },
            # Test lora specification not separated by comma are ignored, only latest specified is used
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4><lora:you-lora:0.2>", [], 3),
                "output": [("you-lora.safetensors", 0.2)],
            },
            {
                 "input": ("<lora:foo:1..2>, <lora:bar:.>, <lora:baz:+> and <lora:quux:>",[], 6),
                 "output": []
            }
        ]

        for test in test_cases:
            promp, loras, loras_limit = test["input"]
            expected = test["output"]
            actual = util.parse_lora_references_from_prompt(promp, loras, loras_limit)
            self.assertEqual(expected, actual)
