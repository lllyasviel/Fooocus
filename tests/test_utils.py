import os
import unittest

import modules.flags
from modules import util


class TestUtils(unittest.TestCase):
    def test_can_parse_tokens_with_lora(self):
        test_cases = [
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4>,  cool   <lora:you-lora:0.2>", [], 5, True),
                "output": (
                    [('hey-lora.safetensors', 0.4), ('you-lora.safetensors', 0.2)], 'some prompt, very cool, cool'),
            },
            # Test can not exceed limit
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4>,  cool   <lora:you-lora:0.2>", [], 1, True),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt, very cool, cool'
                ),
            },
            # test Loras from UI take precedence over prompt
            {
                "input": (
                    "some prompt, very cool, <lora:l1:0.4>, <lora:l2:-0.2>, <lora:l3:0.3>, <lora:l4:0.5>, <lora:l6:0.24>, <lora:l7:0.1>",
                    [("hey-lora.safetensors", 0.4)],
                    5,
                    True
                ),
                "output": (
                    [
                        ('hey-lora.safetensors', 0.4),
                        ('l1.safetensors', 0.4),
                        ('l2.safetensors', -0.2),
                        ('l3.safetensors', 0.3),
                        ('l4.safetensors', 0.5)
                    ],
                    'some prompt, very cool'
                )
            },
            # test correct matching even if there is no space separating loras in the same token
            {
                "input": ("some prompt, very cool, <lora:hey-lora:0.4><lora:you-lora:0.2>", [], 3, True),
                "output": (
                    [
                        ('hey-lora.safetensors', 0.4),
                        ('you-lora.safetensors', 0.2)
                    ],
                    'some prompt, very cool'
                ),
            },
            # test deduplication, also selected loras are never overridden with loras in prompt
            {
                "input": (
                    "some prompt, very cool, <lora:hey-lora:0.4><lora:hey-lora:0.4><lora:you-lora:0.2>",
                    [('you-lora.safetensors', 0.3)],
                    3,
                    True
                ),
                "output": (
                    [
                        ('you-lora.safetensors', 0.3),
                        ('hey-lora.safetensors', 0.4)
                    ],
                    'some prompt, very cool'
                ),
            },
            {
                "input": ("<lora:foo:1..2>, <lora:bar:.>, <test:1.0>, <lora:baz:+> and <lora:quux:>", [], 6, True),
                "output": (
                    [],
                    '<lora:foo:1..2>, <lora:bar:.>, <test:1.0>, <lora:baz:+> and <lora:quux:>'
                )
            }
        ]

        for test in test_cases:
            prompt, loras, loras_limit, skip_file_check = test["input"]
            expected = test["output"]
            actual = util.parse_lora_references_from_prompt(prompt, loras, loras_limit=loras_limit,
                                                            skip_file_check=skip_file_check)
            self.assertEqual(expected, actual)

    def test_can_parse_tokens_and_strip_performance_lora(self):
        lora_filenames = [
            'hey-lora.safetensors',
            modules.flags.PerformanceLoRA.EXTREME_SPEED.value,
            modules.flags.PerformanceLoRA.LIGHTNING.value,
            os.path.join('subfolder', modules.flags.PerformanceLoRA.HYPER_SD.value)
        ]

        test_cases = [
            {
                "input": ("some prompt, <lora:hey-lora:0.4>", [], 5, True, modules.flags.Performance.QUALITY),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt'
                ),
            },
            {
                "input": ("some prompt, <lora:hey-lora:0.4>", [], 5, True, modules.flags.Performance.SPEED),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt'
                ),
            },
            {
                "input": ("some prompt, <lora:sdxl_lcm_lora:1>, <lora:hey-lora:0.4>", [], 5, True, modules.flags.Performance.EXTREME_SPEED),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt'
                ),
            },
            {
                "input": ("some prompt, <lora:sdxl_lightning_4step_lora:1>, <lora:hey-lora:0.4>", [], 5, True, modules.flags.Performance.LIGHTNING),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt'
                ),
            },
            {
                "input": ("some prompt, <lora:sdxl_hyper_sd_4step_lora:1>, <lora:hey-lora:0.4>", [], 5, True, modules.flags.Performance.HYPER_SD),
                "output": (
                    [('hey-lora.safetensors', 0.4)],
                    'some prompt'
                ),
            }
        ]

        for test in test_cases:
            prompt, loras, loras_limit, skip_file_check, performance = test["input"]
            lora_filenames = modules.util.remove_performance_lora(lora_filenames, performance)
            expected = test["output"]
            actual = util.parse_lora_references_from_prompt(prompt, loras, loras_limit=loras_limit, lora_filenames=lora_filenames)
            self.assertEqual(expected, actual)
