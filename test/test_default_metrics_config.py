# MIT License
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""Dependency-free runtime regression checks for the generated metrics catalog."""

import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_COUNTERS = {
    "cache_transfer_graph_plan_hits_total",
    "cache_transfer_graph_plan_misses_total",
    "cache_transfer_graph_backend_batches_total",
    "cache_transfer_graph_backend_shards_total",
}


class DefaultMetricsConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the module and its comprehensions, without importing optional UCM
        # runtime dependencies. AST-only checks miss malformed tuple unpacking.
        cls.module = runpy.run_path(str(REPO_ROOT / "ucm/default_metrics_config.py"))

    def test_catalog_shapes_and_unique_names(self):
        config = self.module["DEFAULT_METRICS_CONFIG"]
        names = []
        for kind in ("counter", "gauge", "histogram"):
            self.assertIsInstance(config[kind], list)
            for metric in config[kind]:
                self.assertIsInstance(metric, dict)
                for field in ("name", "documentation"):
                    self.assertIsInstance(metric[field], str)
                    self.assertTrue(metric[field].strip())
                if kind == "histogram":
                    self.assertIsInstance(metric["buckets"], list)
                    self.assertTrue(metric["buckets"])
                names.append(metric["name"])
        self.assertEqual(len(names), len(set(names)))

    def test_each_graph_counter_has_its_own_pair_and_yaml_entry(self):
        pairs = self.module["_COUNTER_METRICS"]
        for pair in pairs:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
        counters = dict(pairs)
        self.assertTrue(GRAPH_COUNTERS <= counters.keys())
        self.assertIn("cache_load_wait_shards_total", counters)
        source = (REPO_ROOT / "examples/metrics/metrics_configs.yaml").read_text(
            encoding="utf-8"
        )
        for name in GRAPH_COUNTERS:
            expected = f'  - name: "{name}"\n' f'    documentation: "{counters[name]}"'
            self.assertIn(expected, source)

    def test_factory_returns_independent_copies(self):
        get_config = self.module["get_default_metrics_config"]
        first = get_config()
        second = get_config()
        first["counter"][0]["documentation"] = "changed"
        first["consumers"]["vllm_connector"] = False
        self.assertEqual(second, self.module["DEFAULT_METRICS_CONFIG"])
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
