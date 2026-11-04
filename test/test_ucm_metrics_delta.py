import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "ucm_metrics_delta", Path(__file__).parents[1] / "tools/ucm_metrics_delta.py"
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def histogram(count, total, bucket16, bucket32, rank="0"):
    return f"""# TYPE ucm:gdr_h2d_rdma_wr_size_bytes histogram
ucm:gdr_h2d_rdma_wr_size_bytes_bucket{{worker_rank="{rank}",le="16384"}} {bucket16}
ucm:gdr_h2d_rdma_wr_size_bytes_bucket{{worker_rank="{rank}",le="32768"}} {bucket32}
ucm:gdr_h2d_rdma_wr_size_bytes_bucket{{worker_rank="{rank}",le="+Inf"}} {count}
ucm:gdr_h2d_rdma_wr_size_bytes_count{{worker_rank="{rank}"}} {count}
ucm:gdr_h2d_rdma_wr_size_bytes_sum{{worker_rank="{rank}"}} {total}
"""


class MetricsDeltaTest(unittest.TestCase):
    def test_interval_mean_buckets_and_rank_separation(self):
        before = histogram(5, 5 * 18432, 0, 5) + histogram(7, 7 * 18432, 0, 7, "1")
        after = histogram(8, 8 * 18432, 0, 8) + histogram(9, 9 * 18432, 0, 9, "1")
        result = metrics.compare(before, after, 2)
        self.assertEqual(len(result["histograms"]), 2)
        for row, count in zip(result["histograms"], (3, 2)):
            self.assertEqual(row["count"], count)
            self.assertEqual(row["mean"], 18432)
            self.assertEqual(row["unit"], "bytes")
            self.assertEqual(
                row["quantile_bucket_bounds"]["p99"],
                {"lower_exclusive": 16384, "upper_inclusive": 32768},
            )

    def test_counter_rate_and_gauge_are_different(self):
        template = """# TYPE ucm:gdr_h2d_rdma_bytes_total counter
ucm:gdr_h2d_rdma_bytes_total {counter}
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running {gauge}
"""
        result = metrics.compare(
            template.format(counter=100, gauge=2),
            template.format(counter=184420, gauge=1),
            2,
        )
        counter, gauge = result["series"]
        self.assertEqual(counter["delta"], 184320)
        self.assertEqual(counter["window_payload_GBps"], 184320 / 2 / 1e9)
        self.assertEqual(gauge["status"], "snapshot_only")
        self.assertNotIn("delta", gauge)

    def test_missing_is_not_zero_and_empty_interval_has_no_mean(self):
        result = metrics.compare("", histogram(2, 36864, 0, 2))
        self.assertEqual(result["histograms"][0]["status"], "invalid_count_or_sum")
        self.assertTrue(all(r["status"] == "missing_before" for r in result["series"]))
        zero = metrics.compare(histogram(2, 36864, 0, 2), histogram(2, 36864, 0, 2))
        self.assertIsNone(zero["histograms"][0]["mean"])
        self.assertEqual(zero["histograms"][0]["quantile_bucket_bounds"], {})

    def test_resets_and_process_restart(self):
        result = metrics.compare(histogram(5, 92160, 0, 5), histogram(2, 36864, 0, 2))
        self.assertEqual(result["histograms"][0]["status"], "invalid_count_or_sum")
        before = "process_start_time_seconds 1\n" + histogram(1, 18432, 0, 1)
        after = "process_start_time_seconds 2\n" + histogram(8, 147456, 0, 8)
        result = metrics.compare(before, after)
        self.assertTrue(result["process_restart_detected"])
        self.assertEqual(result["histograms"][0]["status"], "invalid_count_or_sum")

    def test_metric_recreation_even_if_counter_increases(self):
        before = (
            histogram(1, 18432, 0, 1)
            + 'ucm:gdr_h2d_rdma_wr_size_bytes_created{worker_rank="0"} 10\n'
        )
        after = (
            histogram(9, 165888, 0, 9)
            + 'ucm:gdr_h2d_rdma_wr_size_bytes_created{worker_rank="0"} 11\n'
        )
        result = metrics.compare(before, after)
        self.assertEqual(result["histograms"][0]["status"], "invalid_count_or_sum")

    def test_invalid_buckets_and_infinite_upper_bound(self):
        with self.assertRaises(ValueError):
            metrics.quantile_buckets([(16, 2), (32, 1), (float("inf"), 3)], 3)
        with self.assertRaises(ValueError):
            metrics.quantile_buckets([(16, 2), (float("inf"), 2)], 3)
        bounds = metrics.quantile_buckets([(1000, 2), (float("inf"), 3)], 3)
        self.assertEqual(
            bounds["p99"], {"lower_exclusive": 1000, "upper_inclusive": None}
        )
        json.dumps(bounds, allow_nan=False)

    def test_duplicate_series_and_invalid_window(self):
        with self.assertRaises(ValueError):
            metrics.parse_metrics("ucm:a_total 1\nucm:a_total 2\n")
        for elapsed in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                metrics.compare("", "", elapsed)

    def test_labels_units_and_raw_file(self):
        raw = 'ucm:count_total{model="a b",worker_rank="0"} 3\n'
        result = metrics.compare(raw, raw.replace(" 3\n", " 5\n"))
        self.assertEqual(result["series"][0]["labels"]["model"], "a b")
        self.assertEqual(result["series"][0]["delta"], 2)
        self.assertEqual(metrics.unit("ucm:save_completion_wait_duration"), "ms")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "before.prom"
            path.write_text(raw, encoding="utf-8")
            self.assertEqual(metrics.read_snapshot(path), (raw, None))

    def test_capture_and_cli_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            before_path = Path(directory) / "before.json"
            after_path = Path(directory) / "after.json"
            with patch.object(
                metrics.urllib.request,
                "urlopen",
                return_value=io.BytesIO(histogram(1, 18432, 0, 1).encode()),
            ):
                metrics.capture("http://localhost:9000/metrics", before_path)
            data = json.loads(before_path.read_text())
            self.assertEqual(data["format"], "ucm-metrics-snapshot-v1")
            self.assertIn("wr_size_bytes", data["metrics"])
            data.update(clock_id="test-host:boot", monotonic_seconds=10)
            before_path.write_text(json.dumps(data), encoding="utf-8")
            data.update(monotonic_seconds=12, metrics=histogram(3, 55296, 0, 3))
            after_path.write_text(json.dumps(data), encoding="utf-8")
            command = [sys.executable, str(Path(metrics.__file__)), "diff"]
            result = subprocess.run(
                command + [str(before_path), str(after_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["elapsed_seconds"], 2)
            self.assertEqual(report["histograms"][0]["mean"], 18432)
            data["clock_id"] = "different-host:boot"
            after_path.write_text(json.dumps(data), encoding="utf-8")
            invalid = subprocess.run(
                command + [str(before_path), str(after_path)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("different hosts", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
