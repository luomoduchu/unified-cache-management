import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gdr_trace_summary", Path(__file__).parents[1] / "tools/gdr_trace_summary.py"
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class GdrTraceSummaryTest(unittest.TestCase):
    def test_overlapping_and_idle_intervals(self):
        self.assertEqual(trace.union_ns([(10, 30), (20, 40), (50, 60)]), 40)

    def test_bandwidth_and_threshold(self):
        rows = [
            dict(
                bytes=size,
                begin_ns=1,
                post_ns=start,
                done_ns=end,
                fallback_host="true",
                fallback_gpu="false",
            )
            for size, start, end in [(65536, 10, 30), (98304, 20, 40), (131072, 50, 60)]
        ]
        result = trace.summarize(rows, 98304)
        self.assertEqual(result["active_GBps"], 294912 / 40)
        self.assertEqual(result["span_GBps"], 294912 / 50)
        self.assertEqual(result["wr_below_threshold"], 1)
        self.assertEqual(result["wr_at_threshold"], 1)
        self.assertEqual(result["wr_above_threshold"], 1)
        self.assertEqual(result["fallback_host_wr"], 3)


if __name__ == "__main__":
    unittest.main()
