"""Summarize UCM_GDR_TRACE logs from one host/run (standard library only)."""

import argparse
import collections
import json
import re


def union_ns(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(rows, threshold):
    intervals = [(r["post_ns"], r["done_ns"]) for r in rows]
    active = union_ns(intervals)
    span = max(b for _, b in intervals) - min(a for a, _ in intervals)
    sizes = collections.Counter(r["bytes"] for r in rows)
    nbytes = sum(r["bytes"] for r in rows)
    latency = sorted(b - a for a, b in intervals)
    return {
        "wr_count": len(rows),
        "bytes": nbytes,
        "size_histogram_bytes": dict(sorted(sizes.items())),
        "wr_below_threshold": sum(n for size, n in sizes.items() if size < threshold),
        "wr_at_threshold": sizes[threshold],
        "wr_above_threshold": sum(n for size, n in sizes.items() if size > threshold),
        "active_ms": active / 1e6,
        "span_ms": span / 1e6,
        "active_GBps": nbytes / active if active else None,
        "span_GBps": nbytes / span if span else None,
        "post_to_poll_p50_us": latency[(len(latency) - 1) // 2] / 1e3,
        "post_to_poll_p99_us": latency[min(len(latency) - 1, int(len(latency) * 0.99))]
        / 1e3,
        "prepare_mean_us": sum(r["post_ns"] - r["begin_ns"] for r in rows)
        / len(rows)
        / 1e3,
        "fallback_host_wr": sum(r["fallback_host"] in ("true", "1") for r in rows),
        "fallback_gpu_wr": sum(r["fallback_gpu"] in ("true", "1") for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--threshold-bytes", type=int, default=96 * 1024)
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    seen = set()
    for path in args.logs:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "UCM_GDR_TRACE pid=" not in line:
                    continue
                r = dict(re.findall(r"(\w+)=(\S+)", line.split("UCM_GDR_TRACE ", 1)[1]))
                for field in ("bytes", "begin_ns", "post_ns", "done_ns"):
                    r[field] = int(r[field])
                if not r["begin_ns"] <= r["post_ns"] <= r["done_ns"]:
                    raise ValueError("Non-monotonic trace timestamps")
                identity = (r["pid"], r["channel"], r["req"], r["post_ns"])
                if identity in seen:
                    continue
                seen.add(identity)
                key = tuple(r[k] for k in ("pid", "gpu", "nic", "channel", "dir"))
                groups[key].append(r)
    if not groups:
        parser.error(
            "No trace records found; enable UCM_GDR_TRACE=1 and capture worker stderr"
        )
    print(
        json.dumps(
            {
                "threshold_bytes": args.threshold_bytes,
                "channels": [
                    dict(
                        zip(("pid", "gpu", "nic", "channel", "direction"), key),
                        **summarize(rows, args.threshold_bytes),
                    )
                    for key, rows in sorted(groups.items())
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
