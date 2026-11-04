#!/usr/bin/env python3
"""Capture Prometheus snapshots and compare one workload, preserving all labels.

Requires prometheus_client (available in the vLLM environment). Missing series
are not assumed to be zero. Rates use the scrape window, not benchmark duration.
"""

import argparse
import json
import math
import platform
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def parse_metrics(text):
    samples = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = (sample.name, tuple(sorted(sample.labels.items())))
            if key in samples:
                raise ValueError(f"Duplicate metric series: {key}")
            samples[key] = (sample.value, family.type, family.name)
    return samples


def unit(name):
    if "bandwidth_gbps" in name:
        return "GB/s"
    if "bytes" in name:
        return "bytes"
    if name.endswith("_us"):
        return "us"
    if name.endswith("_ms") or name in (
        "ucm:save_duration",
        "ucm:save_completion_wait_duration",
    ):
        return "ms"
    if name.endswith("_seconds"):
        return "seconds"
    return "metric_native_unit"


def quantile_buckets(buckets, count):
    """Return bucket bounds, never an interpolated or invented exact quantile."""
    buckets = sorted(buckets)
    if not buckets or not math.isinf(buckets[-1][0]):
        raise ValueError("missing +Inf bucket")
    if buckets[-1][1] != count:
        raise ValueError("+Inf bucket differs from count")
    previous_count = 0
    for bound, cumulative in buckets:
        if math.isnan(bound) or cumulative < previous_count or cumulative < 0:
            raise ValueError("invalid cumulative bucket deltas")
        previous_count = cumulative
    result = {}
    if count == 0:
        return result
    for percentile in (50, 90, 99):
        lower = None
        for upper, cumulative in buckets:
            if cumulative >= count * percentile / 100:
                result[f"p{percentile}"] = {
                    "lower_exclusive": lower,
                    "upper_inclusive": upper if math.isfinite(upper) else None,
                }
                break
            lower = upper
    return result


def compare(before_text, after_text, elapsed_seconds=None, prefixes=("ucm:", "vllm:")):
    if elapsed_seconds is not None and (
        not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
    ):
        raise ValueError("elapsed seconds must be finite and positive")
    before, after = parse_metrics(before_text), parse_metrics(after_text)
    restart = any(
        name == "process_start_time_seconds" and before[key][0] != after[key][0]
        for key in before.keys() & after.keys()
        for name, _ in (key,)
    )
    rows, by_key = [], {}
    for key in sorted(before.keys() | after.keys()):
        name, labels = key
        if not name.startswith(prefixes) or name.endswith("_created"):
            continue
        row = {"metric": name, "labels": dict(labels)}
        rows.append(row)
        by_key[key] = row
        if key not in before or key not in after:
            row["status"] = "missing_before" if key not in before else "missing_after"
            continue
        b, kind, family = before[key]
        a, after_kind, _ = after[key]
        if not math.isfinite(b) or not math.isfinite(a):
            row["status"] = "nonfinite_sample"
            continue
        row.update(before=b, after=a, type=kind)
        if kind != after_kind:
            row["status"] = "type_changed"
            continue
        cumulative = kind in ("counter", "histogram", "summary") or (
            kind == "unknown" and name.endswith(("_total", "_count", "_sum", "_bucket"))
        )
        if not cumulative:
            row["status"] = "snapshot_only"
            continue
        created_key = (family + "_created", labels)
        if name.endswith("_bucket"):
            created_key = (
                family + "_created",
                tuple(x for x in labels if x[0] != "le"),
            )
        recreated = (
            created_key in before
            and created_key in after
            and before[created_key][0] != after[created_key][0]
        )
        if restart or recreated or a < b:
            row["status"] = "reset_or_restart"
            continue
        row.update(status="ok", delta=a - b)
        if elapsed_seconds is not None:
            row["per_second"] = (a - b) / elapsed_seconds
            if name.startswith("ucm:gdr_") and name.endswith("_rdma_bytes_total"):
                row["window_payload_GBps"] = (a - b) / elapsed_seconds / 1e9

    histograms = []
    for (name, labels), count_row in by_key.items():
        if not name.endswith("_count"):
            continue
        base = name[: -len("_count")]
        sum_row = by_key.get((base + "_sum", labels))
        if sum_row is None:
            continue
        hist = {"metric": base, "labels": dict(labels), "unit": unit(base)}
        histograms.append(hist)
        if any(row["status"] != "ok" for row in (count_row, sum_row)):
            hist["status"] = "invalid_count_or_sum"
            continue
        count = count_row["delta"]
        if count == 0 and sum_row["delta"] != 0:
            hist["status"] = "sum_changed_without_count"
            continue
        hist.update(
            status="ok",
            count=count,
            sum=sum_row["delta"],
            mean=sum_row["delta"] / count if count else None,
        )
        bucket_rows = [
            (float(dict(lab)["le"]), row)
            for (metric, lab), row in by_key.items()
            if metric == base + "_bucket"
            and tuple(x for x in lab if x[0] != "le") == labels
            and "le" in dict(lab)
        ]
        if bucket_rows:
            try:
                if any(row["status"] != "ok" for _, row in bucket_rows):
                    raise ValueError("missing/reset bucket")
                hist["quantile_bucket_bounds"] = quantile_buckets(
                    [(bound, row["delta"]) for bound, row in bucket_rows], count
                )
            except ValueError as exc:
                hist["bucket_error"] = str(exc)
    return {
        "elapsed_seconds": elapsed_seconds,
        "process_restart_detected": restart,
        "notes": [
            "All labels are preserved; ranks and processes are never added together.",
            "Missing series are unavailable, not zero; warm up metric series before capture.",
            "Gauge endpoints do not give the maximum or average during the workload.",
            "Rates include all time between scrapes, including idle and drain time.",
            "Resets hidden between endpoints cannot always be detected; do not restart workers.",
            "Histogram sums may overlap in time and must not be summed as wall time.",
        ],
        "series": rows,
        "histograms": histograms,
    }


def capture(url, output):
    start = time.monotonic()
    with urllib.request.urlopen(url, timeout=15) as response:
        metrics = response.read().decode("utf-8")
    end = time.monotonic()
    parse_metrics(metrics)
    boot_file = Path("/proc/sys/kernel/random/boot_id")
    clock_id = (
        platform.node() + ":" + boot_file.read_text().strip()
        if boot_file.exists()
        else None
    )
    data = {
        "format": "ucm-metrics-snapshot-v1",
        "url": url,
        "clock_id": clock_id,
        "monotonic_seconds": (start + end) / 2,
        "scrape_seconds": end - start,
        "captured_unix_seconds": time.time(),
        "metrics": metrics,
    }
    Path(output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("{"):
        snapshot = json.loads(text)
        if snapshot.get("format") != "ucm-metrics-snapshot-v1":
            raise ValueError("Unsupported snapshot JSON format")
        return snapshot["metrics"], snapshot
    return text, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser(
        "snapshot", help="Read /metrics and save text plus capture times"
    )
    snap.add_argument("output")
    snap.add_argument("--url", default="http://localhost:9000/metrics")
    diff = sub.add_parser("diff", help="Compare JSON snapshots or raw .prom files")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.add_argument("--elapsed-seconds", type=float)
    diff.add_argument("--prefix", action="append", help="Metric prefix; repeatable")
    args = parser.parse_args()
    if args.command == "snapshot":
        capture(args.url, args.output)
        return
    before, bm = read_snapshot(args.before)
    after, am = read_snapshot(args.after)
    elapsed = args.elapsed_seconds
    if bm and am:
        if bm["url"] != am["url"]:
            parser.error("Snapshots use different endpoints")
        if bm["clock_id"] and am["clock_id"] and bm["clock_id"] != am["clock_id"]:
            parser.error("Snapshots use different hosts or boot sessions")
        if elapsed is None and bm["clock_id"] and bm["clock_id"] == am["clock_id"]:
            elapsed = am["monotonic_seconds"] - bm["monotonic_seconds"]
    report = compare(before, after, elapsed, tuple(args.prefix or ("ucm:", "vllm:")))
    if bm and am:
        report["scrape_seconds_before_after"] = [
            bm["scrape_seconds"],
            am["scrape_seconds"],
        ]
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
