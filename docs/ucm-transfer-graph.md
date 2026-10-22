# Optional CacheStore Load transfer graph

`cache_transfer_graph` enables the first-stage `UcmTransferGraph` executor in the
actual CacheStore Load queue. It is **false by default**. Existing StoreV1 and
Python load/check/wait interfaces are unchanged. Dump keeps its existing outer
completion semantics. Set the flag back to false to select the existing Load path.

```yaml
ucm_connector_name: UcmPipelineStore
ucm_connector_config:
  store_pipeline: "Cache|Posix"
  storage_backends: ["/your/a800/test-mount"]
  cache_transfer_graph: true
  cache_graph_batch_shards: 32
  cache_graph_batch_bytes: 8388608
  cache_graph_plan_cache_entries: 32
  posix_io_engine: psync  # strict psync is required for the first-stage graph
```

These are starting values for evaluation, not tuned hardware recommendations.
The window is `min(batch_shards, floor(batch_bytes / shard_size))`; resource
pressure may produce smaller actual batches. Shard cap must be 1..65536, byte cap
must hold at least one shard, and plan cache entries must be 0..4096. Zero entries
disables caching while still executing the graph. Negative values, wrong types,
unknown `cache_graph_*` keys and invalid graph limits return InvalidParam during
Setup. Existing unrelated connector options retain their current parsing behavior.
Each Load validates shard indexes and tensor address counts before acquiring any
lease. Variable tensor sizes, zero-sized slots and null ghost-slot destinations
retain CopyStream's skip behavior. Tensor sizes must fit a shard without overflow.

The immutable plan stores only count, rank order and window limits. It uses constant
space even for a very large request and a bounded LRU of plans; running requests
retain shared ownership through eviction. Each replay binds current block IDs,
indexes, addresses, owner state and buffer leases. No file handles or addresses
are cached. Rank order is identical to the original queue.

The dispatcher uses a bounded `TryGet` (bucket try-lock and at most 64 allocation
candidates), submits acquired owner misses in one backend Load, and publishes each
window to the transfer worker. It flushes a partial window before retrying buffer
acquisition. The transfer worker Waits each successful backend submission once,
publishes owner READY/FAILED, copies successful data and fences **every window**
before releasing host leases and destination pointer arrays. Parent failure does
not skip backend drain. A final marker completes the parent once after all earlier
windows. Stop rejects new submissions, cancels resource/shared-reader waits and
drains accepted work in dispatcher/transfer order.

The graph queries the actual backend's optional `LoadWaitCompletion` side
interface; StoreV1's virtual table remains unchanged. Only a successfully set-up
Posix **strict psync Load** backend currently advertises terminal Wait semantics.
The Posix backend must receive `cache_transfer_graph: true` during its own Setup,
not only in the Cache stage. Pipeline configurations propagate this shared option.
HealthBreakerStore forwards the actual wrapped capability and retains its existing
Wait/health handling. Unknown backends, ASU, AIO and non-strict psync are rejected.

Strict psync disables the Load worker timeout/detach monitor. TaskWrapper still
applies its logical timeout and cancels queued work, but Wait does not return until
active reads actually finish. A stuck hard-mounted NFS read can therefore make
Wait and shutdown block indefinitely after the logical deadline: releasing the
buffer while that I/O is still running would be unsafe. Dump worker monitoring
and graph-disabled backend defaults are unchanged. AIO's current forced timeout
completion is not evidence that kernel writes to host memory have drained, so it
is outside the first-stage graph's supported combinations.

Backends must return statuses and honor their advertised terminal/drain contract.
A backend that throws after accepting I/O without returning a handle cannot be
safely recovered by any executor; that behavior is outside this contract. A
dispatcher allocation/submission exception before acceptance fails owned leases
and still publishes the completion marker. Process-wide unrecoverable OOM is not
a supported recovery mechanism.

The graph omits the original speculative, potentially blocking `Prealloc` of the
next shard. This can change eviction and observed hit rates in multi-shard blocks.
Per-window fencing can increase synchronization overhead and batching can increase
first-shard latency. Keep the flag disabled until workload-specific results meet
your latency, throughput and CPU targets. Concurrent producer entry to LoadQueue
is serialized to preserve its SPSC input queue. Local buffer node selection now
uses an atomic cursor for concurrent bucket allocators.

H100 uses the existing CUDA CopyStream implementation; Ascend 910B3 uses its
existing Ascend implementation. This feature does not select a device-specific
copy mode. In particular, do not apply the Ascend-A3 SDMA Direct setting to 910B3.
The first-stage graph rejects effective `cache_io_aggregation`, `use_gdr` and `cache_sdma_direct`
combinations during Setup: those implementations do not guarantee that all
accepted lane/NIC accesses have drained when synchronization reports a failure.
Use the ordinary CUDA, Ascend or simu copy stream for the graph. Existing
aggregation/GDR/SDMA Direct behavior with the graph disabled is unchanged.
OceanStor A800 is accessed through the existing POSIX Linux path, locally mounted
or via NFS, on x86/aarch64. Protocol versions, client/mount details and real
hardware measurements remain to be supplied. No vendor SDK, SPDK or io_uring
dependency was added. See the [A800 evaluation protocol](design/ucm-transfer-graph-a800-evaluation.md).

## Verification

Portable plan checks require only C++17 and always execute checks in Release:

```sh
cmake -S ucm/store/test/transfer_graph -B build/transfer-graph-plan -DCMAKE_BUILD_TYPE=Release
cmake --build build/transfer-graph-plan
ctest --test-dir build/transfer-graph-plan --output-on-failure
# Or: g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. ucm/store/test/transfer_graph/plan_test.cc -o plan_test
```

The portable test covers 161139 rank-order reference cases, byte cap, cache
reuse/eviction/disabled cache, stable ownership and SIZE_MAX index arithmetic.
It verifies the actual plan implementation, not the asynchronous executor.

Linux simu integration and real CacheStore/Posix benchmark build:

```sh
cmake -S . -B build/graph-simu -DRUNTIME_ENVIRONMENT=simu -DBUILD_UCM_SPARSE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_TRANSFER_GRAPH_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/graph-simu -j
ctest --test-dir build/graph-simu --output-on-failure
build/graph-simu/ucm/store/test/ucmstore.test --gtest_filter='GraphQueueTest.*:UCCacheLoadQueueTest.*:UCCacheTransBufferTest.*'
```

`GraphQueueTest` uses the real TransBuffer, dispatcher, transfer worker and simu
copy stream. It covers requests larger than the pool, owner-only batched reads,
duplicates/hits, backend-only bypass, variable tensors, failure propagation,
drain after later submission failure, backend batch failure, independent pointer
bindings, concurrent producers, empty/invalid requests, shutdown resource waits,
bucket contention, one completion on exception and partial H2D failure fencing.
Additional regressions cover both worker-construction failures, stream Setup
failure/exception, rejection after unsuccessful Setup, effective unsupported copy
modes, null/zero ghost slots and a real parent TaskWrapper timeout that must drain
the backend before returning. A separate adequate-pool test checks strict 4+4+1
batching; the pressure test permits any bounded partial batch size.
Posix tests verify strict/non-strict/AIO capability and block a real psync pread
for 1.6 seconds (past the legacy monitor tick) to check that timeout Wait cannot
return early. HealthBreaker tests verify capability forwarding and retained health
error handling. These Linux tests are added but have not run on this workstation.
Simu-specific tests are excluded from other runtime builds. Existing Cache/Posix
tests remain part of ctest. ASan/TSan are recommended Linux follow-up runs.

## Actual CacheStore/Posix comparison benchmark

The executable calls `MakeCacheStore()->Load/Wait`, forwards backend operations to
a counting StoreV1 wrapper and the real `MakePosixStore()`, and uses simu copy
destinations. It requires no Python or torch. It creates a fresh uniquely named
child of an existing **test** directory and retains it for inspection. It never
reads existing cache datasets, clears system caches or changes system settings.

```sh
mkdir -p /your/test-mount/ucm-bench
# PARENT shard_bytes request_shards batch concurrency iterations requested_hit_percent engine io_direct order
build/graph-simu/ucm/store/test/transfer_graph_benchmark /your/test-mount/ucm-bench \
  65536 32 32 4 100 50 psync 0 > comparison.jsonl
# Reverse the mode order for a separate repeat:
build/graph-simu/ucm/store/test/transfer_graph_benchmark /your/test-mount/ucm-bench \
  65536 32 32 4 100 50 psync 0 on-off > comparison-reversed.jsonl
# Small smoke (each mode still uses the CacheStore minimum 1 GiB buffer configuration):
build/graph-simu/ucm/store/test/transfer_graph_benchmark /your/test-mount/ucm-bench \
  4096 4 4 1 2 50 psync 0
```

Each invocation seeds the same deterministic block/offset-dependent data for both
modes, creates a fresh CacheStore per mode, excludes separate warmup requests and
validates every returned byte. Hot keys are private to a serial worker; cold keys
are unique per request. The actual READY hit count is measured as total shards
minus actual backend shards in this workload, which has no shared LOADING readers.
Requested hit percent is rounded to whole shards and may differ from actual hits
if hot keys are evicted. The single-shard block layout intentionally eliminates
the legacy next-shard Prealloc difference. Backend batch counts are actual
successful Load calls, not estimated from B.
Both modes share one strict psync backend with identical worker/timeout policy;
the counting wrapper forwards its actual capability. The CLI rejects aio.

Each JSON line reports mode, backend/path/config, actual submissions/shards/mean
batch, actual hits, warmup, measured bytes, process CPU, throughput, P50/P95/P99,
plan hits/misses and configured resource limits. Unmeasured peak leases and GPU
time are null (unavailable). CPU and wall throughput include request preparation
and readback validation. Latency measures only Load submission through Wait;
`summed_request_load_ms` sums these intervals and can overlap across concurrent
requests, so it is not a wall-clock throughput denominator. This benchmark is an
honest application-loop measurement, not a pure DMA or A800 bandwidth measurement.

Sweep shard sizes 4/16/64/256 KiB, B=1/8/16/32/64, concurrency=1/4/16 and requested
hits=0/50/100. The first-stage comparison supports strict psync only; `io_direct=1` requires actual
filesystem/client support. Record backend CPU concurrency, protocol and mount
parameters, CPU, device, topology and network. The final optional argument selects
`off-on` (default) or `on-off`, recorded as `run_order` in each output row.
Filesystem/client cache warmth can bias results; repeat in separate processes,
alternate both orders, and report that limitation. Buffered
results include OS/client cache effects. Simu on a local filesystem validates
software and must never be labeled H100, 910B3 or A800 performance.

## Current validation record

Windows MinGW g++ 8.1, C++17 `-O2 -Wall -Wextra -Werror`: portable test passed all
161139 reference cases. The independent main-agent rank test also passed 43933
cases.

On 2026-10-22, commit `63ac2591b3fe42f7825c09b48b4922c2ef037450` passed
[GitHub Actions](https://github.com/luomoduchu/unified-cache-management/actions/runs/39743377819)
on Ubuntu 24.04.5 with the simu runtime: the build, all 997 enabled CTest cases,
the Release portable plan test, three Python metrics checks, and 19 graph executor
tests repeated ten times. The existing `UCMetricsStress.TenWritersOneCollector`
test was disabled. The strict psync blocked-read timeout regression passed.
C++/Python lint, toolkit build/install checks and the separate documentation check
also passed. This is Linux software validation, with no physical GPU/NPU/A800.

Both benchmark smoke orders passed full readback validation. With 4 KiB shards,
4 shards/request, batch cap 4, one client and 50% hits, the two measured requests
read four backend shards: graph off issued four backend submissions, graph on
issued two. This confirms batching, not an end-to-end speedup. There are only two
requests per mode; the latency/throughput ordering changes when run order changes.
The [four raw smoke rows](design/ucm-transfer-graph-ci-smoke.jsonl) preserve the
actual output. ASan/TSan, aarch64, CUDA/Ascend builds and H100/910B3/A800 hardware
performance remain untested. See the [independent review](design/ucm-transfer-graph-review.md).
