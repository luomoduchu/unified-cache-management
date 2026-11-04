# 固定 WR 粒度下的长度效应与前缀复用实验

2026-11-04。用户确认：UCM 来自主仓库 develop，vLLM 0.27.1，关闭前缀复用使用 `--no-enable-prefix-caching`。develop 是移动分支，每轮应保存实际 SHA；本地源码定位基线为 `8ce72924b65c07c29c017b9a25ebbb663607659d`，不能据分支名认定服务器与本地相同。

已有原始结果和口径见[模型分析](gdr-model-benchmark-analysis.md)与[WR / CUDA 基线](gdr-wr-and-ce-baseline-analysis.md)。本页提供可证伪的机制与实验步骤；新工具[ucm_metrics_delta.py](../tools/ucm_metrics_delta.py)仅采集/分析 metrics，不修改推理运行逻辑。

## 1. 请求变长、WR 不变，为什么可能反转

固定 18 KiB 只固定一次操作的 payload。一个请求需要多少操作、它们何时提交、多少操作并发、缓存从哪层读取，以及提交后还有多少等待，都可以变化。

用简化的批次传输模型表示：`T(N) ≈ 首次完成延迟 + (N-1)/持续完成速率 + 排队与同步等待`。它说明首次完成快和持续处理能力强是两件事；实际推理还需要看哪些时间暴露在请求关键路径上。此公式中的参数不是从现有数据拟合得到的。

如果两个引擎在相同方向、并发、buffer 和运行状态下，每个 WR 的相对代价始终不变，只增加 N，不足以解释新的非线性恶化。应测下面这些随负载改变的量，而不是把 32K tokens 当成 32 KiB IO：

| 候选机制 | 为什么请求变长会暴露它 | 区分它的证据 |
|---|---|---|
| 提交/CQ/CPU 持续处理能力 | 更多 WR 使 CPU 工作更连续、队列占用更久 | WR/s 停止增长，提交/完成线程 CPU 高；降低并发后回退减轻 |
| GDR D2H READ 方向受限 | load 与 dump 同时增大，READ 和 WRITE 条件不同 | D2H 完成尾延迟/在途量先升高，H2D 单独测试仍快 |
| Cache/Posix/共享 buffer 反压 | 工作集及 buffer 持有时间增大 | Cache 排队、backend wait 或 buffer 获取时间增长 |
| 显存容量、调度与抢占 | 固定并发下每请求占用更多 KV blocks | 实际 running/waiting、preemption、重算、KV 使用率变化 |
| 分层 load/compute/dump 重叠改变 | 每层 load 数量、prefill 分块、尾部 drain 变化 | 设备时间线和 connector wait 显示暴露的等待比例变化 |
| 测试状态没有对齐 | 更长请求改变 HBM/DRAM 命中或延续前轮积压 | 来源计数、warmup/drain、实际复制 bytes 不相等 |

固定客户端并发不代表 GPU 实际 batch 相同。请求变长增加每个请求的工作量和驻留时间，也不意味着 WR 到达速率一定提高；两者都要测。

在 BF16 MLA、block=16、27 层、32 请求、外部命中 90%、每 GPU 逐块恢复的假设下：

| input tokens | 逻辑 H2D 片段数 / GPU | payload / GPU（十进制 GB） |
|---|---:|---:|
| 8000 | 388800 | 7.1664 |
| 16000 | 777600 | 14.3327 |
| 32000 | 1555200 | 28.6654 |

这里只算理想逻辑恢复量，8000 不等于此前另一组 8192；重试、实际布局与去重等要以 Counter 差值核对。20 GB/s 若由 18 KiB WR 承载，约要求每秒 108.5 万个 WR；这是字节数与操作数的换算，不是已测软件处理能力。

已有 Qwen 8K/16K 获益、32K 回退，与上述机制相容。但 DeepSeek 64K 仍有收益、32K 有一次严重慢运行，因此现有数据不能支持跨模型的“越长必然越差”定律。

## 2. 前缀复用改变了什么

关闭 vLLM GPU 前缀复用不会自动关闭 UCM 外部 KV 复用。vLLM 0.27.1 scheduler 分别查询本地和 connector 命中，再合计已计算 tokens；异步外部加载还可能进入 `WAITING_FOR_REMOTE_KVS`，是否走该状态取决于 connector 配置。参见[该版本 scheduler](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)。

你这两组的总命中均约 90%。理想情况下，剩余约 10% tokens 需要新计算。不能把关闭前缀复用说成“重新 prefill 全部 16K”，也不能直接认定它会多保存十倍 KV。

本地 `_generate_dispatch_meta()` 将 HBM 命中排除在 load 区间外；请求元数据的 `token_processed` 从 HBM+外部命中之后开始。layerwise connector 在当前层 load 完成后提交下一层 load，计算后保存当前层。GPU 命中时缺少相同的 load 等待，计算和 dump 可能更集中；外部恢复则可能改变提交节奏、实际 batch 和保存重叠。**后者是可检验的“恢复过程分散保存突发”假设，尚未测到。**

已有四组均值：

| 前缀模式 / 来源 | GDR | TTFT s | E2EL s |
|---|---|---:|---:|
| 开 / GPU 约 90% | OFF | 1.714 | 1.727 |
| 开 / GPU 约 90% | ON | 3.881 | 4.983 |
| 关 / 外部约 90% | OFF | 4.775 | 5.559 |
| 关 / 外部约 90% | ON | 3.498 | 3.982 |

OFF 组 GPU 命中更快符合减少搬运的收益；ON 组反而出现外部恢复更快，需要解释额外等待。优先排查：

1. **保存等待进入 worker 关键路径。** vLLM 0.27.1 的 `_get_kv_connector_output()` 在模型执行的收尾调用 `wait_for_save()`、`get_finished()`；本地 UCM 后者对未完成 dump 可调用阻塞 `wait_dump()`，可能延迟同 worker 的其他请求。源码依据：[vLLM mixin](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/kv_connector_model_runner_mixin.py#L65)、[UCM get_finished](https://github.com/luomoduchu/unified-cache-management/blob/8ce72924b65c07c29c017b9a25ebbb663607659d/ucm/integration/vllm/ucm_connector.py#L2810)。
2. **前置事件和 GDR 软件调度。** GDR scheduler 用 CPU `cudaEventSynchronize()` 等待该 stream 的前置事件；空闲时可 sleep 100 μs，队列满/暂不可提交会 yield 重试。CUDA 路径使用 `cudaStreamWaitEvent`。这些差异有可能改变尾部等待，但不能把每次 100 μs 直接乘所有 WR，也不能据此解释几秒回退。
3. **实际保存量/任务形态不同。** 核对 dump 的实际 bytes、WR、task 数与每 task shard 数。若 GPU 命中组反而保存了大量已命中前缀，需要检查元数据、重复提交和 Cache 所有权；不要先假设这是正常行为。
4. **运行状态差异。** 前轮未完成的 dump、MR 首次注册、CPU/NIC 竞争和抢占可以放大区别；即使只有一个启动参数改变，动态状态也需要控制。

目前最值得看的现成指标是 `save_completion_wait_duration` 和 `connector_get_finished_duration_ms`。`save_duration` 是保存生命周期，不等于阻塞；`wait_for_save()` 名字本身也不代表它一定阻塞。

## 3. 分两阶段做实验

先固定 DeepSeek、TP2、KV dtype/block、两个 GPU/NIC、输出 2、32 请求、并发 32、相同数据与后缀、相同到达方式。明确固定并记录 max-num-batched-tokens、chunked prefill、显存预算、Cache stream 数和 Posix 并发；不要一轮修改多个性能参数。正式性能测试关闭逐 WR trace，诊断另跑短窗口。每配置至少 3 轮，最好 5 轮，交替或平衡 ON/OFF 顺序；warmup 与测量分开。

### 阶段 A：16K 上区分“开关”与“命中来源”

| 场景 | GPU 前缀复用 | 测量前准备 | 实际目标 | GDR |
|---|---|---|---|---|
| H | 开 | 预热目标前缀，等待保存完成 | HBM≈90%、外部≈0 | ON/OFF |
| E | 关 | 外部预热目标前缀 | HBM≈0、外部≈90% | ON/OFF |
| E-control | 开 | 外部预热后仅清 GPU 前缀 | HBM≈0、外部≈90% | ON/OFF |

H 和 E 已有结果，先按统一的 warmup/drain 流程复核并补指标；E-control 是新增的关键两组。E 与 E-control 还必须匹配 UCM DRAM/Posix 来源。每轮使用预热的 90% 前缀和受控的新后缀，避免完整请求预热后命中变成 100%；批内共享前缀也可能造成二次 GPU 命中，必须以实测来源验收。

vLLM 0.27.1 提供只清本地前缀的开发接口，默认 `reset_external=false`，且返回 `success`。在没有业务流量、已启用该开发接口的专用实验实例上，可执行：

```bash
curl -fsS -X POST \
  'http://localhost:9000/reset_prefix_cache?reset_running_requests=false&reset_external=false'
```

必须看到 `{"success":true}`；200 状态本身不够。未完成请求或异步 offload 持有 blocks 时可能失败。接口属于 `VLLM_SERVER_DEV_MODE` 注册的开发路由，默认实例可能 404；若要启用，应另建仅本地访问的实验服务，不在共享对外服务上开启。来源：[0.27.1 reset 路由](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/entrypoints/serve/dev/cache/api_router.py)、[开发路由注册](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/entrypoints/openai/api_server.py#L211)。

判读：E-control 与 E 接近，支持主要差异来自命中路径/时序；二者在 bytes、来源和到达方式相同后仍差很多，则进一步检查前缀开关相关的分配/调度路径。H 仅在 ON 下 `save_completion_wait_duration` 明显增加，则保存完成等待假设增强。该指标很小或本轮没有 D2H，则降低保存假设优先级，查 CPU/事件/模型执行时间线。

### 阶段 B：固定外部来源后扫长度

保持 `--no-enable-prefix-caching`、外部命中≈90%，输入分别 8000/16000/32000，各测 GDR ON/OFF，先得到 6 种配置。两个模式每轮 H2D bytes、命中 tokens、完成请求数应接近；不一致先处理缓存状态，不能直接解释成传输速度差异。

若 32K 回退而 16K 获益，再只补这两个长度的**并发 1**测试，保持总请求数 32、输出 2。若低并发回退明显减轻，优先查队列/资源竞争；若并发 1 仍存在，优先查每请求层间等待和方向差异。不要用客户端 running_max 替代实际 scheduler 时间线。

之后才做固定 18 KiB、同一 pinned buffer、相同 NIC、固定 outstanding/stream 的独立 H2D、D2H、双向微基准，分别增加总 WR 数。若固定条件下 GDR 大批次仍领先，端到端反转应继续在流水线与缓存调度层定位。96 KiB 的 H2D 交叉点不能直接用于 D2H，也不能用增大请求 token 数来模拟增大 memcpy size。

## 4. 每轮采集什么

下表的 UCM 名称均使用 `ucm:` 前缀；先确认服务器实际导出。主仓库 develop 不保证与本地完全一致；只有 HELP、没有 samples 应记为缺失，不是 0。

| 层级 | 指标或记录 | 需要回答的问题 |
|---|---|---|
| 来源 | `total_prefix_query_tokens_total`、`gpu_hbm_hit_tokens_total`、`ucm_hit_tokens_total` | 是否真的保持 GPU/外部命中条件 |
| Store 来源 | `cache_load_shards_total`、`cache_load_wait_shards_total`、`cache_load_backend_shards_total`、Posix 实际读量 | 外部到底来自主机 Cache 还是 SSD；不同计数语义不可混除 |
| 任务 | `cache_load/dump_bytes_total`、`cache_load/dump_duration_ms`、`cache_dump_shards_total` | 逻辑数据量、任务数和任务生命周期 |
| 队列 | `cache_load/dump_queue_wait_duration_ms`、`cache_load/dump_queue_full_total` | 是否被调度队列阻塞 |
| Load 阶段 | `cache_load_backend_submit_duration_ms`、`cache_shard_backend_wait_ms`、`cache_h2d_submit_ms`、`cache_h2d_sync_ms` | 派发、等 shard、CPU 提交、末尾 drain 分别如何变化 |
| Dump 阶段 | `cache_dump_mkbuf_duration_ms`、`cache_d2h_duration_ms`、`cache_dump_backend_submit_duration_ms`、`cache_dump_backend_wait_duration_ms` | buffer/提交/剩余复制/后端反压 |
| Worker 阻塞 | `save_completion_wait_duration`、`connector_get_finished_duration_ms`、`connector_wait_for_layer_load_duration_ms`、`connector_wait_for_save_duration_ms` | 是否真正阻碍模型执行推进 |
| Load 生命周期 | `layerwise_layer_load_duration_ms` | 每层从开始 load 到 wait 返回；包含重叠，区别于 wait 函数自身耗时 |
| GDR 实际传输 | 用户新增的 `gdr_{h2d,d2h}_rdma_bytes_total`、`wr_total`、`wr_size_bytes`、`completion_latency_us` | 实际片段、数量、完成分布；按 rank/方向分开 |
| vLLM | running/waiting 的时间序列、KV 使用率、preemption counter、每步 scheduled tokens、每请求首次/末次输出 | 显存和调度是否改变；接口缺项需要短时事件记录 |
| 系统 | 各 UCM 线程 CPU、切换、磁盘 await/队列、NIC 错误/重试 | CPU/存储/拓扑竞争 |

关键口径：Cache bytes 是对齐后的逻辑计数，可能包含无需实际复制的 shard；`cache_h2d_sync_ms`/`cache_d2h_duration_ms` 主要是末尾同步，不能当完整 DMA 时间。`cache_dump_prereq_wait_ms` 在当前旧 cc 路径才记录，GDR callback 还可能不记录它；缺失不表示前置事件无需等待。原来 OFF 的 1.13 s Dump queue 是另一批 workload，不能移作本轮证据。

同一 rank 的 histogram 计算 `Δsum/Δcount`，分位数用 bucket 差值求所在区间，不取累计值。指标对象有 task、shard、WR 和接口调用之分；不要把不同层级的均值相加，也不要把所有 rank 的等待相加当一条请求延迟。WR/s 及 payload GB/s 使用同一墙钟窗口；任务生命周期可重叠，不能用 bytes/所有 WR latency 之和代替总线带宽。

## 5. 可直接运行的单轮采集

在服务器 vLLM Python 环境运行，需要 `prometheus_client`。采集前记录部署信息：

```bash
git rev-parse HEAD
python -c 'import vllm, ucm; print("vLLM", vllm.__version__); print("UCM", ucm.__file__)'
# 同时保存实际启动命令、UCM YAML、启动时 KV layout/Cache/NIC 日志。
```

每一轮建立独立目录，以下仅是名称示例，替换为实际模式：

```bash
RUN=/tmp/ucm-gdr-experiments/ds16k_ext_gdr_on_r1
mkdir -p "$RUN"

# 先完成对应场景预热；确认上一轮请求、GPU copy 和异步保存已结束。
python tools/ucm_metrics_delta.py snapshot "$RUN/before.json"

# 在这里运行现有 benchmark，保存逐请求结果和总结果到 "$RUN"。

# 客户端 benchmark 返回后立即采集，用来观察业务结束时的状态。
python tools/ucm_metrics_delta.py snapshot "$RUN/returned.json"

# 确认异步传输排空后再采集；不能只凭客户端退出认定已经排空。
python tools/ucm_metrics_delta.py snapshot "$RUN/drained.json"

python tools/ucm_metrics_delta.py diff "$RUN/before.json" "$RUN/returned.json" > "$RUN/business.json"
python tools/ucm_metrics_delta.py diff "$RUN/before.json" "$RUN/drained.json" > "$RUN/full.json"
python tools/ucm_metrics_delta.py diff "$RUN/returned.json" "$RUN/drained.json" > "$RUN/tail.json"
```

这些命令手动执行，注释不能代替真实的 benchmark 和 drain 操作；不要原样连跑后认为完成压测。服务端口变化时给 snapshot 加 `--url`。主机名和 Linux boot ID 相同的 JSON 快照使用单调时钟中点计算窗口，保存 scrape 耗时以衡量采样误差。抓取不是跨 rank 原子快照，应保留每个 rank 的实际值。

`series` 输出 Counter 的本轮 delta 和每秒速率；GDR bytes counter 额外给出 `window_payload_GBps`，包含窗口内 idle/drain，是业务窗口有效带宽。`histograms` 给出本轮 count/sum/mean、单位和 P50/P90/P99 桶区间。Gauge 只给两端值，不编造窗口内最大值。所有标签保留，不自动合并 ranks。

查看最关键的等待统计：

```bash
python - "$RUN/full.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for row in data["histograms"]:
    if any(key in row["metric"] for key in (
        "save_completion_wait", "connector_get_finished", "queue_wait",
        "connector_wait_for_layer_load", "rdma_completion", "rdma_wr_size"
    )):
        print(json.dumps(row, ensure_ascii=False))
PY
```

若仍是原始 `.prom` 文件也可用 `diff before.prom after.prom`；没有准确墙钟间隔时只输出差值，不输出速率。仅在确知两个 scrape 间隔时加 `--elapsed-seconds`，不要把不匹配的 benchmark duration 填进去。

工具将缺失序列标为 `missing_before/after`，将可检测的重启或计数器回退标为 `reset_or_restart`；不会把它们改成 0。预热应使关键 metrics 序列先创建。端点间隐藏的 reset 不一定能检测，特别是 worker 重启而 API 进程未重启，实验期间仍需禁止重启并保存日志。

排空最好以任务/在途数归零或关联完成事件为准。若部署暂时没有这些观测，可检查客户端结束后几次独立快照的 WR/completion/bytes 是否继续增长，同时确认 NIC 无活动；稳定计数只能作为近似，卡住的未完成任务也可能没有计数增长。不要用固定 sleep 秒数宣称已严格排空。

`full` 和 `tail` 可以看后台任务在客户端结束后还完成多少；若一轮开始前没排空，上一轮的完成计数会污染本轮。对 running/waiting、队列长度和 CPU，另做短诊断周期采样（例如 0.2～1 秒，按实际开销调整）。两端快照不能恢复它们的最大值。现有 `running_max=0` 的问题需要这种时间序列或 scheduler 事件验证。

## 6. 仍缺什么计时点

若现有指标已经发现 `get_finished`/save wait 在 ON 下变长，增加少量关联日志即可：统一时钟、rank、request_id、layer、task_id，以及 enqueue、dispatch、first/last copy submit、sync complete、backend complete、worker wait begin/end。任务可覆盖多个请求，不能把整段等待重复分摊给每个请求后求和。

GDR 诊断还需要 stream 入队→出队、前置 event、MR 准备、post→CQ、EAGAIN 次数/等待和在途数。本分支 `UCM_GDR_TRACE` 只覆盖成功完成的 WR 及其部分区间，不能观察之前的 stream 排队与 event，也不记录一直未完成的 WR。短诊断可用：

```bash
# 服务启动前设置，正式性能测试取消并重启。
export UCM_GDR_TRACE=1
# 运行一轮短诊断后：
python tools/gdr_trace_summary.py worker.log > gdr-summary.json
```

CUDA 侧用 Nsight Systems 查看实际 memcpy bytes、GPU start/end、API submit、stream wait 和 kernel overlap；或在批次边界加 CUDA event。**不要在每次 memcpy 后同步**，那会改变原本的流水线。通用诊断入口示例（将现有启动命令原样放进 `start_vllm.sh`，需保持前台进程）：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none -o ds16k_off bash start_vllm.sh
# 只执行短诊断 workload，结束实验服务后分析生成的报告：
nsys stats --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum ds16k_off.nsys-rep
```

先确认两个 GPU worker 的 CUDA 活动都被收集。汇总表看 bytes/次数/时长，跨 stream 有重叠时，物理活跃带宽还需用时间区间并集，不能把 duration 求和当墙钟；纯 RDMA 数据搬运不会自动显示成 CUDA memcpy。Nsight 会产生开销，结果用于定位，正式性能在关闭 profiler 后测。依据：[NVIDIA Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

优先交付的实验数据是：H/E/E-control 各 ON/OFF 的单轮 before、returned、drained 快照和逐请求延迟；其次是外部命中固定后的 8K/16K/32K ON/OFF。它们能先回答来源、实际数据量、队列和 worker 阻塞是否变化，再决定是否需要方向分流、MR、QP 或多 NIC 的代码实验。
