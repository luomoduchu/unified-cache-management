# GDR 的 18 KiB WR、CUDA 复制粒度与保存队列瓶颈

日期：2026-11-04。源码基线：`dc30415d39b4dd9756fcfd144a9abaab6fa1f524`。

后续用户补充：DeepSeek 16K 仅关闭前缀复用后，HBM 命中 0、外部命中 89.92%，ON 三轮 TTFT 平均改善 26.76%、E2EL 改善 28.37%、总吞吐均值提升 39.40%。详见[模型对比第 3.6 节](gdr-model-benchmark-analysis.md)和[六条原始记录](data/gdr-deepseek-16k-external-data.json)。本页原有 WR/队列快照不与这六轮新测试自动对应；以下运行建议已按缓存路径更新。

依据[新分享页](https://chatgpt.com/share/6aead877-0038-83ec-bff5-64b0c5f9423c)后段关于粒度、GDR metrics 和关闭 GDR 基线的讨论，以及本仓库源码。页面中的原始监控以附件出现；本文采用页面回答展示的计数和表格，**尚未独立取得并审计附件中的完整 Prometheus 原始序列**。下文将页面报告值、源码事实、推断分开，数值转录见 [JSON](data/gdr-wr-and-ce-baseline-data.json)。

## 1. 结论与证据边界

- 新页面报告 DeepSeek 的 H2D/D2H **平均 WR 都是 18 KiB**，大小直方图全部落在 `(16, 32] KiB`。这与 BF16 MLA、16-token block、每层一个 latent tensor 的计算一致。桶和均值不能单独证明每一个 WR 都恰好等于 18432 bytes。
- 关闭 GDR 的 Cache 统计是 **20 KiB/shard**，但这不是一次 `cudaMemcpyAsync` 的已测长度。源码将存储 shard 向 4096 bytes 对齐，传输仍使用未填充的 `tensor_size_list`。在相同布局下，18 KiB payload 与 20 KiB 存储计数完全相容。
- 关闭 GDR 时，Load 平均任务耗时约 **39.8 ms**；Dump 约 **1147.22 ms**，其中排队 **1132.20 ms，占 98.69%**。当前最明确的开销位置是 Cache Dump 的调度队列。它是关闭 GDR 的观测，不能用来证明开启 GDR 导致了此前 DeepSeek 16K 的回退。
- 关闭 GDR 的约 **12 μs H2D Sync** 是提交完 shard 后的剩余等待；GDR 的 **38–53 μs** 是页面报告的逐 WR completion 统计。两者不能作直接快慢比较，目前仍缺 CUDA 单次复制的完整执行时间。
- 是否开启 GDR，应取决于外部 KV 恢复的实际收益、D2H 保存和队列开销、CPU 调度及重叠程度。**小于 96 KiB 是候选条件，不是端到端加速保证。**

## 2. 新页面的环境与数据范围

配置问答中 JSON 的 `model` 写了 Qwen3-32B，但同一段实际启动命令是 `/models/DeepSeek-V2-Lite-Chat`，服务名 `DeepSeekv2`，后续监控也标注这个服务名。因此本次 18/20 KiB 数据按 DeepSeek 分析，不能当作 Qwen 的实测 WR。

该段明确提供 TP=2、PP=1、BF16、block size=16、blocks_per_chunk=1。后两项没有在启动 YAML 中显式设置，应通过运行日志确认。GPU 为物理 2/3，NIC 列表 `mlx5_4,mlx5_6`；Cache|Posix、psync、Posix 并发 128、O_DIRECT；aggregation 和 sdma_direct 均关闭。GPU prefix caching、chunked prefill 开启，max-num-seqs=64，显存预算比例 0.5。完整背景见[此前模型对比](gdr-model-benchmark-analysis.md)。

最后十个提问依次讨论配置推算、请求长度、接口 bytes 来源、运行参数、首批 GDR 数据、OFF 指标、追加 GDR 数据、Counter 差值、Cache 指标查看、OFF 四组 workload。**其中采集命令和建议不属于新的实验结果。**

新 GDR 快照先描述“8K/32 并发及三组 32K/32 并发”，随后追加“8K/32 并发及三组 16K/32 并发”。没有给出逐组 before/after 数值，不能把后一张累计快照全部归于新增 16K。

最后一条明确提供的 OFF workload 是：

| 组 | input_len | output_len | 请求数 | 目标命中率 | 并发 | seed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 128 | 16 | 90% | 16 | 140000 |
| 2 | 32000 | 2 | 32 | 90% | 32 | 15590 |
| 3 | 32000 | 2 | 32 | 90% | 32 | 35681 |
| 4 | 32000 | 2 | 32 | 90% | 32 | 29212 |

OFF 第一组实际是并发 16、输出 128；不能沿用前面“8K/32 并发”的文字，也不能与旧页 16K/输出 2 的结果直接拼接。目标命中率 90% 不代表已经验证的 HBM/DRAM/SSD 来源比例。

## 3. 实际 WR、存储 shard 与任务大小

### 3.1 GDR 数据

页面后一次 GDR 汇总报告：

| 方向 | 报告 WR 数 | 报告总量（约，十进制 GB） | 平均 WR | 直方图范围 |
|---|---:|---:|---:|---|
| H2D | 7,219,611 | 133.07 | 18 KiB | 全部在 `(16,32] KiB` |
| D2H | 5,664,573 | 104.41 | 18 KiB | 全部在 `(16,32] KiB` |

按均值反算，`WR 数 × 18432` 分别为 133,071,869,952 和 104,409,409,536 bytes，与页面四舍五入后的总量相容；这不是另一次独立测量。保留原 rank 口径，未将 H2D 数字再乘 2。两个 rank 的计数相同并不证明重复导出：MLA 的两个 GPU 可能确实各加载一份 latent cache，应核对 exporter/process/rank 后再决定是否合计。

模型公式：

```text
单层、单 token： (512 latent + 64 RoPE) × 2 bytes = 1152 bytes
单层、16 tokens：16 × 1152 = 18432 bytes = 18 KiB
27 层全模型：   27 × 18432 = 497664 bytes = 486 KiB / block / GPU
```

这里的 486 KiB 是完整逻辑 block 总量，不是本次看到的单 WR。MLA latent 在常规 TP=2 下复制，不能再把 18 KiB 除以 2。模型维度来源见[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/raw/main/config.json)。

本地 `Stream` 向量接口逐个调用 `sizes[i]`，GDR 每个非零片段形成一个 WR/SGE；没有按用户输入长度将所有 layers 合并为一个 WR。增加 input_len 通常增加 block/WR 数量。`tensor_size_list * blocks_per_chunk` 在 Python 中也是列表重复，并不自动把每项乘大或合并成一次复制。

Qwen 在相同 BF16、block=16、TP=2 下，单层 K+V 合计 32 KiB；如果 K/V 分开传输，可能是两个 16 KiB 片段，如果布局合并则可能是 32 KiB。**本次 DeepSeek 的 18 KiB 数据不能代替 Qwen 的实际 WR 分布。**

### 3.2 为什么关闭 GDR 看到 20 KiB

OFF 页面计数重算：

| 项目 | Load（每 rank） | Dump（页面所列序列） |
|---|---:|---:|
| Cache bytes | 95,551,488,000 | 110,699,827,200 |
| Cache shards | 4,665,600 | 5,405,265 |
| Task 数 | 2,592 | 10,962 |
| bytes/shard | **20,480 = 20 KiB** | **20,480 = 20 KiB** |
| 平均 shards/task | 1,800 | 493.09 |
| 平均 bytes/task | 35.15625 MiB | 9.63069 MiB |

源码中 `_get_store_io_sizes()` 默认 alignment=4096，而创建 Store 时分别设置：

```text
tensor_size_list = 未填充的 tensor 大小列表
shard_size      = 向 4096 bytes 对齐后的逻辑 shard 大小

ceil(18432 / 4096) × 4096 = 20480 bytes
```

`TransManager` 的 Cache bytes 指标采用 `shardSize × desc.size()`；`Stream` 和 `CudaStream` 的复制长度采用 `tensorSizes[i]`。因此，若服务器与此布局相同，实际 CUDA memcpy 仍可为 **18 KiB/次**，Cache 记账却为 **20 KiB/shard**。每 shard 多计的 2 KiB 是存储对齐空间，占 payload 的 11.11%，不代表 GPU 总线上多搬了 11.11%。

35.16/9.63 MiB 说明一个上层任务包含许多 shard，**不能解释为一次同等大小的 CUDA DMA**。关闭 `cache_io_aggregation` 后，当前普通路径仍逐 tensor 调用 `cudaMemcpyAsync`。

一个有用的数值交叉检查是：

```text
32000 × 90% / 16 = 1800 blocks / request
3 组 × 32 requests × 27 layers = 2592 tasks
2592 × 1800 = 4665600 shards
```

这恰好吻合 OFF Load 计数，与三组 32K 的 90% 外部恢复量相容。但仅靠累计总数，不能独立证明 8K 组没有 Load，也不能证明每个 task 都有 1800 shards。

## 4. 延迟应怎样比较

### 4.1 GDR completion

页面后一次累计统计如下，分位数是桶区间，不是精确数值：

| 方向 / rank | 平均 μs | P50 μs | P90 μs | P99 μs |
|---|---:|---|---|---|
| H2D / 0 | 38.38 | `(10,20]` | `(50,100]` | `(200,500]` |
| H2D / 1 | 42.21 | `(10,20]` | `(100,200]` | `(200,500]` |
| D2H / 0 | 53.00 | `(10,20]` | `(100,200]` | `(500,1000]` |

D2H 的均值和尾部更高，值得优先排查。这里没有同口径 CUDA 延迟，不能直接断言 D2H RDMA 比 CUDA 慢。也不能用 `18 KiB / 53 μs` 宣称 NIC 持续带宽只有约 0.35 GB/s：WR 可能并发，CPU completion 观测还包含调度/排队。

服务器自定义的 `gdr_*_rdma_completion_latency_us` 埋点代码未提供，本仓库没有同名 Prometheus 指标，不能擅自认定它和本分支 trace 的时间起点一致。本分支 trace 的 `post_ns → done_ns` 是 CPU 提交到 CQ 被轮询的间隔，仍不是纯硬件 DMA 时间。

### 4.2 OFF Load

| 指标 | rank 0 | rank 1 | 含义 |
|---|---:|---:|---|
| Load duration | 39.85 ms | 39.76 ms | Cache 任务耗时，含等待 |
| Queue wait | 18.57 ms | 18.49 ms | 等待 Load 调度 |
| Backend submit | 9.58 ms | 9.97 ms | 分配/查找 buffer、逐 shard 派发后端等循环 |
| H2D Sync | 12.2 μs | 12.1 μs | 最后提交后的剩余同步等待 |
| 平均任务带宽 | 1.193 GB/s | 1.196 GB/s | 每任务 bytes/duration 的统计 |

rank 0 的排队约占任务均值 46.60%。backend submit 不是 SSD 读取完成延迟，也不是纯 CUDA 复制耗时；其他流水级可能与它重叠。末尾同步很短，表明复制在前面的流水过程中可能已完成大部分，不能据此获得整次传输的起止时间。

### 4.3 OFF Dump：开销主要在哪里

| 指标 | 页面报告均值 |
|---|---:|
| Dump duration | 1147.22 ms |
| Queue wait | 1132.20 ms |
| MkBuf | 14.70 ms |
| Prerequisite wait | 0.205 ms |
| Backend submit | 0.238 ms |
| Backend wait | 4.325 ms |
| 平均任务带宽 | 0.388 GB/s |

在 duration 和 queue histogram 覆盖相同任务集合的条件下：

```text
1132.20 / 1147.22 = 98.69%    # 等待调度占比
1147.22 - 1132.20 = 15.02 ms # 出队后的任务耗时，仍不是纯复制
```

源码在 `DumpQueue::DispatchOneTask` / `DumpQ::DispatchOneTask` 开始处记录 `Now - waiter.startTp`，之后才执行本任务的 buffer 获取、D2H 和后端提交。因此 **1.13 秒发生在本任务被消费前的队列等待**。队列积压可能由前序任务复制慢、buffer/锁争用、CPU 调度慢或后端反压造成；队列指标定位了等待位置，没有区分这些根因。

MkBuf 包含 buffer 查找、所有权/就绪判断、复制提交循环；旧实现还从前置 event 入队前开始计时。它不是“分配内存花了 14.7 ms”的充分证据。旧实现的 prerequisite callback 时间可与 MkBuf 重叠，backend wait 又在独立线程运行，不能把这几个均值直接相加还原 1147.22 ms。

页面称 `cache_d2h_duration_ms` 只有 HELP、没有实际 count/sum；缺失不等于 0。即使有此指标，当前两套实现也主要测末尾 stream synchronize，仍不等于全部 D2H 设备执行时间。

### 4.4 这些数字还不能给出实际链路 GB/s

`cache_*_bandwidth_gbps` 当前源码按 bytes/seconds/1e9 计算，单位实际是 GB/s。页面报告的是任务比值的均值 `mean(B_i/T_i)`；`sum(B_i)/sum(T_i)` 与它不同，更不等于多个任务并发执行时的 `sum(B_i)/wall_time`。

例如用 OFF 页面均值反算，Load 得到约 0.925 GB/s，Dump 约 0.00880 GB/s，与 1.193/0.388 的任务带宽均值不同；这本身不证明计数错误。还需检查各指标 count 是否同一集合。Cache bytes 又可能包含对齐和已就绪 shard，不能当作硬件实际搬运字节。

实际业务窗口有效带宽需要 `Δ实际复制 bytes / Δ墙钟时间`。还应另报传输活跃时间内的带宽，明确是否包含 idle/排队；多 stream 的重叠区间不能重复计时，H2D 和 D2H 分开报告。

## 5. 96 KiB 以下为何仍可能劣化

此前“96 KB”的十进制/二进制单位尚未完全核实；这里按 96 KiB（98304 bytes）讨论，18 KiB 在两种口径下都远小于交叉点。实际分流阈值仍应以微基准使用的字节数为准。

微基准的小 IO 收益通常来自固定软件开销差异。对同一方向、相同排队条件，可用 `固定提交/完成开销 + bytes/有效带宽` 理解交叉点。数据小的时候，较低的固定开销可以胜过更高的大块持续带宽。UCM 的 RDMA 路径绕过每片段的 CUDA memcpy 调用，但增加自己的 scheduler、MR 查找、WR 提交和 CQ completion 工作，真实服务的净收益需要重测。

有四个不能省略的条件：

1. **方向一致。** 当前 H2D 是 RDMA WRITE，D2H 是 RDMA READ；每 QP 的 READ outstanding 参数为 1。之前展示的微基准是 `host_to_device_ce/gdr`，不能把 H2D 的 96 KiB 交叉点直接用于 D2H。
2. **同一计时口径。** 页面微基准有 Count=512 的批次。批次 Copy 时间、逐 WR completion、CUDA API 返回时间、CUDA event 时间、末尾 Sync 是不同指标。
3. **在请求关键路径上。** 外部命中为 0 时，没有同等的 H2D 恢复收益；如果 CUDA 复制已经与读取/计算充分重叠，缩短复制也未必缩短 TTFT。
4. **不被排队和保存抵消。** 更多 WR 可以使某个流水级饱和，或放大事件等待和完成长尾。固定 WR 大小下，请求变长不会自动“跨过 96 KiB”；性能反转应检查并发、负载、缓存来源与重叠变化。

普通 pinned-host `cudaMemcpyAsync` 同样可使用复制引擎，不需要默认假设它额外经过一次 CPU staging 或占用计算 SM。不能以“GDR 不用 SM”单独解释两者差异。

## 6. 开启和关闭分别在什么情况下有用

| 场景 | 当前建议 | 原因与条件 |
|---|---|---|
| 外部 KV 命中高，小 tensor，H2D 等待明显，NIC/MR/CPU 状态良好 | 优先测 GDR ON | 有可兑现的恢复收益，需同时观察 Dump 队列和尾延迟 |
| 大量 HBM 命中，几乎没有外部 H2D，短输出 | 优先以 OFF 为基线 | H2D 收益空间小，保存和调度开销更容易影响结果 |
| 同方向微基准已证实大连续复制超过交叉点后 CUDA 更快 | 优先测 OFF | 前提是实际复制变大，不能使用 request token 数代替 bytes |
| ON 时 D2H、事件等待、CQ 长尾或队列明显恶化 | 暂用 OFF，定位保存路径 | 小 WR 并不能补偿软件等待和反压 |
| SSD/CPU/Cache 队列主导，CUDA 复制已被重叠隐藏 | 先优化主导流水级 | 仅换传输引擎不一定改善 TTFT/E2EL |

结合此前同一模型的端到端结果，当前运行选择可暂定：

| 模型 / 输入 | 暂定选择 | 已有依据 |
|---|---|---|
| Qwen3-32B 8K、16K | ON | 旧页汇总 TTFT 约改善 20.52%、11.33% |
| Qwen3-32B 32K | OFF | 旧页汇总 ON 的 TTFT 约恶化 14.45% |
| DeepSeek 16K，旧 GPU 命中组，前缀复用开启 | OFF | 三轮 ON 的 TTFT/E2EL 均明显回退，HBM 命中约 90%、外部 0 |
| DeepSeek 16K，新外部命中组，前缀复用关闭 | **ON** | 外部命中 89.92%、HBM 0；三轮 TTFT 改善 26.76%、E2EL 改善 28.37% |
| DeepSeek 32K | 稳定性优先时 OFF | ON 前两轮更快，但第三轮出现严重慢运行 |
| DeepSeek 64K | ON 作为候选 | 三轮 TTFT 平均改善 7.22%，仍需更多重复验证 |

这些是已有 workload 下的选择，不是按模型名称或输入长度硬编码的通用路由规则。Qwen 原始逐次数据仍缺失；新页累计监控不能替代相同 workload 的 ON/OFF 配对实验。

## 7. 性能劣化的位置与下一步优先级

**已经定位的现象：本页旧 OFF 快照的 Dump 调度队列积压。尚未定位的因果：ON 相对 OFF 为何增加了旧 GPU 命中 DeepSeek 16K 的几秒延迟。** 新外部命中 16K 已有明确收益，不能再将回退泛化为模型或长度的固有性质；旧队列快照也不能直接解释新六轮的改善。

当前源码支持的候选传导链为：

```text
保存流水级服务变慢 / 资源争用
    → Dump 队列积压
    → pending dump 无法及时结束
    → get_finished() 中 wait_dump() 阻塞
    → 模型 worker 后续推进受影响
```

`get_finished()` 中存在同步等待，但实际是否影响本次客户端 TTFT/末 token，应由服务版本和时间线确认。新页的 OFF 1.13 s queue wait 不能用来解释旧页 ON 首 token 后增加的约 1.09 s：模式、workload、统计层级均不同，数值接近不构成因果证据。

最小定位顺序：

1. 固定服务器 UCM commit、加载的 `.so`、vLLM 版本和 Cache 实现。新页有 `cache_dump_prereq_wait_ms` 样本，而本地 v2 不记录它，旧 cc 才记录；这提示部署可能不同，不能仅凭本地 CUDA 默认 v2 就断言服务使用 Cache2。`cache_load_backend_only` 的有效性也必须按部署版本判断。
2. 同 workload 做 ON/OFF before/after；每轮前确认上一轮保存结束，分别保存 HBM/外部来源、rank、进程和时间窗口。正式性能对比关闭逐 WR 日志，并保持 metrics 开销可比。
3. 对齐请求、层、Load/Dump task：入队、出队、buffer 获取、首次/末次 copy 提交、同步完成、backend 完成、`get_finished()/wait_dump()`。优先检查队列等待是否在 ON 中进一步增长，以及 worker 等待是否与慢请求重合。
4. CUDA 侧记录 `cudaMemcpyAsync` 实参 bytes/count 和方向；API 调用计时只表示提交开销。完整设备时间使用短时 CUDA event 批次测量或 GPU 时间线，保留正常异步行为，不在每次复制后强制同步。
5. GDR 侧分别记录 stream 排队、event 等待、MR 准备、post→CQ、失败/重试与在途 WR。当前 trace 没覆盖前两段，不能因 WR completion 看起来正常就排除 GDR 软件队列。
6. 若证据指向 D2H，再做 H2D-only GDR / D2H-only GDR 的诊断分流。**当前配置没有这种方向开关，需实现并保证事件、buffer 生命周期和完成顺序正确。** 若证据指向 CPU/Posix 则先试并发或亲和性，若指向 MR 则修注册复用，不先盲目加 NIC。

`DEVICE_GDR_NICS=mlx5_4,mlx5_6` 是两个可见 GPU 各选一个 NIC，不是每 GPU 双 NIC；20/25/37 GB/s 微基准不能直接代入当前单 NIC 的业务结果。增加 NIC 也不会消除已经在 Cache 队列中等待的时间。

## 8. 源码依据

以下固定到本次复核基线；不暗示目标服务器安装了这个版本：

- [存储大小对齐](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/integration/vllm/ucm_connector.py#L167)；[tensor size 与 Store shard 分开配置](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/integration/vllm/ucm_connector.py#L1886)。
- [逐 tensor 复制](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/shared/trans/stream.h#L64)；[CUDA memcpy 实参](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/shared/trans/cuda/cuda_stream.cc#L63)。
- [Cache2 任务 bytes 与 bandwidth 计时](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/v2/trans_manager.h#L54)；[旧 Cache 对应计时](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/cc/trans_manager.h#L53)。
- [旧 Dump 队列与 MkBuf、prerequisite 计时](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/cc/dump_queue.cc#L98)；[Cache2 Dump](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/v2/dump_queue.h#L152)。
- [Load 派发计时](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/v2/load_queue.h#L152)；[末尾 H2D 同步](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/store/cache/v2/load_queue.h#L366)。
- [完成阶段等待保存](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/integration/vllm/ucm_connector.py#L2810)。
- [每 QP READ 深度](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/shared/trans/cuda/gdr/gdr_copy.cc#L118)；[单 SGE 的 READ/WRITE](https://github.com/luomoduchu/unified-cache-management/blob/dc30415d39b4dd9756fcfd144a9abaab6fa1f524/ucm/shared/trans/cuda/gdr/gdr_copy.cc#L688)。

本次更新的是分析和证据记录，没有更改传输逻辑，也没有在本地 Windows 上复现 H100 性能。
