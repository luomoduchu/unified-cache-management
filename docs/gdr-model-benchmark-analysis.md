# Qwen3-32B 与 DeepSeek-V2-Lite-Chat 的 GDR 实测复核

日期：2026-11-04。

数据来源：[用户提供的分享页](https://chatgpt.com/share/6aeaa0a9-aa1c-83ec-a2a8-7b13b198c150)，以及本次对话补充的原始表头和硬件信息。已读取分享页全部可见消息，包括 DeepSeek 的 18 条逐次记录、启动命令、UCM 配置，以及 Qwen 的汇总表。**分享页未包含 Qwen 的逐次原始表；Qwen 数值来自页面中已有的 AI 汇总，不能当作已重新审计的原始结果。**

完整数字转录见 [JSON 数据](data/gdr-model-benchmark-data.json)。服务端地址省略，其余 DeepSeek 数据列均保留；字段名依据用户补充的表头映射。其中 `reported_OutputTokenThroughput` 对应无空格的原始列名 `OutputTokenThroughput`，不与后面的整体 `Output Token Throughput` 混用，前者公式尚未核实。代码检查基线为本仓库 `477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2`。目标服务器安装的 UCM commit 未提供，因此下文源码机制需要与服务器版本核对。

## 1. 当前证据支持的结论

1. **两个模型的 16K 测试走了不同的缓存读取路径。** DeepSeek 16K 是 HBM 命中约 90%、外部命中 0；Qwen 汇总中的 16K 是 HBM 命中 0、外部命中约 90%。不能用这两组结果直接推导“GDR 对 GQA 有益、对 MLA 有害”。
2. **DeepSeek 16K 是需要修复或解释的端到端回退。** 虽然它不适合衡量外部 KV 的 H2D 读取加速，但不能排除 GDR 的 D2H 保存、线程调度及同步影响。三轮均明显回退，不能因外部命中为 0 而忽略。
3. **当前源码中存在可定位的候选阻塞链：慢 D2H → pending dump → `get_finished()` 等待 → 模型 worker 推进受阻。** 这比单纯猜测“小模型计算少”更具体，但仍需要时间线证明它造成了本次回退。
4. **DeepSeek 32K 同时有常态加速和一次严重慢运行。** 保留全部三轮后，整体完成吞吐约下降 50%；不能直接丢弃第三轮、只报告中位数加速。
5. **DeepSeek 64K 三轮方向一致，TTFT 平均下降 7.22%、E2EL 下降 7.70%。** 这是当前数据中相对一致的收益，但三轮样本仍不足以证明长期稳定。
6. **96 KB 是微基准中的单次 IO 交叉点；16K/32K/64K 是请求 token 数。** 本次数据没有 WR 粒度、实际 H2D/D2H 字节数或传输时间线，不能据此实现可靠的大小分流阈值。

## 2. 测试环境与配置

### 2.1 硬件与实际使用范围

| 项目 | 已提供信息 |
|---|---|
| 整机 | 8 张 H100；每张显存容量、CPU/NUMA 型号未给出 |
| 此服务使用 GPU | `CUDA_VISIBLE_DEVICES=2,3`，TP=2；不是单个模型同时使用 8 卡 |
| NIC 映射 | `DEVICE_GDR_NICS=mlx5_4,mlx5_6` |
| GPU/NIC 路径 | 涉及 PCIe 4；每跳链路宽度、ACS/IOMMU、完整拓扑未提供 |
| 最近单 NIC 微基准 | 20 GB/s |
| 最近两 NIC，共享上游 | 合计 25 GB/s |
| 最近加远端 NIC | 合计 37 GB/s；方向、聚合方式待确认 |
| 微基准交叉点 | 小于约 96 KB，RDMA 更快；大于则 CUDA memcpy 更快；方向、队列深度、KB/KiB 口径未提供 |

按当前源码的可见 CUDA 设备编号索引规则，预期逻辑 GPU 0 → `mlx5_4`，逻辑 GPU 1 → `mlx5_6`。应由启动日志和设备 PCI bus ID 核对物理 GPU 映射。**这表示每个 GPU 一个 NIC，不表示同一 GPU 同时使用两张 NIC。** 25/37 GB/s 不能直接作为当前单 GPU 传输应达到的带宽。

### 2.2 页面给出的 DeepSeek 启动设置

| 参数 | 值 |
|---|---|
| 模型路径 / 服务名 | `/models/DeepSeek-V2-Lite-Chat` / `DeepSeekv2` |
| `UCM_LOG_LEVEL` | `info` |
| `ENABLE_UCM_PATCH` / `PYTHONHASHSEED` | `1` / `0` |
| `--tensor-parallel-size` | 2 |
| `--max-model-len` | 65536 |
| `--port` | 9000 |
| `--max-num-seqs` | 64 |
| `--gpu-memory-utilization` | 0.5 |
| GPU Prefix Caching | `--enable-prefix-caching` |
| Chunked Prefill | `--enable-chunked-prefill` |
| KV Connector | `UCMConnector` |
| 模块 | `ucm.integration.vllm.ucm_connector` |
| `kv_role` | `kv_both` |
| 配置文件 | `/unified-cache-management/examples/ucm_gdr_config.yaml` |
| 日志 | `/vllm-workspace/logs/vllm_ucm_时间戳.log`，stdout/stderr 合并 |

用户说明 Qwen 启动命令“基本类似”，没有给出独立完整命令，不能认定全部选项逐字相同。页面标题中的 `0.27h100` 不能独立证明 vLLM 的精确版本。两种模型使用同一 UCM，但具体 commit、已安装动态库与源码是否一致仍未知。

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Cache|Posix
      storage_backends: /zsj/data
      posix_capacity_gb: 10240
      posix_io_engine: psync
      posix_data_trans_concurrency: 128
      io_direct: true
      cache_load_backend_only: true
      use_gdr: false  # ON/OFF 唯一主动切换项
      cache_io_aggregation: false
      cache_sdma_direct: false
```

配置中的两点边界：

- `io_direct: true` 控制 Posix 的 O_DIRECT，不能绕过 vLLM GPU Prefix Cache，也不能绕过 UCM 自身 DRAM Cache。
- 本仓库 CUDA 构建使用 `cache/v2`。`cache_load_backend_only` 只在 `cache/cc` 读取，v2 不消费这个键。**若服务器使用本分支版本，这个配置不会强制从 SSD 读取。** 即使使用支持该键的旧 Cache，它也无法绕过更上层的 HBM 命中。

### 2.3 压测口径

用户补充表头确认：`num_requests=32`、`concurrency=32`、目标 `hit_rate=90`、`output_len=2`，每种长度每个模式 3 轮。DeepSeek 的 `Total Requests=32`、`Success Requests=32`、`Failed Requests=0`、`Total Generated Tokens=64`，所有 18 轮一致。服务的 `max-num-seqs=64` 是上限，不是这次客户端并发量。

DeepSeek 实际平均输入分别约 16014.5、32014.6、64014.6 tokens，不是 16384/32768/65536。两种模式同轮的总输入 token 数完全一致。增加的约 14.6 tokens 的来源未给出，可能与请求包装有关，不能擅自扣除。

尚缺：压测工具/版本、请求到达时间、逐 token 流事件、warmup/重启顺序、实际 KV dtype、block size、attention backend、`max-num-batched-tokens`、Cache 容量、CPU/NUMA 绑定、trace 开关状态。默认项可能随模型和软件版本变化，不能把表头中的并发 32 误当 block size。

## 3. 数值复核

以下 ON/OFF 均指开启/关闭 GDR；延迟是各轮报表均值再取算术平均。百分比统一为 `(ON/OFF - 1) × 100%`：延迟负数为改善，吞吐正数为改善。

### 3.1 Qwen：页面已有汇总，尚缺逐次原始记录

| 输入 | HBM / 外部命中 | TTFT ON / OFF (s) | TTFT 变化 | E2EL ON / OFF (s) | E2EL 变化 | 总 token/s ON / OFF | 吞吐变化 |
|---|---|---|---|---|---|---|---|
| 8K | 0 / 89.82% | 5.331 / 6.707 | -20.52% | 6.313 / 7.986 | -20.95% | 35353.23 / 28459.34 | +24.22% |
| 16K | 0 / 89.91% | 15.886 / 17.915 | -11.33% | 17.995 / 20.105 | -10.49% | 20658.34 / 18872.00 | +9.47% |
| 32K | 0 / 89.95% | 29.809 / 26.045 | +14.45% | 31.773 / 27.969 | +13.60% | 20767.61 / 22524.87 | -7.80% |

其余 Qwen 汇总列（TPOT、ITL、duration、request/prefill/input/output throughput）全部保存在 JSON 的 `qwen_reported_means`。页面称 32K 第一轮 GDR 更快、后两轮更慢，但本分享未展示对应原始数值，无法重新计算其分布。

### 3.2 DeepSeek：缓存命中来源

| 输入 | HBM 命中 | 外部命中 | 命中计数 / 查询计数范围 |
|---|---|---|---|
| 16K | 89.92% | 0 | HBM 460800 / 512463～512472 |
| 32K | 0 | 89.96% | 外部 921600 / 1024466～1024469 |
| 64K | 0 | 89.98% | 外部 1843200 / 2048463～2048474 |

这些查询计数与 `Total Input Tokens` 相同，命中计数又分别等于 `32 × input_len × 90%`，数值上符合 **token 计数**。不能沿用页面 AI 回复中的“460800 个 HBM blocks”说法；它们不是已验证的物理 KV block 或 WR 数量。

### 3.3 DeepSeek：全部逐次核心结果

| 输入 | GDR | 轮次 | running_max | E2EL ms | TTFT ms | TPOT ms | ITL ms | 总耗时 s | req/s |
|---|---|---|---|---|---|---|---|---|---|
| 16K | ON | 1 | 16 | 4762.1 | 3621.9 | 1140.3 | 380.1 | 11.0957 | 2.8840 |
| 16K | ON | 2 | 20 | 4905.8 | 3869.9 | 1035.9 | 345.3 | 9.8464 | 3.2499 |
| 16K | ON | 3 | 23 | 5282.5 | 4149.7 | 1132.8 | 377.6 | 10.6438 | 3.0064 |
| 16K | OFF | 1 | 2 | 2153.1 | 2118.2 | 34.9 | 11.6 | 3.8896 | 8.2271 |
| 16K | OFF | 2 | 0 | 1507.7 | 1505.7 | 2.0 | 0.7 | 2.9683 | 10.7806 |
| 16K | OFF | 3 | 2 | 1519.9 | 1517.7 | 2.2 | 0.7 | 2.9698 | 10.7750 |
| 32K | ON | 1 | 20 | 21394.8 | 19506.2 | 1888.6 | 629.5 | 30.8394 | 1.0376 |
| 32K | ON | 2 | 20 | 24445.6 | 21979.2 | 2466.4 | 822.1 | 37.9007 | 0.8443 |
| 32K | ON | 3 | 14 | 105199.6 | 92965.9 | 12233.7 | 4077.9 | 175.4144 | 0.1824 |
| 32K | OFF | 1 | 6 | 29730.3 | 27394.2 | 2336.1 | 778.7 | 40.2764 | 0.7945 |
| 32K | OFF | 2 | 6 | 30245.0 | 27910.0 | 2334.9 | 778.3 | 40.6171 | 0.7878 |
| 32K | OFF | 3 | 7 | 30404.4 | 28046.3 | 2358.1 | 786.0 | 41.0656 | 0.7792 |
| 64K | ON | 1 | 4 | 49007.4 | 47191.0 | 1816.4 | 605.5 | 65.1271 | 0.4913 |
| 64K | ON | 2 | 4 | 52947.0 | 50865.0 | 2082.0 | 694.0 | 72.0005 | 0.4444 |
| 64K | ON | 3 | 4 | 60310.1 | 57996.2 | 2313.9 | 771.3 | 80.8407 | 0.3958 |
| 64K | OFF | 1 | 4 | 58224.6 | 55678.8 | 2545.8 | 848.6 | 85.4920 | 0.3743 |
| 64K | OFF | 2 | 4 | 55469.6 | 53127.0 | 2342.6 | 780.9 | 80.9139 | 0.3955 |
| 64K | OFF | 3 | 3 | 62109.6 | 59395.2 | 2714.4 | 904.8 | 90.2725 | 0.3545 |

`running_max` 的全名是 `vllm:num_requests_running_max`。16K ON 为 16/20/23、OFF 为 2/0/2，是需要核查的调度/采样差异。OFF 第二轮成功完成 32 个请求却报告最大运行数 0，说明采样窗口或指标采集口径必须核对；不能把 0 解释为模型没有执行，也不能用该列直接证明同一时刻 GPU 上的实际 batch size。

### 3.4 DeepSeek：三轮均值

| 指标 | 16K ON / OFF | 32K ON / OFF | 64K ON / OFF |
|---|---|---|---|
| E2EL ms | 4983.47 / 1726.90 | 50346.67 / 30126.57 | 54088.17 / 58601.27 |
| TTFT ms | 3880.50 / 1713.87 | 44817.10 / 27783.50 | 52017.40 / 56067.00 |
| TPOT ms | 1103.00 / 13.03 | 5529.57 / 2343.03 | 2070.77 / 2534.27 |
| ITL ms | 367.67 / 4.33 | 1843.17 / 781.00 | 690.27 / 844.77 |
| 总耗时 s | 10.53 / 3.28 | 81.38 / 40.65 | 72.66 / 85.56 |
| Request Throughput req/s | 3.0468 / 9.9276 | 0.6881 / 0.7872 | 0.4438 / 0.3748 |
| Prefill Token Throughput token/s | 4139.69 / 9582.90 | 1147.40 / 1152.41 | 1239.60 / 1144.14 |
| Input Token Throughput token/s | 48793.10 / 158986.07 | 22030.00 / 25201.86 | 28414.57 / 23989.87 |
| Output Token Throughput token/s | 6.09 / 19.86 | 1.38 / 1.57 | 0.89 / 0.75 |
| Total Token Throughput token/s | 48799.19 / 159005.93 | 22031.38 / 25203.43 | 28415.46 / 23990.62 |

16K 的 TTFT 增加 126.42%，E2EL 增加 188.58%，逐轮吞吐均值下降 69.31%。64K 的 TTFT 下降 7.22%，E2EL 下降 7.70%，总 token 吞吐均值提升 18.44%。

### 3.5 32K 的慢运行必须保留；平均吞吐有两种不同含义

三轮 GDR ON 的总耗时为 30.839 / 37.901 / 175.414 s；OFF 为 40.276 / 40.617 / 41.066 s。前两轮 ON 的 TTFT 分别更低 28.79%、21.25%，第三轮却从 OFF 的 28.046 s 增至 ON 的 92.966 s。

| 统计量 | GDR ON | GDR OFF | ON 相对 OFF |
|---|---|---|---|
| TTFT 均值 | 44.817 s | 27.784 s | +61.31% |
| TTFT 中位数 | 21.979 s | 27.910 s | -21.25% |
| 逐轮 req/s 算术平均 | 0.6881 | 0.7872 | -12.59% |
| 三轮总请求 / 三轮总耗时 | 0.3932 req/s | 0.7871 req/s | **-50.05%** |

最后一行按 `96 / Σ duration_seconds` 计算。两个吞吐指标回答不同的问题；页面的 -12.59% 算术正确，但会低估把这三轮顺序执行时慢运行对总完成效率的影响。中位数用于描述典型运行，不能据此删除第三轮，更不能将它直接认定为无关系统噪声：它也可能由 GDR 路径本身触发。

64K 三轮 TTFT 收益依次为 15.24%、4.26%、2.36%，方向一致但幅度收窄；不能据此断言随时间退化，也不宜称为已证明稳定。三轮合并完成吞吐提升约 17.76%，与逐轮总 token/s 均值的 18.44% 口径不同。

## 4. 为什么两个模型呈现不同现象

### 4.1 最直接的已知差异是 HBM 驻留，而非 RDMA 的单次粒度

在 BF16 KV、常规 TP=2、压缩 MLA、没有额外上下文并行分片等假设下：

- Qwen 每 GPU 每 token KV = `64 层 × 2(K/V) × (8/2) KV heads × 128 × 2 bytes` = **128 KiB**。
- DeepSeek 压缩 MLA 每 GPU 每 token KV = `27 层 × (512 + 64) × 2 bytes` = **30.375 KiB**；该 latent cache 在 TP ranks 间复制，不能简单再除以 2。

模型维度依据 [Qwen 官方配置](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)、[DeepSeek 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/raw/main/config.json)。压缩 MLA 的缓存维度与复制投影可参考 [vLLM 实现](https://docs.vllm.ai/en/v0.9.2/api/vllm/model_executor/models/deepseek_v2.html)；该参考版本不代表用户实际安装版本。

由此推算，32 个独立 16000-token 请求的完整逻辑 KV，每 GPU 分别约 62.5 GiB 与 14.83 GiB。这里不是实际同时驻留量：共享前缀去重、调度分批、缓存精度和后端布局都会改变实占空间。DeepSeek 更小的 KV 和模型权重，与 16K 仍可 HBM 命中、32K 转外部命中的现象相符。`gpu-memory-utilization=0.5` 是总显存预算比例，不能保证两个模型有相同的 KV 容量。

所以 Qwen 16K 有大量外部 KV 恢复可以从 GDR 获益，而 DeepSeek 16K 的 HBM 命中部分没有相同的 H2D 加速空间。**这解释收益机会的不同，但还不足以解释 DeepSeek 增加几秒的开销。**

### 4.2 DeepSeek 16K：优先排查保存完成等待，而不是外部读取带宽

从原始数据直接计算：

| 指标 | GDR ON | GDR OFF | 增量 |
|---|---|---|---|
| TTFT 均值 | 3880.50 ms | 1713.87 ms | +2166.63 ms |
| E2EL - TTFT | 约 1102.97 ms | 约 13.03 ms | **约 +1089.93 ms** |

只输出 2 tokens，首 token 后却出现约 1.1 s 的额外等待，不能用“省下的 H2D 数据较少”解释。当前源码给出一个可检验的机制：

1. `_generate_dispatch_meta()` 仍为新计算的完整 prompt blocks 生成 dump。约 90% HBM 命中不等于没有剩余 prompt KV 要保存。
2. `save_kv_layer()` 按层提交保存；Cache2 `DumpQ` 将 GPU KV 收集到 host buffer，再提交 Posix 写。
3. `use_gdr=true` 同时影响 load 和 dump。H2D 使用 RDMA WRITE；D2H 使用 RDMA READ，且当前每 QP 的 `max_rd_atomic=1`、`max_dest_rd_atomic=1`。
4. `wait_for_save()` 通常先检查已完成任务，不能仅凭方法名断定它每次同步等待。但是 **`get_finished()` 对已结束且有异步 dump 的请求，会直接调用 `_wait_pending_dump_task()`，进而 `wait_dump()`**；此处没有先做非阻塞完成检查。
5. 如果 D2H 落后于生成速度，模型 worker 可能在这里等保存完成，影响后续迭代以及其他请求。短输出、小模型、chunked prefill 和 32 并发会改变可用于重叠的时间窗口。

这是一条源码支持的候选阻塞链，不是已经测到的因果链。其实际位置相对于客户端首/末 token 的发送取决于 vLLM 版本。也不能把 1.1 s 全部当作 dump 时间：服务调度、输出流缓冲、CPU 竞争也会贡献。

Cache2 的 dump 完成条件是 D2H 完成且后端写入已提交，并非每次等待 SSD 写完；但后端慢写仍可占用 buffer slots、压住队列并间接形成反压。应同时测 D2H、队列和后端，不能简单归结为同步落盘。

若 trace 显示本轮 D2H 字节数也接近 0，则保存假设应降级，转查 GDR 空闲线程、CPU affinity、计时/采样窗口和其他同机负载。若 D2H 明显存在且 `get_finished()` 时间与慢请求重合，才有依据继续做方向隔离。

### 4.3 MLA 默认共享缓冲带来另一个需要控制的变量

当前 connector 使用 `config.setdefault("share_buffer_enable", self.is_mla)`。同样省略该配置时，Qwen GQA 与 DeepSeek MLA 的运行配置可能不同。MLA 的 dump 按绝对 block 位置分给 TP ranks，load 可以访问共享缓冲的不同 rank 段。

这会改变 host NUMA → NIC → GPU 路径、MR 注册范围和跨 rank 等待。即使 NIC 最近，数据所在的 host 内存也不一定本地。它是模型间差异的候选放大因素；同一模型 ON/OFF 仍使用相同共享策略，所以不能仅凭此默认值就断言根因。

### 4.4 Qwen 32K 和 DeepSeek 32K/64K 不能用同一条“长请求”规则解释

Qwen 32K 的外部读取量可能更大，单 NIC 持续带宽、WR 提交速率、队列与后端资源会更重要。DeepSeek 32K/64K 开始具备外部恢复的收益空间，而 32K 第三轮表现出另外的长尾问题。

固定模型、KV dtype、block size 和 tensor layout 时，输入变长通常首先增加 WR 个数，而非每个 WR 的 bytes。若每 WR 大小不变，简单模型 `N × (固定开销 + bytes/带宽)` 中只增加 N 并不能使两条路径排序翻转；必须进一步看到饱和、排队、重叠、缓存来源或运行状态的变化。

## 5. 压测指标需要澄清的三个问题

1. **TPOT 与 ITL 相差约 3 倍。** DeepSeek 18 条记录全部如此，Qwen 汇总也出现同样关系。输出 2 tokens 时，常规 token 间隔口径下只有一个有效间隔，不应自然出现固定 3 倍。可能有空 SSE chunk、角色/终止事件或不同分母，但没有压测源码不能定论。现阶段直接用 `E2EL - TTFT` 描述首 token 后的客户端耗时，不把 ITL 当作纯 GPU decode latency。
2. **多项指标并非独立证据。** DeepSeek 的 `Prefill Token Throughput` 数值近似等于 `InputTokens / TTFT_seconds`，其 input 包含命中前缀，并非 GPU 真正重新计算的 token/s。`Concurrency` 数值又近似等于 `Request Throughput × E2EL_seconds`，不能当独立测量的 server batch size。输入/输出/总 token throughput 在 token 数相同的情况下也主要是总耗时的不同缩放。
3. **running_max 是观测指标，采样必须对齐。** OFF 16K 某轮 running_max=0 而请求成功，已经提示 scrape 周期或窗口问题。ON 的较大 running_max 可能是请求滞留更久的结果，也可能反映批处理变化；仅凭最大值不能确定因果方向。

## 6. 下一轮最有区分度的定位顺序

### 6.1 先复现 DeepSeek 16K，不先调整 96 KB 阈值

固定 32 请求、并发 32、输入 16000、输出 2，以及相同 HBM/外部命中。每轮确认前一轮保存已结束；交替或平衡顺序执行 ON/OFF，记录绝对延迟、逐请求时间线及运行顺序。正式性能测试关闭 `UCM_GDR_TRACE`；单独做短诊断运行。

| 观察或实验 | 要回答的问题 |
|---|---|
| 分别统计 `dir=H2D` 与 `dir=D2H` 的 WR 数、bytes、耗时 | 外部命中 0 时，是否实际有 GDR 保存？ |
| 记录 `get_finished()` 进入/退出、pending dump 数、每次 `wait_dump()` 时间 | 是否在等待保存完成，是否与客户端长等待重合？ |
| 看 `cache_dump_queue_wait_duration_ms`、`cache_d2h_duration_ms`、后端写入等待 | 慢在传输、排队还是后端反压？ |
| 固定驻留状态，诊断构建临时不提交新 dump | 去掉保存后是否还回退？此实验改变缓存写入语义，只用于定位 |
| 分方向比较 H2D GDR / D2H CUDA 与全 CUDA、全 GDR | 是否为 D2H 专属问题？当前单一 `use_gdr` 开关尚不支持此矩阵，需另加诊断实现 |
| 并发 1/8/32，采集 CPU 线程状态、NUMA、NIC/CQ 计数 | 固定开销还是高并发放大？ |

不要直接删除完成等待：它保护 GPU KV blocks 在复制完成前不被回收复用。若确认阻塞，应研究正确的异步完成通知或方向选择，保留生命周期保证。

### 6.2 再验证外部读取，以及真正的 SSD 读取

将 GPU Prefix Cache、UCM DRAM Cache、Posix 后端区分开。需要研究外部 H2D 时，可在固定预热数据后让两组都不命中 GPU prefix，并确认 UCM 外部 hit 与 H2D bytes；这并不自动保证 SSD 读取。

要研究 SSD，还需观察 `cache_load_backend_shards_total`、Posix 实际读字节/IOPS 和 `cache_shard_backend_wait_ms`。当前 Cache2 不能通过现有 `cache_load_backend_only` 开关强制后端读取；需使用经验证的驱逐流程或为 v2 实现明确的诊断开关。不能通过直接移除 Cache、把 GPU 地址交给 Posix 来替代 staging。

### 6.3 保留并重现 32K 第三轮

同步记录 request admission、GPU preemption/重算、CPU run queue、UCM load/dump 队列、Posix 等待、MR 注册、CQ 完成、NIC 错误/重试以及实际缓存来源。先判断超长空档发生在哪一段，再决定是否改 QP 深度、线程调度、注册或传输选路。

当前 trace 只记录成功 CQ 完成的 WR，`begin_ns` 不含之前的 stream 排队、CUDA event 等待和 EAGAIN 重试。没有完成的 WR 也没有完成记录。因此“trace 内 post→done 很短”并不能排除 GDR 调度瓶颈。需要把模型/connector/queue 时间线与 trace 对齐。

已有工具可用于短诊断日志：

```bash
# 在启动服务前设置，确保所有 worker 继承；保留 stderr。
export UCM_GDR_TRACE=1
python tools/gdr_trace_summary.py deepseek-16k-on.log > deepseek-16k-on-summary.json
# 正式 A/B 测试前取消变量并重启服务。
unset UCM_GDR_TRACE
```

工具详情与计时限制见 [GDR 观测说明](gdr-observability.md)。默认 96 KiB = 98304 bytes；若微基准中的 96 KB 是十进制，改用 `--threshold-bytes 96000`。本次没有目标服务器运行日志，未测得实际 GB/s 或 WR 粒度。

## 7. 源码定位

以下链接固定在本次检查的提交，不暗示服务器已安装该版本。

- [CUDA Cache 版本选择](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/store/cache/CMakeLists.txt#L1)：CUDA → v2。
- [仅旧 Cache 读取 backend-only](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/store/cache/cc/cache_store.cc#L141)；[v2 命中时跳过后端读取](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/store/cache/v2/load_queue.h#L166)。
- [MLA dump 分工](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/integration/vllm/ucm_connector.py#L1835)；[共享缓冲默认值](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/integration/vllm/ucm_connector.py#L1874)。
- [load/dump 元数据分开生成](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/integration/vllm/ucm_connector.py#L2283)。
- [完成阶段等待 dump](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/integration/vllm/ucm_connector.py#L2810)；[层级保存中的非阻塞检查](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/integration/vllm/ucm_connector.py#L3130)。
- [D2H 完成与后端提交](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/store/cache/v2/dump_queue.h#L226)。
- [QP READ 并行度](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/shared/trans/cuda/gdr/gdr_copy.cc#L118)；[H2D WRITE / D2H READ](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/shared/trans/cuda/gdr/gdr_copy.cc#L698)。
- [NIC 列表按可见设备 ID 索引](https://github.com/luomoduchu/unified-cache-management/blob/477ceffe3d84c07fb2c32d5ad5df2875cdc8daa2/ucm/shared/trans/cuda/gdr/gdr_config.cc#L60)。

本次新增的是数据复核与定位文档，没有改变 GDR 运行逻辑，也没有将候选原因当作已复现的根因。
