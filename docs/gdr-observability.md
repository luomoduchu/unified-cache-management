# GDR 实测分析（learngdr）

## 已确认的代码路径

- 当前 CUDA 构建选用 Cache v2：`cache/v2/load_queue.h` / `dump_queue.h` → `CopyStream` → `GdrStream` → `GdrCopyChannelImpl` → verbs。`cache/cc` 是本提交为 ascend-a5 选择的另一套实现，不能将其专有配置直接套用到 CUDA。
- PC Store: `TransQueue` 按 `ioSize_` 和 shard 数调用 stream。不要把此路径的粒度直接套到 Cache Store。
- Cache Store 传递 `tensorSizes_`；实际 WR 粒度应以 trace 的 `bytes` 为准，而非请求 token 数、整个 KV block 或整个 shard 大小。
- `GdrStream` 的数组接口逐项提交；每次非零 copy 对应一个 WR、一个 SGE、一个 signaled completion，没有自动合并或双 NIC striping。大于 uint32 范围的 copy 被拒绝，不自动切分。
- H2D 使用 RDMA WRITE，D2H 使用 RDMA READ。QP 以自身 endpoint 连接，实现本机 Host/GPU 数据传输，不是跨服务器 KV 拉取的网络带宽。
- 每个 stream 有 scheduler/completion 线程，操作环容量 8192，scheduler 每批最多取 128 个操作（不是合并成一个 WR）；空闲退避有 100 us sleep。QP 请求发送深度最多 4096，实际在途上限由设备返回值减一决定。
- 预注册 MR 命中时复用；未命中先尝试注册 registry 中的整个已知缓冲并复用，无法解析时才按传输临时注册并在完成后释放。`fallback=false` 不排除首次注册整个缓冲的开销。
- `DEVICE_GDR_NICS` 按 CUDA device ID 索引选择一个 NIC；未设置则所有设备默认 `mlx5_0`。须注意 CUDA_VISIBLE_DEVICES 重映射。增加 stream 数仍使用同一个 NIC。
- Cache Store GDR 与 IO aggregation、SDMA Direct 配置互斥；比较开关时应记录完整配置。

## 启用和运行

在 Linux H100 机器按项目既有安装流程重新构建，构建环境包含 `PLATFORM=cuda ENABLE_GDR=1`，并启用 store 的 `use_gdr: true`。

```bash
export UCM_GDR_TRACE=1
# 根据实际 nvidia-smi topo -m / PCIe 拓扑填写八个 NIC，不能盲用示例映射。
export DEVICE_GDR_NICS='<GPU0 NIC>,<GPU1 NIC>,<GPU2 NIC>,<GPU3 NIC>,<GPU4 NIC>,<GPU5 NIC>,<GPU6 NIC>,<GPU7 NIC>'
# 用既有命令启动服务，确保 worker 继承以上变量，并保存 stderr（例如 2>&1）。
python tools/gdr_trace_summary.py run-8k.log > summary-8k.json
python tools/gdr_trace_summary.py run-16k.log > summary-16k.json
python tools/gdr_trace_summary.py run-32k.log > summary-32k.json
```

按同一机器、同一次运行传入日志；不要混合不同实验。每个结果按 pid/GPU/NIC/channel/方向分组。可传多个 worker 日志，重复记录会去重。不包含成功 CQ completion 的失败/未完成 WR 不计入统计；出现 GDR error 时该次实验不能当作完整吞吐结果。

默认分界点为 98304 字节（96 KiB）；如果微基准实际使用十进制 96 KB，传 `--threshold-bytes 96000`。

## 统计口径和限制

- `bytes`: 实际单个 WR 的 payload；直方图给出精确大小和次数，不是 RDMA packet/PCIe TLP 大小。
- `begin_ns`: 本次提交尝试在查找/注册 MR 前；`post_ns`: 发布 slot、调用 verbs 前；`done_ns`: 成功 CQ 轮询后、释放 MR 前。时钟为单调主机时钟。
- `prepare_mean_us`: 成功提交尝试的 MR 查找/注册、slot 获取等软件开销。不包括 stream 排队、之前的 EAGAIN 重试、CUDA event 等待、完成后的注销耗时。
- `post_to_poll_*`: 提交前到软件观察完成的延迟，包含排队、传输、completion 调度延迟，不能等同硬件 DMA 时间。
- `active_GBps`: 完成 payload 总字节 / 所有 post→done 区间的并集时间；正确处理并发区间，不累加重叠延迟。`span_GBps`: 总字节 / 首次 post 到最后 done 的时间，包含期间空闲。GB/s 为十进制。
- 两者均是 channel 的软件有效带宽，不是模型端到端 IO 带宽或硬件链路利用率。不能直接相加各 channel 的带宽得到 GPU 总带宽；需要在统一时间窗口合并区间和字节。
- `fallback_host_wr` / `fallback_gpu_wr`: 临时 MR 的 WR 数量；prepare 高且 fallback 多时优先检查预注册覆盖。
- trace 默认关闭，只有环境变量精确为 `1` 时启用。每个完成 WR 同步写一条 stderr 记录，绕过普通 UCM 日志的限流及异步覆盖，不受 UCM_LOG_LEVEL 影响，也不会自动进入 UCM 的旋转日志文件。需保存所有 worker 的 stderr。同步输出会拖慢 completion 线程并可能反压提交，只用于短时诊断，不能拿开启 trace 的模型性能和原始结果直接比较。没有硬件计数器，也没有 CUDA 基线路径计时。

## 对当前数据的解释及实验顺序

用户测量：8×H100，GPU/RDMA PCIe 4；最近单 NIC 20 GB/s、最近两 NIC 共享链路合计 25 GB/s、最近+远端合计 37 GB/s。8k/16k GDR 提升 20%/10%，32k 落后 10%。用户补充观察指标包括 TTFT、吞吐、TPOT、E2EL、ITL；尚无逐项原始数值，不能假定所有指标改善幅度完全相同，也不能把这些百分比直接换算为 IO 带宽。

1. 固定模型、TP、KV dtype、block size、blocks_per_chunk、缓存命中率、并发、stream 数以及 NIC 映射。8k/16k/32k 分别记录热身后短测；保留无 trace 的 GDR 开/关基准。
2. 看 WR 直方图是否变化。若大小不变而次数增加，96 KiB 分界不能直接解释长度拐点；更值得检查持续带宽瓶颈、排队、CPU 线程竞争及计算与传输重叠。
3. 若大小跨过分界，才有依据研究按大小选择 GDR/CUDA。微基准的分界必须在相同方向、并发度、内存类型和计算负载下复核，不能直接硬编码为生产阈值。
4. 先验证每个 GPU 真正绑定最近 NIC。当前单 NIC 实现应以单 NIC 20 GB/s 测量作为相关参考；25/37 GB/s 不能作为现有单 stream 应达目标。
5. 最近+远端快于最近双卡，与共享 PCIe 上游瓶颈的假设相符，但尚不能证明模型 32k 退化的原因。特别是 37 GB/s：若全部单向 payload 必经同一条 Gen4 x16 链路，就超过其约 31.5 GB/s 的编码后理论值，应先复核方向、链路代际及聚合计时口径。双 NIC 需要新增 channel 分流及完成顺序管理，不是改一个 NIC 列表就能实现。

本次只增加诊断，不改变传输选路、合并、并发度或 96 KiB 路由策略。实际性能仍需目标服务器验证。

## 已提供的运行配置及进一步定位

```yaml
ucm_connector_name: UcmPipelineStore
ucm_connector_config:
  store_pipeline: Cache|Posix
  storage_backends: /zsj/data
  posix_capacity_gb: 10240
  posix_io_engine: psync
  posix_data_trans_concurrency: 128
  use_gdr: true
  cache_io_aggregation: false
  cache_sdma_direct: false
```

本配置中 GDR 覆盖 Host Cache ↔ GPU，不会让 Posix 文件 IO 直接变成磁盘到 GPU 的 GDR。Cache 命中与 Posix 回源必须分开分析。

`posix_data_trans_concurrency` 控制 Posix psync 工作池，不是 GDR 在途 WR 数。当前 `posix/cc/trans_queue.cc` 为每个初始化的 TransQueue 分别配置 128 个 load worker 和 128 个 dump worker。实际实例数由部署决定，不能直接认为全机只有 128 个线程，也不能仅凭 8 卡就断言有 2048 个线程。CPU 争用是待测假设，应结合线程数、CPU 亲和性、上下文切换、磁盘等待观测。

未显式设置 `cache_stream_number`，当前 Cache 实现配置默认值为 4；这与 Posix 的 128 独立。每个 GDR stream 又有 scheduler/completion 两个后台线程，准确 stream 总数应按启动日志和实际 queue 实例核实。多个 stream 仍映射到同一 GPU 所配置的那个 NIC。

建议先保持此配置采集粒度和映射，再单独改变一个变量：Posix 并发 32/64/128，或 cache stream 数 1/2/4。分别运行内存 Cache 热命中和 Posix 回源场景；冷/热状态需通过命中与回源记录确认，不要仅凭重复请求推定。每组固定输出长度、请求并发/到达率及缓存命中 token 数。

吞吐提升用 `(on-off)/off`，延迟改善用 `(off-on)/off`；TTFT/TPOT/E2EL/ITL 分别报告均值及 p50/p99。TPOT/ITL 变化还可能反映后台保存、资源争用和调度变化，不能只用加载带宽解释。

## 去掉 Cache 与绕过 Cache 命中的区别

`store_pipeline: Posix` 有对应 builder，但该层的 psync 直接对传入地址执行 pread/pwrite，期望的是主机内存；自身不建立 GDR stream，也不读取 use_gdr 来做 GPU 搬运。适合调用方提供 Host buffer 的场景，不能当作当前 GPU KV 指针场景的等价替换。

更正此前建议：`cache_load_backend_only: true` 只在旧 `cache/cc` 中解析并实现；当前 CUDA 选择的 `cache/v2` 没有读取该配置，不能用它来保证 H100 测试强制回源。保留 `Cache|Posix` 和 GDR 的回源实验，应通过可控数据集、进程/缓存生命周期及来源指标验证；如需开关式 bypass，需要另行实现并测试 v2 支持。
