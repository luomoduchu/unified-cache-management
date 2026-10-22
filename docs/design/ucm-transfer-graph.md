# UcmTransferGraph：设计与第一阶段实现约束

状态：第一阶段代码、可移植测试、Linux simu 集成测试及真实 CacheStore/Posix 对照基准已落地；可移植测试已执行，Linux/设备验证待 CI 与独立审查。基线为 `9cfffebd`。

## 1. 目标和实际硬件边界

将反复执行的 KV 传输流程拆为一次准备的不可变计划与每次独立绑定的执行状态，减少 CPU 任务构造、后端提交和等待管理的开销，改善存储与 H2D 的流水重叠。

华为 OceanStor A800 是分布式全闪存储系统，不是一个可以默认由主机直接控制的本地 NVMe SSD。当前暂按已有文件路径接口设计；实际协议、客户端、挂载参数、网络和计算设备尚待确认。NFS/OceanFS 客户端、专用 SDK、裸块设备是不同后端，不能因介质为 SSD 就直接切换 SPDK 或推断 io_uring 可用。保留 UCM 的 Posix psync/libaio 以及其他 StoreV1 后端；不新增厂商 SDK 或必需的内核依赖。

官方资料：
- https://www.huawei.com/en/news/2024/2/mwc2024-innovative-data-storage-solutions-for-the-ai-era
- https://e.huawei.com/en/products/storage/ai-storage/oceanstor-a800

不预设 CPU 品牌、NUMA 节点数、内存容量或单机可以达到 A800 整机标称吞吐。真实瓶颈可能是网络、协议、文件元数据、后端并发、内存或设备链路。

## 2. 从现有源码定位切入点

- `ucm/store/ucmstore_v1.h`：已有批量 TaskDesc、Load/Dump、Check/Wait；保持其 ABI 和 Python load/dump/check/wait 接口。
- `ucm/store/cache/cc/load_queue.cc::DispatchOneTask`：逐 shard 重新计算访问顺序，针对 owner 且未 READY 的 shard，分别构造单元素 TaskDesc 并调用 backend Load。
- `TaskWrapper::Submit`：每次后端提交创建 task/latch 与 map 条目。Cache 的逐 shard 提交使这些开销按 miss 数增长。
- `LoadQueue::TransferOneTask`：逐 shard Wait，然后提交 H2D；现有 holder 在整个任务末尾同步后释放。大任务与小 buffer 池组合要特别避免资源等待环。
- `TransBuffer`：已有共享/本地池、owner、引用计数、LOADING/READY/FAILED 状态。Get 的分配循环可等待，不能在已持有一批待提交 buffer 时无条件再 Get。
- `DumpQueue`：已经批量提交 backend Dump，且外层任务完成与后端持久化完成并非同一语义。第一阶段不改写该语义。
- `Posix`：psync 和 libaio 已经接受多 shard TaskDesc。批量 Load 减少的是 Store 层管理开销，不代表多个文件的 read 自动合并成一个系统调用或一个磁盘命令。
- `Dram TaskManager`：归一化、路由和分批是后续计划复用点；本阶段不缓存可能失效的路由或地址。

## 3. 架构

```text
现有 Python / C++ 调用
          |
      StoreV1 / CacheStore
          |
  [开关关闭] 原 LoadQueue
  [开关开启] UcmTransferGraph
          |
   不可变计划 / 有界计划缓存
   shard 顺序、批量上限、固定阶段与布局验证
          |
   每次执行独立的绑定状态
   当前 block、shard、目标地址、buffer lease、后端句柄、错误和完成计数
          |
  Resolve/Acquire -> SubmitReadBatch -> AwaitRead
                         |                 |
                 后续批次读取         CopyH2D -> Fence -> Release
          |
  既有 StoreV1 后端 + 既有 CopyStream
```

这是特化的传输图，不引入通用 DAG 调度框架。固定节点依赖由代码状态机表达，减少节点分配和回调对象。计划可复用，执行状态不可跨请求复用。

第一阶段交付真正连接到 CacheStore Load 的可选路径、正确性测试、对照基准与配置说明。Dump、Dram 路由计划复用、厂商 I/O 后端与 CUDA Graph 嵌套作为后续扩展，不以空接口冒充已经实现。

## 4. 计划与运行时绑定

建议独立的轻量 C++17 计划模块放在 `ucm/store/cache/cc/transfer_graph.h`（可配 .cc），不依赖 CUDA、Ascend、libaio 或第三方任务图框架。固定配置在实例 Setup 时绑定。

- Compile/Prepare：预计算与旧路径等价的 shard 访问顺序、窗口边界；缓存按请求 shard 数区分的计划。设备/rank、张量布局、窗口限制属于实例配置，变更需重新 Setup。
- Bind/Replay：每次读取最新 block ID、shard index、目标地址与 TransBuffer 状态。不得把原始地址、缓存命中结果、文件描述符或路由结果固化在计划中。
- Cache：有界 LRU 或简单有界替换，限制条目数与索引总内存；超大 shape 使用无缓存执行，不能让请求尺寸无限占据计划缓存。
- 运行中保有计划的稳定所有权，后续缓存淘汰不得影响已提交的执行。
- 动态 shard 数、尾批次、重复 block/shard、非连续目标地址和 variable tensor sizes 均按实际绑定处理。

## 5. 执行算法和性能机制

1. 校验请求和固定配置；空请求直接完成。配置关闭完全使用原路径。
2. 取得计划；按既有 rank 交错顺序逐个解析缓存状态。
3. 只将新取得 ownership 且未 READY 的 shard 加入后端读批次。已有 READY 的 shard 不读后端；非 owner 的 LOADING shard 等待现有 owner。
4. 同一请求内重复 shard 只由 owner 提交一次；所有目标地址仍要正确填充。
5. 窗口同时受 shard 数、字节数及可用 buffer 限制。采用有界尝试获取 buffer，无法继续时先提交现有小批；空批无法获取时允许可取消的等待。禁止拿着尚未提交的整批 buffer 永久阻塞。
6. 一次 backend Load 提交多 shard 描述符；同一后端句柄只能 Wait 一次。批次失败时不能将任何未确认有效的 buffer 标为 READY，不能 H2D 读取失败数据。
7. 后端完成后依赖才就绪，再发起相应 H2D。批次完成后 fence，再释放本批引用，允许下一窗口复用内存；不能把所有窗口的引用一直保留到整个请求结束。
8. dispatcher 可准备/提交下一批，transfer worker 消费前批，以现有双阶段结构重叠存储和设备传输。有界运行队列提供反压，不依赖积攒不同用户请求来凑批。
9. 一个父任务只有一个最终完成通知；最后批次、提前失败和取消都必须先收敛已提交工作，再完成父任务。

理论上，M 个 owner miss 的后端任务数由 M 降为约 ceil(M/B)。这只描述管理调用数，不能据此宣称等比例吞吐提升；buffer 压力、混合命中和尾批会降低实际 batch size。

批量等待可能增加首个 shard 可用延迟；每窗口 fence 也可能增加同步次数。B 应可调，必须同时测 P99 与吞吐。纯 DRAM 命中与 B=1 必须作为回归样本。

## 6. 正确性与生命周期（实现必须满足）

- 每个异步后端任务的 buffer lease 必须保留到后端停止访问。Timeout、failureSet、调用者不再等待都不能作为释放证据。TaskWrapper 的 latch drain 不自动等于真实 I/O drain；第一阶段须通过 LoadWaitCompletion 查询已启用 strict psync 的真实后端能力，不能将 AIO force-complete 当作终止证据。
- 即使父任务已经失败，也必须 drain 已经成功提交的后端句柄。不能靠跳过 transfer 分支释放仍被 I/O 使用的内存。
- H2D 部分提交后发生错误，也必须执行 fence/drain 后才能释放已使用的 host buffer 或向上层报告可复用目标地址。
- owner 负责发布 READY/FAILED。共享读者不得把别人的 LOADING 状态擅自覆盖为 FAILED。
- 后端批次只有整体状态时，采用保守的整批失败语义，避免把部分读取的数据当作命中。
- 每个成功提交的 handle 至多 Wait 一次；每个 parent waiter 恰好完成一次；任何路径不得遗失已经入队的完成标记。
- 停机停止接纳新任务，唤醒资源等待，排空已提交的批次，最后销毁队列、流和 buffer。保证 dispatcher 与 transfer 的退出顺序不会互相等候。
- producer 并发调用不能破坏 SPSC 队列；对入队执行适当串行化，或采用已有合适的多生产者机制。
- 不缓存文件句柄，不跳过健康熔断，不改变缓存 key、GC 或磁盘布局，不把 Dump 的外层 Wait 改成持久化保证。
- 若某后端不满足必要完成语义，显式限制配置或保持旧路径，并记录原因；不得失败后悄悄重复已提交写操作。

## 7. 配置草案

配置名称可由实现结合项目习惯微调，最终文档须与实际一致：

```yaml
ucm_connector_name: UcmPipelineStore
ucm_connector_config:
  store_pipeline: "Cache|Posix"
  # storage_backends: 填写真实 A800 客户端挂载路径
  cache_transfer_graph: true        # 默认 false，灰度开关
  cache_graph_batch_shards: 32      # 显式调优起点，不是 A800 最佳值
  cache_graph_batch_bytes: 8388608  # 最大 8 MiB 数据窗口
  cache_graph_plan_cache_entries: 32
```

默认关闭。未知配置、0 batch、整数溢出、tensor 个数不符、非法 shard index 均应清楚报错。无需重编译即可通过配置回退。CPU 绑核和 NUMA 使用项目已有配置，不硬编码设备专属常量。

## 8. 验证计划与测量口径

### 可移植计划/绑定测试

无需 GPU/厂商库运行：访问顺序与旧算法等价、0/1/大 shard 数、尾批、byte cap、cache hit/eviction、跨请求地址更新、溢出与非法配置。可提供独立 C++17 测试目标，在 Windows 编译器上也能执行；不能用它代替 Linux 集成测试。

### Linux simu 集成测试

使用现有 MockStore、TransBuffer 和 simu runtime：
- 全 miss 的 batch 提交计数与实际数据正确性；全 hit 不读后端；混合命中与重复 shard。
- 多 tensor/变长 tensor、rank 交错、大于池容量的请求、紧张 buffer 与共享 owner。
- 提交失败、后端 Wait 失败、部分 H2D 提交失败；证明先 drain 再释放和一次完成。
- 多请求并发、stop/drain、空请求、关闭开关的既有行为。
- 保留原 Cache/Posix 相关测试并运行回归；有条件运行 ASan/TSan，明确哪些已执行。

### 对照基准

提供真实调用 UCM 的基准，graph off/on 使用相同数据、命中分布、后端、线程数和内存。预热单独统计，读后校验数据。输出 JSON/CSV 包含后端提交次数、实际批量、吞吐、CPU 时间、P50/P95/P99、plan 命中及资源上限。

矩阵：shard 4/16/64/256 KiB、batch 1/8/16/32/64、并发 1/4/16、DRAM 命中 0/50/100%；第一阶段图路径仅支持 strict psync，aio 需后续完善真实 I/O drain 后另行评估。A800 测试记录协议、客户端、网络速率、挂载参数、io_direct、CPU/设备型号和并发配置。

测试目录必须新建独立子目录，不读写已有生产缓存文件。无需清空系统 page cache、修改块设备或全局内核参数。buffered 结果要注明可能包含客户端/系统缓存收益；不能以本地/tmp 成绩称为 A800 成绩。

设计验收：数据与生命周期正确；全 miss 条件下管理提交数明显下降；不存在无界缓存或资源等待环；真实吞吐/CPU/P99 不符合目标时保留默认关闭并报告。

## 9. 分工和交付

按用户指定顺序：主代理完成设计；`gpt-6.1-sol` 实现代码、配置、测试和基准；`gpt-6-astra` 独立检查并执行可用验证；发现的问题回交实现代理修复，再复核。保留审查记录和可复现命令。

当前工作机是 Windows，发现 MinGW g++ 与 CMake，未发现已安装 WSL Linux 或 Docker；尚无 A800/计算设备远程环境。可以立即完成代码、可移植验证与基准工具，Linux/GPU/A800 验证须如实记录未运行原因，不将设计预期写成实测提升。

## 10. 第一阶段实现记录

- 用户已确认 A800 通过 POSIX Linux 文件路径接入（本地挂载或 NFS，具体协议版本后补），CPU 为 x86/aarch64，设备为 H100 和昇腾 910B3，目前没有测试机器。复用 CUDA/Ascend CopyStream；不为 910B3 默认启用仅适用于 Ascend-A3 的 SDMA Direct。
- `transfer_graph.h` 的不可变计划使用常量空间的 rank 顺序索引，而非预存 O(shard 数) 的索引数组。因此超大 shape 也不会撑大计划缓存；LRU 上限 4096 项，0 项表示不缓存，批次 shard 上限 65536。运行中保有 shared_ptr。
- 实际 `CacheStore -> TransManager -> LoadQueue` 通过默认关闭的配置选取图路径。两阶段执行有界窗口，TryGet 使用 bucket try-lock 和最多 64 个候选扫描；压力下先提交已取得的窗口，再重试。空窗口等待可被 stop/父失败取消。
- 每批后端句柄只交由 transfer worker Wait 一次。父失败后仍 drain，owner 才发布 READY/FAILED，保守整批失败。每窗口始终 fence 后释放 host lease 与目标地址数组。独立末尾标记保证完成顺序；停机先停止接纳并排空 dispatcher，再排空 transfer。
- 图路径省去旧路径的 speculative Prealloc，因为该接口可能在持有待提交窗口时阻塞。此差异可能影响多 shard block 的缓存状态与命中率；基准使用单 shard block 消除此额外变量，并报告实际读后端 shard 数与 READY 命中。
- 对 LoadQueue 入队做 producer 串行化，修复 LocalBufferStrategy nodeCursor 的并发数据竞争；没有改 StoreV1 ABI、后端布局、健康检查或 Dump 语义。
- 审查修复：worker 构造或 stream Setup 失败后停止接纳，发布 dispatcherDone 并 join 已启动的 transfer，避免构造失败时析构等待；图路径显式拒绝有效 IO aggregation 与 GDR 模式，因为当前错误返回不保证所有 lane/NIC 访问 drain。原关闭图路径语义不变。
- 完成语义更正：原先认为所有 TaskWrapper Wait（包括 AIO）都能 drain 的假设不成立。新增独立 `LoadWaitCompletion` 能力接口，不改 StoreV1 虚表；真实 Posix 对象仅在成功 Setup 且 strict psync Load 模式下声明能力，HealthBreaker/基准计数包装器真实转发。未知、ASU、AIO、非 strict psync 后端拒绝图路径；同时拒绝未经验证的有效 SDMA Direct。
- strict psync 由后端 Setup 的 `cache_transfer_graph` 开启，仅取消 loadPool 的 active worker timeout/detach monitor，dumpPool 不改。逻辑 timeout 仍取消排队工作并等待活跃 read 返回；hard NFS mount I/O 若永久不返回，Wait/停机也可能永久等待，不能提前释放 buffer。新增实际阻塞 pread 跨越 1.6 秒监控周期的回归及能力/健康包装转发测试，Linux 执行结果待 CI。
- 新增可移植计划测试、Linux simu 实际执行器故障/资源测试，以及 `MakeCacheStore()->Load/Wait -> MakePosixStore()` 的 C++ JSONL 对照基准。基准预热分离、实际计数、全数据校验；CPU/吞吐包含校验，延迟仅计 Load-through-Wait，未采集项明确输出 null。
- Windows g++ 8.1 C++17 可移植测试已通过 161139 个参考用例；主代理独立 rank 顺序核查另通过 43933 组。Linux 集成、基准运行、ASan/TSan、H100/910B3/A800 尚未在本机运行，等待 CI/审查补充，不据此宣称性能提升。

配置、运行命令、测试覆盖及限制见 [使用与验证说明](../ucm-transfer-graph.md)，硬件实测规范见 [A800 评估说明](ucm-transfer-graph-a800-evaluation.md)。后续审查发现与实际执行结果应继续追加。
