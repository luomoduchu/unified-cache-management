# UcmTransferGraph 独立审查记录

审查者：用户指定的 gpt-6-astra 代理。基线：`9cfffebd`；审查对象为 graph 分支工作区（含未跟踪的新文件）。2026-10-21 初审，2026-10-22 继续。源码审查已闭环，可提交 CI；不能视为 Linux/设备验证通过。

## 已执行验证

- 独立读取设计、图执行器、TransBuffer、SPSC、TaskWrapper、CopyStream 与其 CUDA/Ascend/GDR 实现、Posix psync/aio 实际完成路径、健康包装层、测试、基准和 CI。
- 独立以 Windows MinGW g++ 8.1 编译 `plan_test.cc`：`-std=c++17 -O2 -Wall -Wextra -Werror -I.`，执行输出 `161139 reference cases passed`。输出程序位于忽略目录 `build/transfer-graph-validation/reviewer_plan_test.exe`。
- 初审实际执行 `runpy.run_path('ucm/default_metrics_config.py')`，复现 `ValueError: too many values to unpack (expected 2)`。修复后独立再次执行导入、counter 名称唯一性、四个图指标存在性检查，全部通过。
- 本审查代理未在本机运行 Linux 集成测试、真实 CacheStore/Posix benchmark、ASan/TSan、CUDA/Ascend 编译或设备测试。后续 CI 结果须单独记录来源与提交版本。

## 发现与修复状态

1. **P1 / metrics 导入失败（已源码复核及本机运行复核）**：四个新指标被插入一个原二元组，模块初始化解包失败，影响 graph 关闭的调用方。已拆成独立二元组；独立运行复核通过。
2. **P2 / 紧池测试不稳定（已源码复核）**：两槽池在逐个 lease 释放、bucket 竞争时允许单元素 flush，9 shard 不保证最多5次提交。改为合法范围，另用充足池验证严格批量计数。
3. **P2 / 完成回调测试竞态（已源码复核）**：Latch 先发布 counter=0 再执行 epilog，仅 WaitForDuration 不能保证测试的回调计数已更新。新增 epilog future 同步。
4. **P1 / dispatcher 线程启动失败析构挂死（已源码复核，Linux故障注入待运行）**：transfer 已启动而 dispatcher 构造抛异常时，原 graph transfer 永远等待 dispatcherDone。修复启动异常收敛、失败后拒绝接纳，并新增线程/stream启动故障注入。
5. **P1 / IO aggregation 与 GDR 同步失败不保证 drain（已限制组合，源码复核通过）**：AscendShardIOAggregator::Synchronize 在第一 lane/event 失败提前返回；GdrStream::Synchronized 在 HasAsyncError 时提前返回，未等待已接受的 NIC 请求终止。图不能在这些返回后无条件释放 host leases。第一阶段应明确拒绝这些组合，保留现有默认路径。
6. **P1 / backend Wait 并非普遍终止保证（已严格限定并修复 psync Load，源码复核通过）**：
   - Posix AIO `io_engine_aio.h::ForceComplete` 在 CancelTask 后 Abort latch；`aio_impl.cc::CancelTask` 对不能取消的 I/O 仅保留记录，不等待内核完成。
   - Posix psync 的 ThreadPool monitor 在运行中超时调用 OnIoUnitTimeout（Done latch），随后 detach worker；阻塞 pread 仍可能继续写 buffer。
   - ASU 的 ClientTaskManager::WaitContext 超时直接返回，未 drain。
   - 因此不能只依赖 TaskWrapper 外层等待循环。建议用不改 StoreV1 虚表的独立能力接口检查实际 backend；仅成功配置严格 psync Load 完成模式的 Posix 可声明能力，健康包装层转发，未知/ASU/AIO拒绝。严格 psync 模式禁用 Load pool 的运行中超时 detach，保留逻辑超时、排队取消和真实完成等待；Dump/默认路径保持既有行为。

## 核心路径判断与验证边界

- 常量空间 plan、有限 LRU、shared_ptr 淘汰所有权及每次地址重绑定符合设计。配置有 shard/entry 上限，字节窗口用除法避免乘法溢出。
- 图 dispatcher 先 TryGet，再提交已经取得的窗口；不拿着未提交整窗执行阻塞 Get。transfer 独立推进，可释放前窗缓解池压力。
- 每个成功 backend handle 由已发布 batch 接管；父失败仍 Wait；owner 才发布 READY/FAILED；整体 Wait 失败不启动该批 H2D。
- 部分 H2D 提交失败仍调用全部 CopyStream 的同步；host lease 与目的地址数组在该调用返回前存活。真实设备 fatal error 下是否已经终止仍依赖驱动契约，simu无法证明硬件故障恢复。
- 完成标记与 batch 共用 FIFO，dispatcher 结束后 transfer 才退出；producer 入队串行化。外部调用者仍必须保证对象生命期，不得在对象已销毁后 Submit。
- 普通 CUDA/Ascend 与 simu 为第一阶段目标；复杂 copy 模式宜拒绝。未经实际测试不声称 CUDA/Ascend 编译成功。
- 严格 drain 不是有界内核取消：NFS hard mount 或阻塞系统调用可能使 Wait/停机长时间等待。安全释放优先于虚假的 timeout 完成。

## 基准与测试范围

基准实际使用 CacheStore/Posix/simu，计数来自真正 backend Load，读取后逐字验证。单 shard block 消除旧 Prealloc 对对照的额外影响；CPU/吞吐包含请求准备与校验，延迟仅 Load-through-Wait，文档已区分。初审版本 off 总先于 on；复审版本已支持逆序参数（见下文），仍不能据单次结果宣称 A800 提升。未测峰值和 GPU 时间为 null。

新增/补充的集成测试覆盖资源压力、重复/命中、批次失败、父失败后 drain、真实父 timeout、部分 H2D 失败、ghost slots、启动失败、并发 producer、停机与异常完成。源码覆盖不等于已经执行，须等待 Linux CI。

## 2026-10-22 修复闭环复审（提交 CI 前）

已复核实现代理最终批次，前述第 4–6 项已在源码中收敛：

- Setup 捕获两次线程创建与 stream Setup 错误，停止接纳，发布 dispatcherDone，join 已启动 transfer；故障注入覆盖四种启动失败。
- 图同时拒绝 IO aggregation、GDR 与 SDMA Direct；常规 CUDA/Ascend/simu 路径保留。
- 新 `load_wait_completion.h` 为独立多态 side interface，不新增 StoreV1 虚函数。CacheStore 与 LoadQueue 都检查实际 backend；Posix 仅在成功 Setup、传输开启、psync 且 strictLoadCompletion 时返回 true。HealthBreaker 和基准包装器查询真实对象能力。
- strict psync 仅不注册 Load pool 的 active-worker timeout hook，使其不启动 detach monitor；TaskWrapper 仍取消排队并等待真实 read 返回，Dump pool 与默认关闭路径保持原策略。
- 新 Posix 回归用 pread hook 阻塞 1600 ms，超过 500 ms 逻辑超时与原 1 s monitor 周期，要求 release 前 Wait 不返回、release 后返回 Timeout 且实际数据完整。该测试源码已审查，尚待 Linux 执行。hook 受 UCM_ENABLE_TEST_HOOKS 保护，生产构建仍直接 pread。
- 独立运行 `python test/test_default_metrics_config.py`：3 项通过。曾尝试模块形式 unittest 因本机同名 test 包导入失败，随后直接脚本运行成功；不把前者计作通过。
- 基准新增 `off-on|on-off` 参数和 run_order 字段，两模式共享同一个 strict psync backend；已消除初审所记“无法逆序”的限制。当前基准拒绝 aio。读取校验仍包含在 CPU/墙钟吞吐中，延迟范围已明确，不能据此宣称设备或 A800 提升。

结论：在第一阶段限定的 strict Posix psync 与普通 copy stream 组合内，未发现尚未处理的阻塞级源码问题，可以提交 Ubuntu simu CI 做编译与执行验证。此结论不代表 Linux 测试已通过，也不代表 CUDA/Ascend/真实设备验证完成。新增 side-interface 的跨动态库查询由真实 CacheStore/Posix 基准 smoke 检验；健康包装转发有独立源码用例。CI 失败需继续修复并追加记录。



## CI 执行结果（由主代理追加）

待实际运行后记录提交 SHA、工作流链接、测试/基准结果与后续修复。当前未填写通过结论。
