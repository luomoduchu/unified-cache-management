# UCM GDR 混合分流：初次实验指南

> 分支 `GDRchange` @ `2e1233e`
> 两阶段：① `cache_bw.py` 机制验证（不启 vLLM，分钟级）② vLLM 端到端 TTFT
>
> **一句话判据**：只有当 `cache_load_wait_shards_total ≈ 0`（数据已在 host DRAM、纯 H2D）时，
> 你测的才是这个设计的作用面。带 SSD 等待的轮次，预期三组无差异——那不是在测它。

---

## 0. 部署与自检（三条全过再往下）

### 0.1 构建

```bash
git clone -b GDRchange https://github.com/luomoduchu/unified-cache-management.git
cd unified-cache-management
# 沿用你原来的构建命令，确认带 -DENABLE_GDR=1（即 UCM_ENABLE_GDR_STREAM=ON）
```

### 0.2 三条自检

**① 新代码真的生效**

```bash
# 启动后（任一阶段）
grep "UseGdr to" <ucm日志>
# 期望：Set ...::UseGdr to false (hybrid=true, gdrStreams=1, weights=0.46:0.54).
```

看不到 `weights=` 就是在跑老库——老库里 `cache_gdr_mode` 会被**静默忽略**，配置写了也不报错。这是最容易踩的坑。

**② NIC 绑定正确**

```bash
grep "Enable GDR stream on device" <ucm日志>
# 期望：每张卡一行，nic 互不相同（2 卡即 mlx5_4 / mlx5_6）
```

`DEVICE_GDR_NICS` 只有 2 个条目本身就是一次验证：若 `cudaGetDevice()` 返回物理序号 2/3，`ResolveNicName` 会直接抛
`missing GDR NIC name for device(2)`，进程起不来。能起来就说明按 ordinal 索引的假设成立。

**③ shard 大小**

```bash
grep "ShardSize" <ucm日志>
```

**必须是 144 KB（147456）**，否则 46:54 的分流权重不成立（见 §3.3）。TP 或模型一变，shard 大小就会变。

---

## 1. 第一阶段：`cache_bw.py` 机制验证

### 1.1 为什么先做这个

`ucm/store/test/e2e/cache_bw.py` 直接压 UCM 的 load 路径，**不启 vLLM**，几十秒一组，读的是 C++ 侧
`cache_load_bytes_total / cache_load_duration_ms`。它有两个 mode，正好对应我们要区分的两个供给 regime：

| mode | 管线 | 数据来源 | 含义 |
|---|---|---|---|
| `cache-hit` | `Cache` | Cache buffer 中已 `Ready` | **纯 H2D，η=1，本设计的靶心** |
| `posix-hit` | `Posix` seed → `Cache\|Posix` load | 磁盘 | 走 SSD，控制组，预期无差异 |

### 1.2 先看计划（不需要 Torch / UCM 运行时）

```bash
python ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --device cuda --model custom \
  --workers 1 --layers 1 --io-size 147456 --blocks 512 \
  --cache-capacity-gb 32 --epochs 10 --warmup 2 \
  --cache-copy hybrid --cache-gdr-streams 1 --hybrid-gdr-percent 46 \
  --dry-run
```

检查输出里的 `shard_size` 是否为 `147456`、`cache_copy` 是否为 `hybrid`、`hybrid_gdr_percent` 是否为 `46`。

### 1.3 跑 8 组

```bash
CACHE_BW=/path/to/unified-cache-management/ucm/store/test/e2e/cache_bw.py
OUT=./bw-results; mkdir -p $OUT

run_case () {          # $1=mode  $2=tag  其余参数传给 cache_bw.py
  local mode=$1 tag=$2; shift 2
  echo "===== $mode / $tag ====="
  python $CACHE_BW \
    --mode $mode --device cuda --model custom \
    --workers 1 --layers 1 --io-size 147456 --blocks 512 \
    --cache-capacity-gb 32 --epochs 10 --warmup 2 \
    --direct-io --posix-engine psync --posix-concurrency 128 \
    --storage-backends /zsj/bw-data \
    "$@" 2>&1 | tee $OUT/${mode}_${tag}.log
}

for mode in cache-hit posix-hit; do
  run_case $mode ce     --cache-copy plain
  run_case $mode gdr4   --cache-copy gdr    --cache-gdr-streams 4
  run_case $mode gdr1   --cache-copy gdr    --cache-gdr-streams 1
  run_case $mode hybrid --cache-copy hybrid --cache-gdr-streams 1 --hybrid-gdr-percent 46
done
```

四个变体的含义：

| tag | 实际配置 | 说明 |
|---|---|---|
| `ce` | `use_gdr: false` | 纯 CE 基线 |
| `gdr4` | `use_gdr: true` + `cache_gdr_stream_number: 4` | 改动前的等价行为（4 通道 × 2 自旋线程） |
| `gdr1` | `use_gdr: true` + `cache_gdr_stream_number: 1` | **第 2 项改动** |
| `hybrid` | `cache_gdr_mode: hybrid` + 1 通道 + 46% | **一期设计** |

### 1.4 先跑一遍带校验的（防数据损坏）

路由改动理论上只影响"哪个引擎拷"，但一定要证明它不破坏数据。**每个 copy 模式先带 `--verify` 跑一次**：

```bash
python $CACHE_BW --mode cache-hit --device cuda --model custom \
  --workers 1 --layers 1 --io-size 147456 --blocks 64 \
  --cache-capacity-gb 8 --epochs 2 --warmup 0 \
  --cache-copy hybrid --cache-gdr-streams 1 --verify
```

`--verify` 会在计时区间之外逐个校验 tensor。**校验失败就不要继续测性能了。**

### 1.5 判读

看每组日志末尾汇总里的 `cache_load_bandwidth_gbps`：

| tag | `cache-hit` 期望（单卡，144 KB shard） | `posix-hit` 期望 |
|---|---|---|
| `ce` | ≈ 27 GB/s | 由磁盘决定 |
| `gdr4` | ≈ 21–23 GB/s（4 组忙等线程可能拉低） | 同上 |
| `gdr1` | ≥ `gdr4`，≈ 22–23 GB/s | 同上 |
| `hybrid` | **≈ 38–41 GB/s** | 与上面接近 |

- `hybrid` 在 `cache-hit` 拿不到 ~41 → 先查 §0.2 三条自检，再看 `cache_stream_number` 是否为 4（CE 流太少会喂不满）。
- `posix-hit` 三组出现明显差异 → 说明那块盘的读带宽高于 23 GB/s，SSD 没成为瓶颈，这个结果要记录（它意味着冷读场景也能吃到收益）。

### 1.6 再跑 8 卡看线性

```bash
python $CACHE_BW --mode cache-hit --device cuda --model custom \
  --workers 8 --layers 1 --io-size 147456 --blocks 512 \
  --cache-capacity-gb 128 --epochs 10 --warmup 2 \
  --cache-copy hybrid --cache-gdr-streams 1
```

期望 ≈ 8 × 41.5 = 332 GB/s，但**很可能先撞 host DRAM 读带宽墙**（双路 DDR4 实际 200–250 GB/s）。
如果 8 卡结果明显低于 8×单卡，把这个数记下来——那是下一个天花板，也是后续要不要做 NUMA 绑核的依据。

---

## 2. 第二阶段：vLLM 端到端

### 2.1 环境变量

```bash
export UCM_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=2,3          # 物理 GPU2,3 → 逻辑 ordinal 0,1
export DEVICE_GDR_NICS=mlx5_4,mlx5_6     # GPU2↔mlx5_4, GPU3↔mlx5_6（按 ordinal 索引）
export ENABLE_UCM_PATCH=1
export PYTHONHASHSEED=0
KV_CACHE_DIR=/zsj/data; mkdir -p ${KV_CACHE_DIR}
```

**不要清空 `KV_CACHE_DIR`**：hit/miss 由"是否重复同一个 prompt"控制，与磁盘状态无关；清盘只会让 miss 组变慢，浪费时间。

### 2.2 四份 config（脚本自动生成，不用手写四份）

`/unified-cache-management/examples/` 下建四个文件，只有末尾的 copy 段不同：

```bash
make_config () {        # $1=输出文件  $2=copy 段
cat > "$1" <<EOF
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/zsj/data"
      posix_capacity_gb: 10240
      posix_io_engine: "psync"
      posix_data_trans_concurrency: 128
      io_direct: true
      cache_io_aggregation: false
      cache_sdma_direct: false
$2
EOF
}

EX=/unified-cache-management/examples
make_config $EX/ucm_gdr_ce.yaml     '      use_gdr: false'
make_config $EX/ucm_gdr_gdr4.yaml   '      use_gdr: true
      cache_gdr_stream_number: 4'
make_config $EX/ucm_gdr_gdr1.yaml   '      use_gdr: true
      cache_gdr_stream_number: 1'
make_config $EX/ucm_gdr_hybrid.yaml '      cache_gdr_mode: "hybrid"
      cache_gdr_stream_number: 1
      cache_hybrid_gdr_percent: 46'
```

> **原配置里的 `cache_load_backend_only` 已删除**：当前 develop 源码里不存在这个键，
> 写了会被静默忽略；若你的旧版确实支持它，它会把 Cache 绕开强制读盘，正好压掉本设计的收益。

四个文件里 `hybrid` 那份**故意不写 `use_gdr: true`**：`cache_gdr_mode: hybrid` 内部会把 `useGdr` 置为 false，
让 dump 侧继续走 CE（GDR READ 从未标定，一期不碰）。

### 2.3 启停脚本

```bash
#!/usr/bin/env bash
# 用法: ./run_vllm.sh start ce|gdr4|gdr1|hybrid
#       ./run_vllm.sh stop
set -uo pipefail

VAR=${2:-}
LOG=/vllm-workspace/logs/ucm_${VAR}_$(date +%Y%m%d-%H%M%S).log

case "${1:-}" in
start)
  [ -z "$VAR" ] && { echo "缺少变体"; exit 1; }
  export UCM_LOG_LEVEL=info
  export CUDA_VISIBLE_DEVICES=2,3
  export DEVICE_GDR_NICS=mlx5_4,mlx5_6
  export ENABLE_UCM_PATCH=1
  export PYTHONHASHSEED=0
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv > /tmp/gpu_before.txt

  vllm serve /models/Qwen3.8-27B \
      --served-model-name Qwen3 \
      --tensor-parallel-size 2 \
      --max-model-len 65536 \
      --block-size 128 \
      --port 9000 \
      --max-num-seqs 4 \
      --gpu-memory-utilization 0.5 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --kv-transfer-config '{
          "kv_connector": "UCMConnector",
          "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
          "kv_role": "kv_both",
          "kv_connector_extra_config": {
              "UCM_CONFIG_FILE": "/unified-cache-management/examples/ucm_gdr_'${VAR}'.yaml"
          }
      }' > "${LOG}" 2>&1 &
  echo $! > /tmp/vllm.pid
  echo "PID=$(cat /tmp/vllm.pid) LOG=${LOG}"

  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:9000/health > /dev/null && { echo "ready (${i}x2s)"; break; }
    sleep 2
  done
  # 自检
  grep -m1 "UseGdr to" "${LOG}" || echo "!! 日志里没有 UseGdr —— 检查是否装的是 GDRchange 分支"
  grep "Enable GDR stream on device" "${LOG}"
  grep -m1 "ShardSize" "${LOG}"
  ;;
stop)
  [ -f /tmp/vllm.pid ] && kill "$(cat /tmp/vllm.pid)" 2>/dev/null
  for i in $(seq 1 30); do
    kill -0 "$(cat /tmp/vllm.pid)" 2>/dev/null || break
    sleep 1
  done
  echo "stopped"
  ;;
*)
  echo "用法: $0 start <ce|gdr4|gdr1|hybrid> | $0 stop"; exit 1;;
esac
```

**两个参数改动的原因**：

- `--max-num-seqs 4`（原 64）：并发会打乱 load 路径的串行结构，纯净测量要压到个位数。
- `--no-enable-chunked-prefill`：chunked prefill 让 H2D 与 compute 重叠，会把我们要测的拷贝差异冲淡。
  正式跑完可再开着一组看真实场景（若该版本默认开启，`--no-enable-chunked-prefill` 即关闭）。

### 2.4 测法：同一个 prompt 连发 5 次

每次**重启 vLLM** → Cache 是进程内的内存，重启后必然为空 → 第 1 次请求必 miss（走 Posix），第 2 次起 hit（纯 H2D）。
`--no-enable-prefix-caching` 让 vLLM 自己的 prefix cache 让位，复用只能由 UCM 提供。

把下面存成 `bench_same_prompt.py`：

```python
import json, time, urllib.request, sys

URL    = "http://127.0.0.1:9000/v1/completions"
N      = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000   # 约多少 word，越长约好测

# 固定内容 → 每次都是同一个 prompt；hash 决定 UCM 的 key
PROMPT = " ".join(f"w{i:05d}" for i in range(TOKENS))

def one():
    body = json.dumps({"model": "Qwen3", "prompt": PROMPT, "max_tokens": 16,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line and ttft is None:
                ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0

for i in range(1, N + 1):
    ttft, total = one()
    tag = "miss(预期)" if i == 1 else "hit(预期)"
    print(f"req{i} {tag:10s} ttft={ttft*1000:8.2f} ms  total={total*1000:9.2f} ms", flush=True)
```

```bash
python bench_same_prompt.py 5 8000
```

**取第 3–5 次**作为 hit 组数据（前两次可能还有显存分配、JIT、CPU 缓存预热）。

### 2.5 查看性能

**① UCM 侧指标**（进程级计数器，重启即归零，所以直接读绝对值就是本轮总量）

```bash
curl -s http://127.0.0.1:9000/metrics | grep -E \
 '^ucm:(cache_load_shards_total|cache_load_wait_shards_total|cache_load_backend_shards_total|cache_load_success_shards_total|cache_posix_load_success_shards_total|cache_load_bytes_total|cache_load_bandwidth_gbps_(sum|count)|cache_load_duration_ms_(sum|count))'
```

**② 一键算三个关键比值**

```bash
cat > ucm_stats.py <<'PY'
import urllib.request, re, sys
txt = urllib.request.urlopen("http://127.0.0.1:9000/metrics", timeout=10).read().decode()
m = dict(re.findall(r'^ucm:(\w+)\s+([0-9.eE+-]+)$', txt, re.M))
g = lambda k: float(m.get(k, 0))
shards = g("cache_load_shards_total")
hit    = g("cache_load_success_shards_total")
posix  = g("cache_posix_load_success_shards_total")
wait   = g("cache_load_wait_shards_total")
bw_n   = g("cache_load_bandwidth_gbps_count")
bw_s   = g("cache_load_bandwidth_gbps_sum")
print(f"shards_total        = {shards:.0f}")
print(f"hit  (fromReady)    = {hit:.0f}")
print(f"posix(wait backend) = {posix:.0f}")
print(f"wait_shards_total   = {wait:.0f}")
print(f"命中率              = {hit/(hit+posix)*100 if hit+posix else 0:.1f}%")
print(f"饥饿率 wait/shards  = {wait/shards*100 if shards else 0:.1f}%   <- 必须≈0")
print(f"平均 load 带宽      = {bw_s/bw_n if bw_n else 0:.2f} GB/s <- 期望 23→41")
PY
python ucm_stats.py
```

**③ 日志侧**

```bash
grep -c "Enable GDR stream on device" $LOG     # 应等于卡数
grep -E "ShardSize|UseGdr to|GDR stream" $LOG
```

**④ 端到端时延**：`bench_same_prompt.py` 的 `ttft` / `total` 就是 TTFT 与 E2EL。
每变体跑 3 轮取中位数，别取单次。

### 2.6 判读矩阵

| 观察 | 结论 |
|---|---|
| hit 组 `饥饿率 ≈ 0`，`平均带宽` hybrid ≈ 41、ce ≈ 27、gdr ≈ 23 | **设计成立**，再比 TTFT |
| hit 组 `饥饿率 > 20%` | 没命中。查 `cache_buffer_capacity_gb` 是否够大、prompt 是否超过 Cache 容量 |
| miss 组三变体无差异 | **符合预期**，不是失败（冷读场景本就是 SSD 绑定） |
| `gdr1 ≥ gdr4` | 第 2 项成立（忙等线程确实在抢资源），照此发布 |
| `gdr4 > gdr1` | 我关于"忙等线程抢 CPU"的假设错了，需重新分析 |
| `hybrid < gdr1` | 46:54 权重不适用，先确认 shard 是否 144 KB，再考虑重标 |

### 2.7 常见坑

1. **配置写了但日志没有 `weights=`** → 装的是老库，`cache_gdr_mode` 被静默忽略。
2. **`wait_shards_total > 0` 还在比 TTFT** → 这一轮没命中，测的是 SSD 绑定，结论无效。
3. **第一次请求当数据用** → 它必 miss 且含预热，丢掉。
4. **并发 > 8 或 chunked prefill 开着** → 拷贝差异被 compute 掩盖。
5. **改了 TP / 模型 / block-size** → shard 大小变了，46:54 失效；先看 `ShardSize` 日志。
6. **`DEVICE_GDR_NICS` 条目数与可见卡数不一致** → 直接起不来（`missing GDR NIC name for device(N)`），这是预期行为。

---

## 3. 结论记录表（照抄填空）

### 3.1 机制层（cache_bw.py，单卡，144 KB）

| tag | cache-hit GB/s | posix-hit GB/s | 备注 |
|---|---|---|---|
| ce |  |  |  |
| gdr4 |  |  |  |
| gdr1 |  |  |  |
| hybrid |  |  |  |

### 3.2 8 卡扩展

| tag | 8 卡 GB/s | 相对 8×单卡 | 备注（是否撞 DRAM 墙） |
|---|---|---|---|
| gdr1 |  |  |  |
| hybrid |  |  |  |

### 3.3 端到端（vLLM，取第 3–5 次中位数）

| 变体 | 饥饿率 | 平均带宽 GB/s | TTFT ms | E2EL ms |
|---|---|---|---|---|
| ce |  |  |  |  |
| gdr4 |  |  |  |  |
| gdr1 |  |  |  |  |
| hybrid |  |  |  |  |

### 3.4 结论

- 机制层：hybrid 是否达到 ~41 GB/s？__________
- 第 2 项：1 通道 vs 4 通道谁更好？__________
- 端到端：hybrid 相对 ce 的 TTFT 改善？__________
- 是否与预测一致（hit 组 25–40%，miss 组 ≤3%）？__________
