/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include "load_wait_completion.h"
#include "metrics_api.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeCacheStore();
extern "C" UC::StoreV1* MakePosixStore();

namespace {
using Clock = std::chrono::steady_clock;
using namespace UC;
struct Options {
    std::string parent, engine{"psync"}, order{"off-on"};
    size_t bytes{65536}, shards{32}, batch{32}, concurrency{1}, iterations{20}, hit{0};
    bool direct{false};
};
void Require(bool ok, const char* message)
{
    if (!ok) { throw std::runtime_error(message); }
}
void Good(const Status& s) { Require(s.Success(), "UCM operation failed (see UCM log)"); }
UC::Detail::BlockId Key(size_t id)
{
    UC::Detail::BlockId key{};
    std::memcpy(key.data(), &id, sizeof(id));
    return key;
}
uint64_t Pattern(size_t id, size_t offset)
{
    uint64_t word = uint64_t(id) + uint64_t(offset / 8) * 0x9e3779b97f4a7c15ULL;
    word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
    word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
    return word ^ (word >> 31);
}

// Counts actual successful StoreV1 backend submissions, rather than inferring batches.
class CountedStore : public StoreV1, public LoadWaitCompletion {
    std::unique_ptr<StoreV1> store_{MakePosixStore()};

public:
    bool LoadWaitDrains() const noexcept override { return HasDrainingLoadWait(store_.get()); }
    std::atomic<size_t> calls{0}, shards{0};
    Status Setup(const UC::Detail::Dictionary& c) override { return store_->Setup(c); }
    std::string Readme() const override { return store_->Readme(); }
    Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId* b, size_t n) override
    {
        return store_->Lookup(b, n);
    }
    Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId* b, size_t n) override
    {
        return store_->LookupOnPrefix(b, n);
    }
    Expected<ssize_t> LookupOnReverse(const UC::Detail::BlockId* b, size_t n) override
    {
        return store_->LookupOnReverse(b, n);
    }
    void Prefetch(const UC::Detail::BlockId* b, size_t n) override { store_->Prefetch(b, n); }
    Status CheckHealth() override { return store_->CheckHealth(); }
    Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc task) override
    {
        const auto n = task.size();
        auto result = store_->Load(std::move(task));
        if (result) {
            ++calls;
            shards += n;
        }
        return result;
    }
    Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc task) override
    {
        return store_->Dump(std::move(task));
    }
    Expected<bool> Check(UC::Detail::TaskHandle h) override { return store_->Check(h); }
    Status Wait(UC::Detail::TaskHandle h) override { return store_->Wait(h); }
};

void Seed(CountedStore& backend, size_t count, size_t bytes)
{
    void* ptr = nullptr;
    Require(posix_memalign(&ptr, 4096, bytes) == 0, "seed allocation failed");
    std::unique_ptr<void, decltype(&std::free)> buffer(ptr, &std::free);
    for (size_t id = 0; id != count; ++id) {
        for (size_t offset = 0; offset != bytes; offset += 8) {
            const auto word = Pattern(id, offset);
            std::memcpy(static_cast<unsigned char*>(ptr) + offset, &word, sizeof(word));
        }
        UC::Detail::TaskDesc desc{
            {Key(id), 0, {ptr}}
        };
        auto result = backend.Dump(std::move(desc));
        Require(bool(result), "seed submit failed");
        Good(backend.Wait(result.Value()));  // real backend persistence completion
    }
}

void Run(CountedStore& backend, const Options& options, const std::string& path, bool graph)
{
    std::unique_ptr<StoreV1> cache(MakeCacheStore());
    UC::Detail::Dictionary config;
    config.Set("store_backend", static_cast<StoreV1*>(&backend));
    config.Set("unique_id", std::string("transfer-graph-benchmark-") + (graph ? "on" : "off"));
    config.SetNumber("device_id", 0);
    config.SetNumber("tensor_size", options.bytes);
    config.SetNumber("shard_size", options.bytes);
    config.SetNumber("block_size", options.bytes);
    config.Set("share_buffer_enable", false);
    config.Set("io_direct", options.direct);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("cache_load_exclusive_buffer_number", 64);
    config.SetNumber("waiting_queue_depth", 256);
    config.SetNumber("running_queue_depth", 256);
    config.SetNumber("local_rank_size", 1);
    config.SetNumber("timeout_ms", 600000);
    config.Set("cache_transfer_graph", graph);
    config.SetNumber("cache_graph_batch_shards", options.batch);
    config.SetNumber("cache_graph_batch_bytes", options.batch * options.bytes);
    config.SetNumber("cache_graph_plan_cache_entries", 32);
    Good(cache->Setup(config));
    const size_t hot = options.shards * options.hit / 100;
    const size_t cold = options.shards - hot;
    const size_t hotTotal = hot * options.concurrency;
    auto execute = [&](size_t worker, size_t iteration) {
        std::vector<std::vector<unsigned char>> data(options.shards,
                                                     std::vector<unsigned char>(options.bytes));
        UC::Detail::TaskDesc desc;
        std::vector<size_t> keys;
        for (size_t i = 0; i != options.shards; ++i) {
            const auto id =
                i < hot ? worker * hot + i
                        : hotTotal + (iteration * options.concurrency + worker) * cold + i - hot;
            keys.push_back(id);
            desc.push_back({Key(id), 0, {data[i].data()}});
        }
        const auto start = Clock::now();
        auto result = cache->Load(std::move(desc));
        Require(bool(result), "cache submit failed");
        Good(cache->Wait(result.Value()));
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        for (size_t i = 0; i != data.size(); ++i) {
            for (size_t offset = 0; offset != options.bytes; offset += 8) {
                const auto word = Pattern(keys[i], offset);
                Require(std::memcmp(data[i].data() + offset, &word, sizeof(word)) == 0,
                        "read verification failed");
            }
        }
        return ms;
    };
    // Warm the same request shape and requested hot keys; use separate cold keys.
    const auto warmStart = Clock::now();
    for (size_t worker = 0; worker != options.concurrency; ++worker) {
        execute(worker, options.iterations);
    }
    const auto warmMs = std::chrono::duration<double, std::milli>(Clock::now() - warmStart).count();
    backend.calls = 0;
    backend.shards = 0;
    Metrics::GetAllStatsAndClear();
    std::vector<double> latency;
    std::mutex mutex;
    std::exception_ptr failure;
    std::vector<std::thread> workers;
    const auto cpuStart = std::clock();
    const auto wallStart = Clock::now();
    for (size_t worker = 0; worker != options.concurrency; ++worker) {
        workers.emplace_back([&, worker] {
            try {
                for (size_t i = 0; i != options.iterations; ++i) {
                    const auto ms = execute(worker, i);
                    std::lock_guard<std::mutex> lock(mutex);
                    latency.push_back(ms);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure) { failure = std::current_exception(); }
            }
        });
    }
    for (auto& worker : workers) { worker.join(); }
    if (failure) { std::rethrow_exception(failure); }
    const auto wall = std::chrono::duration<double>(Clock::now() - wallStart).count();
    const auto cpu = double(std::clock() - cpuStart) / CLOCKS_PER_SEC;
    std::sort(latency.begin(), latency.end());
    double summedLoadMs = 0;
    for (const auto ms : latency) { summedLoadMs += ms; }
    auto percentile = [&](double p) {
        return latency[static_cast<size_t>((latency.size() - 1) * p)];
    };
    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    auto counter = [&](const char* name) {
        auto it = counters.find(name);
        return it == counters.end() ? 0.0 : it->second;
    };
    const auto total = options.shards * options.iterations * options.concurrency;
    // Hot keys are private to each serial worker and cold keys are unique, so this
    // workload has no shared LOADING observers. Every backend shard is a real miss.
    const auto waitShards = static_cast<double>(backend.shards.load());
    std::cout << std::setprecision(9) << "{\"scope\":\"CacheStore-Load/Posix/simu\",\"graph\":"
              << (graph ? "true" : "false") << ",\"dataset\":" << std::quoted(path)
              << ",\"engine\":" << std::quoted(options.engine)
              << ",\"run_order\":" << std::quoted(options.order)
              << ",\"io_direct\":" << (options.direct ? "true" : "false")
              << ",\"shard_bytes\":" << options.bytes << ",\"request_shards\":" << options.shards
              << ",\"batch_cap\":" << options.batch << ",\"concurrency\":" << options.concurrency
              << ",\"requested_hit_percent\":" << options.hit << ",\"measured_shards\":" << total
              << ",\"backend_submissions\":" << backend.calls.load()
              << ",\"backend_shards\":" << backend.shards.load() << ",\"actual_mean_batch\":"
              << (backend.calls.load() ? double(backend.shards.load()) / backend.calls.load() : 0)
              << ",\"ready_hits\":" << double(total) - waitShards
              << ",\"wait_shards\":" << waitShards << ",\"warmup_ms\":" << warmMs
              << ",\"wall_seconds_including_validation\":" << wall
              << ",\"process_cpu_seconds\":" << cpu << ",\"GiB_per_second_including_validation\":"
              << double(total) * options.bytes / wall / (1ULL << 30)
              << ",\"p50_ms\":" << percentile(.50) << ",\"p95_ms\":" << percentile(.95)
              << ",\"p99_ms\":" << percentile(.99)
              << ",\"latency_scope\":\"Load-submit-through-Wait\",\"summed_request_load_ms\":"
              << summedLoadMs
              << ",\"plan_hits\":" << counter("cache_transfer_graph_plan_hits_total")
              << ",\"plan_misses\":" << counter("cache_transfer_graph_plan_misses_total")
              << ",\"buffer_capacity_bytes\":1073741824,\"waiting_queue_depth\":256,\"running_"
                 "queue_depth\":256"
              << ",\"plan_cache_entries_cap\":32,\"peak_leases\":null,\"gpu_time\":null}\n";
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        Require(argc >= 2,
                "usage: transfer_graph_benchmark PARENT [shard_bytes shards batch concurrency "
                "iterations hit_percent psync io_direct_0_or_1 off-on|on-off]");
        Options o;
        o.parent = argv[1];
        size_t* values[] = {&o.bytes, &o.shards, &o.batch, &o.concurrency, &o.iterations, &o.hit};
        for (int i = 2; i < argc && i < 8; ++i) { *values[i - 2] = std::stoull(argv[i]); }
        if (argc > 8) { o.engine = argv[8]; }
        if (argc > 9) { o.direct = std::string(argv[9]) == "1"; }
        if (argc > 10) { o.order = argv[10]; }
        Require(
            argc <= 11 && (argc <= 9 || std::string(argv[9]) == "0" || std::string(argv[9]) == "1"),
            "invalid benchmark arguments");
        Require(o.bytes >= 4096 && o.bytes <= 1048576 && o.bytes % 4096 == 0 && o.shards &&
                    o.shards <= 4096 && o.batch && o.batch <= 65536 && o.concurrency &&
                    o.concurrency <= 64 && o.iterations && o.iterations <= 100000 && o.hit <= 100 &&
                    o.engine == "psync" && (o.order == "off-on" || o.order == "on-off"),
                "invalid benchmark options");
        Require(o.shards * o.concurrency * 2 <= (1ULL << 30) / o.bytes,
                "workload exceeds conservative legacy-path buffer budget");
        Require(std::filesystem::is_directory(o.parent),
                "PARENT must be an existing test directory");
        const auto path = std::filesystem::path(o.parent) /
                          ("ucm-transfer-graph-" + std::to_string(getpid()) + "-" +
                           std::to_string(Clock::now().time_since_epoch().count()));
        Require(std::filesystem::create_directory(path), "cannot create fresh benchmark directory");
        std::cerr << "Dedicated dataset retained at " << path << '\n';
        CountedStore backend;
        UC::Detail::Dictionary config;
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", o.bytes);
        config.SetNumber("shard_size", o.bytes);
        config.SetNumber("block_size", o.bytes);
        config.Set("storage_backends", std::vector<std::string>{path.string()});
        config.Set("posix_io_engine", o.engine);
        config.Set("cache_transfer_graph", true);  // same terminal backend semantics in both modes
        config.Set("io_direct", o.direct);
        config.SetNumber("data_dir_shard_bytes", 0);
        config.SetNumber("timeout_ms", 600000);
        Good(backend.Setup(config));
        const auto hot = o.shards * o.hit / 100;
        Seed(backend, o.concurrency * (hot + (o.iterations + 1) * (o.shards - hot)), o.bytes);
        Metrics::SetUp();
        for (auto name :
             {"cache_load_wait_shards_total", "cache_load_shards_total",
              "cache_transfer_graph_plan_hits_total", "cache_transfer_graph_plan_misses_total"}) {
            Metrics::CreateStats(name, "counter");
        }
        const bool graphFirst = o.order == "on-off";
        Run(backend, o, path.string(), graphFirst);
        Run(backend, o, path.string(), !graphFirst);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
