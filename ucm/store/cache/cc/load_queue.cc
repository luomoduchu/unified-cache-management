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
#include "load_queue.h"
#include <algorithm>
#include "load_wait_completion.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

LoadQueue::~LoadQueue()
{
    {
        std::lock_guard<std::mutex> lock(submitMutex_);
        accepting_ = false;
        stop_.store(true);
    }
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    {
        std::lock_guard<std::mutex> lock(submitMutex_);
        if (setupAttempted_) {
            return Status::InvalidParam("load queue Setup may only be called once");
        }
        setupAttempted_ = true;
    }
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    nShardPerBlock_ = config.shardSize ? config.blockSize / config.shardSize : 0;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cpuAffinityCores_ = config.cpuAffinityCores;
    localRankSize_ = config.localRankSize;
    graphEnabled_ = config.cacheTransferGraph;
    if (graphEnabled_ && (cacheIOAggregation_ || useGdr_ || cacheSdmaDirect_)) {
        return Status::InvalidParam(
            "transfer graph requires ordinary copy streams; aggregation, GDR and SDMA Direct are "
            "unsupported");
    }
    if (graphEnabled_ && !HasDrainingLoadWait(backend_)) {
        return Status::InvalidParam(
            "transfer graph backend must advertise terminal Load Wait (strict Posix psync "
            "required)");
    }
    if (graphEnabled_) {
        if (!backend_ || !buffer_ || !failureSet_ || deviceId_ < 0 || !localRankSize_ ||
            !config.shardSize || !config.blockSize || config.blockSize % config.shardSize ||
            config.bufferCapacity < config.shardSize || config.waitingQueueDepth < 2 ||
            config.runningQueueDepth < 2 || tensorSizes_.empty()) {
            return Status::InvalidParam("invalid transfer graph layout or queue limits");
        }
        size_t total = 0;
        for (auto bytes : tensorSizes_) {
            if (bytes > config.shardSize - total) {
                return Status::InvalidParam("invalid transfer graph tensor sizes");
            }
            total += bytes;
        }
        if (!total) { return Status::InvalidParam("empty transfer graph tensor layout"); }
        try {
            graph_.Setup(deviceId_, localRankSize_, config.cacheGraphBatchShards,
                         config.cacheGraphBatchBytes, config.shardSize,
                         config.cacheGraphPlanCacheEntries);
        } catch (const std::exception& e) {
            return Status::InvalidParam(e.what());
        }
    }
    waiting_.Setup(config.waitingQueueDepth);
    if (graphEnabled_) {
        graphRunning_.Setup(config.runningQueueDepth);
    } else {
        running_.Setup(config.runningQueueDepth);
    }
    holder_.reserve(1024);
    std::promise<Status> started;
    auto fut = started.get_future();
    auto status = Status::OK();
    try {
        transfer_ = MakeWorker([this, &started] { TransferStage(started); });
        status = fut.get();
        if (status.Success()) {
            dispatcher_ = MakeWorker([this] { DispatchStage(); });
        }
    } catch (const std::exception& e) {
        status = Status::Error(e.what());
    }
    if (status.Failure()) {
        stop_.store(true);
        dispatcherDone_.store(true, std::memory_order_release);
        if (transfer_.joinable()) { transfer_.join(); }
        return status;
    }
    {
        std::lock_guard<std::mutex> lock(submitMutex_);
        accepting_ = true;
    }
    return Status::OK();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    std::lock_guard<std::mutex> lock(submitMutex_);  // waiting_ has a single producer
    waiter->Up();
    auto success = accepting_ && !stop_.load() && waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
    RecordFailedShards(task->desc.size());
    task->Fail(Status::Error("load queue is unavailable or full"));
    if (failureSet_) { failureSet_->Insert(task->id); }
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM load dispatcher name.", nameStatus);
    }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    if (!graphEnabled_) {
        waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
        return;
    }
    for (;;) {
        TaskPair pair;
        if (waiting_.TryPop(pair)) {
            DispatchGraphTask(std::move(pair));
        } else if (stop_.load(std::memory_order_acquire)) {
            if (waiting_.TryPop(pair)) {
                DispatchGraphTask(std::move(pair));
            } else {
                break;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    dispatcherDone_.store(true, std::memory_order_release);
}

static std::vector<size_t> RearrangeIndex(size_t n, size_t iProc, size_t nProc)
{
    std::vector<size_t> order;
    order.reserve(n);
    for (size_t r = 0; r < nProc; ++r) {
        size_t slice = (iProc + r) % nProc;
        for (size_t j = 0;; ++j) {
            size_t i = slice + j * nProc;
            if (i >= n) { break; }
            order.push_back(i);
        }
    }
    return order;
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    size_t backendSubmitCount = 0;
    size_t waitShardCount = 0;
    const auto indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[indexes[i]];
        ShardTask shardTask;
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, true);
        shardTask.backendTaskHandle = 0;
        shardTask.fromPosix = !shardTask.bufferHandle.Ready();
        if (shardTask.fromPosix) { waitShardCount++; }
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            auto res = backend_->Load(std::move(backendTask));
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
                RecordLoadSourceShards(i + 1, waitShardCount);
                RecordFailedShards(nShard - i);
                shardTask.bufferHandle.MarkFailed(res.Error());
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            backendSubmitCount++;
        }
        shardTask.task = task;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    auto tpDispatch = NowTime::Now();
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[indexes[i]];
        if (shard.index + 1 != nShardPerBlock_) {
            buffer_->Prealloc(shard.owner, shard.index + 1, true);
        }
    }
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    RecordLoadSourceShards(nShard, waitShardCount);
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = Status::OK();
    try {
        s = SetupCopyStream(stream);
    } catch (const std::exception& e) {
        s = Status::Error(e.what());
    }
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    if (!graphEnabled_) {
        running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
        return;
    }
    for (;;) {
        GraphBatch batch;
        if (graphRunning_.TryPop(batch)) {
            TransferGraphBatch(stream, std::move(batch));
        } else if (dispatcherDone_.load(std::memory_order_acquire)) {
            // Recheck after acquiring the producer's final publication.
            if (graphRunning_.TryPop(batch)) {
                TransferGraphBatch(stream, std::move(batch));
            } else {
                break;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

Status LoadQueue::SetupCopyStream(CopyStream& stream)
{
    if (cacheIOAggregation_) { return stream.SetupIoAggregation(deviceId_, useGdr_); }
    if (cacheSdmaDirect_) { return stream.SetupSdmaDirect(deviceId_, useGdr_); }
    return stream.Setup(deviceId_, streamNumber_, useGdr_);
}

void LoadQueue::DispatchGraphTask(TaskPair&& pair)
{
    const auto& task = pair.first;
    auto fail = [&](const Status& s) {
        task->Fail(s);
        failureSet_->Insert(task->id);
    };
    GraphBatch batch;
    std::shared_ptr<const UcmTransferPlan> plan;
    try {
        // Validate before touching buffers or moving any destination arrays.
        for (const auto& shard : task->desc) {
            if (shard.index >= nShardPerBlock_ || shard.addrs.size() != tensorSizes_.size()) {
                fail(Status::InvalidParam("invalid graph shard index or tensor addresses"));
                break;
            }
        }
        const auto hitsBefore = graph_.Hits();
        plan = graph_.Prepare(task->desc.size());
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_transfer_graph_plan_hits_total"),
                                 graph_.Hits() - hitsBefore);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_transfer_graph_plan_misses_total"),
                                 graph_.Hits() == hitsBefore ? 1.0 : 0.0);
        graphPlanHits_.store(graph_.Hits());
        size_t position = 0;
        while (position < plan->Size() && !failureSet_->Contains(task->id)) {
            if (stop_.load()) {
                fail(Status::Error("load queue stopping"));
                break;
            }
            batch = GraphBatch{};
            batch.task = task;
            batch.plan = plan;
            batch.shards.reserve(plan->Window());
            Detail::TaskDesc reads;
            reads.brief = "GraphBackend2Cache";
            reads.reserve(plan->Window());
            while (position < plan->Size() && batch.shards.size() < plan->Window()) {
                auto& shard = task->desc[plan->Index(position)];
                auto lease = buffer_->TryGet(shard.owner, shard.index, true, true);
                if (!lease) { break; }  // flush before waiting for more buffers
                ShardTask bound;
                bound.task = task;
                bound.fromPosix = !lease.Ready();
                bound.bufferHandle = std::move(lease);
                bound.shard = std::move(shard);
                batch.shards.push_back(std::move(bound));
                auto& acquired = batch.shards.back();
                if (acquired.bufferHandle.Owner() && !acquired.bufferHandle.Ready()) {
                    reads.push_back({acquired.shard.owner,
                                     acquired.shard.index,
                                     {acquired.bufferHandle.Data()}});
                }
                ++position;
            }
            if (batch.shards.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            if (!reads.empty()) {
                auto count = reads.size();
                auto result = backend_->Load(std::move(reads));
                if (result) {
                    batch.backendHandle = result.Value();
                    batch.submitted = true;  // handle zero is not used as a sentinel
                    graphBackendSubmits_.fetch_add(1);
                    graphBackendShards_.fetch_add(count);
                    UC::Metrics::UpdateStats(
                        NAME_TO_METRIC_ID("cache_transfer_graph_backend_batches_total"), 1.0);
                    UC::Metrics::UpdateStats(
                        NAME_TO_METRIC_ID("cache_transfer_graph_backend_shards_total"), count);
                } else {
                    for (auto& shard : batch.shards) {
                        if (shard.bufferHandle.Owner() && !shard.bufferHandle.Ready()) {
                            shard.bufferHandle.MarkFailed(result.Error());
                        }
                    }
                    fail(result.Error());
                }
            }
            graphRunning_.Push(std::move(batch));
            batch.submitted = false;  // ownership of the handle moved to the transfer queue
        }
        if (stop_.load() && !failureSet_->Contains(task->id)) {
            fail(Status::Error("load queue stopping"));
        }
    } catch (const std::exception& e) {
        // Unpublished batch: drain before any owner lease can be destroyed.
        auto error = Status::Error(e.what());
        if (batch.submitted) {
            backend_->Wait(batch.backendHandle);
            batch.submitted = false;
        }
        for (auto& shard : batch.shards) {
            if (shard.bufferHandle.Owner() && !shard.bufferHandle.Ready()) {
                shard.bufferHandle.MarkFailed(error);
            }
        }
        fail(error);
    }
    GraphBatch end;
    end.task = task;
    end.waiter = pair.second;
    end.plan = std::move(plan);
    graphRunning_.Push(std::move(end));
}

void LoadQueue::TransferGraphBatch(CopyStream& stream, GraphBatch&& batch)
{
    if (batch.waiter) {
        batch.waiter->Done();
        return;
    }
    auto status = Status::OK();
    // Always drain every accepted backend handle, including after parent failure.
    if (batch.submitted) {
        status = backend_->Wait(batch.backendHandle);
        for (auto& shard : batch.shards) {
            auto& lease = shard.bufferHandle;
            if (lease.Owner() && !lease.Ready()) {
                if (status.Success()) {
                    lease.MarkReady();
                } else {
                    lease.MarkFailed(status);
                }
            }
        }
    }
    if (status.Success() && !failureSet_->Contains(batch.task->id)) {
        for (auto& shard : batch.shards) {
            // Owner reads have completed above. Readers never publish owner state.
            while (shard.bufferHandle.GetState() == TransBuffer::State::LOADING &&
                   !failureSet_->Contains(batch.task->id) && !stop_.load()) {
                std::this_thread::yield();
            }
            if (failureSet_->Contains(batch.task->id) || stop_.load()) {
                status = Status::Error("graph load cancelled");
                break;
            }
            if (shard.bufferHandle.GetState() == TransBuffer::State::FAILED) {
                status = shard.bufferHandle.FailureStatus();
                break;
            }
            auto host =
                cacheSdmaDirect_ ? shard.bufferHandle.DeviceData() : shard.bufferHandle.Data();
            status = HostToDeviceAsync(stream, host, shard.shard.addrs.data());
            if (status.Failure()) { break; }
        }
    }
    // Also fence on partial submission failure. Leases AND address arrays live until here.
    auto fence = stream.Synchronize();
    if (status.Success()) { status = fence; }
    const bool success = status.Success() && !failureSet_->Contains(batch.task->id);
    RecordLoadSourceShards(batch.shards.size(),
                           std::count_if(batch.shards.begin(), batch.shards.end(),
                                         [](const ShardTask& s) { return s.fromPosix; }));
    RecordShardResults(batch.shards, nullptr, success);
    if (status.Failure()) {
        batch.task->Fail(status);
        failureSet_->Insert(batch.task->id);
    }
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& task)
{
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    if (failureSet_->Contains(taskHandle)) {
        RecordFailedShards(1);
        if (task.waiter) {
            holder_.clear();
            task.waiter->Done();
        }
        return;
    }

    auto s = Status::OK();
    auto waiter = task.waiter;
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] {
            RecordShardResults(holder_, &task, false);
            break;
        }
        auto tpBackendReady = NowTime::Now();
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);

        auto* host = cacheSdmaDirect_ ? task.bufferHandle.DeviceData() : task.bufferHandle.Data();
        s = HostToDeviceAsync(stream, host, task.shard.addrs.data());
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            RecordShardResults(holder_, &task, false);
            break;
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        s = stream.Synchronize();
        auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
        RecordH2dSyncMetrics(h2dSyncMs);
        RecordShardResults(holder_, &task, s.Success());
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] {
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
    }
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
            task.bufferHandle.MarkFailed(s);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    for (;;) {
        auto state = task.bufferHandle.GetState();
        if (state == TransBuffer::State::READY) { return Status::OK(); }
        if (state == TransBuffer::State::FAILED) { return task.bufferHandle.FailureStatus(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_);
}

void LoadQueue::RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                                   bool success) const
{
    size_t cache = 0;
    size_t posix = 0;
    for (const auto& task : tasks) {
        if (task.fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (extra != nullptr) {
        if (extra->fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (!success) {
        RecordFailedShards(cache + posix);
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_success_shards_total"),
                             static_cast<double>(cache));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_posix_load_success_shards_total"),
                             static_cast<double>(posix));
}

void LoadQueue::RecordFailedShards(size_t count) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_failed_shards_total"),
                             static_cast<double>(count));
}

void LoadQueue::RecordLoadSourceShards(size_t total, size_t wait) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(total));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_wait_shards_total"),
                             static_cast<double>(wait));
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

}  // namespace UC::CacheStore
