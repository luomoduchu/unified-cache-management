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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_LOAD_QUEUE_H
#define UNIFIEDCACHE_CACHE_STORE_CC_LOAD_QUEUE_H

#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include "copy_stream.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "transfer_graph.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct ShardTask {
        TaskPtr task;
        Detail::Shard shard;
        TransBuffer::Handle bufferHandle;
        Detail::TaskHandle backendTaskHandle;
        WaiterPtr waiter;
        bool fromPosix{false};
    };
    struct GraphBatch {
        TaskPtr task;
        WaiterPtr waiter;  // only the final marker owns the parent's completion
        std::shared_ptr<const UcmTransferPlan> plan;
        std::vector<ShardTask> shards;
        Detail::TaskHandle backendHandle{0};
        bool submitted{false};
    };

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    std::vector<size_t> tensorSizes_{};
    size_t nShardPerBlock_{0};
    size_t streamNumber_{1};
    bool useGdr_{false};
    bool cacheIOAggregation_{false};
    bool cacheSdmaDirect_{false};
    std::vector<ssize_t> cpuAffinityCores_{};
    size_t localRankSize_{};
    bool graphEnabled_{false};
    UcmTransferGraph graph_;
    std::mutex submitMutex_;
    bool accepting_{false};  // protected by submitMutex_
    bool setupAttempted_{false};
    std::atomic_bool dispatcherDone_{false};
    SpscRingQueue<GraphBatch> graphRunning_;
    std::atomic<size_t> graphBackendSubmits_{0}, graphBackendShards_{0}, graphPlanHits_{0};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::vector<ShardTask> holder_;

public:
    virtual ~LoadQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);
    size_t GraphBackendSubmits() const { return graphBackendSubmits_.load(); }
    size_t GraphBackendShards() const { return graphBackendShards_.load(); }
    size_t GraphPlanHits() const { return graphPlanHits_.load(); }

protected:
    virtual Status SetupCopyStream(CopyStream& stream);
    virtual std::thread MakeWorker(std::function<void()> worker)
    {
        return std::thread(std::move(worker));
    }

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void DispatchGraphTask(TaskPair&& pair);
    void TransferGraphBatch(CopyStream& stream, GraphBatch&& batch);
    void TransferStage(std::promise<Status>& started);
    void TransferOneTask(CopyStream& stream, ShardTask&& task);
    Status WaitBackendTaskReady(ShardTask& task);
    Status HostToDeviceAsync(CopyStream& stream, void* host, void** device);
    void RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                            bool success) const;
    void RecordLoadSourceShards(size_t total, size_t wait) const;
    void RecordFailedShards(size_t count) const;
    void RecordH2dSyncMetrics(double h2dSyncMs) const;
};

}  // namespace UC::CacheStore

#endif
