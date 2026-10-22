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
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include "cache/cc/load_queue.h"
#include "cache/cc/trans_manager.h"
#include "detail/mock_store.h"
#include "load_wait_completion.h"
#include "trans/simu/simu_stream.h"

extern "C" UC::StoreV1* MakeCacheStore();

namespace {
using namespace UC;
using namespace UC::CacheStore;
using namespace testing;

class GraphCapableMock : public UC::Test::Detail::MockStore, public LoadWaitCompletion {
public:
    bool LoadWaitDrains() const noexcept override { return true; }
};

class GraphQueueTest : public testing::Test {
protected:
    NiceMock<GraphCapableMock> backend;
    HashSet<UC::Detail::TaskHandle> failures;
    Config config;
    TransBuffer buffer;
    std::atomic<size_t> submissions{0}, reads{0}, waits{0};
    void SetUp() override
    {
        config.storeBackend = &backend;
        config.deviceId = 0;
        config.shardSize = 128;
        config.blockSize = 128 * 32;
        config.tensorSizes = {31, 65};
        config.bufferCapacity = 128 * 2;  // deliberately smaller than a request/window
        config.loadExclusiveBufferNumber = 0;
        config.shareBufferEnable = false;
        config.ioDirect = false;
        config.cacheTransferGraph = true;
        config.cacheGraphBatchShards = 8;
        config.cacheGraphBatchBytes = 1024;
        config.waitingQueueDepth = 64;
        config.runningQueueDepth = 4;
        config.localRankSize = 1;
        config.uniqueId = "transfer-graph-test";
        ON_CALL(backend, Load)
            .WillByDefault(
                Invoke([&](UC::Detail::TaskDesc desc) -> Expected<UC::Detail::TaskHandle> {
                    reads += desc.size();
                    for (const auto& shard : desc) {
                        std::memset(shard.addrs[0], static_cast<int>(shard.index + 17),
                                    config.shardSize);
                    }
                    return ++submissions;
                }));
        ON_CALL(backend, Wait).WillByDefault(Invoke([&](UC::Detail::TaskHandle) {
            ++waits;
            return Status::OK();
        }));
        ASSERT_TRUE(buffer.Setup(config).Success());
    }
    struct Request {
        std::vector<std::vector<unsigned char>> data;
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter = std::make_shared<Latch>();
        explicit Request(const std::vector<size_t>& indexes)
        {
            UC::Detail::TaskDesc desc;
            data.resize(indexes.size(), std::vector<unsigned char>(96, 0));
            for (size_t i = 0; i < indexes.size(); ++i) {
                desc.push_back({
                    {},
                    indexes[i], {data[i].data(), data[i].data() + 31}
                });
            }
            task = std::make_shared<TransTask>(TransTask::Type::LOAD, std::move(desc));
        }
        void Check(const std::vector<size_t>& indexes) const
        {
            for (size_t i = 0; i < indexes.size(); ++i) {
                for (auto byte : data[i]) { ASSERT_EQ(byte, indexes[i] + 17); }
            }
        }
    };
    void Run(LoadQueue& queue, Request& request)
    {
        queue.Submit(request.task, request.waiter);
        ASSERT_TRUE(request.waiter->WaitForDuration(3000));
        ASSERT_FALSE(failures.Contains(request.task->id));
    }
};

TEST_F(GraphQueueTest, LargeRequestFlushesBeforeResourceWaitAndRebindsAddresses)
{
    config.localRankSize = 3;
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    const std::vector<size_t> indexes{0, 1, 2, 3, 4, 5, 6, 7, 8};
    Request request(indexes);
    Run(queue, request);
    request.Check(indexes);
    EXPECT_EQ(reads, 9);
    EXPECT_EQ(waits, submissions);
    EXPECT_GE(submissions, 1);
    EXPECT_LE(submissions, 9);
    Request rebound({7, 8});
    Run(queue, rebound);
    rebound.Check({7, 8});
    EXPECT_EQ(queue.GraphBackendShards(), reads.load());
}

TEST_F(GraphQueueTest, SufficientBufferBatchesAllMissesAtConfiguredLimit)
{
    config.bufferCapacity = config.shardSize * 16;
    config.cacheGraphBatchShards = 4;
    ASSERT_TRUE(buffer.Setup(config).Success());
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    const std::vector<size_t> indexes{0, 1, 2, 3, 4, 5, 6, 7, 8};
    Request request(indexes);
    Run(queue, request);
    request.Check(indexes);
    EXPECT_EQ(reads, 9);
    EXPECT_EQ(submissions, 3);  // 4 + 4 + 1, with no pool pressure or competing allocator
    EXPECT_EQ(waits, submissions);
}

TEST_F(GraphQueueTest, DuplicateAndAllHitOnlyOwnersReadOnce)
{
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request first({0, 1, 0});
    Run(queue, first);
    first.Check({0, 1, 0});
    EXPECT_EQ(reads, 2);
    EXPECT_EQ(submissions, 1);
    Request hit({0, 1, 0});
    Run(queue, hit);
    hit.Check({0, 1, 0});
    EXPECT_EQ(reads, 2);
    EXPECT_EQ(queue.GraphPlanHits(), 1);
}

TEST_F(GraphQueueTest, MixedHitsAndBackendOnlyBypass)
{
    config.cacheLoadBackendOnly = true;
    ASSERT_TRUE(buffer.Setup(config).Success());
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request first({0}), mixed({0, 1}), repeat({0, 1});
    Run(queue, first);
    Run(queue, mixed);
    Run(queue, repeat);
    repeat.Check({0, 1});
    EXPECT_EQ(reads, 5);
}

TEST_F(GraphQueueTest, SharedReaderDoesNotPublishFailedState)
{
    auto owner = buffer.Get({}, 0, true, true);
    EXPECT_CALL(backend, Load).Times(0);
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0});
    queue.Submit(request.task, request.waiter);
    owner.MarkFailed(Status::NotFound());
    ASSERT_TRUE(request.waiter->WaitForDuration(3000));
    EXPECT_EQ(request.task->FailureStatus(), Status::NotFound());
    EXPECT_EQ(owner.GetState(), TransBuffer::State::FAILED);
}

TEST_F(GraphQueueTest, FailureAfterEarlierSubmitDrainsBeforeParentCompletion)
{
    config.cacheGraphBatchShards = 1;
    std::promise<void> entered, release, failed;
    auto gate = release.get_future().share();
    EXPECT_CALL(backend, Load)
        .WillOnce(Return(UC::Detail::TaskHandle(1)))
        .WillOnce(Invoke([&](UC::Detail::TaskDesc) -> Expected<UC::Detail::TaskHandle> {
            failed.set_value();
            return Status::NotFound();
        }));
    EXPECT_CALL(backend, Wait).WillOnce(Invoke([&](UC::Detail::TaskHandle handle) {
        EXPECT_EQ(handle, 1);
        entered.set_value();
        gate.wait();
        return Status::OK();
    }));
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0, 1, 2});
    queue.Submit(request.task, request.waiter);
    const auto waiting = entered.get_future().wait_for(std::chrono::seconds(3));
    const auto failure = failed.get_future().wait_for(std::chrono::seconds(3));
    EXPECT_EQ(waiting, std::future_status::ready);
    EXPECT_EQ(failure, std::future_status::ready);
    EXPECT_FALSE(request.waiter->Check());
    auto observer = buffer.TryGet({}, 0, true, true);
    EXPECT_TRUE(observer);
    if (observer) { EXPECT_FALSE(observer.Owner()); }  // lease retained during drain
    release.set_value();
    ASSERT_TRUE(request.waiter->WaitForDuration(3000));
    EXPECT_EQ(request.task->FailureStatus(), Status::NotFound());
}

TEST_F(GraphQueueTest, BackendBatchFailurePublishesFailureToEveryOwner)
{
    EXPECT_CALL(backend, Wait).WillOnce(Return(Status::NotFound()));
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0, 1});
    queue.Submit(request.task, request.waiter);
    ASSERT_TRUE(request.waiter->WaitForDuration(3000));
    EXPECT_TRUE(failures.Contains(request.task->id));
    for (const auto& dst : request.data) {
        EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](unsigned char x) { return x == 0; }));
    }
    EXPECT_EQ(submissions, 1);
}

TEST_F(GraphQueueTest, ConcurrentProducersAndEmptyRequestsComplete)
{
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    std::vector<std::unique_ptr<Request>> requests;
    for (size_t i = 0; i != 16; ++i) { requests.emplace_back(new Request({0, 1})); }
    std::vector<std::thread> producers;
    for (auto& request : requests) {
        auto ptr = request.get();
        producers.emplace_back([&queue, ptr] { queue.Submit(ptr->task, ptr->waiter); });
    }
    for (auto& producer : producers) { producer.join(); }
    for (auto& request : requests) {
        ASSERT_TRUE(request->waiter->WaitForDuration(3000));
        request->Check({0, 1});
    }
    Request empty({});
    Run(queue, empty);
    EXPECT_EQ(reads, 2);
}

TEST_F(GraphQueueTest, ShutdownDrainsAcceptedTasksAndCancelsResourceWait)
{
    auto held0 = buffer.Get({}, 0, true, true);
    auto held1 = buffer.Get({}, 1, true, true);
    auto queue = std::make_unique<LoadQueue>();
    ASSERT_TRUE(queue->Setup(config, &failures, &buffer).Success());
    Request request({2, 3});
    queue->Submit(request.task, request.waiter);
    queue.reset();
    EXPECT_TRUE(request.waiter->Check());
    EXPECT_TRUE(failures.Contains(request.task->id));
    EXPECT_EQ(submissions, 0);
}

TEST_F(GraphQueueTest, InvalidBindingsFailAndDoNotSubmit)
{
    EXPECT_CALL(backend, Load).Times(0);
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request index({32}), layout({0});
    layout.task->desc[0].addrs.pop_back();
    for (auto request : {&index, &layout}) {
        queue.Submit(request->task, request->waiter);
        ASSERT_TRUE(request->waiter->WaitForDuration(3000));
        EXPECT_TRUE(failures.Contains(request->task->id));
    }
}

TEST_F(GraphQueueTest, CacheStoreRejectsInvalidAndUnknownGraphConfiguration)
{
    for (const auto& key : {"cache_graph_batch_shards", "cache_graph_batch_bytes",
                            "cache_graph_plan_cache_entries", "cache_graph_typo"}) {
        UC::Detail::Dictionary settings;
        settings.Set("store_backend", static_cast<StoreV1*>(&backend));
        settings.Set("unique_id", std::string("graph-invalid-config"));
        settings.SetNumber("device_id", 0);
        settings.SetNumber("tensor_size", 128);
        settings.SetNumber("shard_size", 128);
        settings.SetNumber("block_size", 128);
        settings.Set("cache_transfer_graph", true);
        settings.SetNumber(key, -1);
        std::unique_ptr<StoreV1> cache(MakeCacheStore());
        EXPECT_TRUE(cache->Setup(settings).Failure());
    }
    auto invalid = config;
    invalid.cacheGraphBatchShards = 0;
    LoadQueue queue;
    EXPECT_TRUE(queue.Setup(invalid, &failures, &buffer).Failure());
}

TEST_F(GraphQueueTest, SubmissionExceptionFailsOwnerAndCompletesOnce)
{
    EXPECT_CALL(backend, Load).WillOnce(Throw(std::runtime_error("injected allocation failure")));
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0, 1});
    std::atomic<size_t> done{0};
    std::promise<void> epilog;
    request.waiter->SetEpilog([&] {
        ++done;
        epilog.set_value();
    });
    queue.Submit(request.task, request.waiter);
    ASSERT_TRUE(request.waiter->WaitForDuration(3000));
    ASSERT_EQ(epilog.get_future().wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(failures.Contains(request.task->id));
    EXPECT_EQ(done, 1);
}

TEST_F(GraphQueueTest, GraphRejectsCopyModesWithoutFailureDrainGuarantee)
{
    for (size_t mode : {0, 1, 2}) {
        auto unsupported = config;
        unsupported.cacheIOAggregation = mode == 0;
        unsupported.useGdr = mode == 1;
        unsupported.cacheSdmaDirect = mode == 2;
        LoadQueue queue;
        EXPECT_TRUE(queue.Setup(unsupported, &failures, &buffer).Failure());
    }
}

TEST_F(GraphQueueTest, GraphRejectsBackendWithoutTerminalWaitCapability)
{
    NiceMock<UC::Test::Detail::MockStore> unknown;
    auto unsupported = config;
    unsupported.storeBackend = &unknown;
    LoadQueue queue;
    EXPECT_TRUE(queue.Setup(unsupported, &failures, &buffer).Failure());
    EXPECT_FALSE(HasDrainingLoadWait(&unknown));
    EXPECT_FALSE(HasDrainingLoadWait(nullptr));
    UC::Detail::Dictionary settings;
    settings.Set("store_backend", static_cast<StoreV1*>(&unknown));
    settings.Set("unique_id", std::string("graph-unknown-backend"));
    settings.SetNumber("device_id", 0);
    settings.SetNumber("tensor_size", 128);
    settings.SetNumber("shard_size", 128);
    settings.SetNumber("block_size", 128);
    settings.Set("cache_transfer_graph", true);
    std::unique_ptr<StoreV1> cache(MakeCacheStore());
    EXPECT_TRUE(cache->Setup(settings).Failure());
}

class StartupFailureQueue : public LoadQueue {
    size_t mode_, workers_{0};

public:
    explicit StartupFailureQueue(size_t mode) : mode_(mode) {}

protected:
    std::thread MakeWorker(std::function<void()> worker) override
    {
        if (++workers_ == mode_) {
            throw std::system_error(
                std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        return LoadQueue::MakeWorker(std::move(worker));
    }
    Status SetupCopyStream(CopyStream& stream) override
    {
        if (mode_ == 3) { return Status::Error("injected stream Setup failure"); }
        if (mode_ == 4) { throw std::runtime_error("injected stream Setup exception"); }
        return LoadQueue::SetupCopyStream(stream);
    }
};

TEST_F(GraphQueueTest, StartupFailuresJoinWorkersAndRejectSubmissions)
{
    for (size_t mode : {1, 2, 3, 4}) {
        StartupFailureQueue queue(mode);
        EXPECT_TRUE(queue.Setup(config, &failures, &buffer).Failure());
        Request request({0});
        queue.Submit(request.task, request.waiter);
        EXPECT_TRUE(request.waiter->Check());
        EXPECT_TRUE(failures.Contains(request.task->id));
    }
    EXPECT_EQ(submissions, 0);
}

TEST_F(GraphQueueTest, NullAndZeroGhostSlotsPreserveTensorLayout)
{
    config.tensorSizes = {31, 0, 65, 32};
    LoadQueue queue;
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0});
    request.task->desc[0].addrs = {request.data[0].data(), nullptr, request.data[0].data() + 31,
                                   nullptr};
    Run(queue, request);
    request.Check({0});
    EXPECT_EQ(reads, 1);
}

class TimeoutGraphManager : public TransManager {
public:
    bool Failed(UC::Detail::TaskHandle id) { return failureSet_.Contains(id); }
};

TEST_F(GraphQueueTest, ParentTaskWrapperTimeoutDrainsBeforeReturning)
{
    config.timeoutMs = 1;
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    EXPECT_CALL(backend, Wait).WillOnce(Invoke([&](UC::Detail::TaskHandle) {
        entered.set_value();
        gate.wait();
        return Status::OK();
    }));
    TimeoutGraphManager manager;
    ASSERT_TRUE(manager.Setup(config, &buffer).Success());
    Request request({0});
    auto submitted = manager.Submit(TransTask(TransTask::Type::LOAD, request.task->desc));
    ASSERT_TRUE(submitted);
    const auto handle = submitted.Value();
    EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(3)), std::future_status::ready);
    auto waiting = std::async(std::launch::async, [&] { return manager.Wait(handle); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!manager.Failed(handle) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(manager.Failed(handle));  // real TaskWrapper timeout cancellation observed
    EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(1)), std::future_status::timeout);
    auto lease = buffer.TryGet({}, 0, true, true);
    EXPECT_TRUE(lease);
    if (lease) { EXPECT_FALSE(lease.Owner()); }
    release.set_value();
    ASSERT_EQ(waiting.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(waiting.get(), Status::Timeout());
    EXPECT_TRUE(std::all_of(request.data[0].begin(), request.data[0].end(),
                            [](unsigned char byte) { return byte == 0; }));
}

TEST_F(GraphQueueTest, BoundedTryGetDoesNotWaitBehindBlockingAllocator)
{
    auto first = buffer.Get({}, 0, true, true);
    auto second = buffer.Get({}, 1, true, true);
    std::promise<void> started;
    auto blocked = std::async(std::launch::async, [&] {
        started.set_value();
        return buffer.Get({}, 2, true, true);
    });
    started.get_future().wait();
    // Whether the allocator has acquired the bucket yet or not, TryGet must return.
    auto attempt = std::async(std::launch::async, [&] { return buffer.TryGet({}, 2, true, true); });
    const auto state = attempt.wait_for(std::chrono::milliseconds(100));
    second = {};  // always unblock allocator before reporting an assertion failure
    auto acquired = blocked.get();
    acquired.MarkReady();
    EXPECT_EQ(state, std::future_status::ready);
    auto observed = attempt.get();
    EXPECT_FALSE(observed);  // no free node, or bucket busy
}

class PartialFailureStream : public UC::Trans::SimuStream {
public:
    std::promise<void> entered, release;
    std::shared_future<void> gate = release.get_future().share();
    size_t calls{0};
    Status HostToDeviceAsync(void* host, void* device, size_t size) override
    {
        if (++calls == 2) { return Status::Error("injected second tensor failure"); }
        AppendCallback([this](bool) {
            entered.set_value();
            gate.wait();
        });
        return SimuStream::HostToDeviceAsync(host, device, size);
    }
};
class InjectedGraphQueue : public LoadQueue {
    std::shared_ptr<UC::Trans::Stream> stream_;

public:
    explicit InjectedGraphQueue(std::shared_ptr<UC::Trans::Stream> stream)
        : stream_(std::move(stream))
    {
    }

protected:
    Status SetupCopyStream(CopyStream& copy) override { return copy.Setup(stream_); }
};

TEST_F(GraphQueueTest, PartialH2dFailureFencesBeforeLeaseReleaseAndCompletion)
{
    auto stream = std::make_shared<PartialFailureStream>();
    ASSERT_TRUE(stream->Setup().Success());
    InjectedGraphQueue queue(stream);
    ASSERT_TRUE(queue.Setup(config, &failures, &buffer).Success());
    Request request({0});
    queue.Submit(request.task, request.waiter);
    EXPECT_EQ(stream->entered.get_future().wait_for(std::chrono::seconds(3)),
              std::future_status::ready);
    EXPECT_FALSE(request.waiter->Check());
    auto observer = buffer.TryGet({}, 0, true, true);
    EXPECT_TRUE(observer);
    if (observer) { EXPECT_FALSE(observer.Owner()); }
    stream->release.set_value();
    ASSERT_TRUE(request.waiter->WaitForDuration(3000));
    EXPECT_TRUE(failures.Contains(request.task->id));
    for (size_t i = 0; i != 31; ++i) { EXPECT_EQ(request.data[0][i], 17); }
    for (size_t i = 31; i != 96; ++i) { EXPECT_EQ(request.data[0][i], 0); }
}
}  // namespace
