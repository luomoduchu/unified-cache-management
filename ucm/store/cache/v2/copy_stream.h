/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "trans/device.h"

namespace UC::Cache2 {

class CopyStream {
    static constexpr size_t kMaxStreamNumber = 32;

    int32_t deviceId_{-1};
    size_t streamNumber_{0};
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;
    /* Hybrid routing state. weights_/inflight_ share streams_' indices and stay
     * empty on every non-hybrid path, so the existing rotation keeps working. */
    bool hybrid_{false};
    std::vector<double> weights_;
    std::vector<size_t> inflight_;

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const bool useGdr = false)
    {
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        hybrid_ = false;
        weights_.clear();
        inflight_.clear();
        streams_.clear();
        streams_.reserve(streamNumber);
        for (size_t i = 0; i < streamNumber; ++i) {
            auto stream = useGdr ? device.MakeGdrStream() : device.MakeSharedStream();
            if (!stream) {
                UC_ERROR("Failed to make stream on device({}).", deviceId);
                return Status::Error();
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        streamIndex_ = 0;
        return Status::OK();
    }

    /* CE and GDR side by side, shards routed by weighted in-flight bytes. The
     * task boundary already drains every stream (Synchronize), so the counters
     * only have to be cleared there -- no completion callbacks are needed. */
    Status SetupHybrid(const int32_t deviceId, const size_t ceStreamNumber,
                       const size_t gdrStreamNumber, const double weightGdr,
                       const double weightCe)
    {
        if (ceStreamNumber == 0 || gdrStreamNumber == 0) {
            return Status::InvalidParam("hybrid needs both ce({}) and gdr({}) streams",
                                        ceStreamNumber, gdrStreamNumber);
        }
        if (ceStreamNumber + gdrStreamNumber > kMaxStreamNumber) {
            return Status::InvalidParam("hybrid stream number({}+{}) exceeds {}", ceStreamNumber,
                                        gdrStreamNumber, kMaxStreamNumber);
        }
        if (weightGdr <= 0.0 || weightCe <= 0.0) {
            return Status::InvalidParam("hybrid weights must be positive({}, {})", weightGdr,
                                        weightCe);
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        streams_.clear();
        weights_.clear();
        streams_.reserve(ceStreamNumber + gdrStreamNumber);
        for (size_t i = 0; i < gdrStreamNumber; ++i) {
            auto stream = device.MakeGdrStream();
            if (!stream) {
                UC_ERROR("Failed to make gdr stream on device({}).", deviceId);
                return Status::Unsupported();
            }
            streams_.push_back(std::move(stream));
            weights_.push_back(weightGdr / static_cast<double>(gdrStreamNumber));
        }
        for (size_t i = 0; i < ceStreamNumber; ++i) {
            auto stream = device.MakeSharedStream();
            if (!stream) {
                UC_ERROR("Failed to make stream on device({}).", deviceId);
                return Status::Error();
            }
            streams_.push_back(std::move(stream));
            weights_.push_back(weightCe / static_cast<double>(ceStreamNumber));
        }
        inflight_.assign(streams_.size(), 0);
        hybrid_ = true;
        deviceId_ = deviceId;
        streamNumber_ = streams_.size();
        streamIndex_ = 0;
        return Status::OK();
    }

    /* IO aggregation submits batches through one stream. */
    Status SetupIoAggregation(const int32_t deviceId, const bool useGdr)
    {
        if (useGdr) {
            return Status::InvalidParam("GDR stream is incompatible with cache IO aggregation");
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        auto stream = device.MakeIoAggregationStream();
        if (!stream) {
            UC_ERROR("Cache IO aggregation is not available on device({}).", deviceId);
            return Status::Unsupported();
        }
        hybrid_ = false;
        weights_.clear();
        inflight_.clear();
        streams_.clear();
        streams_.push_back(std::move(stream));
        deviceId_ = deviceId;
        streamNumber_ = 1;
        streamIndex_ = 0;
        return Status::OK();
    }

    /* SDMA Direct shards a host segment straight into device fragments, so it
     * needs streamNumber descriptors ready for rotation and covers all of them
     * on wait and sync, exactly like the plain path. A platform whose runtime
     * cannot make such a stream reports it here rather than silently falling
     * back to fragment-by-fragment copies. */
    Status SetupSdmaDirect(const int32_t deviceId, const size_t streamNumber, const bool useGdr)
    {
        if (useGdr) {
            return Status::InvalidParam("GDR stream is incompatible with cache SDMA Direct");
        }
        if (streamNumber == 0 || streamNumber > kMaxStreamNumber) {
            return Status::InvalidParam("invalid sdma direct stream number({})", streamNumber);
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        hybrid_ = false;
        weights_.clear();
        inflight_.clear();
        streams_.clear();
        streams_.reserve(streamNumber);
        for (size_t i = 0; i < streamNumber; ++i) {
            auto stream = device.MakeSdmaDirectStream();
            if (!stream) {
                UC_ERROR("Cache SDMA Direct is not available on device({}).", deviceId);
                return Status::Unsupported();
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        streamIndex_ = 0;
        return Status::OK();
    }

    /* Trans::Stream has no batch form for device-to-device, so this walks the
     * fragments itself. */
    Status DeviceToDeviceScatterAsync(void* src, void** dst,
                                      const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && dst[i] != nullptr) {
                auto* pSrc = static_cast<void*>(static_cast<int8_t*>(src) + offset);
                auto s = stream->DeviceToDeviceAsync(pSrc, dst[i], sizes[i]);
                if (s.Failure()) {
                    SyncAfterFailure(*stream);
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status DeviceToDeviceGatherAsync(void** src, void* dst,
                                     const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && src[i] != nullptr) {
                auto* pDst = static_cast<void*>(static_cast<int8_t*>(dst) + offset);
                auto s = stream->DeviceToDeviceAsync(src[i], pDst, sizes[i]);
                if (s.Failure()) {
                    SyncAfterFailure(*stream);
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    /* The whole fragment list goes to the stream in one piece, as in the cc
     * core: Trans::Stream's default implementation walks it fragment by
     * fragment, while SDMA Direct and IO aggregation override it to issue a
     * single batch. Which of the two happens is the stream's business, so there
     * is nothing to branch on here -- only the failure path still has to drain. */
    Status HostToDeviceScatterAsync(void* host, void** device,
                                    const std::vector<size_t>& sizes) noexcept
    {
        auto stream = PickStream(SumSizes(sizes));
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        auto s = stream->HostToDeviceAsync(host, device, sizes);
        if (s.Failure()) [[unlikely]] { SyncAfterFailure(*stream); }
        return s;
    }

    Status DeviceToHostGatherAsync(void** device, void* host,
                                   const std::vector<size_t>& sizes) noexcept
    {
        auto stream = PickStream(SumSizes(sizes));
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        auto s = stream->DeviceToHostAsync(device, host, sizes);
        if (s.Failure()) [[unlikely]] { SyncAfterFailure(*stream); }
        return s;
    }

    Status WaitEvent(const Trans::Event& event) noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->WaitEvent(event);
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to wait event on stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }

    Status Synchronize() noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->Synchronized();
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to synchronize stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        /* Every task ends with a full drain, so the routing counters restart
         * from zero here. A failed synchronize is terminal for the task and the
         * caller re-drains, so clearing anyway is safe: the counters are only a
         * scheduling hint, never a correctness input. */
        ResetInflight();
        return status;
    }

private:
    /* Trans::Stream submits a batch fragment by fragment and returns on the
     * first failure, so a failed copy can still have earlier fragments in
     * flight. The caller releases its buffer slot as soon as it sees the error,
     * and that slot can then be handed to another task while the device is
     * still writing to it, so drain the stream before handing the error back. */
    void SyncAfterFailure(Trans::Stream& stream) noexcept
    {
        auto s = stream.Synchronized();
        if (s.Failure()) {
            UC_ERROR("Failed({}) to synchronize stream on device({}).", s, deviceId_);
        }
    }

    std::shared_ptr<Trans::Stream> NextStream() noexcept
    {
        if (streamNumber_ == 0) [[unlikely]] { return nullptr; }
        auto& stream = streams_[streamIndex_];
        streamIndex_ = (streamIndex_ + 1) % streamNumber_;
        return stream;
    }

    static size_t SumSizes(const std::vector<size_t>& sizes) noexcept
    {
        size_t total = 0;
        for (const auto size : sizes) { total += size; }
        return total;
    }

    void ResetInflight() noexcept
    {
        for (auto& bytes : inflight_) { bytes = 0; }
    }

    /* Weighted earliest-finish: the score is the bytes this channel would be
     * holding, so a channel with twice the weight takes twice the bytes before
     * it looks loaded. This converges to the split measured on the hardware
     * (GDR 46 : CE 54), whereas a "GDR first, spill to CE" threshold would
     * hard-code the wrong priority: at 144 KB CE is the faster engine
     * (27.59 vs 23.25 GB/s), and both share one per-GPU PCIe ceiling anyway. */
    std::shared_ptr<Trans::Stream> PickStream(const size_t bytes) noexcept
    {
        if (streamNumber_ == 0) [[unlikely]] { return nullptr; }
        if (!hybrid_) { return NextStream(); }
        size_t best = 0;
        double bestScore = 0.0;
        for (size_t i = 0; i < streamNumber_; ++i) {
            const double score = static_cast<double>(inflight_[i] + bytes) / weights_[i];
            if (i == 0 || score < bestScore) {
                bestScore = score;
                best = i;
            }
        }
        inflight_[best] += bytes;
        return streams_[best];
    }
};

}  // namespace UC::Cache2
