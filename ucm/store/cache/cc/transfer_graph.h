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
#ifndef UCM_CACHE_TRANSFER_GRAPH_H
#define UCM_CACHE_TRANSFER_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <stdexcept>

namespace UC::CacheStore {

// An immutable, constant-space program. No request addresses, leases or hit state.
// Index() is the old rank-interleaved order, without an O(request size) vector.
class UcmTransferPlan {
    const size_t count_, rank_, ranks_, window_;
    size_t Prefix(size_t slices) const
    {
        const size_t tail = std::min(slices, ranks_ - rank_);
        const size_t head = slices - tail;
        const size_t remainder = count_ % ranks_;
        const size_t extras =
            (rank_ < remainder ? std::min(tail, remainder - rank_) : 0) + std::min(head, remainder);
        return (count_ / ranks_) * slices + extras;
    }

public:
    UcmTransferPlan(size_t count, size_t rank, size_t ranks, size_t window)
        : count_(count), rank_(ranks ? rank % ranks : 0), ranks_(ranks), window_(window)
    {
        if (!ranks || !window) { throw std::invalid_argument("invalid transfer graph limits"); }
    }
    size_t Size() const { return count_; }
    size_t Window() const { return window_; }
    size_t Index(size_t position) const
    {
        if (position >= count_) { throw std::out_of_range("transfer graph index"); }
        size_t lo = 0, hi = ranks_;
        while (lo + 1 < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (Prefix(mid) <= position) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Addition expressed without overflow even for very large rank counts.
        const size_t slice = lo >= ranks_ - rank_ ? lo - (ranks_ - rank_) : rank_ + lo;
        return slice + (position - Prefix(lo)) * ranks_;
    }
};

// Owned exclusively by the dispatcher. Eviction never invalidates an in-flight plan.
class UcmTransferGraph {
    size_t rank_{0}, ranks_{1}, window_{1}, capacity_{0};
    std::list<std::shared_ptr<const UcmTransferPlan>> plans_;
    size_t hits_{0}, misses_{0};

public:
    void Setup(size_t rank, size_t ranks, size_t shards, size_t bytes, size_t shardBytes,
               size_t entries)
    {
        if (!ranks || !shards || !shardBytes || bytes < shardBytes || entries > 4096 ||
            shards > 65536) {
            throw std::invalid_argument("invalid transfer graph configuration");
        }
        rank_ = rank;
        ranks_ = ranks;
        window_ = std::min(shards, bytes / shardBytes);
        capacity_ = entries;
        plans_.clear();
        hits_ = misses_ = 0;
    }
    std::shared_ptr<const UcmTransferPlan> Prepare(size_t count)
    {
        for (auto it = plans_.begin(); it != plans_.end(); ++it) {
            if ((*it)->Size() == count) {
                auto plan = *it;
                plans_.splice(plans_.begin(), plans_, it);
                ++hits_;
                return plan;
            }
        }
        ++misses_;
        auto plan = std::make_shared<const UcmTransferPlan>(count, rank_, ranks_, window_);
        if (capacity_) {
            plans_.push_front(plan);
            if (plans_.size() > capacity_) { plans_.pop_back(); }
        }
        return plan;
    }
    size_t Entries() const { return plans_.size(); }
    size_t Hits() const { return hits_; }
    size_t Misses() const { return misses_; }
};
}  // namespace UC::CacheStore
#endif
