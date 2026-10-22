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
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "ucm/store/cache/cc/transfer_graph.h"

using UC::CacheStore::UcmTransferGraph;
using UC::CacheStore::UcmTransferPlan;

#define CHECK(condition)                                            \
    do {                                                            \
        if (!(condition)) { throw std::runtime_error(#condition); } \
    } while (false)

int main()
{
    size_t cases = 0;
    for (size_t n = 0; n < 257; ++n) {
        for (size_t ranks = 1; ranks <= 33; ++ranks) {
            for (size_t rank = 0; rank < ranks + 2; ++rank) {
                std::vector<size_t> expected;
                for (size_t r = 0; r < ranks; ++r) {
                    for (size_t i = (rank + r) % ranks; i < n; i += ranks) {
                        expected.push_back(i);
                    }
                }
                UcmTransferPlan plan(n, rank, ranks, 7);
                CHECK(plan.Size() == expected.size());
                for (size_t i = 0; i < n; ++i) { CHECK(plan.Index(i) == expected[i]); }
                ++cases;
            }
        }
    }
    UcmTransferGraph graph;
    graph.Setup(1, 3, 8, 15, 4, 2);
    auto stable = graph.Prepare(10);
    CHECK(stable->Window() == 3);
    CHECK(graph.Prepare(10) == stable && graph.Hits() == 1);
    graph.Prepare(11);
    graph.Prepare(12);
    CHECK(graph.Entries() == 2 && stable->Size() == 10);
    CHECK(graph.Prepare(10) != stable);
    graph.Setup(0, 1, 8, 32, 4, 0);
    auto uncached = graph.Prepare(5);
    CHECK(graph.Prepare(5) != uncached && graph.Entries() == 0);
    const auto huge = std::numeric_limits<size_t>::max();
    UcmTransferPlan large(huge, huge - 1, huge, 1);
    CHECK(large.Index(0) == huge - 1 && large.Index(1) == 0);
    for (int which = 0; which != 5; ++which) {
        bool threw = false;
        try {
            graph.Setup(0, which == 0 ? 0 : 1, which == 1 ? 0 : 8, which == 2 ? 3 : 32, 4,
                        which == 3 ? 4097 : 0);
            if (which == 4) { graph.Prepare(0)->Index(0); }
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }
    std::cout << "transfer graph plan: " << cases << " reference cases passed\n";
}
