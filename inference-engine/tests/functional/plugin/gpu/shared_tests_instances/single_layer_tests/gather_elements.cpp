// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset5.hpp>

#include "single_layer_tests/gather_elements.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset5;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
};

// set1
const auto gatherElementsArgsSubset1 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {2, 2}, {2, 3, 4} })),                                // Data shape
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {2, 1}, {2, 1, 1} })),                                // Indices shape
    ::testing::ValuesIn(std::vector<int>({ 0, 1 }))             // Batch dims
);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set1, GatherElementsLayerTest,
    ::testing::Combine(
        gatherElementsArgsSubset1,
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values<Config>({})),
    GatherElementsLayerTest::getTestCaseName);

// set2
const auto gatherElementsArgsSubset2 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {15, 12, 20, 15, 2}, {15, 12, 18, 7, 17} })),         // Data shape
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {15, 12, 2}, {15, 12, 5, 9, 1, 3} })),                // Indices shape
    ::testing::ValuesIn(std::vector<int>({ 1, 2 }))             // Batch dims
);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set2, GatherElementsLayerTest,
    ::testing::Combine(
        gatherElementsArgsSubset2,
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values<Config>({})),
    GatherElementsLayerTest::getTestCaseName);

// set3
const auto gatherElementsArgsSubset3 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {4, 3, 2, 5, 5, 2}, {4, 3, 2, 5, 7, 2} })),           // Data shape
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(
        { {4, 3, 2, 5, 1}, {4, 3, 2, 5, 6, 2} })),              // Indices shape
    ::testing::ValuesIn(std::vector<int>({ 3, 4 }))             // Batch dims
);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set3, GatherElementsLayerTest,
    ::testing::Combine(
        gatherElementsArgsSubset3,
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values<Config>({})),
    GatherElementsLayerTest::getTestCaseName);

}  // namespace
