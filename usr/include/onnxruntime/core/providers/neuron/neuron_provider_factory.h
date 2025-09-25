// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime/core/session/onnxruntime_c_api.h"

enum NeuronFlags {
  NEURON_FLAG_USE_NONE = 0x000,
  NEURON_FLAG_USE_FP16 = 0x001,
  NEURON_FLAG_USE_NCHW = 0x002,
  NEURON_FLAG_CPU_DISABLED = 0x004,
  NEURON_FLAG_CPU_ONLY = 0x008,
  NEURON_FLAG_MDLA_ONLY = 0x010,
  NEURON_FLAG_LAST = NEURON_FLAG_CPU_ONLY,
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Neuron,
                          _In_ OrtSessionOptions* options, _In_ uint32_t neuron_flags);

#ifdef __cplusplus
}
#endif
