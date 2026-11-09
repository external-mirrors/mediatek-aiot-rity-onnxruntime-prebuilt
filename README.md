# ONNXRUNTIME PREBUILT

This repository provides the onnxruntime prebuilt shared libraries, python package and examples
for Genio (aarch64, Yocto scarthgap: glibc 2.39 / GCC 13 / Python 3.12).

Version: 1.26.0 (release tag `v1.26.0-ep0.3.1`; NeuroPilot EP + op-capability pair release 0.3.1)

## Contents

| Tree | Installed with | Contents |
|---|---|---|
| `ort_w_ep/`  | `ORT_EP=1` (ENABLE_NEURON_EP=1, NPU SoCs) | ORT + MediaTek NeuroPilot plugin EP + benchmark tools |
| `ort_wo_ep/` | `ORT_EP=0` | ORT only (CPU / XNNPACK) |

`usr/lib`:
- `libonnxruntime.so.1.26.0` (+ `.so`, `.so.1` symlinks), `libonnxruntime_providers_shared.so`
- `libonnxruntime_providers_neuropilot.so` — NeuroPilot **plugin** execution provider
  (`NeuroPilotExecutionProvider`, ORT plugin EP API, negotiates ORT API 23..26). Loaded by ORT via
  `RegisterExecutionProviderLibrary`; **not** linked into libonnxruntime.
- `libneuron_op_capability.so` — confidential per-op capability table, `dlopen`ed by the EP. Must
  always be deployed together with the EP it was built with (same commit pair).
- `python3.12/site-packages/onnxruntime/` — python package built from the same tree.

`usr/share/onnxruntime_benchmark`: `onnxruntime_benchmark.py` (partition / latency / accuracy, single
model or a whole directory), `onnxruntime_perf_test`, sample quantized models.
`usr/share/onnxruntime_example`: `label_image.py` (efficientnet-lite4) with `-e NeuroPilotExecutionProvider`.

## Source

- onnxruntime: https://github.com/microsoft/onnxruntime tag `v1.26.0`
  (`build.py --build_shared_lib --build_wheel --use_xnnpack --config Release`, native aarch64,
  Ubuntu 24.04 container).
- NeuroPilot plugin EP: `onnxruntime-ep-neuropilot` 0.3.0 source drop + Genio patches (GridSample via
  MTKEXT on NeuroPilot < 9, Squeeze/Unsqueeze raw_data axes, IN/BN NHWC domain, partition controls),
  built against the ORT v1.26.0 headers.
- op-capability: `libneuron-op-capability` 0.3.0 source drop + IN/BN NCHW-claim patch (C ABI v1).
- **The EP and op-capability libraries are an ABI-locked pair.** 0.3.x uses the op-capability C ABI v1
  (`NeuronCap*`); 0.2.x and older used a C++ interface. Never mix libraries from different releases:
  the EP refuses to initialize.
- Full provenance (40-char hashes, submodules, toolchain, base-image digest, sha256 of every binary):
  `BUILD-INFO.yaml` in this repo, installed as `/usr/share/onnxruntime-prebuilt/BUILD-INFO.yaml`.
- Runtime requirement: NeuroPilot runtime >= 8.2.16 (`neuropilot-bin`, provides
  `libneuronusdk_adapter.mtk.so`, dlopen'ed by the EP).

## Using the NPU EP (python)

```python
import onnxruntime as ort
ort.register_execution_provider_library("neuropilot", "/usr/lib/libonnxruntime_providers_neuropilot.so")
devices = [d for d in ort.get_ep_devices() if d.ep_name == "NeuroPilotExecutionProvider"]
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {"NEURON_FLAG_USE_FP16": "1"})
sess = ort.InferenceSession("model.onnx", so)
```

`NEURON_FLAG_USE_FP16=1` is required. Other options: `NEURON_FLAG_MIN_GROUP_SIZE`,
`NEURON_FLAG_OPTIMIZATION_STRING` (e.g. `--opt=2 --num-mdla=1 --reshape-to-4d`),
`NEURON_FLAG_FORCE_SKIP_OPS_STRING`, `NEURON_FLAG_CPU_ONLY`, `NEURON_FLAG_MDLA_ONLY`, `NEURON_FLAG_USE_NCHW`,
`NEURON_FLAG_MAX_PARTITION_BYTES`, `NEURON_FLAG_PARTITION_COUNT`, `NEURON_FLAG_MAX_PARTITION_NODES`,
`NEURON_FLAG_COMPUTE_LIGHT_MIN_NODES` (default 16; `0` restores the 0.2.x behaviour of running every
partition without Conv/MatMul on CPU), `NEURON_FLAG_SKIP_NODE_PREFIXES`, `NEURON_FLAG_GRIDSAMPLE_ABS_GRID`,
`NEURON_FLAG_SKIP_CAPABILITY_PROBE`.

The legacy in-tree `NeuronExecutionProvider` of the 1.20.2 release no longer exists.
