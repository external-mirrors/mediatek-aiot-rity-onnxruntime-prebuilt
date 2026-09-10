#!/usr/bin/env python3
"""
onnxruntime_benchmark.py - Genio NPU evaluation tool for ONNX Runtime.

For one model (-m) or every *.onnx under a directory (--all -d DIR) it reports:

  * partition  : NPU partitions, nodes claimed by the NPU / total nodes (from the EP's
                 GetCapability log) and the per-provider node assignment measured by
                 ORT profiling (which ops actually fell back to the CPU).
  * performance: NPU and CPU latency (mean / p50 / p95 / min / max, FPS) and speed-up.
  * accuracy   : NPU vs CPU on the SAME model (execution fidelity: cosine / MAE / max err),
                 optionally NPU vs a reference FP32 model on the CPU (--ref, end-to-end
                 quantization + NPU error), and YOLO detection matching (--task yolo).

Works with both EP generations:
  * plugin EP  (ORT >= 1.23): libonnxruntime_providers_neuropilot.so registered at runtime,
                 EP name NeuroPilotExecutionProvider (default --ep-lib /usr/lib/...).
  * in-tree EP (ORT 1.20.x): NeuronExecutionProvider built into libonnxruntime.

Examples
  onnxruntime_benchmark.py -m yolo11s_480_int8.onnx --ref yolo11s_480.onnx --input bus_480.npy --task yolo
  onnxruntime_benchmark.py --all -d /usr/share/onnxruntime_benchmark --runs 50 --json all.json
  onnxruntime_benchmark.py -m model.onnx --ep cpu            # CPU only
  onnxruntime_benchmark.py -m model.onnx --flags '{"NEURON_FLAG_USE_FP16":"1"}'
"""

import argparse
import collections
import contextlib
import glob
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PLUGIN_EP_NAME = "NeuroPilotExecutionProvider"     # ORT >= 1.23 plugin EP ABI
BUILTIN_EP_NAME = "NeuronExecutionProvider"        # ORT 1.20.x in-tree EP
NPU_EP_NAMES = (PLUGIN_EP_NAME, BUILTIN_EP_NAME)
DEFAULT_EP_LIB_CANDIDATES = [
    os.environ.get("NEURON_EP_LIB", ""),
    "/usr/lib/libonnxruntime_providers_neuropilot.so",
    "/usr/lib/onnxruntime-neuron/lib/libonnxruntime_providers_neuropilot.so",
]
DEFAULT_FLAGS = {
    "NEURON_FLAG_USE_FP16": "1",           # required: compiler rejects quantized graphs without it
    "NEURON_FLAG_MIN_GROUP_SIZE": "0",
    "NEURON_FLAG_OPTIMIZATION_STRING": "--opt=2 --num-mdla=1 --reshape-to-4d --interval-coloring-converage=1.0",
}
NP_DTYPE = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(int32)": np.int32,
    "tensor(int64)": np.int64, "tensor(bool)": np.bool_,
}
PART_RE = re.compile(r"partitions=(\d+)(?:\s*\(proposed=(\d+)\))?\s+supported_nodes=(\d+)\s+total_nodes=(\d+)")


# --------------------------------------------------------------------------- utils
@contextlib.contextmanager
def captured_fds():
    """Capture fd 1/2 (incl. native EP/compiler output); yields a callable returning the text."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1); os.dup2(tmp.fileno(), 2)
    text = {"v": ""}
    try:
        yield lambda: text["v"]
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1])
        tmp.flush(); tmp.seek(0)
        text["v"] = tmp.read().decode("utf-8", "replace")
        tmp.close()


def is_dynamic(d):
    return isinstance(d, str) or d is None or d < 0


def resolve_shape(shape, batch=1):
    return [batch if is_dynamic(d) else int(d) for d in shape]


def stats(lat):
    a = np.asarray(lat, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std()), "min_ms": float(a.min()),
            "max_ms": float(a.max()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "fps": float(1000.0 / np.percentile(a, 50)),
            "runs": int(a.size)}


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def perf_mode_active():
    try:
        return open("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read().strip() == "performance"
    except OSError:
        return None


# --------------------------------------------------------------------------- inputs
def make_feeds(session, args):
    """Return (list_of_feed_dicts, description)."""
    inputs = session.get_inputs()
    if args.input:                                            # single .npy for the first input
        x = np.load(args.input)
        inp = inputs[0]
        x = coerce(x, inp)
        feed = {inp.name: x}
        for extra in inputs[1:]:
            feed[extra.name] = random_tensor(extra, args.batch, seed=1)
        return [feed], f"npy {args.input}"
    if args.images:
        feeds = image_feeds(inputs[0], args.images, args.batch, args.samples, args.norm)
        return feeds, f"{len(feeds)} images from {args.images} (norm {args.norm})"
    feeds = []
    for i in range(max(1, args.samples)):
        feeds.append({inp.name: random_tensor(inp, args.batch, seed=100 + i) for inp in inputs})
    return feeds, f"{len(feeds)} seeded random tensors"


def coerce(x, inp):
    """Convert an input array to what the session expects; image-like float [0,1] <-> uint8 [0,255]."""
    dtype = NP_DTYPE.get(inp.type, np.float32)
    if dtype == np.uint8 and np.issubdtype(x.dtype, np.floating):
        x = np.clip(np.round(x * 255.0), 0, 255)
    elif np.issubdtype(dtype, np.floating) and x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    x = x.astype(dtype)
    if x.ndim == len(inp.shape) - 1:
        x = x[None]
    return x


def random_tensor(inp, batch, seed):
    rng = np.random.default_rng(seed)
    shape = resolve_shape(inp.shape, batch)
    dtype = NP_DTYPE.get(inp.type, np.float32)
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    if dtype is np.bool_:
        return rng.integers(0, 2, shape).astype(dtype)
    return rng.integers(0, 128, shape).astype(dtype)


def image_feeds(inp, image_dir, batch, limit, norm):
    from PIL import Image
    shape = resolve_shape(inp.shape, batch)
    if len(shape) != 4:
        raise ValueError(f"--images needs a 4-D image input, model input is {inp.shape}")
    nhwc = shape[3] in (1, 3, 4)
    h, w, c = (shape[1], shape[2], shape[3]) if nhwc else (shape[2], shape[3], shape[1])
    files = sorted(f for f in Path(image_dir).iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))[:max(1, limit)]
    if not files:
        raise ValueError(f"no images in {image_dir}")
    feeds = []
    for f in files:
        im = Image.open(f).convert("L" if c == 1 else "RGB").resize((w, h))
        a = np.asarray(im, dtype=np.float32)
        if a.ndim == 2:
            a = a[..., None]
        if norm == "0-1":
            a = a / 255.0
        elif norm == "-1-1":
            a = a / 127.5 - 1.0
        a = a[None] if nhwc else a.transpose(2, 0, 1)[None]
        a = np.repeat(a, shape[0], axis=0)
        feeds.append({inp.name: coerce(a, inp)})
    return feeds


# --------------------------------------------------------------------------- sessions
class NpuBackend:
    """Abstracts plugin-EP (ORT>=1.23) vs in-tree (ORT 1.20) NPU providers."""

    def __init__(self, ep_lib, ep_name):
        self.kind, self.devices, self.lib, self.name = None, None, None, ep_name
        if hasattr(ort, "register_execution_provider_library"):
            lib = next((p for p in ([ep_lib] if ep_lib else DEFAULT_EP_LIB_CANDIDATES) if p and Path(p).is_file()), None)
            if lib:
                ort.register_execution_provider_library("neuropilot", str(Path(lib).absolute()))
                self.devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
                if self.devices:
                    self.kind, self.lib = "plugin", lib
                else:
                    print(f"WARNING: {ep_name} registered no device from {lib} "
                          "(is libneuron_op_capability.so beside it and the NPU driver loaded?)")
        if self.kind is None and BUILTIN_EP_NAME in ort.get_available_providers():
            self.kind, self.name = "builtin", BUILTIN_EP_NAME

    def available(self):
        return self.kind is not None

    def session(self, model, flags, so=None):
        so = so or ort.SessionOptions()
        if self.kind == "plugin":
            so.add_provider_for_devices(self.devices, flags)
            return ort.InferenceSession(model, so)
        return ort.InferenceSession(model, so, providers=[(self.name, flags), "CPUExecutionProvider"])


def cpu_session(model, so=None):
    so = so or ort.SessionOptions()
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


def apply_session_args(so, args):
    if args.num_threads:
        if args.thread_mode == "parallel":
            so.intra_op_num_threads = args.num_threads
        else:
            so.inter_op_num_threads = args.num_threads
    so.graph_optimization_level = {
        "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[args.optimize]
    return so


# --------------------------------------------------------------------------- partition
def parse_partitions(log_text):
    m = None
    for m in PART_RE.finditer(log_text):
        pass
    if not m:
        return None
    return {"partitions": int(m.group(1)), "proposed": int(m.group(2)) if m.group(2) else None,
            "claimed_nodes": int(m.group(3)), "total_nodes": int(m.group(4))}


def node_assignment(model, factory, feed):
    """Run one profiled inference; count nodes per provider and CPU-fallback op types."""
    prof_dir = tempfile.mkdtemp(prefix="ortprof_")
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(prof_dir, "p")
    with captured_fds():                                   # hide the compiler's per-session chatter
        s = factory(so)
        s.run(None, feed)
        path = s.end_profiling()
    counts, cpu_ops = collections.Counter(), collections.Counter()
    try:
        for e in json.load(open(path)):
            if e.get("cat") != "Node":
                continue
            a = e.get("args", {})
            prov = a.get("provider")
            if not prov or not e.get("name", "").endswith("_kernel_time"):
                continue
            counts[prov] += 1
            if prov not in NPU_EP_NAMES:
                cpu_ops[a.get("op_name", "?")] += 1
    except Exception:
        pass
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)
        with contextlib.suppress(OSError):
            os.rmdir(prof_dir)
    return dict(counts), dict(cpu_ops.most_common())


# --------------------------------------------------------------------------- accuracy
def compare(a_outs, b_outs, names):
    res = {}
    for n, u, v in zip(names, a_outs, b_outs):
        u = np.asarray(u, np.float64).ravel(); v = np.asarray(v, np.float64).ravel()
        if u.size != v.size:
            res[n] = {"cosine": float("nan"), "note": f"shape mismatch {u.size} vs {v.size}"}
            continue
        d = np.abs(u - v)
        res[n] = {"cosine": cosine(u, v), "mae": float(d.mean()), "max_abs_err": float(d.max()),
                  "ref_range": float(v.max() - v.min())}
    return res


def is_e2e(o):
    return o.ndim == 3 and o.shape[2] < o.shape[1] and o.shape[2] <= 64


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def yolo_decode(o, conf):
    a = o[0]
    if is_e2e(o):                                              # end-to-end top-k head [N, 6]
        m = a[:, 4] > conf
        d = a[m]
        return d[:, :4], d[:, 4], d[:, 5].astype(int)
    sc = a[4:].max(0); cl = a[4:].argmax(0); m = sc > conf   # raw head [4+nc, N]
    cx, cy, w, h = a[0][m], a[1][m], a[2][m], a[3][m]
    box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    sc, cl = sc[m], cl[m]
    keep = []
    for k in np.argsort(-sc):
        if all(not (cl[k] == cl[j] and _iou(box[k], box[j]) > 0.5) for j in keep):
            keep.append(k)
    keep = np.array(keep, dtype=int)
    return box[keep], sc[keep], cl[keep]


def yolo_match(ref_out, npu_out, conf):
    rb, rs, rc = yolo_decode(ref_out, conf); gb, gs, gc = yolo_decode(npu_out, conf)
    matched, dsc, dbx, used = 0, 0.0, 0.0, set()
    for i in range(len(rb)):
        cand = [(_iou(rb[i], gb[j]), j) for j in range(len(gb)) if j not in used and gc[j] == rc[i]]
        cand = [c for c in cand if c[0] > 0.5]
        if cand:
            _, j = max(cand); used.add(j); matched += 1
            dsc = max(dsc, abs(float(rs[i] - gs[j]))); dbx = max(dbx, float(np.abs(rb[i] - gb[j]).max()))
    return {"ref": int(len(rb)), "npu": int(len(gb)), "matched": int(matched),
            "max_dscore": dsc, "max_dbox_px": dbx}


# --------------------------------------------------------------------------- per model
def bench(session, feeds, warmup, runs):
    for _ in range(warmup):
        session.run(None, feeds[0])
    lat = []
    for i in range(runs):
        t0 = time.perf_counter()
        session.run(None, feeds[i % len(feeds)])
        lat.append((time.perf_counter() - t0) * 1000.0)
    return stats(lat)


def evaluate(model_path, npu, flags, args):
    r = {"model": str(model_path), "size_mb": round(model_path.stat().st_size / 1e6, 2)}
    print("=" * 78)
    print(f"Model      : {model_path}")
    cpu = cpu_session(str(model_path), apply_session_args(ort.SessionOptions(), args))
    for i in cpu.get_inputs():
        print(f"Input      : {i.name} {i.shape} {i.type}")
    for o in cpu.get_outputs():
        print(f"Output     : {o.name} {o.shape} {o.type}")
    feeds, src = make_feeds(cpu, args)
    r["input_source"] = src
    out_names = [o.name for o in cpu.get_outputs()]
    use_npu = npu is not None and npu.available() and args.ep != "cpu"

    npu_sess, log_text = None, ""
    if use_npu:
        r["neuron_flags"] = flags
        t0 = time.perf_counter()
        with captured_fds() as get_log:
            try:
                npu_sess = npu.session(str(model_path), flags, apply_session_args(ort.SessionOptions(), args))
                npu_sess.run(None, feeds[0])                    # first run = NPU compile + validation
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:300]}"
                npu_sess = None
        log_text = get_log()
        r["npu_session_s"] = round(time.perf_counter() - t0, 2)
        if npu_sess is None:
            r["npu_error"] = err
            print(f"NPU        : FAILED  {err}")
            if args.verbose:
                print(log_text[-2000:])
        else:
            part = parse_partitions(log_text)
            r["partition"] = part
            counts, cpu_ops = node_assignment(str(model_path),
                                              lambda so: npu.session(str(model_path), flags, so), feeds[0])
            r["node_assignment"] = counts
            r["cpu_fallback_ops"] = cpu_ops
            npu_n = sum(v for k, v in counts.items() if k in NPU_EP_NAMES)
            cpu_n = sum(v for k, v in counts.items() if k not in NPU_EP_NAMES)
            if part:
                print(f"Partition  : {part['partitions']} NPU partition(s), claimed {part['claimed_nodes']}/{part['total_nodes']} nodes"
                      + (f" (proposed {part['proposed']})" if part["proposed"] is not None else ""))
            print(f"Assignment : {npu_n} NPU node(s) [fused], {cpu_n} CPU node(s)"
                  + (f"  CPU ops: {cpu_ops}" if cpu_ops else ""))
            if npu_n == 0:
                print("WARNING    : no node ran on the NPU - whole graph fell back to CPU")
                r["npu_error"] = "no node on NPU"
        if args.verbose and log_text:
            sys.stderr.write(log_text)

    # ---- performance
    if npu_sess is not None:
        r["npu"] = bench(npu_sess, feeds, args.warmup, args.runs)
        s = r["npu"]
        print(f"NPU perf   : p50 {s['p50_ms']:.2f} ms  mean {s['mean_ms']:.2f}  p95 {s['p95_ms']:.2f}  min {s['min_ms']:.2f}  ({s['fps']:.1f} FPS, {s['runs']} runs)")
    if not args.no_cpu:
        r["cpu"] = bench(cpu, feeds, min(args.warmup, 3), max(3, args.runs // (5 if npu_sess else 1)))
        s = r["cpu"]
        print(f"CPU perf   : p50 {s['p50_ms']:.2f} ms  mean {s['mean_ms']:.2f}  min {s['min_ms']:.2f}  ({s['fps']:.1f} FPS, {s['runs']} runs)")
        if npu_sess is not None:
            r["speedup"] = r["cpu"]["p50_ms"] / r["npu"]["p50_ms"]
            print(f"Speed-up   : {r['speedup']:.2f}x (p50)")

    # ---- accuracy
    if npu_sess is not None and not args.no_compare:
        acc = collections.defaultdict(list)
        yolo = None
        if args.task == "yolo" and not (len(cpu.get_outputs()) >= 1 and len(resolve_shape(cpu.get_outputs()[0].shape)) == 3):
            args.task = "generic"                                    # e.g. segmentation logits [1,C,H,W]: no boxes to match
        ref_sess = cpu_session(args.ref) if args.ref else None
        ref_acc = collections.defaultdict(list)
        for feed in feeds[:args.samples]:
            npu_out = npu_sess.run(None, feed)
            cpu_out = cpu.run(None, feed)
            for n, v in compare(npu_out, cpu_out, out_names).items():
                acc[n].append(v)
            if ref_sess is not None:
                rfeed = {ref_sess.get_inputs()[0].name: coerce(next(iter(feed.values())), ref_sess.get_inputs()[0])}
                ref_out = ref_sess.run(None, rfeed)
                for n, v in compare(npu_out, ref_out, out_names).items():
                    ref_acc[n].append(v)
                if args.task == "yolo" and yolo is None:
                    yolo = yolo_match(ref_out[0], npu_out[0], args.conf)
            elif args.task == "yolo" and yolo is None:
                yolo = yolo_match(cpu_out[0], npu_out[0], args.conf)

        def summarize(d):
            out = {}
            for n, lst in d.items():
                out[n] = {k: float(np.nanmean([x.get(k, np.nan) for x in lst])) for k in ("cosine", "mae", "max_abs_err")}
            return out
        r["accuracy_npu_vs_cpu"] = summarize(acc)
        for n, v in r["accuracy_npu_vs_cpu"].items():
            skip = args.task == "yolo" and n == out_names[0] and is_e2e(npu_out[0])
            print(f"Acc vs CPU : {n[:28]:<28} cosine {v['cosine']:.5f}  MAE {v['mae']:.5f}  max|err| {v['max_abs_err']:.4f}"
                  + ("  (e2e top-k head: cosine not meaningful)" if skip else ""))
        if ref_sess is not None:
            r["accuracy_npu_vs_ref"] = summarize(ref_acc)
            r["ref_model"] = args.ref
            for n, v in r["accuracy_npu_vs_ref"].items():
                print(f"Acc vs REF : {n[:28]:<28} cosine {v['cosine']:.5f}  MAE {v['mae']:.5f}  max|err| {v['max_abs_err']:.4f}")
        if yolo:
            r["yolo_detections"] = yolo
            print(f"Detections : ref {yolo['ref']} / npu {yolo['npu']} / matched {yolo['matched']}   "
                  f"max dScore {yolo['max_dscore']:.4f}   max dBox {yolo['max_dbox_px']:.2f} px")
    return r


# --------------------------------------------------------------------------- main
def build_flags(args):
    flags = dict(DEFAULT_FLAGS)
    # legacy options from the 1.20.2 script
    if args.neuron_flag_use_fp16 is not None:
        flags["NEURON_FLAG_USE_FP16"] = args.neuron_flag_use_fp16
    if args.neuron_flag_min_group_size is not None:
        flags["NEURON_FLAG_MIN_GROUP_SIZE"] = args.neuron_flag_min_group_size
    if args.opt_string is not None:
        if args.opt_string == "":
            flags.pop("NEURON_FLAG_OPTIMIZATION_STRING", None)
        else:
            flags["NEURON_FLAG_OPTIMIZATION_STRING"] = args.opt_string
    if args.skip_ops:
        flags["NEURON_FLAG_FORCE_SKIP_OPS_STRING"] = args.skip_ops
    if args.flags:
        flags.update(json.loads(args.flags))
    return flags


def main():
    for stream in (sys.stdout, sys.stderr):              # keep logs readable when redirected to a file
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("models")
    g.add_argument("--model", "-m", help="single ONNX model")
    g.add_argument("--all", "--auto", action="store_true", help="evaluate every *.onnx under --directory (recursive)")
    g.add_argument("--directory", "-d", default=".", help="directory for --all")
    g.add_argument("--ref", help="reference FP32 model run on CPU for end-to-end accuracy (single-model mode)")
    g = p.add_argument_group("inputs")
    g.add_argument("--input", help=".npy tensor for the first input")
    g.add_argument("--images", help="directory of images for the first (4-D) input")
    g.add_argument("--norm", default="0-1", choices=["0-1", "-1-1", "raw"], help="image normalisation (default 0-1)")
    g.add_argument("--samples", type=int, default=3, help="inputs used for accuracy comparison (default 3)")
    g.add_argument("--batch", type=int, default=1, help="value substituted for dynamic dims")
    g = p.add_argument_group("execution")
    g.add_argument("--ep", default="auto", choices=["auto", "npu", "cpu"], help="auto/npu: NPU + CPU baseline, cpu: CPU only")
    g.add_argument("--ep-lib", help="plugin EP .so (default: $NEURON_EP_LIB or /usr/lib/libonnxruntime_providers_neuropilot.so)")
    g.add_argument("--ep-name", default=PLUGIN_EP_NAME)
    g.add_argument("--flags", help="JSON dict of NEURON_FLAG_* provider options (merged over defaults)")
    g.add_argument("--opt-string", help="NEURON_FLAG_OPTIMIZATION_STRING ('' to unset)")
    g.add_argument("--skip-ops", help="NEURON_FLAG_FORCE_SKIP_OPS_STRING, e.g. Pad,Resize")
    g.add_argument("--neuron_flag_use_fp16", help=argparse.SUPPRESS)
    g.add_argument("--neuron_flag_min_group_size", help=argparse.SUPPRESS)
    g.add_argument("--num_threads", type=int, default=0, help="ORT thread count (0 = default)")
    g.add_argument("--thread_mode", default="parallel", choices=["parallel", "sequential"])
    g.add_argument("--optimize", default="ENABLE_ALL", choices=["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"])
    g = p.add_argument_group("measurement")
    g.add_argument("--runs", "--num_runs", "-n", type=int, default=50)
    g.add_argument("--warmup", type=int, default=5)
    g.add_argument("--no-cpu", action="store_true", help="skip the CPU baseline")
    g.add_argument("--no-compare", action="store_true", help="skip accuracy comparison")
    g.add_argument("--task", default="auto", choices=["auto", "generic", "yolo"], help="yolo: add detection matching")
    g.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    g = p.add_argument_group("output")
    g.add_argument("--json", help="write full results to this JSON file")
    g.add_argument("--verbose", "-v", action="store_true", help="show EP / compiler logs")
    args = p.parse_args()

    if args.all:
        models = sorted(Path(m) for m in glob.glob(os.path.join(args.directory, "**", "*.onnx"), recursive=True))
        if args.ref:
            print("NOTE: --ref is ignored in --all mode")
            args.ref = None
    elif args.model:
        models = [Path(args.model)]
    else:
        p.error("give --model MODEL or --all [-d DIR]")
    models = [m for m in models if m.is_file()]
    if not models:
        sys.exit("no model found")

    ort.set_default_logger_severity(1 if args.verbose else 3)
    npu = NpuBackend(args.ep_lib, args.ep_name) if args.ep != "cpu" else None
    flags = build_flags(args)
    print(f"ONNX Runtime : {ort.__version__}   providers {ort.get_available_providers()}")
    if npu and npu.available():
        print(f"NPU backend  : {npu.kind} EP '{npu.name}'" + (f" from {npu.lib}" if npu.lib else ""))
        print(f"Neuron flags : {json.dumps(flags)}")
    elif args.ep != "cpu":
        print("NPU backend  : NOT AVAILABLE (CPU only)")
    pm = perf_mode_active()
    if pm is False:
        print("NOTE: CPU governor is not 'performance'; numbers may fluctuate (echo performance > .../scaling_governor)")

    results = []
    for m in models:
        a = argparse.Namespace(**vars(args))
        if a.task == "auto":
            a.task = "yolo" if "yolo" in m.name.lower() else "generic"
        try:
            results.append(evaluate(m, npu, flags, a))
        except Exception as exc:
            print(f"ERROR      : {m}: {type(exc).__name__}: {exc}")
            results.append({"model": str(m), "error": f"{type(exc).__name__}: {exc}"})

    if len(results) > 1 or args.all:
        print("\n" + "=" * 100)
        print(f"{'Model':34} {'Parts':>5} {'Claimed':>11} {'NPU p50':>9} {'CPU p50':>9} {'Speedup':>8} {'Cos(CPU)':>9} {'Status'}")
        print("-" * 100)
        for r in results:
            name = os.path.basename(r["model"])[:34]
            part = r.get("partition") or {}
            parts = str(part.get("partitions", "-"))
            claimed = f"{part['claimed_nodes']}/{part['total_nodes']}" if part else "-"
            npu_p50 = f"{r['npu']['p50_ms']:.2f}" if r.get("npu") else "-"
            cpu_p50 = f"{r['cpu']['p50_ms']:.2f}" if r.get("cpu") else "-"
            sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "-"
            cos = "-"
            if r.get("accuracy_npu_vs_cpu"):
                vals = [v["cosine"] for v in r["accuracy_npu_vs_cpu"].values() if not np.isnan(v["cosine"])]
                cos = f"{min(vals):.4f}" if vals else "-"
            status = r.get("error") or r.get("npu_error") or "ok"
            print(f"{name:34} {parts:>5} {claimed:>11} {npu_p50:>9} {cpu_p50:>9} {sp:>8} {cos:>9} {status[:30]}")

    if args.json:
        out = {"onnxruntime_version": ort.__version__, "npu_backend": npu.kind if npu else None,
               "neuron_flags": flags, "perf_mode": pm, "results": results}
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWritten: {args.json}")
    return 0 if all("error" not in r and "npu_error" not in r for r in results) else 2


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    # The NPU runtime can leave a non-daemon native thread behind; skip interpreter
    # teardown so the process does not hang as a zombie after printing the results.
    os._exit(rc)
