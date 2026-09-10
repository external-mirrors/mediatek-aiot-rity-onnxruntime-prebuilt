# Example Script performing image classification using efficientnet-lite4 ONNX model

import numpy as np
import math
import onnxruntime as rt
import cv2
import argparse
import os
import time

def img_stats(a, name={}):
    return {
        "name": name,
        "size": a.shape,
        "mean": "{:.2f}".format(a.mean()),
        "std": "{:.2f}".format(a.std()),
        "max": a.max(),
        "min": a.min(),
        "median": "{:.2f}".format(np.median(a)),
    }

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def pre_process(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", default='kitten.jfif', help="path of image to be processed")
    parser.add_argument("--labels", "-l", default='labels_map.txt', help="path to file containing labels")
    parser.add_argument("--model", "-m", default='/usr/share/onnxruntime_example/efficientnet-lite4.onnx', help="Effecientnet-lite4 model path")
    parser.add_argument("--execution_provider", "-e", default='CPUExecutionProvider',
                        choices=['CPUExecutionProvider', 'XnnpackExecutionProvider', 'NeuronExecutionProvider',
                                 'NeuroPilotExecutionProvider'],
                        help="onnxruntime execution provider (NeuroPilotExecutionProvider = plugin NPU EP, "
                             "NeuronExecutionProvider = legacy in-tree NPU EP on ORT 1.20)")
    parser.add_argument("--ep-lib", default=os.environ.get("NEURON_EP_LIB", "/usr/lib/libonnxruntime_providers_neuropilot.so"),
                        help="plugin EP shared library (NeuroPilotExecutionProvider only)")
    parser.add_argument("--profiling", "-p", type=int, default=0, help="1 to enable model profiling, default is 0")
    parser.add_argument("--neuron_flag_use_fp16", type=str, default=None, help="Set NEURON_FLAG_USE_FP16 for NeuronExecutionProvider")
    parser.add_argument("--neuron_flag_min_group_size", type=str, default=None, help="Set NEURON_FLAG_MIN_GROUP_SIZE for NeuronExecutionProvider")
    args = parser.parse_args()

    # Load labels
    with open(args.labels, 'r') as f:
        labels = [l.rstrip() for l in f]

    if args.profiling == 1:
        rt.set_default_logger_severity(0)

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pre-process the image like mobilenet and resize it to 224x224
    img = pre_process(img, (224, 224, 3))

    # create a batch of 1 (that batch size is baked into the saved_model)
    img_batch = np.expand_dims(img, axis=0)

    # Build provider options
    provider_options = {"NEURON_FLAG_USE_FP16": "1"}
    if args.neuron_flag_use_fp16 is not None:
        provider_options["NEURON_FLAG_USE_FP16"] = args.neuron_flag_use_fp16
    if args.neuron_flag_min_group_size is not None:
        provider_options["NEURON_FLAG_MIN_GROUP_SIZE"] = args.neuron_flag_min_group_size

    so = rt.SessionOptions()
    if args.execution_provider == "NeuroPilotExecutionProvider":
        # Plugin EP (ORT >= 1.23): register the library, then select its device(s).
        rt.register_execution_provider_library("neuropilot", args.ep_lib)
        devices = [d for d in rt.get_ep_devices() if d.ep_name == "NeuroPilotExecutionProvider"]
        if not devices:
            raise SystemExit("NeuroPilotExecutionProvider not available (check --ep-lib and libneuron_op_capability.so)")
        so.add_provider_for_devices(devices, provider_options)
        sess = rt.InferenceSession(args.model, so)
    elif args.execution_provider == "NeuronExecutionProvider":
        sess = rt.InferenceSession(args.model, so, providers=[("NeuronExecutionProvider", provider_options), "CPUExecutionProvider"])
    else:
        sess = rt.InferenceSession(args.model, so, providers=[args.execution_provider])
    input_name = sess.get_inputs()[0].name

    # run inference and print results
    start_time = time.time()
    results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
    stop_time = time.time()

    result = reversed(results[0].argsort()[-5:])
    for r in result:
        print(results[0][r], labels[r])

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
