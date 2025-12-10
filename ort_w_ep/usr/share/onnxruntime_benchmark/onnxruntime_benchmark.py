import argparse
import onnxruntime as ort
import numpy as np
import time
import os
import glob

def generate_dummy_inputs(session):
    inputs = {}
    for input_meta in session.get_inputs():
        shape = []
        for dim in input_meta.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(1)
            else:
                shape.append(int(dim))
        if input_meta.type == 'tensor(float)':
            dtype = np.float32
            data = np.random.randn(*shape).astype(dtype)
        elif input_meta.type == 'tensor(int64)':
            dtype = np.int64
            data = np.random.randint(0, 10, size=shape, dtype=dtype)
        elif input_meta.type == 'tensor(int32)':
            dtype = np.int32
            data = np.random.randint(0, 10, size=shape, dtype=dtype)
        elif input_meta.type == 'tensor(bool)':
            dtype = np.bool_
            data = np.random.choice([True, False], size=shape)
        else:
            dtype = np.float32
            data = np.random.randn(*shape).astype(dtype)
        inputs[input_meta.name] = data
    return inputs

def benchmark_model(model_path, provider, sess_options, provider_options=None, num_runs=10):
    if provider_options is not None:
        providers = [(provider, provider_options)]
    else:
        providers = [provider]
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
    except Exception as e:
        print(f"Failed to load model {model_path} with provider {provider}: {e}")
        return None
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    dummy_inputs = generate_dummy_inputs(session)
    # Warm-up
    try:
        _ = session.run(output_names, dummy_inputs)
    except Exception as e:
        print(f"Failed to run model {model_path} with provider {provider}: {e}")
        return None
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = session.run(output_names, dummy_inputs)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    return avg_time

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Benchmarking Script")
    parser.add_argument("--model", "-m", help="Path to ONNX model")
    parser.add_argument("--auto", action="store_true", help="Auto-benchmark all ONNX models in directory")
    parser.add_argument("--directory", "-d", type=str, default=".", help="Directory to search for ONNX models (used with --auto)")
    parser.add_argument("--num_runs", "-n", type=int, default=10, help="Number of inference runs for benchmarking")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel/sequential execution")
    parser.add_argument("--thread_mode", type=str, default="sequential", choices=["parallel", "sequential"],
                        help="Threading mode: 'parallel' sets intra_op_num_threads, 'sequential' sets inter_op_num_threads")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX Runtime profiling")
    parser.add_argument("--optimize", type=str, default="ENABLE_ALL",
                        choices=["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"],
                        help="Set ONNX Runtime graph optimization level")
    parser.add_argument("--neuron_flag_use_fp16", type=str, default="1",
                        help="Set NEURON_FLAG_USE_FP16 for NeuronExecutionProvider")
    parser.add_argument("--execution_provider", type=str, default="CPUExecutionProvider",
                                                 help="ONNX Runtime execution provider (e.g., CPUExecutionProvider, XnnpackExecutionProvider, NeuronExecutionProvider)")
    parser.add_argument("--neuron_flag_min_group_size", type=str, default="1",
                        help="Set NEURON_FLAG_MIN_GROUP_SIZE for NeuronExecutionProvider")
    args = parser.parse_args()

    available_providers = ort.get_available_providers()
    provider_map = {
        "CPUExecutionProvider": "CPUExecutionProvider",
        "XnnpackExecutionProvider": "XnnpackExecutionProvider",
        "NeuronExecutionProvider": "NeuronExecutionProvider"
    }

    opt_map = {
        "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    }

    if args.auto:
        # Find all ONNX models in the directory
        model_files = glob.glob(os.path.join(args.directory, "*.onnx"))
        if not model_files:
            print(f"No ONNX models found in directory {args.directory}")
            return
        print(f"Found {len(model_files)} ONNX models in {args.directory}")
        results = []
        for model_path in model_files:
            print(f"\nBenchmarking model: {model_path}")
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = opt_map[args.optimize]
            if args.num_threads is not None:
                if args.thread_mode == "parallel":
                    sess_options.intra_op_num_threads = args.num_threads
                    #sess_options.inter_op_num_threads = 1
                else:
                    sess_options.inter_op_num_threads = args.num_threads
                    #sess_options.intra_op_num_threads = 1
            # CPU
            if "CPUExecutionProvider" not in available_providers:
                print("CPUExecutionProvider not available!")
                cpu_time = None
            else:
                cpu_time = benchmark_model(model_path, "CPUExecutionProvider", sess_options, num_runs=args.num_runs)
                print(f"  CPUExecutionProvider avg time: {cpu_time*1000:.2f} ms" if cpu_time else "  Failed on CPU")
            # Neuron
            if "NeuronExecutionProvider" not in available_providers:
                print("NeuronExecutionProvider not available!")
                neuron_time = None
            else:
                neuron_options = {}
                if args.neuron_flag_use_fp16 is not None:
                    neuron_options["NEURON_FLAG_USE_FP16"] = args.neuron_flag_use_fp16
                if args.neuron_flag_min_group_size is not None:
                    neuron_options["NEURON_FLAG_MIN_GROUP_SIZE"] = args.neuron_flag_min_group_size
                neuron_time = benchmark_model(model_path, "NeuronExecutionProvider", sess_options, provider_options=neuron_options, num_runs=args.num_runs)
                print(f"  NeuronExecutionProvider avg time: {neuron_time*1000:.2f} ms" if neuron_time else "  Failed on Neuron")
            # Speedup
            speedup = (cpu_time / neuron_time) if cpu_time and neuron_time else None
            results.append({
                "model": os.path.basename(model_path),
                "cpu_time": cpu_time,
                "neuron_time": neuron_time,
                "speedup": speedup
            })
        # Print summary table
        print("\nSummary (Neuron vs CPU):")
        print(f"{'Model':30} {'CPU (ms)':>12} {'Neuron (ms)':>14} {'Speedup':>10}")
        print("-"*70)
        for r in results:
            cpu_str = f"{r['cpu_time']*1000:.2f}" if r['cpu_time'] else "N/A"
            neuron_str = f"{r['neuron_time']*1000:.2f}" if r['neuron_time'] else "N/A"
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
            print(f"{r['model']:30} {cpu_str:>12} {neuron_str:>14} {speedup_str:>10}")
    else:
        # Single model mode (as before)
        if not args.model:
            print("Please specify --model <model_path> or use --auto")
            return
        actual_provider = provider_map.get(args.execution_provider, args.execution_provider)
        if actual_provider not in available_providers:
            print(f"Execution provider '{actual_provider}' is not available on this system.")
            print(f"Available providers: {available_providers}")
            return
        sess_options = ort.SessionOptions()
        if args.num_threads is not None:
            if args.thread_mode == "parallel":
                sess_options.intra_op_num_threads = args.num_threads
            else:
                sess_options.inter_op_num_threads = args.num_threads
        if args.profile:
            sess_options.enable_profiling = True
        sess_options.graph_optimization_level = opt_map[args.optimize]
        if args.execution_provider == "NeuronExecutionProvider":
            provider_options = {}
            if args.neuron_flag_use_fp16 is not None:
                provider_options["NEURON_FLAG_USE_FP16"] = args.neuron_flag_use_fp16
            if args.neuron_flag_min_group_size is not None:
                provider_options["NEURON_FLAG_MIN_GROUP_SIZE"] = args.neuron_flag_min_group_size
            providers = [(actual_provider, provider_options)]
        else:
            providers = [actual_provider]
        session = ort.InferenceSession(args.model, sess_options, providers=providers)
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        print("Model Inputs:")
        for input_meta in session.get_inputs():
            print(f"  {input_meta.name}: shape={input_meta.shape}, type={input_meta.type}")
        dummy_inputs = generate_dummy_inputs(session)
        _ = session.run(output_names, dummy_inputs)
        times = []
        for i in range(args.num_runs):
            start_time = time.time()
            _ = session.run(output_names, dummy_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Run {i+1}: {times[-1]*1000:.2f} ms")
        print("\nBenchmark Results:")
        print(f"Provider: {args.execution_provider}")
        print(f"Model: {args.model}")
        print(f"Runs: {args.num_runs}")
        print(f"Threads: {args.num_threads if args.num_threads is not None else 'default'}")
        print(f"Thread Mode: {args.thread_mode}")
        print(f"Optimization Level: {args.optimize}")
        print(f"Average Inference Time: {np.mean(times)*1000:.2f} ms")
        print(f"Min Inference Time: {np.min(times)*1000:.2f} ms")
        print(f"Max Inference Time: {np.max(times)*1000:.2f} ms")
        if args.profile:
            profile_file = session.end_profiling()
            print(f"\nONNX Runtime profiling enabled.")
            print(f"Profiling file saved to: {profile_file}")
            if os.path.exists(profile_file):
                with open(profile_file, "r") as f:
                    print("\n--- Profiling file (first 20 lines) ---")
                    for i, line in enumerate(f):
                        print(line.rstrip())
                        if i >= 19:
                            break
                    print("--- End of snippet ---")

if __name__ == "__main__":
    main()
