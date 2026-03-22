import time
import numpy as np
import os
from .tensorrt_export import TensorRTExporter
from .edge_runner import EdgeRunner
from models.pinn_module import PINNModule
import torch


def profile_latency():
    batch_size = 1
    seq_len = 10
    state_dim = 6
    num_runs = 100

    dummy_input_torch = torch.randn(batch_size, seq_len, state_dim)
    dummy_input_np = dummy_input_torch.numpy()

    device = torch.device("cpu")
    pinn_model = PINNModule().to(device)
    pinn_model.eval()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pinn_model(dummy_input_torch)
    end = time.time()
    torch_latency = (end - start) / num_runs * 1000

    print(f"PyTorch CPU Inference Latency: {torch_latency:.2f} ms")

    onnx_path = "assets/results/pinn_hgv.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    exporter = TensorRTExporter(pinn_model, output_dir="assets/results", filename="pinn_hgv.onnx")
    exporter.export(dummy_input_torch)

    runner = EdgeRunner(model_path=onnx_path)

    for _ in range(10):
        runner.predict(dummy_input_np)

    start = time.time()
    for _ in range(num_runs):
        runner.predict(dummy_input_np)
    end = time.time()
    onnx_latency = (end - start) / num_runs * 1000

    print(f"ONNX Runtime Inference Latency: {onnx_latency:.2f} ms")
    print(f"Speedup: {torch_latency / onnx_latency:.2f}x")


if __name__ == "__main__":
    profile_latency()
