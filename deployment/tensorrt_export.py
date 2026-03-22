import torch
import os
from models.pinn_module import PINNModule


class TensorRTExporter:
    def __init__(self, model, output_dir="assets/results", filename="pinn_hgv.onnx"):
        self.model = model
        self.output_dir = output_dir
        self.filename = filename

        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, filename)

    def export(self, dummy_input):
        self.model.eval()

        torch.onnx.export(
            self.model,
            dummy_input,
            self.filepath,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["state_pred", "ne_pred"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "state_pred": {0: "batch_size", 1: "seq_len"},
                "ne_pred": {0: "batch_size", 1: "seq_len"},
            },
        )
        print(f"Model exported to {self.filepath}")


if __name__ == "__main__":
    model = PINNModule()
    dummy_input = torch.randn(1, 10, 6)
    exporter = TensorRTExporter(model)
    exporter.export(dummy_input)
