import os
import numpy as np
import onnxruntime as ort


class EdgeRunner:
    def __init__(self, model_path="assets/results/pinn_hgv.onnx"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)

        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, 0)

        outputs = self.session.run(None, {self.input_name: input_data})

        return outputs[0], outputs[1]
