import joblib
import json
from typing import Any
import bentoml
import numpy as np
import onnxruntime as ort


@bentoml.service
class Vulcanology:
    def __init__(self) -> None:
        # Load ONNX model
        self.session = ort.InferenceSession(
            "../model.onnx", providers=["CPUExecutionProvider"]
        )

        # Load preprocessing metadata
        with open("../meta/columns_process.json", "r") as f:
            self.columns_process = json.load(f)

        with open("../meta/scaler.pkl", "rb") as f:
            self.scaler = joblib.load(f)

    def preprocess(self, raw_data: dict[str, Any]) -> np.ndarray:
        """
        Converts raw input dict into scaled feature vector for ONNX model
        """
        data_array = []
        data_missing = []
        data_extra = []

        for col in self.columns_process:
            # Track missing
            if col["name"] in raw_data:
                data_missing.append(0)
            else:
                data_missing.append(1)

            data_value = raw_data.get(col["name"], 0)
            action_type = col["Action"]["type"]

            if action_type == "none":
                data_array.append(data_value)

            elif action_type == "Description":
                data_array.append(data_value)
                if data_missing[-1] == 1:
                    data_array.append(0)
                    data_missing.append(1)
                    continue
                for key, value in col["Action"]["Mapping"].items():
                    min_v, max_v = value[0], value[1]
                    if max_v == -1:
                        max_v = float("inf")
                    if min_v < data_value <= max_v:
                        data_array.append(int(key))
                        data_missing.append(0)
                        break

            elif action_type == "one-hot":
                one_hot_vector = [0] * (len(col["Action"]["categories"]) + 1)
                for idx, category in enumerate(col["Action"]["categories"]):
                    if data_value == category:
                        one_hot_vector[idx] = 1
                        break
                if sum(one_hot_vector) == 0:
                    one_hot_vector[-1] = 1
                data_extra.extend(one_hot_vector)

            elif action_type == "multi-hot":
                if data_missing[-1] == 1:
                    categories = []
                else:
                    categories = str(data_value).split(",")
                multi_hot_vector = [0] * len(col["Action"]["categories"])
                for idx, category in enumerate(col["Action"]["categories"]):
                    if category in categories:
                        multi_hot_vector[idx] = 1
                data_extra.extend(multi_hot_vector)

        # Scale
        data_transformed = self.scaler.transform([data_array])
        # Concatenate all features
        data = np.concatenate([data_transformed[0], data_extra, data_missing]).astype(
            np.float32
        )
        # ONNX expects 2D array: [batch_size, features]
        return np.expand_dims(data, axis=0)

    @bentoml.api
    def predict(self, payload: dict[str, Any]) -> float:
        # Preprocess input
        input_array = self.preprocess(payload)
        # Get ONNX model input name
        input_name = self.session.get_inputs()[0].name
        # Run inference
        pred = self.session.run(None, {input_name: input_array})
        # ONNX Runtime returns a list of outputs
        return float(pred[0][0])
