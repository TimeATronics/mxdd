"""
    This file is part of the MXDD distribution (https://github.com/TimeATronics/mxdd).
    Copyright (c) 2025 Aradhya Chakrabarti

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but 
    WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
    General Public License for more details.

    You should have received a copy of the GNU General Public License 
    along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf
mobilenet = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
import tf2onnx
onnx_model_path = "mobilenetv2.onnx"

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(mobilenet, input_signature=spec, opset=13)
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model saved to {onnx_model_path}")
