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
import cv2
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

MODEL_PATH = "mnet_xgboost.pkl"
SCALER_PATH = "scaler_norm.pkl"
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
def extract_deep_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = mobilenet.predict(image)
    return features.flatten()

def classify_image(image_path):
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image!")
        return
    feature_vector = extract_deep_features(image)
    feature_vector = scaler.transform([feature_vector]) # Normalize features
    prediction = clf.predict(feature_vector)[0]
    label = "FAKE" if prediction == 1 else "REAL"
    print(f"Image: {image_path} -> Prediction: {label}")

deepfake_root_dir = "C:\\Users\\AradhyaPC\\Desktop\\deepfake_detection\\dataset\\ffhq\\validation"
image_paths = []
for root, _, files in os.walk(deepfake_root_dir):
    for file in files:
        if file.lower().endswith(".png"):  # Check for PNG files
            image_paths.append(os.path.join(root, file))
for i in image_paths:
    classify_image(i)

#classify_image("C:\\Users\\AradhyaPC\\Desktop\\img4.png")