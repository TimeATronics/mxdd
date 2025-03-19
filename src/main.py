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
import sys
import cv2
import os
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSplashScreen, QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QMenuBar, QMenu, QDialog, QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

MODEL_PATH = "mnet_xgboost1.pkl"
SCALER_PATH = "scaler_norm1.pkl"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

script_dir = os.path.dirname(os.path.abspath(__file__))
splash_screen_path = os.path.join(script_dir, 'img', 'splash_screen.png')
icon_path = os.path.join(script_dir, 'img', 'icon.png')

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_deep_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = mobilenet.predict(image)
    return features.flatten()

def classify_image(image):
    feature_vector = extract_deep_features(image)
    feature_vector = scaler.transform([feature_vector])
    probability = clf.predict_proba(feature_vector)[0]  # Get probabilities
    return probability[0], probability[1]  # (real_prob, fake_prob)

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, (299, 299))
    return None

class TextPopup(QDialog):
    def __init__(self, title, content):
        super().__init__()
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setText(content)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        self.setLayout(layout)

class ResultPopup(QDialog):
    def __init__(self, results):
        super().__init__()
        self.setWindowTitle("Classification Results")
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setRowCount(len(results))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File Path", "Real %", "Fake %"])
        
        for i, (file_name, real_prob, fake_prob) in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(file_name))  # Full path
            self.table.setItem(i, 1, QTableWidgetItem(f"{real_prob * 100:.2f}%"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{fake_prob * 100:.2f}%"))
        
        layout.addWidget(self.table)
        
        save_button = QPushButton("Save as CSV", self)
        save_button.clicked.connect(lambda: self.save_results_as_csv(results))
        layout.addWidget(save_button)
        
        self.setLayout(layout)

    def save_results_as_csv(self, results):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)", options=options)
        if file_path:
            df = pd.DataFrame(results, columns=["File Path", "Real %", "Fake %"])
            df.to_csv(file_path, index=False)

class DeepfakeDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.dark_mode = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle("MXDD v0.1")
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(100, 100, 600, 500)

        # Menu Bar
        menubar = QMenuBar(self)
        file_menu = QMenu("File", self)
        view_menu = QMenu("View", self)
        help_menu = QMenu("Help", self)

        file_menu.addAction("Open Image", self.load_image)
        file_menu.addAction("Open Video", self.load_video)
        file_menu.addAction("Process Folder", self.process_folder)  # New action
        file_menu.addAction("Exit", self.close)

        view_menu.addAction("Toggle Dark/Light Mode", self.toggle_theme)
        view_menu.addAction("Reset", self.reset_ui)  # New reset action

        help_menu.addAction("About", lambda: self.show_popup("About", "This is a deepfake detection tool."))
        help_menu.addAction("Readme", lambda: self.show_popup("Readme", "Instructions on how to use the tool."))
        help_menu.addAction("License", lambda: self.show_popup("License", "This software is licensed under XYZ."))

        menubar.addMenu(file_menu)
        menubar.addMenu(view_menu)
        menubar.addMenu(help_menu)

        # Image display
        self.image_label = QLabel("No Image Loaded", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray;")

        # File selection buttons
        self.btn_open_image = QPushButton("Open Image", self)
        self.btn_open_image.clicked.connect(self.load_image)
        
        self.btn_open_video = QPushButton("Open Video", self)
        self.btn_open_video.clicked.connect(self.load_video)

        # Classification Result
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px;")
        
        # Certainty Panel
        self.real_label = QLabel("REAL: ", self)
        self.fake_label = QLabel("FAKE: ", self)
        self.real_label.setStyleSheet("font-size: 16px;")
        self.fake_label.setStyleSheet("font-size: 16px;")
        
        self.panel = QFrame(self)
        panel_layout = QHBoxLayout()
        panel_layout.addWidget(self.real_label)
        panel_layout.addWidget(self.fake_label)
        self.panel.setLayout(panel_layout)
        self.panel.setStyleSheet("background: lightgray; padding: 5px;")
        
        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)  # Set range for determinate mode
        self.progress_bar.setVisible(False)  # Initially hidden

        # Layout
        layout = QVBoxLayout()
        layout.setMenuBar(menubar)
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_open_image)
        layout.addWidget(self.btn_open_video)
        layout.addWidget(self.result_label)
        layout.addWidget(self.panel)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                real_prob, fake_prob = classify_image(image)
                self.display_result(real_prob, fake_prob, image)
            else:
                self.result_label.setText("Error: Could not classify image")

    def load_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)", options=options)
        if file_path:
            cap = cv2.VideoCapture(file_path)
            frames_processed = 0
            while cap.isOpened() and frames_processed < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                face = extract_face(frame)
                if face is not None:
                    real_prob, fake_prob = classify_image(face)
                    self.display_result(real_prob, fake_prob, face)
                    frames_processed += 1
            cap.release()

    def process_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder_path:
            self.results = []
            self.progress_bar.setVisible(True)  # Show progress bar
            self.progress_bar.setRange(0, 0)  # Indeterminate mode
            self.progress_bar.setWindowTitle("Processing")
            self.progress_bar.setWindowModality(Qt.WindowModal)
            self.progress_bar.show()

            # Process files in the folder
            self.process_files_in_folder(folder_path)

    def process_files_in_folder(self, folder_path):
        import os

        class Worker(QThread):
            update_progress = pyqtSignal(int)
            finished = pyqtSignal()

            def __init__(self, folder_path, results):
                super().__init__()
                self.folder_path = folder_path
                self.results = results

            def run(self):
                files = os.listdir(self.folder_path)
                total_files = len(files)
                for i, file_name in enumerate(files):
                    file_path = os.path.join(self.folder_path, file_name)
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image = cv2.imread(file_path)
                        if image is not None:
                            real_prob, fake_prob = classify_image(image)
                            self.results.append((file_path, real_prob, fake_prob))  # Full path
                    elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                        cap = cv2.VideoCapture(file_path)
                        frames_processed = 0
                        while cap.isOpened() and frames_processed < 5:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            face = extract_face(frame)
                            if face is not None:
                                real_prob, fake_prob = classify_image(face)
                                self.results.append((file_path, real_prob, fake_prob))  # Full path
                                frames_processed += 1
                        cap.release()
                    self.update_progress.emit(int((i + 1) / total_files * 100))
                self.finished.emit()

        self.results = []
        self.worker = Worker(folder_path, self.results)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_results(self):
        self.progress_bar.setVisible(False)  # Hide progress bar
        result_popup = ResultPopup(self.results)
        result_popup.exec_()

    def display_result(self, real_prob, fake_prob, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        bytes_per_line = channels * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(q_img)
        pixmap = QPixmap.fromImage(q_img).scaled(int(400 * self.zoom_factor), int(400 * self.zoom_factor), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        if real_prob > fake_prob:
            self.real_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
            self.fake_label.setStyleSheet("font-size: 18px; font-weight: normal; color: black;")
        else:
            self.fake_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            self.real_label.setStyleSheet("font-size: 18px; font-weight: normal; color: black;")
        self.real_label.setText(f"REAL: {real_prob * 100:.2f}%")
        self.fake_label.setText(f"FAKE: {fake_prob * 100:.2f}%")
        self.result_label.setText("Classified Successfully")

    def reset_ui(self):
        self.image_label.setPixmap(QPixmap())  # Clear the image
        self.result_label.setText("")  # Clear the result label
        self.real_label.setText("REAL: ")  # Reset real label
        self.fake_label.setText("FAKE: ")  # Reset fake label
        self.progress_bar.setVisible(False)  # Hide progress bar
        self.results = []  # Clear results

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.update_theme()

    def update_theme(self):
        if self.dark_mode:
            self.setStyleSheet("background-color: #2E2E2E; color: white;")
            self.panel.setStyleSheet("background: #2E2E2E; padding: 5px;")
        else:
            self.setStyleSheet("background-color: white; color: black;")
            self.panel.setStyleSheet("background: white; padding: 5px;")
        
        # Update menu hover text color
        self.setStyleSheet(self.styleSheet() + " QMenu::item:selected { background-color: lightgray; color: black; }")

    def show_popup(self, title, content):
        popup = TextPopup(title, content)
        popup.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = QSplashScreen(QPixmap(splash_screen_path))
    splash.show()

    def start_main():
        splash.close()
        global window
        window = DeepfakeDetector()
        window.show()

    QTimer.singleShot(3000, start_main)
    sys.exit(app.exec())