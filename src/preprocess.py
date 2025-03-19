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
import random

# Input and output folder paths
VIDEO_FOLDER = "C:\\Users\\AradhyaPC\\Desktop\\videos_fake"
OUTPUT_FOLDER = "C:\\Users\\AradhyaPC\\Desktop\\deepfake_detection\\fake_new"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_face_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        return False
    random_frame_indices = random.sample(range(frame_count), min(5, frame_count))

    face_extracted = False
    for idx in random_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (299, 299))
            cv2.imwrite(output_path, face_resized)
            face_extracted = True
            break
        if face_extracted:
            break

    cap.release()
    return face_extracted

def process_videos(video_folder, output_folder):
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            
            success = extract_face_frame(video_path, output_path)
            if success:
                print(f"Extracted face frame from: {filename}")
            else:
                print(f"No face found in: {filename}")

process_videos(VIDEO_FOLDER, OUTPUT_FOLDER)