import cv2
import numpy as np
from clickbait import *
from pathlib import Path

def process_and_save_frames(input_video_path, output_dir):
   Path(output_dir).mkdir(parents=True, exist_ok=True)
   
   cap = cv2.VideoCapture(input_video_path)
   backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
   
   frame_count = 0
   while True:
       ret, frame = cap.read()
       if not ret:
           break
           
       fg_mask = backSub.apply(frame)
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
       fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
       fg_image = cv2.bitwise_and(frame, frame, mask=fg_mask)
       
       cv2.imwrite(f"{output_dir}/frame_{frame_count:06d}.jpg", fg_image)
       frame_count += 1
       
   cap.release()

"""
Process video
"""
# Load filename
data_dir = f'data/'
datasets, sessions, files = scan_dataset(data_dir, min_size_bytes=1e9, filetype='.avi')
data_path = f"{data_dir}{datasets[0]}/{sessions[0]}/{files[0]}"

# Get video filename
video_filename = f"{data_path}.avi"

process_and_save_frames(video_filename, 'output')