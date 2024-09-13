import cv2
import os
import csv
import numpy as np

def scan_directories(data_directory, min_size_bytes, filetype='.avi'):
    dataset_list, session_list, file_list = [], [], []

    for root, _, files in os.walk(data_directory):
        if any(x in root.lower() for x in ["exclude", "movement", "missing"]):
            continue

        for file in files:
            if file.endswith(filetype):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) < min_size_bytes:
                    continue

                path_parts = os.path.normpath(root).split(os.sep)
                if len(path_parts) >= 3:
                    dataset_list.append(path_parts[-2])
                    session_list.append(path_parts[-1])
                    file_list.append(file)

    return dataset_list, session_list, file_list

def process_chunk(chunk, background_subtractor):
    centroids = []
    for frame in chunk:
        # Downsample the frame
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the mouse)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))

    return centroids

def process_video(video_filename, csv_filename):
    video = cv2.VideoCapture(video_filename)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    chunk_size = 100  # Process 100 frames at a time
    all_centroids = []

    try:
        while True:
            chunk = []
            for _ in range(chunk_size):
                ret, frame = video.read()
                if not ret:
                    break
                chunk.append(frame)

            if not chunk:
                break

            centroids = process_chunk(chunk, background_subtractor)
            all_centroids.extend(centroids)

    except Exception as e:
        print(f"Error processing {video_filename}: {str(e)}")

    finally:
        video.release()

    print(f'Saving {csv_filename}')
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_centroids)

def main():
    data_dir = 'T:/clickbait/data/no_implant'
    min_file_size = 1e9  # 1 GB
    datasets, sessions, files = scan_directories(data_dir, min_file_size, filetype='.avi')
    print(f"Located {len(files)} sessions")

    for ii, video_file in enumerate(files):
        video_filename = os.path.join(data_dir, datasets[ii], sessions[ii], files[ii])
        csv_filename = os.path.join(data_dir, datasets[ii], sessions[ii], 'centroid.csv')
        print(f'Processing session {ii}: {video_filename}')
        process_video(video_filename, csv_filename)

if __name__ == "__main__":
    main()