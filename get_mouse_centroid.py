import cv2
import os
import csv
import numpy as np

def scan_directories(data_directory, min_size_bytes, filetype='.tif'):
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

def process_video(video_filename, csv_filename):
    video = cv2.VideoCapture(video_filename)
    centroid_list = []
    frame_count = 0

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                continue

            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_frame = cv2.GaussianBlur(grey_frame, (27, 27), 20)
            _, thresh_frame = cv2.threshold(blur_frame, 32, 255, cv2.THRESH_BINARY)
            thresh_frame = np.abs(thresh_frame - 255)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_frame, connectivity=8)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            centroid = centroids[largest_label]
            centroid_list.append(centroid)

    except Exception as e:
        print(f"Error processing {video_filename}: {str(e)}")

    finally:
        video.release()

    print(f'Saving {csv_filename}')
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(centroid_list)

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