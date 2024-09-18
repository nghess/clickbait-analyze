import cv2
import numpy as np
import pandas as pd
import grid_maze as gm

num = '5'
file_dir = f'C:/centroid_extract_temp/1003/Clickbait {num}/'

# Load video
video = cv2.VideoCapture(f'{file_dir}video_session{num}.avi')
# Get the total number of frames
video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_factor = 1
# Load video timestamps
video_ts = pd.read_csv(f'{file_dir}video_ts_session{num}.csv')
video_ts.columns = ['timestamp']
# Load event CSV
col_names = ['trial_number', 'water_left', 'water_right', 'iti', 'reward_state', 'timestamp', 'target_cell']
event_data = pd.read_csv(f'{file_dir}events_session{num}.csv', usecols=range(7))
event_data.columns = col_names

# Load centroid CSV
centroid = pd.read_csv(f'{file_dir}centroid.csv', header=None)
#print(centroid.head())
# Load click frames CSV
click = pd.read_csv(f'{file_dir}click_frames_session{num}.csv', header=None).T
#print(click.head())
# Use click values and indices to pull out centroid values and add them to a list
click_centroids = []

for ii in click[0]:
    click_centroids.append([centroid[0][ii]*2, centroid[1][ii]*2])


#Set Types
event_data = event_data.astype({
    'trial_number': 'uint8',
    'water_left': 'bool',
    'water_right': 'bool',
    'iti': 'bool',
    'reward_state': 'bool',
    'timestamp': 'datetime64[ns]',
    'target_cell': 'float'
})

# Resample event_data to match the length of video
video_ts = video_ts.set_index('timestamp')
event_data = event_data.set_index('timestamp')
event_data = event_data.reindex(video_ts.index, method='nearest')
video_ts = video_ts.reset_index()
event_data = event_data.reset_index()

# Print debugging
print(f"video frames:{video_len}")
print(f"video_timestamp.csv length: {len(video_ts)}")
print(f"event_data.csv length: {len(event_data)}")
print(f"centroid.csv length: {len(centroid)}")
# print(event_data.head())

# Generate grid maze
grid_maze = gm.GridMaze((video_width,video_height), (10,4))

# Iterate through targets, and create list of each target that belonged to a particular trial_number
# all_targets = []
# for trial_num in event_data['trial_number'].unique():
#     trial_targets = event_data[event_data['trial_number'] == trial_num]['target_cell']
#     all_targets.extend(trial_targets)

all_targets = event_data['target_cell'].unique()


# Draw all targets from event_data on a blank canvas
canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
for tgt in all_targets:
    if pd.notna(tgt):
        tgt_idx = int(float(tgt))
        tgt_cell = grid_maze.cells[tgt_idx]
        canvas = gm.draw_grid_cell(canvas, tgt_cell[0], tgt_cell[1], (0,255,0), -1, opacity=.25)

# Display click centroids on canvas
for click in click_centroids:
    canvas = cv2.circle(canvas, (int(click[0]), int(click[1])), 5, (255, 0, 0), -1)


# Display Targets
cv2.imshow(f'clickbait session {num} targets', canvas)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows


