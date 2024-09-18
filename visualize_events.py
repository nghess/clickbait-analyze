import cv2
import numpy as np
import pandas as pd
import grid_maze as gm

num = '9'
file_dir = f'C:/centroid_extract_temp/1003/Clickbait {num}/'

# Load video
video = cv2.VideoCapture(f'{file_dir}video_session{num}.avi')
# Get the total number of frames
video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_factor = .5
# Load video timestamps
video_ts = pd.read_csv(f'{file_dir}video_ts_session{num}.csv')
video_ts.columns = ['timestamp']
# Load event CSV
col_names = ['trial_number', 'water_left', 'water_right', 'iti', 'reward_state', 'timestamp', 'target_cell']
event_data = pd.read_csv(f'{file_dir}events_session{num}.csv', usecols=range(7))
event_data.columns = col_names
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
print(event_data.head())

# Generate grid maze
grid_maze = gm.GridMaze((video_width//2,video_height//2), (10,4))

# Initialize counter
ii = 0
state_color = (0,0,0)

video.set(cv2.CAP_PROP_POS_FRAMES, ii)

# Display video with event overlay
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame to save memory
    height, width, depth = frame.shape
    resized_frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
    height, width, depth = resized_frame.shape

    # Draw target cell
    if pd.notna(event_data['target_cell'][ii]):
        tgt_idx = int(float(event_data['target_cell'][ii]))
        tgt_cell = grid_maze.cells[tgt_idx]
        resized_frame = gm.draw_grid_cell(resized_frame, tgt_cell[0], tgt_cell[1], (0,255,0), -1, opacity=.25)

    
    grey_frame = cv2.cvtColor(resized_frame, code=cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(grey_frame, (27,27), 20, 20)
    _, thresh_frame = cv2.threshold(blur_frame, 32, 255, type=cv2.THRESH_BINARY) #cv2.TH
    thresh_frame = np.abs(thresh_frame - 255)
    #thresh_frame += grey_frame
    thresh_frame = cv2.cvtColor(thresh_frame, code=cv2.COLOR_GRAY2BGR)

    # Change trial number text color based on trial state
    if event_data['reward_state'][ii]:
        state_color = (0,255,0)
    elif not event_data['reward_state'][ii] and (pd.isna(event_data['target_cell'][ii]) or event_data['target_cell'][ii] == ''):
        state_color = (0,0,255)
    else:
        state_color = (255,0,0)

    # Print trial number at top left corner
    cv2.putText(resized_frame, str(event_data['trial_number'][ii]), (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, state_color, 1, cv2.LINE_AA) 
    # Print frame number at top left corner
    cv2.putText(resized_frame, str(ii), (10,40), cv2.FONT_HERSHEY_SIMPLEX, .5, state_color, 1, cv2.LINE_AA) 
    # Print event_data timestamp at bottom left corner
    cv2.putText(resized_frame, str(event_data['timestamp'][ii])[:22], (10,height-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (128,128,128), 1, cv2.LINE_AA) 
    # Print video_ts timestamp at bottom left corner
    cv2.putText(resized_frame, str(video_ts['timestamp'][ii])[:22], (10,height-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (128,128,128), 1, cv2.LINE_AA) 

    # Blend threshold/centroid with original video
    opacity = .5 
    comp_frame = cv2.addWeighted(thresh_frame, opacity, resized_frame, 1 - opacity, 0)

    # Display Video
    cv2.imshow('clickbait synch test', resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Increment frame counter
    ii += 1

    # If we've reached the end of the video, reset to the beginning and clear the counter
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ii = 0
        continue


