import csv
import numpy as np
import pandas as pd


# Params
num = '5'
file_dir = f'C:/centroid_extract_temp/1003/Clickbait {num}/'

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

event_data['click'] = list(np.zeros(len(event_data)))

click_frame = []

for ii in range(len(event_data)-1):
    if event_data['reward_state'][ii] == False and event_data['reward_state'][ii+1] == True:
        event_data['reward_state'][ii] = 1
        click_frame.append(ii)

# Write click_frame to CSV
with open(f'{file_dir}click_frames_session{num}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(click_frame)

print(len(click_frame))

print(event_data.head())