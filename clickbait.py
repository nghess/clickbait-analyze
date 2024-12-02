import os
import re
import cv2
import ast
import datetime
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.colors as colors
import plotly.graph_objects as go

'''
File system
'''
def scan_dataset(data_directory, min_size_bytes, filetype='.tif'):
    """
    Scans a directory for files of a specific type and organizes them into dataset/session/file lists.
    
    Args:
        data_directory (str): Root directory to scan
        min_size_bytes (int): Minimum file size in bytes to include
        filetype (str): File extension to filter by (default: '.tif')
    
    Returns:
        tuple: Three lists containing:
            - mouse_list: IDs of mice (parent directories)
            - session_list: Names of sessions (sub-directories)
            - file_list: Names of files (without extensions)
    
    Expected directory structure:
        root_directory/
        └── dataset/
            └── session/
                └── file.tif
    """
    def remove_file_extension(filename:str):
        
        """Removes file extension from a filename"""
        return re.sub(r'\..*$', '', filename)

    mouse_list, session_list, file_list = [], [], []

    for root, _, files in os.walk(data_directory):
        # Skip directories containing specific keywords
        if any(x in root.lower() for x in ["exclude", "movement", "missing"]):
            continue

        for file in files:
            if file.endswith(filetype):
                file_path = os.path.join(root, file)
                # Skip files smaller than minimum size
                if os.path.getsize(file_path) < min_size_bytes:
                    continue

                # Split path into components and verify structure
                path_parts = os.path.normpath(root).split(os.sep)
                #assert len(path_parts) == 3, "Directory structure should be: dataset/session/file."

                # Store the dataset (parent dir), session (sub-dir), and file names
                mouse_list.append(path_parts[-2])
                session_list.append(path_parts[-1])
                file_list.append(remove_file_extension(file))

    return mouse_list, session_list, file_list

'''
Data
'''
class BehaviorSession:
    def __init__(self, data_dir, mouse_id, session_id, file_id):
        self.data_dir = data_dir
        self.mouse_id = mouse_id
        self.session_id = session_id
        self.file_id = file_id
        self.data_path = f"{data_dir}{mouse_id}/{session_id}/{file_id}"
        
        # Initialize data containers
        self.video_ts = None
        self.event_data = None
        self.video_filename = None
        
        # Load the session data
        self._load_session_data()
        
    def _load_session_data(self):
        """Internal method to load all session data"""
        self.video_filename = f"{self.data_path}.avi"
        
        # Load and process timestamps
        self._load_timestamps()
        
        # Load and process events
        self._load_events()
        
        # Add computed columns
        self._add_computed_columns()
        
    def _load_timestamps(self):
        """Load video timestamps"""
        self.video_ts = pd.read_csv(f"{self.data_path}_video_timestamp.csv")
        self.video_ts.columns = ['timestamp']
        self.video_ts = self.video_ts.astype({'timestamp': 'datetime64[ns]'})
        
    def _load_events(self):
        """Load and process event data"""
        # Load events A
        col_names_a = ['trial_number', 'timestamp', 'poke_left', 'poke_right', 
                      'centroid_x', 'centroid_y', 'target_cell']
        event_data_a = pd.read_csv(f"{self.data_path}_eventsA.csv")
        event_data_a.columns = col_names_a
        
        # Load events B
        col_names_b = ['iti', 'reward_state', 'water_left', 'water_right', 'click']
        event_data_b = pd.read_csv(f"{self.data_path}_eventsB.csv")
        event_data_b.columns = col_names_b
        
        # Combine and process
        self._combine_events(event_data_a, event_data_b)
        
    def _combine_events(self, event_data_a, event_data_b):
        """Combine and validate event data"""
        if len(event_data_a) != len(event_data_b):
            print("Event dataframes must contain same number of rows")
            min_length = min(len(event_data_a), len(event_data_b))
            max_length = max(len(event_data_a), len(event_data_b))
            print(f"Trimmed long dataframe by {max_length-min_length} rows.")
            event_data_a = event_data_a.iloc[:min_length]
            event_data_b = event_data_b.iloc[:min_length]
            
        self.event_data = pd.concat([event_data_a, event_data_b], axis=1)
        self._set_datatypes()
        
    def _set_datatypes(self):
        """Set proper datatypes for all columns"""
        self.event_data = self.event_data.astype({
            'trial_number': 'uint8',
            'timestamp': 'datetime64[ns]',
            'poke_left': 'bool',
            'poke_right': 'bool',
            'centroid_x': 'uint16',
            'centroid_y': 'uint16',
            'target_cell': 'str',
            'iti': 'bool',
            'water_left': 'bool',
            'water_right': 'bool',
            'reward_state': 'bool',
            'click': 'bool'
        })
        self.event_data['target_cell'] = self.event_data['target_cell'].apply(ast.literal_eval)
        
    def _add_computed_columns(self):
        """Add computed columns to event_data"""
        # Add distance column
        self.event_data['distance'] = np.sqrt(
            (self.event_data['centroid_x'] - self.event_data['centroid_x'].shift(1))**2 + 
            (self.event_data['centroid_y'] - self.event_data['centroid_y'].shift(1))**2)
            
        # Add frame_ms column
        self.event_data['frame_ms'] = self.event_data['timestamp'].diff().dt.total_seconds() * 1000
        
        # Add gap column
        gap_thresh = 100
        self.event_data['gap'] = (self.event_data['distance'] >= gap_thresh).astype(np.uint8)
        
    def synchronize_timestamps(self):
        """Synchronize video timestamps with event data"""
        if len(self.video_ts) < len(self.event_data):
            self._sync_events_to_video()
        elif len(self.video_ts) > len(self.event_data):
            self._trim_video_timestamps()
            
    def get_session_info(self):
        """Print session information"""
        print(f"Mouse: {self.mouse_id} Session: {self.session_id}")
        print(f"Video length: {len(self.video_ts)} frames")
        print(f"Events Data Length: {len(self.event_data)} rows")
        print(f"Video length at 50.6 FPS: {len(self.video_ts)/50.6/60:.2f} minutes")

class BehaviorExperiment:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sessions = {}
        
    def load_session(self, mouse_id, session_id, file_id):
        """Load a single session"""
        session_key = f"{mouse_id}_{session_id}"
        print(f"Loading session: {session_key}")
        self.sessions[session_key] = BehaviorSession(self.data_dir, mouse_id, session_id, file_id)
        
    def load_all_sessions(self, mice, sessions, files):
        """Load all sessions from provided lists"""
        for idx in range(len(mice)):
            self.load_session(mice[idx], sessions[idx], files[idx])
            
    def get_session(self, mouse_id, session_id):
        """Retrieve a specific session"""
        session_key = f"{mouse_id}_{session_id}"
        return self.sessions.get(session_key)

'''
Grid
'''
class GridMaze: 
    def __init__(self, maze_dims, maze_cells, border=False):
            assert len(maze_dims) == 2, "Maze bounds must be 2-dimensional list or tuple."
            assert len(maze_cells) == 2, "Maze dimensions must be 2-dimensional (rows * columns)."

            self.shape = maze_dims
            self.cell_shape = maze_cells
            self.rows = maze_cells[0]
            self.cols = maze_cells[1]
            self.task_idx = []
            self.arena_idx = []

            self.cellsize_i = maze_dims[0]//maze_cells[0]
            self.cellsize_j = maze_dims[1]//maze_cells[1]

            self.start = 1 if border else 0
            self.end_i = self.rows - (1 if border else 0)
            self.end_j = self.cols - (1 if border else 0)

            self.cells = [[[[0,0],[0,0]] for j in range(self.cols)] for i in range(self.rows)]
            self.cells_count = len(self.cells) + len(self.cells[0])

            # Single loop structure for both cases
            for i in range(self.rows):
                for j in range(self.cols):
                    self.cells[i][j] = [
                        [j * self.cellsize_j, i * self.cellsize_i],
                        [(j + 1) * self.cellsize_j, (i + 1) * self.cellsize_i]]
                    
                    self.arena_idx.append(self.cells[i][j])
                
    def draw_grid(self, frame, color=(0,0,0), opacity=.5):
        overlay = frame.copy()
        self.cells_count = 0
        for i in range(self.start, self.end_i):
                for j in range(self.start, self.end_j):
                    cv2.rectangle(overlay, self.cells[i][j][0], self.cells[i][j][1], color, thickness=1)
                    self.task_idx.append((self.cells[i][j][0], self.cells[i][j][1]))
                    self.cells_count += 1
        return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    def get_target_cell(self, cell_idx):
        self.tgt_cells = []
        if len(cell_idx) > 0: 
            for i in range(self.start, self.end_i):
                for j in range(self.start, self.end_j):
                    self.tgt_cells.append(self.cells[i][j])
            return self.tgt_cells[cell_idx[0]]
        else:
            return [[-10,-10],[-5,-5]]

    def draw_cell(self, frame, i, j, color, thickness, opacity):
        # Create a separate image for the rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, self.cells[i][j][0], self.cells[i][j][1], color, thickness)
        # Blend the overlay with the original frame
        return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    def draw_target_cell(self, frame, pt1, pt2, color, thickness, opacity):
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, thickness)
        # Blend the overlay with the original frame
        return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    

    def get_mouse_cell(self, centroid_x, centroid_y):    
        # Calculate grid position accounting for border
        grid_i = int(centroid_y // self.cellsize_i)  # Subtract 1 to account for border
        grid_j = int(centroid_x // self.cellsize_j)  # Subtract 1 to account for border

        return grid_i, grid_j

'''
Video
'''

def slice_video_timestamp(df_to_trim, reference_df, timestamp_col='timestamp'):
    # Get the first timestamp from the reference DataFrame
    reference_start = reference_df[timestamp_col].iloc[0]
    
    # Find the index of the closest matching timestamp in df_to_trim
    slice_idx = (df_to_trim[timestamp_col] - reference_start).abs().idxmin()
    
    # Return the trimmed DataFrame starting from the closest match
    return df_to_trim.iloc[slice_idx:], slice_idx

'''
Data Visualization
'''
def visualize_trial(df, trial_number, color_code="trial_number", target_frame=True, grid_x=4, grid_y=9, dim_x=894, dim_y=1952, opacity=1.0):
    assert color_code in(['trial_number', 'frame_number', 'velocity']), "Valid color codes are 'trial_number', 'frame_number', and 'velocity'."
    
    df = df.loc[df['trial_number'].isin(trial_number)].copy()

    if color_code == "frame_number":
            df = df.reset_index(drop=True)
            color_code = df.index

    # Create the scatter plot with Plotly
    fig = px.scatter(df, x='centroid_x', y='centroid_y', color=color_code, opacity=opacity, color_continuous_scale="Bluered",
        title=f"2D Scatter Plot of Centroid Coordinates<br><b>Trials: {str(trial_number)[1:-1]}</b>")
    
    # Add target frame locations
    if target_frame:
        target_idx = df.index[df['reward_state'] & ~df['reward_state'].shift(1).fillna(False)]
        print(target_idx)
        fig.add_scatter(
        x=df['centroid_x'][target_idx],
        y=df['centroid_y'][target_idx],
        mode='markers',
        opacity=.25,
        marker=dict(
            size=50,
            color='green',
            symbol='circle'  # 'circle', 'square', 'diamond', etc.
        ),
        name='Target Found'
        )
    
    fig.update_layout(
        xaxis=dict(tick0=0, dtick=dim_x//grid_x, gridwidth=1),
        yaxis=dict(tick0=0, dtick=dim_y//grid_y, gridwidth=1),
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        yaxis_scaleanchor='x',
        width=dim_x,
        height=dim_y
        )

    # Show the plot
    fig.show()

def generate_grid_location(df, grid_x=4, grid_y=9, dim_x=894, dim_y=1952):
    grid_loc_x = df['centroid_x']
    grid_loc_y = df['centroid_y']