import os
import re
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors
import plotly.io as pio
import plotly.express as px


'''
File system
'''
def scan_dataset(data_directory, min_size_bytes, filetype='.tif'):

    def remove_file_extension(filename:str):
        return re.sub(r'\..*$', '', filename)

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
                assert len(path_parts) == 3, "Directory structure should be: dataset/session/file."

                dataset_list.append(path_parts[-2])
                session_list.append(path_parts[-1])
                file_list.append(remove_file_extension(file))

    return dataset_list, session_list, file_list

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