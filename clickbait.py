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
            self.rows = maze_cells[0]
            self.cols = maze_cells[1]
            self.cells = [[None for x in range(self.cols)] for y in range(self.rows)]
            self.cellsize_i = maze_dims[0]//maze_cells[0]
            self.cellsize_j = maze_dims[1]//maze_cells[1]

            # Fill the matrix with coordinates
            for i in range(self.rows):
                for j in range(self.cols):
                    self.cells[i][j] = (
                        (j * self.cellsize_j, i * self.cellsize_i),
                        ((j + 1) * self.cellsize_j, (i + 1) * self.cellsize_i))
            # else:
            #     self.cells = [[
            #         ((x * cellsize_x, y * cellsize_y), 
            #         ((x + 1) * cellsize_x, (y + 1) * cellsize_y))
            #         for y in range(1, self.shape[1]-1)]
            #         for x in range(1, self.shape[0]-1)]
                
    def draw_grid(self, frame, color=(0,0,0)):
        canvas = frame.copy()
        for i in range(self.rows):
                for j in range(self.cols):
                    cv2.rectangle(canvas, self.cells[i][j][0], self.cells[i][j][1], color, thickness=1)
        return canvas

    def get_cell(self, x, y):
        if 0 <= y < len(self.cells) and 0 <= x < len(self.cells[0]):
            return self.cells[y][x]
        return None

    def draw_mouse_cell(self, frame, pt1, pt2, color, thickness, opacity):
            # Create a separate image for the rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, pt1, pt2, color, thickness)
            # Blend the overlay with the original frame
            return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    def draw_target_cell(self, frame, pt1, pt2, color, thickness, opacity):
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, thickness)
        # Blend the overlay with the original frame
        return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

'''
Visualization
'''
def visualize_trial(df, trial_number, color_code="trial_number", grid_x=4, grid_y=9, dim_x=894, dim_y=1952, opacity=1.0):
    assert color_code in(['trial_number', 'frame_number', 'velocity']), "Valid color codes are 'trial_number', 'frame_number', and 'velocity'."
    
    df = df.loc[df['trial_number'].isin(trial_number)].copy()

    if color_code == "frame_number":
            df = df.reset_index(drop=True)
            color_code = df.index

    # Create the scatter plot with Plotly
    fig = px.scatter(df, x='centroid_x', y='centroid_y', color=color_code, opacity=opacity, color_continuous_scale="Bluered",
                    title=f"2D Scatter Plot of Centroid Coordinates<br><b>Trials: {str(trial_number)[1:-1]}</b>")
    
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