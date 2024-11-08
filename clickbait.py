import os
import re

from numpy import isin
import plotly.graph_objects as go
import plotly.colors as colors
import plotly.io as pio
import plotly.express as px

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
                assert len(path_parts) == 3, "Directory structure should be dataset/session/file"

                dataset_list.append(path_parts[-2])
                session_list.append(path_parts[-1])
                file_list.append(remove_file_extension(file))

    return dataset_list, session_list, file_list



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