import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.colors as colors
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

'''
Data Visualization
'''

def visualize_occupancy_heatmap(df, grid_x=9, grid_y=20, dim_x=894, dim_y=1952, colorscale='viridis', title="Occupancy Heatmap", sigma=2, log_scale=False, normalize=False, show_ticks=False, title_size=12, label_size=10, tick_size=8):
    # Get min/max coordinates
    x_min, x_max = df['centroid_x'].min(), df['centroid_x'].max()
    y_min, y_max = df['centroid_y'].min(), df['centroid_y'].max()

    # Create histogram
    hist, x_edges, y_edges = np.histogram2d(
        df['centroid_x'], 
        df['centroid_y'],
        bins=[grid_x, grid_y],
        #range=[[0, dim_x], [0, dim_y]]  # Set range from 0 to dimensions
        range=[[x_min, x_max], [y_min, y_max]]  # Set range from min to max centroids
    )

    # Convert to probability density if requested
    if normalize:
        hist = hist / hist.sum()

    # Apply Gaussian smoothing
    hist_smooth = gaussian_filter(hist.T, sigma=sigma)

    # Apply log scaling if requested
    if log_scale:
        hist_smooth = np.log1p(hist_smooth)  # log1p adds 1 before taking log to handle zeros

    # Create the plot
    fig, ax = plt.subplots(figsize=(6.3, 10))
    im = ax.imshow(hist_smooth, 
                   cmap=colorscale,
                   extent=(x_min, x_max, y_min, y_max),  # Set extent from min to max
                   aspect='auto',
                   origin='lower')

    # Add colorbar with appropriate label
    if normalize and log_scale:
        label = 'Log Probability Density'
    elif normalize:
        label = 'Probability Density'
    elif log_scale:
        label = 'Log Occupancy Count'
    else:
        label = 'Occupancy Count'
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, size=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    # Set labels and title
    ax.set_xlabel('X Coordinate', fontsize=label_size)
    ax.set_ylabel('Y Coordinate', fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    # Set ticks if show_ticks is True
    if show_ticks:
        ax.set_xticks(np.linspace(x_min, x_max, grid_x+1))
        ax.set_yticks(np.linspace(y_min, y_max, grid_y+1))
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def visualize_trial_trajectory(df, trial_number, color_code="trial_number", target_frame=True, grid_x=4, grid_y=9, dim_x=894, dim_y=1952, opacity=1.0):
    assert color_code in(df.columns.tolist()), "Use a column name from the dataframe for color_code."
    
    df = df.loc[df['trial_number'].isin(trial_number)].copy()

    if color_code == "frame_number":
            df = df.reset_index(drop=True)
            color_code = df.index

    # Copy index to a new column for hover data
    df['index'] = df.index

    # Create the scatter plot with Plotly
    fig = px.scatter(df, x='centroid_x', y='centroid_y', color=color_code, opacity=opacity, color_continuous_scale="Bluered",
        title=f"2D Scatter Plot of Centroid Coordinates<br><b>Trials: {str(trial_number)[1:-1]}</b>",
        # Add the index as a column
        hover_data={'index': True})  # Show index in hover data
    
    # Add target frame locations
    if target_frame:
        target_idx = df.index[df['reward_state'] & ~df['reward_state'].shift(1).fillna(False)]
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


def linear_regression_plot(df, dv, iv, dv_name:str, iv_name:str, error_type='CI'):
    """
    Plot linear regression with error bars.
    error_type: 'CI' for 95% confidence interval, 'SEM' for standard error of the mean
    """
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=df[iv], 
            y=df[dv],
            mode='markers',
            name=dv_name,
            showlegend=True,
            marker=dict(
                color='gray', #df['mouse_id'].astype('category').cat.codes,
                #colorscale='Turbo',
                symbol=df['mouse_id'].astype('category').cat.codes
            )
        )
    )

    # Fit line
    z = np.polyfit(df[iv], df[dv], 1)
    p = np.poly1d(z)
    
    # Calculate error metrics
    n = len(df)
    x_mean = df[iv].mean()
    x_sq = np.sum((df[iv] - x_mean) ** 2)
    y_hat = p(df[iv])
    std_err = np.sqrt(np.sum((df[dv] - y_hat) ** 2) / (n-2))
    
    # Calculate points for plotting
    x_new = np.linspace(df[iv].min(), df[iv].max(), 100)
    y_new = p(x_new)
    
    # Calculate error bars based on type
    if error_type == 'CI':
        from scipy import stats
        error_margin = stats.t.ppf(0.975, n-2) * std_err * np.sqrt(1/n + (x_new - x_mean)**2 / x_sq)
        error_name = '95% CI'
    else:  # SEM
        error_margin = std_err * np.sqrt(1/n + (x_new - x_mean)**2 / x_sq)
        error_name = 'SEM'
    
    # Add fitted line
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=y_new,
            mode='lines',
            name='Fitted Line',
            line=dict(color='red')
        )
    )
    
    # Add error bands
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=y_new + error_margin,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=y_new - error_margin,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            name=error_name,
            fillcolor='rgba(255,0,0,0.2)'
        )
    )

    # Update layout
    fig.update_layout(
        title=f'{dv_name} by {iv_name}',
        xaxis_title=iv_name,
        yaxis_title=dv_name,
        hovermode='x unified'
    )
    
    fig.show()


from plotly.subplots import make_subplots

def linear_regression_plot_grid(df, mouse_ids, dv_col, iv_col, error_type='SEM'):
    """
    Plot linear regressions for multiple mice in a 2x2 grid.
    
    Args:
        df: DataFrame containing the data
        mouse_ids: List of 4 mouse IDs to plot
        dv_col: Name of the dependent variable column
        iv_col: Name of the independent variable column
        error_type: Type of error bars to display ('SEM' or '95CI')
    """
    # Create 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=[f'<b>Mouse {mouse_id}</b>' for mouse_id in mouse_ids]
    )

    for idx, mouse_id in enumerate(mouse_ids):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Filter data for current mouse
        mouse_data = df[df['mouse_id'] == mouse_id].copy()
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=mouse_data[iv_col], 
                y=mouse_data[dv_col],
                mode='markers',
                name=f'Mouse {mouse_id}',
                showlegend=False,
                marker=dict(color='gray')
            ),
            row=row, col=col
        )

        # Fit line
        z = np.polyfit(mouse_data[iv_col], mouse_data[dv_col], 1)
        p = np.poly1d(z)
        
        # Calculate error metrics
        n = len(mouse_data)
        x_mean = mouse_data[iv_col].mean()
        x_sq = np.sum((mouse_data[iv_col] - x_mean) ** 2)
        y_hat = p(mouse_data[iv_col])
        std_err = np.sqrt(np.sum((mouse_data[dv_col] - y_hat) ** 2) / (n-2))
        
        # Calculate points for plotting
        x_new = np.linspace(mouse_data[iv_col].min(), mouse_data[iv_col].max(), 100)
        y_new = p(x_new)
        
        # Calculate error margins based on type
        if error_type == '95CI':
            from scipy import stats
            error_margin = stats.t.ppf(0.975, n-2) * std_err * np.sqrt(1/n + (x_new - x_mean)**2 / x_sq)
            error_name = '95% CI'
        else:  # SEM
            error_margin = std_err * np.sqrt(1/n + (x_new - x_mean)**2 / x_sq)
            error_name = 'SEM'
        
        # Add fitted line
        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_new,
                mode='lines',
                name='Fitted Line',
                line=dict(color='red'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add error bands
        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_new + error_margin,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_new - error_margin,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                name=error_name,
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=False
            ),
            row=row, col=col
        )

        # Update axes labels
        fig.update_xaxes(title_text='Session Number', row=row, col=col)
        fig.update_yaxes(title_text='Trials Completed', row=row, col=col)

    # Update layout
    fig.update_layout(
        height=1000,
        width=1600,
        showlegend=False,
        title_text=f"<b>Trials Completed vs Session Number by Mouse</b>",
        hovermode='closest'
    )
    
    fig.show()