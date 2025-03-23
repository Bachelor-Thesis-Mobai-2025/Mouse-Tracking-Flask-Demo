import glob
import json
import os
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Use TkAgg backend for compatibility
matplotlib.use('TkAgg')


# ========================
# DATA LOADING FUNCTIONS
# ========================

def _get_column_names(df, use_normalized=True):
    """
    Get appropriate column names for x, y, and velocity.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    use_normalized : bool
        Whether to use normalized columns

    Returns:
    --------
    tuple : (x_column, y_column, velocity_column)
    """
    if use_normalized:
        return (
            'x_normalized' if 'x_normalized' in df.columns else 'x',
            'y_normalized' if 'y_normalized' in df.columns else 'y',
            'velocity'
        )
    return 'x', 'y', 'velocity'


def load_and_preprocess_json_file(file_path, normalize_time=True, normalize_xy=True):
    """
    Load a mouse tracking JSON file and preprocess it.

    Parameters:
    -----------
    file_path : str
        Path to the JSON file
    normalize_time : bool
        Whether to normalize time to 0-1 range
    normalize_xy : bool
        Whether to normalize x,y coordinates

    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    dict
        Trajectory metrics
    """
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract trajectory and metrics
    trajectory = data.get('trajectory', [])
    trajectory_metrics = data.get('trajectory_metrics', {})

    # Create DataFrame from trajectory
    df = pd.DataFrame(trajectory)

    # Normalize time to 0-1 range if requested
    if normalize_time and 'timestamp' in df.columns:
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        df['time_normalized'] = (df['timestamp'] - min_time) / (max_time - min_time) if max_time > min_time else 0

    # Normalize x,y coordinates if requested
    if normalize_xy:
        for coord in ['x', 'y']:
            min_val = df[coord].min()
            max_val = df[coord].max()
            col_name = f'{coord}_normalized'
            df[col_name] = (df[coord] - min_val) / (max_val - min_val) if max_val > min_val else 0

    # Extract truthfulness from the file path
    df['is_truthful'] = 'truthful' in file_path.lower()
    df['answer'] = 'yes' if '_yes.' in file_path.lower() else 'no'

    # Add pause detection - a pause is when dx=0 and dy=0 (no movement)
    df['is_paused'] = (df['dx'] == 0) & (df['dy'] == 0)

    # Add a column to indicate when a pause starts (transition from movement to pause)
    shifted_paused = df['is_paused'].shift(1)
    if shifted_paused.isna().any():
        shifted_paused = shifted_paused.astype(bool)  # Ensure consistent type
    df['pause_start'] = df['is_paused'] & ~shifted_paused

    # Add a column to indicate when a pause ends (transition from pause to movement)
    df['pause_end'] = ~df['is_paused'] & shifted_paused

    return df, trajectory_metrics


def get_files_by_type(data_directory, truthful=True, max_files=None, file_ext='.json'):
    """
    Get mouse tracking files of a specified type.

    Parameters:
    -----------
    data_directory : str
        Path to the data directory
    truthful : bool
        Whether to get truthful or deceptive files
    max_files : int, optional
        Maximum number of files to return
    file_ext : str
        File extension to look for

    Returns:
    --------
    list
        List of file paths
    """
    subfolder = 'truthful' if truthful else 'deceptive'
    folder_path = os.path.join(data_directory, subfolder)

    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory {folder_path} not found")

    files = glob.glob(os.path.join(folder_path, f"*{file_ext}"))

    return files if max_files is None else files[:min(max_files, len(files))]


def find_decision_point(df):
    """
    Find the decision point in the trajectory (last click).

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    int
        Index of decision point, or -1 if not found
    """
    if 'click' in df.columns:
        click_indices = df.index[df['click'] == 1].tolist()
        return click_indices[-1] if click_indices else -1
    return -1


def create_average_trajectory(files, normalize=True, resolution=100):
    """
    Create an average trajectory from multiple files, truncating at decision points.

    Parameters:
    -----------
    files : list
        List of file paths to process
    normalize : bool
        Whether to normalize x,y coordinates
    resolution : int
        Number of points to interpolate to

    Returns:
    --------
    pandas.DataFrame
        Average trajectory with normalized time, x, y, velocity, and acceleration
    dict
        Average metrics
    """
    if not files:
        return None, {}

    # Common time grid for interpolation
    time_grid = np.linspace(0, 1, resolution)

    # Initialize lists to store truncated trajectories and metrics
    truncated_trajectories = []
    all_metrics = []

    # Process each file
    for file_path in files:
        try:
            # Load and preprocess the file
            df, metrics = load_and_preprocess_json_file(file_path, normalize_time=True, normalize_xy=normalize)
            all_metrics.append(metrics)

            # Find and truncate at decision point
            decision_idx = find_decision_point(df)
            decision_idx = len(df) - 1 if decision_idx < 0 else decision_idx

            # Truncate the dataframe
            df_truncated = df.iloc[:decision_idx+1].copy()

            # Re-normalize time to 0-1 after truncation
            df_truncated['time_normalized'] = (df_truncated['timestamp'] - df_truncated['timestamp'].min()) / \
                                              (df_truncated['timestamp'].max() - df_truncated['timestamp'].min())

            # Get column names
            x_col, y_col, velocity_col = _get_column_names(df_truncated)

            # Prepare trajectory data
            traj = {
                'time_normalized': df_truncated['time_normalized'],
                x_col: df_truncated[x_col],
                y_col: df_truncated[y_col],
                velocity_col: df_truncated[velocity_col],
                'acceleration': df_truncated['acceleration'],
                'curvature': df_truncated['curvature'],
                'jerk': df_truncated['jerk'],
                'is_paused': df_truncated['is_paused']
            }
            truncated_trajectories.append(traj)

        except Exception as processing_error:
            print(f"Error processing {file_path}: {processing_error}")
            continue

    # If no valid trajectories, return None
    if not truncated_trajectories:
        return None, {}

    # Interpolate each trajectory to the common time grid
    x_values = np.zeros((len(truncated_trajectories), resolution))
    y_values = np.zeros((len(truncated_trajectories), resolution))
    velocities = np.zeros((len(truncated_trajectories), resolution))
    accelerations = np.zeros((len(truncated_trajectories), resolution))
    curvatures = np.zeros((len(truncated_trajectories), resolution))
    jerks = np.zeros((len(truncated_trajectories), resolution))
    is_paused = np.zeros((len(truncated_trajectories), resolution))

    for i, traj in enumerate(truncated_trajectories):
        x_values[i] = np.interp(time_grid, traj['time_normalized'], traj['x_normalized'])
        y_values[i] = np.interp(time_grid, traj['time_normalized'], traj['y_normalized'])
        velocities[i] = np.interp(time_grid, traj['time_normalized'], traj['velocity'])
        accelerations[i] = np.interp(time_grid, traj['time_normalized'], traj['acceleration'])
        curvatures[i] = np.interp(time_grid, traj['time_normalized'], traj['curvature'])
        jerks[i] = np.interp(time_grid, traj['time_normalized'], traj['jerk'])
        is_paused[i] = np.interp(time_grid, traj['time_normalized'], traj['is_paused'].astype(float))

    # Calculate min and max for each metric
    min_velocities = np.min(velocities, axis=0)
    max_velocities = np.max(velocities, axis=0)
    min_accelerations = np.min(accelerations, axis=0)
    max_accelerations = np.max(accelerations, axis=0)
    min_curvatures = np.min(curvatures, axis=0)
    max_curvatures = np.max(curvatures, axis=0)
    min_jerks = np.min(jerks, axis=0)
    max_jerks = np.max(jerks, axis=0)

    # Compute average metrics from all_metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m.get(key, 0) for m in all_metrics])

    # Create a DataFrame for the average trajectory with additional columns for min/max
    return pd.DataFrame({
        'time_normalized': time_grid,
        'x_normalized': np.mean(x_values, axis=0),
        'y_normalized': np.mean(y_values, axis=0),
        'velocity': np.mean(velocities, axis=0),
        'velocity_min': min_velocities,
        'velocity_max': max_velocities,
        'acceleration': np.mean(accelerations, axis=0),
        'acceleration_min': min_accelerations,
        'acceleration_max': max_accelerations,
        'curvature': np.mean(curvatures, axis=0),
        'curvature_min': min_curvatures,
        'curvature_max': max_curvatures,
        'jerk': np.mean(jerks, axis=0),
        'jerk_min': min_jerks,
        'jerk_max': max_jerks,
        'is_paused': np.mean(is_paused, axis=0) > 0.5
    }), avg_metrics


def extract_pauses(files):
    """
    Extract pause information from multiple files.

    Parameters:
    -----------
    files : list
        List of file paths to process

    Returns:
    --------
    list of dict
        List of pause information dictionaries
    """
    all_pauses = []

    for file_path in files:
        try:
            # Load and preprocess the file
            df, _ = load_and_preprocess_json_file(file_path, normalize_time=True, normalize_xy=True)

            # Find continuous pauses
            is_paused = df['is_paused'].astype(int)
            pause_groups = (is_paused.diff() != 0).cumsum()

            for group_id, group_df in df.groupby(pause_groups):
                if group_df['is_paused'].iloc[0]:
                    # This is a pause group
                    start_time = group_df['time_normalized'].iloc[0]
                    end_time = group_df['time_normalized'].iloc[-1]
                    duration = end_time - start_time

                    # Only consider pauses with duration > 0
                    if duration > 0:
                        pause_info = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'file_path': file_path,
                            'is_truthful': 'truthful' in file_path.lower()
                        }
                        all_pauses.append(pause_info)

        except Exception as processing_error:
            print(f"Error extracting pauses from {file_path}: {processing_error}")

    return all_pauses


# ========================
# VISUALIZATION FUNCTIONS
# ========================

def plot_2d_trajectory(ax, data, metrics, is_truthful=True, color_metric='velocity'):
    """
    Plot a single 2D trajectory with optimal path indicator and colored by a metric.
    Fixed version to ensure proper coloring with extreme value ranges.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pandas.DataFrame
        Trajectory data
    metrics : dict
        Trajectory metrics
    is_truthful : bool
        Whether the data is truthful (True) or deceptive (False)
    color_metric : str
        Metric to use for coloring the trajectory (e.g., 'velocity', 'curvature')
    """
    # Set variables based on data type
    base_color = 'blue' if is_truthful else 'red'
    label_prefix = 'Truthful' if is_truthful else 'Deceptive'
    title_prefix = 'Truthful' if is_truthful else 'Deceptive'

    # Check if selected metric exists
    if color_metric in data.columns:
        # Make a copy to avoid modifying the original data
        plot_data = data.copy()

        # Check for outliers - values that are extremely high compared to most
        metric_values = plot_data[color_metric].values

        # Winsorize extreme values to improve coloring
        # This caps very high values to something more reasonable for visualization
        p99 = np.percentile(metric_values, 99)  # 99th percentile as upper bound
        if metric_values.max() > p99 > 0:
            plot_data[color_metric] = np.minimum(plot_data[color_metric], p99)

        # Apply log scale for better visualization if range is very large
        max_val = plot_data[color_metric].max()
        min_val = plot_data[color_metric].min()

        if max_val > min_val * 1000 and min_val >= 0:  # Very large range
            # Add small epsilon to avoid log(0)
            epsilon = 1e-6
            plot_data[color_metric] = np.log1p(plot_data[color_metric] + epsilon)

        # Create a colored line segment plot
        points = np.array([plot_data['x_normalized'], plot_data['y_normalized']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with the specified colormap
        from matplotlib.collections import LineCollection
        # Manually set min and max values for better coloring
        min_color = plot_data[color_metric].min()
        max_color = plot_data[color_metric].max()

        # Ensure some difference between min and max
        if min_color == max_color:
            max_color = min_color + 1.0

        norm = plt.Normalize(min_color, max_color)
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=3, alpha=0.8)

        # Use the values to color
        lc.set_array(plot_data[color_metric][:-1])

        # Add the colored line segments to the plot
        line = ax.add_collection(lc)

        # Add a colorbar
        label_suffix = " (log scale)" if max_val > min_val * 1000 and min_val >= 0 else ""
        plt.colorbar(line, ax=ax, label=color_metric.replace('_', ' ').title() + label_suffix)

    else:
        # Fallback to solid color line if metric not available
        ax.plot(data['x_normalized'], data['y_normalized'],
                color=base_color, alpha=0.7, linewidth=2, label=f'{label_prefix} Path')

    # Plot the optimal path
    ax.plot([data['x_normalized'].iloc[0], data['x_normalized'].iloc[-1]],
            [data['y_normalized'].iloc[0], data['y_normalized'].iloc[-1]],
            'g--', linewidth=2, alpha=0.7, label='Optimal Path')

    # Add start and end points
    ax.scatter(data['x_normalized'].iloc[0], data['y_normalized'].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(data['x_normalized'].iloc[-1], data['y_normalized'].iloc[-1],
               color=base_color, s=100, label='End')

    # Add a text box with key metrics
    metrics_text = (
        f"Avg time to answer: {metrics.get('total_time', 'N/A'):.2f}s\n"
        f"Avg time to first move: {metrics.get('time_to_first_movement', 'N/A'):.2f}s\n"
        f"Avg hesitation time: {metrics.get('hesitation_time', 'N/A'):.2f}s\n"
        f"Avg hover time: {metrics.get('hover_time', 'N/A'):.2f}s"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set labels and title
    ax.set_xlabel('Normalized X Position')
    ax.set_ylabel('Normalized Y Position')
    ax.set_title(f'Average {title_prefix} 2D Trajectory (Colored by {color_metric.replace("_", " ").title()})')

    # Add legend
    ax.legend(loc='lower right')

    return ax


# ========================
# PATH EFFICIENCY FUNCTIONS
# ========================

def plot_path_efficiency_metrics(ax, truthful_metrics, deceptive_metrics):
    """
    Create a bar chart comparing path efficiency metrics between truthful and deceptive trajectories.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_metrics : dict
        Dictionary of truthful trajectory metrics
    deceptive_metrics : dict
        Dictionary of deceptive trajectory metrics
    """
    # Define metrics to plot
    metrics_to_plot = [
        'decision_path_efficiency',
        'final_decision_path_efficiency',
        'hesitation_time',
        'time_to_first_movement',
        'total_pause_time'
    ]

    # Filter to metrics that actually exist in the data
    metrics_to_plot = [m for m in metrics_to_plot if m in truthful_metrics and m in deceptive_metrics]

    # If no metrics to plot, show message
    if not metrics_to_plot:
        ax.text(0.5, 0.5, "No path efficiency metrics available",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Path Efficiency Metrics Comparison")
        return ax

    # Set up bar positions
    bar_width = 0.35
    x_indices = np.arange(len(metrics_to_plot))

    # Create bars for truthful metrics
    truthful_values = [truthful_metrics.get(metric, 0) for metric in metrics_to_plot]
    deceptive_values = [deceptive_metrics.get(metric, 0) for metric in metrics_to_plot]

    # Plot bars
    ax.bar(x_indices - bar_width/2, truthful_values, width=bar_width,
           color='blue', alpha=0.7, label='Truthful')
    ax.bar(x_indices + bar_width/2, deceptive_values, width=bar_width,
           color='red', alpha=0.7, label='Deceptive')

    # Add text labels
    for i, (t_val, d_val) in enumerate(zip(truthful_values, deceptive_values)):
        ax.text(x_indices[i] - bar_width/2, t_val + 0.02, f"{t_val:.2f}",
                ha='center', va='bottom', fontsize=8)
        ax.text(x_indices[i] + bar_width/2, d_val + 0.02, f"{d_val:.2f}",
                ha='center', va='bottom', fontsize=8)

    # Add additional metrics as annotations if available
    annotation = []

    if 'hesitation_count' in truthful_metrics and 'hesitation_count' in deceptive_metrics:
        annotation.append(f"Hesitation Count: Truthful={truthful_metrics['hesitation_count']:.2f}, "
                          f"Deceptive={deceptive_metrics['hesitation_count']:.2f}")

    if 'direction_changes' in truthful_metrics and 'direction_changes' in deceptive_metrics:
        annotation.append(f"Direction Changes: Truthful={truthful_metrics['direction_changes']:.2f}, "
                          f"Deceptive={deceptive_metrics['direction_changes']:.2f}")

    if annotation:
        ax.text(0.5, 0.01, '\n'.join(annotation), ha='center', va='bottom', transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Customize plot
    ax.set_ylabel('Metric Value')
    ax.set_title("Path Efficiency Metrics Comparison")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot],
                       rotation=45, ha='right')
    ax.legend()

    return ax


# ========================
# PAUSE ANALYSIS FUNCTIONS
# ========================

def plot_pause_analysis(ax, truthful_pauses, deceptive_pauses):
    """
    Plot pause length over normalized time.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_pauses : list
        List of truthful pause information dictionaries
    deceptive_pauses : list
        List of deceptive pause information dictionaries
    """
    # Extract truthful pause data
    truthful_times = [p['start_time'] for p in truthful_pauses]
    truthful_durations = [p['duration'] for p in truthful_pauses]

    # Extract deceptive pause data
    deceptive_times = [p['start_time'] for p in deceptive_pauses]
    deceptive_durations = [p['duration'] for p in deceptive_pauses]

    # Plot pause data
    ax.scatter(truthful_times, truthful_durations, color='blue', alpha=0.7, label='Truthful')
    ax.scatter(deceptive_times, deceptive_durations, color='red', alpha=0.7, label='Deceptive')

    # Calculate averages
    truthful_avg_duration = np.mean(truthful_durations) if truthful_durations else 0
    deceptive_avg_duration = np.mean(deceptive_durations) if deceptive_durations else 0

    truthful_min_duration = min(truthful_durations) if truthful_durations else 0
    truthful_max_duration = max(truthful_durations) if truthful_durations else 0

    deceptive_min_duration = min(deceptive_durations) if deceptive_durations else 0
    deceptive_max_duration = max(deceptive_durations) if deceptive_durations else 0

    # Add horizontal lines for average pause durations
    ax.axhline(y=truthful_avg_duration, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=deceptive_avg_duration, color='red', linestyle='--', alpha=0.5)

    # Add text annotations for statistics
    stats_text = (
        f"Truthful: {len(truthful_pauses)} pauses\n"
        f"Avg: {truthful_avg_duration:.3f}, Min: {truthful_min_duration:.3f}, Max: {truthful_max_duration:.3f}\n\n"
        f"Deceptive: {len(deceptive_pauses)} pauses\n"
        f"Avg: {deceptive_avg_duration:.3f}, Min: {deceptive_min_duration:.3f}, Max: {deceptive_max_duration:.3f}"
    )

    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Pause Duration')
    ax.set_title('Pause Analysis: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_pause_regression(ax, truthful_pauses, deceptive_pauses):
    """
    Plot polynomial regression for pause data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_pauses : list
        List of truthful pause information dictionaries
    deceptive_pauses : list
        List of deceptive pause information dictionaries
    """
    # Extract truthful pause data
    truthful_times = np.array([p['start_time'] for p in truthful_pauses]).reshape(-1, 1)
    truthful_durations = np.array([p['duration'] for p in truthful_pauses])

    # Extract deceptive pause data
    deceptive_times = np.array([p['start_time'] for p in deceptive_pauses]).reshape(-1, 1)
    deceptive_durations = np.array([p['duration'] for p in deceptive_pauses])

    # Plot raw data points
    ax.scatter(truthful_times, truthful_durations, color='blue', alpha=0.7, label='Truthful Data')
    ax.scatter(deceptive_times, deceptive_durations, color='red', alpha=0.7, label='Deceptive Data')

    # Perform polynomial regression if we have enough data points
    if len(truthful_times) > 3:
        truthful_model = make_pipeline(PolynomialFeatures(3), LinearRegression(), memory=None)
        truthful_model.fit(truthful_times, truthful_durations)

        # Generate prediction range
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        truthful_pred = truthful_model.predict(x_range)

        # Plot regression line
        ax.plot(x_range, truthful_pred, 'b-', linewidth=2, label='Truthful Trend')

    if len(deceptive_times) > 3:
        deceptive_model = make_pipeline(PolynomialFeatures(3), LinearRegression(), memory=None)
        deceptive_model.fit(deceptive_times, deceptive_durations)

        # Generate prediction range
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        deceptive_pred = deceptive_model.predict(x_range)

        # Plot regression line
        ax.plot(x_range, deceptive_pred, 'r-', linewidth=2, label='Deceptive Trend')

    # Calculate and display statistics
    truthful_avg = np.mean(truthful_durations) if len(truthful_durations) > 0 else 0
    deceptive_avg = np.mean(deceptive_durations) if len(deceptive_durations) > 0 else 0

    # Add text annotations for statistics
    stats_text = (
        f"Truthful: {len(truthful_pauses)} pauses, Avg Duration: {truthful_avg:.3f}\n"
        f"Deceptive: {len(deceptive_pauses)} pauses, Avg Duration: {deceptive_avg:.3f}"
    )

    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Pause Duration')
    ax.set_title('Pause Regression Analysis: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_acceleration_ranges(ax, truthful_data, deceptive_data):
    """
    Plot acceleration min/max ranges.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Plot min/max ranges
    ax.fill_between(time,
                    truthful_data['acceleration_min'],
                    truthful_data['acceleration_max'],
                    color='blue', alpha=0.3, label='Truthful Range')

    ax.fill_between(time,
                    deceptive_data['acceleration_min'],
                    deceptive_data['acceleration_max'],
                    color='red', alpha=0.3, label='Deceptive Range')

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Acceleration')
    ax.set_title('Acceleration Ranges: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_acceleration_averages(ax, truthful_data, deceptive_data):
    """
    Plot only the average accelerations without ranges.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Plot only average accelerations
    ax.plot(time, truthful_data['acceleration'], 'b-', label='Truthful Avg', linewidth=2)
    ax.plot(time, deceptive_data['acceleration'], 'r-', label='Deceptive Avg', linewidth=2)

    # Calculate difference between accelerations
    accel_diff = np.abs(truthful_data['acceleration'] - deceptive_data['acceleration'])

    # Find and highlight significant difference points
    threshold = accel_diff.mean() + accel_diff.std()
    significant_points = time[accel_diff > threshold]

    # Highlight regions of significant difference
    for point in significant_points:
        ax.axvline(x=point, color='grey', linestyle='--', alpha=0.2)

    # Add annotation for significant differences
    if len(significant_points) > 0:
        ax.text(0.05, 0.95, f'Significant differences: {len(significant_points)} points',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Acceleration')
    ax.set_title('Average Accelerations: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_acceleration_distribution(ax, truthful_data, deceptive_data):
    """
    Plot acceleration distribution comparison.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    # Get acceleration data
    truthful_accel = truthful_data['acceleration']
    deceptive_accel = deceptive_data['acceleration']

    # Create density plots
    try:
        truthful_density = gaussian_kde(truthful_accel)
        deceptive_density = gaussian_kde(deceptive_accel)

        # Create x range for plotting
        x_range = np.linspace(min(truthful_accel.min(), deceptive_accel.min()),
                              max(truthful_accel.max(), deceptive_accel.max()),
                              1000)

        # Plot densities
        ax.plot(x_range, truthful_density(x_range), 'b-', linewidth=2, label='Truthful')
        ax.plot(x_range, deceptive_density(x_range), 'r-', linewidth=2, label='Deceptive')
    except np.linalg.LinAlgError:
        # Fallback to histograms if KDE fails
        ax.hist(truthful_accel, bins=30, alpha=0.5, color='blue', density=True, label='Truthful')
        ax.hist(deceptive_accel, bins=30, alpha=0.5, color='red', density=True, label='Deceptive')

    # Add mean lines
    ax.axvline(truthful_accel.mean(), color='blue', linestyle='--',
               label=f'Truthful Mean: {truthful_accel.mean():.2f}')
    ax.axvline(deceptive_accel.mean(), color='red', linestyle='--',
               label=f'Deceptive Mean: {deceptive_accel.mean():.2f}')

    # T-test for statistical comparison
    _, p_value = stats.ttest_ind(truthful_accel, deceptive_accel)
    significance = "Significant" if p_value < 0.05 else "Not Significant"

    # Add t-test result annotation
    ax.text(0.5, 0.95, f"T-test: p={p_value:.4f} ({significance})",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Acceleration')
    ax.set_ylabel('Density')
    ax.set_title('Acceleration Distribution: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_3d_average_trajectory(ax, data, metrics, is_truthful=True, color_metric='velocity', z_metric=None):
    """
    Plot a single 3D trajectory with optimal path indicator and custom metrics.
    Fixed version to ensure proper coloring with extreme value ranges.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on (must be 3D)
    data : pandas.DataFrame
        Trajectory data
    metrics : dict
        Trajectory metrics
    is_truthful : bool
        Whether the data is truthful (True) or deceptive (False)
    color_metric : str
        Metric to use for coloring the trajectory
    z_metric : str, optional
        If provided, use this metric for the z-axis instead of y_normalized
    """
    # Set color based on data type
    base_color = 'blue' if is_truthful else 'red'
    title_prefix = 'Truthful' if is_truthful else 'Deceptive'

    # Determine z-axis values
    if z_metric and z_metric in data.columns:
        z_values = data[z_metric]
        z_label = z_metric.replace('_', ' ').title()
    else:
        z_values = data['y_normalized']
        z_label = 'Normalized Y Position'
        z_metric = 'y_normalized'  # For title consistency

    # Create a colormap based on the selected metric
    if color_metric in data.columns:
        # Make a copy to avoid modifying the original data
        plot_data = data.copy()

        # Check for outliers - values that are extremely high compared to most
        metric_values = plot_data[color_metric].values

        # Winsorize extreme values to improve coloring
        # This caps very high values to something more reasonable for visualization
        p99 = np.percentile(metric_values, 99)  # 99th percentile as upper bound
        if metric_values.max() > p99 > 0:
            plot_data[color_metric] = np.minimum(plot_data[color_metric], p99)

        # Apply log scale for better visualization if range is very large
        max_val = plot_data[color_metric].max()
        min_val = plot_data[color_metric].min()

        if max_val > min_val * 1000 and min_val >= 0:  # Very large range
            # Add small epsilon to avoid log(0)
            epsilon = 1e-6
            plot_data[color_metric] = np.log1p(plot_data[color_metric] + epsilon)

        # Ensure some difference between min and max
        min_color = plot_data[color_metric].min()
        max_color = plot_data[color_metric].max()
        if min_color == max_color:
            max_color = min_color + 1.0

        # Create the normalization and colormap
        norm = plt.Normalize(min_color, max_color)
        colors = plt.cm.viridis(norm(plot_data[color_metric]))

        # Plot the 3D trajectory with metric-based coloring
        for i in range(len(plot_data) - 1):
            ax.plot(
                plot_data['time_normalized'].iloc[i:i+2],
                plot_data['x_normalized'].iloc[i:i+2],
                z_values.iloc[i:i+2],
                color=colors[i],
                linewidth=2
            )

        # Create a scalar mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])

        # Add a colorbar with appropriate label
        label_suffix = " (log scale)" if max_val > min_val * 1000 and min_val >= 0 else ""
        plt.colorbar(sm, ax=ax, label=color_metric.replace('_', ' ').title() + label_suffix)
    else:
        # Fallback to solid color if metric not available
        ax.plot(
            data['time_normalized'],
            data['x_normalized'],
            z_values,
            color=base_color,
            linewidth=2,
            label=title_prefix
        )

    # Add start and end points
    ax.scatter(data['time_normalized'].iloc[0],
               data['x_normalized'].iloc[0],
               z_values.iloc[0],
               color='green', s=100, label='Start')

    ax.scatter(data['time_normalized'].iloc[-1],
               data['x_normalized'].iloc[-1],
               z_values.iloc[-1],
               color=base_color, s=100, label='End')

    # Add optimal path line
    ax.plot(
        [data['time_normalized'].iloc[0], data['time_normalized'].iloc[-1]],
        [data['x_normalized'].iloc[0], data['x_normalized'].iloc[-1]],
        [z_values.iloc[0], z_values.iloc[-1]],
        'g--', linewidth=2, alpha=0.7, label='Optimal Path'
    )

    # Add a text box with key metrics
    metrics_text = (
        f"Avg time to answer: {metrics.get('total_time', 'N/A'):.2f}s\n"
        f"Avg time to first move: {metrics.get('time_to_first_movement', 'N/A'):.2f}s\n"
        f"Avg hesitation time: {metrics.get('hesitation_time', 'N/A'):.2f}s\n"
        f"Avg hover time: {metrics.get('hover_time', 'N/A'):.2f}s"
    )
    ax.text2D(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set labels and title
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Normalized X Position')
    ax.set_zlabel(z_label)

    # Adjust title based on metrics used
    if z_metric == 'y_normalized':
        ax.set_title(f'Average {title_prefix} 3D Trajectory (Colored by {color_metric.replace("_", " ").title()})')
    else:
        ax.set_title(f'Average {title_prefix} 3D Trajectory\n(Z: {z_label},'
                     f' Color: {color_metric.replace("_", " ").title()})')

    ax.legend()

    return ax


# ========================
# VELOCITY PLOT FUNCTIONS
# ========================

def plot_velocity_ranges(ax, truthful_data, deceptive_data):
    """
    Plot velocity min/max ranges without average lines.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Plot min/max ranges only (without average lines)
    ax.fill_between(time,
                    truthful_data['velocity_min'],
                    truthful_data['velocity_max'],
                    color='blue', alpha=0.3, label='Truthful Range')

    ax.fill_between(time,
                    deceptive_data['velocity_min'],
                    deceptive_data['velocity_max'],
                    color='red', alpha=0.3, label='Deceptive Range')

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity Ranges: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_velocity_averages(ax, truthful_data, deceptive_data):
    """
    Plot only the average velocities without ranges.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Plot only average velocities
    ax.plot(time, truthful_data['velocity'], 'b-', label='Truthful Avg', linewidth=2)
    ax.plot(time, deceptive_data['velocity'], 'r-', label='Deceptive Avg', linewidth=2)

    # Calculate difference between velocities
    velocity_diff = np.abs(truthful_data['velocity'] - deceptive_data['velocity'])

    # Find and highlight significant difference points
    threshold = velocity_diff.mean() + velocity_diff.std()
    significant_points = time[velocity_diff > threshold]

    # Highlight regions of significant difference
    for point in significant_points:
        ax.axvline(x=point, color='grey', linestyle='--', alpha=0.2)

    # Add annotation for significant differences
    if len(significant_points) > 0:
        ax.text(0.05, 0.95, f'Significant differences: {len(significant_points)} points',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Velocity')
    ax.set_title('Average Velocities: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_normalized_velocity_comparison(ax, truthful_data, deceptive_data):
    """
    Plot normalized velocity comparison for clearer visualization.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Normalize velocities to 0-1 range for each dataset
    def normalize_series(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val) if max_val > min_val else series

    truthful_velocity_norm = normalize_series(truthful_data['velocity'])
    deceptive_velocity_norm = normalize_series(deceptive_data['velocity'])

    # Plot normalized velocities
    ax.plot(time, truthful_velocity_norm, 'b-', label='Truthful', linewidth=2)
    ax.plot(time, deceptive_velocity_norm, 'r-', label='Deceptive', linewidth=2)

    # Add vertical lines at significant difference points
    diff = np.abs(truthful_velocity_norm - deceptive_velocity_norm)
    significant_points = time[diff > diff.mean() + diff.std()]

    for point in significant_points:
        ax.axvline(x=point, color='grey', linestyle='--', alpha=0.3)

    # Add annotation for significant differences
    if len(significant_points) > 0:
        ax.text(0.05, 0.95, f'Significant differences: {len(significant_points)} points',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Normalized Velocity')
    ax.set_title('Normalized Velocity Comparison: Truthful vs Deceptive')
    ax.legend()

    return ax


def plot_velocity_distribution(ax, truthful_data, deceptive_data):
    """
    Plot velocity distribution comparison.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    # Get velocity data
    truthful_vel = truthful_data['velocity']
    deceptive_vel = deceptive_data['velocity']

    # Create density plots
    try:
        truthful_density = gaussian_kde(truthful_vel)
        deceptive_density = gaussian_kde(deceptive_vel)

        # Create x range for plotting
        x_range = np.linspace(min(truthful_vel.min(), deceptive_vel.min()),
                              max(truthful_vel.max(), deceptive_vel.max()),
                              1000)

        # Plot densities
        ax.plot(x_range, truthful_density(x_range), 'b-', linewidth=2, label='Truthful')
        ax.plot(x_range, deceptive_density(x_range), 'r-', linewidth=2, label='Deceptive')
    except np.linalg.LinAlgError:
        # Fallback to histograms if KDE fails
        ax.hist(truthful_vel, bins=30, alpha=0.5, color='blue', density=True, label='Truthful')
        ax.hist(deceptive_vel, bins=30, alpha=0.5, color='red', density=True, label='Deceptive')

    # Add mean lines
    ax.axvline(truthful_vel.mean(), color='blue', linestyle='--',
               label=f'Truthful Mean: {truthful_vel.mean():.2f}')
    ax.axvline(deceptive_vel.mean(), color='red', linestyle='--',
               label=f'Deceptive Mean: {deceptive_vel.mean():.2f}')

    # T-test for statistical comparison
    _, p_value = stats.ttest_ind(truthful_vel, deceptive_vel)
    significance = "Significant" if p_value < 0.05 else "Not Significant"

    # Add t-test result annotation
    ax.text(0.5, 0.95, f"T-test: p={p_value:.4f} ({significance})",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Velocity')
    ax.set_ylabel('Density')
    ax.set_title('Velocity Distribution: Truthful vs Deceptive')
    ax.legend()

    return ax


# ========================
# ACCELERATION PLOT FUNCTIONS
# ========================

def plot_acceleration_comparison(ax, truthful_data, deceptive_data):
    """
    Plot acceleration comparison between truthful and deceptive data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    """
    time = truthful_data['time_normalized']

    # Plot average accelerations
    ax.plot(time, truthful_data['acceleration'], 'b-', label='Truthful', linewidth=2)
    ax.plot(time, deceptive_data['acceleration'], 'r-', label='Deceptive', linewidth=2)

    # Calculate difference between accelerations
    accel_diff = np.abs(truthful_data['acceleration'] - deceptive_data['acceleration'])

    # Find and highlight significant difference points
    threshold = accel_diff.mean() + accel_diff.std()
    significant_points = time[accel_diff > threshold]

    for point in significant_points:
        ax.axvline(x=point, color='grey', linestyle='--', alpha=0.3)

    # Add annotation for significant differences
    if len(significant_points) > 0:
        ax.text(0.05, 0.95, f'Significant differences: {len(significant_points)} points',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Acceleration')
    ax.set_title('Average Acceleration Comparison: Truthful vs Deceptive')
    ax.legend()

    return ax


# ========================
# MAIN ORCHESTRATION FUNCTIONS
# ========================

def create_trajectory_plots(data_directory, save_directory=None, file_ext='.json'):
    """
    Create comprehensive visualizations of mouse tracking data and save individual plots.

    Parameters:
    -----------
    data_directory : str
        Path to the data directory
    save_directory : str, optional
        Directory to save individual plots. If None, plots are not saved.
    file_ext : str
        File extension to look for

    Returns:
    --------
    None
    """
    # Get all files
    truthful_files = get_files_by_type(data_directory, truthful=True, file_ext=file_ext)
    deceptive_files = get_files_by_type(data_directory, truthful=False, file_ext=file_ext)

    print(f"Found {len(truthful_files)} truthful files and {len(deceptive_files)} deceptive files")

    if not truthful_files or not deceptive_files:
        print("Not enough data to create visualizations")
        return None

    # Create average trajectories
    truthful_avg, truthful_metrics = create_average_trajectory(truthful_files)
    deceptive_avg, deceptive_metrics = create_average_trajectory(deceptive_files)

    # Extract pause information
    truthful_pauses = extract_pauses(truthful_files)
    deceptive_pauses = extract_pauses(deceptive_files)

    # Create directory for saving plots if it doesn't exist
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Base plot functions
    plot_functions = [
        # 2D Trajectories with velocity coloring
        (plot_2d_trajectory, [truthful_avg, truthful_metrics, True, 'velocity'],
         "2D_Truthful_Trajectory_Velocity"),
        (plot_2d_trajectory, [deceptive_avg, deceptive_metrics, False, 'velocity'],
         "2D_Deceptive_Trajectory_Velocity"),

        # 3D Trajectories with standard configuration
        (plot_3d_average_trajectory, [truthful_avg, truthful_metrics, True, 'velocity', None],
         "3D_Truthful_Trajectory"),
        (plot_3d_average_trajectory, [deceptive_avg, deceptive_metrics, False, 'velocity', None],
         "3D_Deceptive_Trajectory"),

        # Velocity visualizations
        (plot_normalized_velocity_comparison, [truthful_avg, deceptive_avg], "Normalized_Velocity_Comparison"),
        (plot_velocity_ranges, [truthful_avg, deceptive_avg], "Velocity_Ranges"),
        (plot_velocity_averages, [truthful_avg, deceptive_avg], "Velocity_Averages"),
        (plot_velocity_distribution, [truthful_avg, deceptive_avg], "Velocity_Distribution"),

        # Acceleration visualizations
        (plot_acceleration_comparison, [truthful_avg, deceptive_avg], "Acceleration_Comparison"),
        (plot_acceleration_ranges, [truthful_avg, deceptive_avg], "Acceleration_Ranges"),
        (plot_acceleration_averages, [truthful_avg, deceptive_avg], "Acceleration_Averages"),
        (plot_acceleration_distribution, [truthful_avg, deceptive_avg], "Acceleration_Distribution"),

        # Pause analysis
        (plot_pause_analysis, [truthful_pauses, deceptive_pauses], "Pause_Analysis"),
        (plot_pause_regression, [truthful_pauses, deceptive_pauses], "Pause_Regression"),

        # Path efficiency metrics
        (plot_path_efficiency_metrics, [truthful_metrics, deceptive_metrics], "Path_Efficiency_Metrics")
    ]

    # Create each plot
    for plot_func, args, filename in plot_functions:
        try:
            fig = plt.figure(figsize=(10, 8))
            # Use 3D projection only for 3D trajectory plots
            if '3D' in filename:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            plot_func(ax, *args)
            plt.tight_layout()

            if save_directory:
                plt.savefig(os.path.join(save_directory, f"{filename}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
        except Exception as plot_error:
            print(f"Error creating {filename} plot: {plot_error}")
            traceback.print_exc()


def main():
    """
    Main function to run the visualization.
    """
    try:
        # Default data directory
        data_directory = "data"

        # Create output directory for individual plots
        graphs_directory = "graphs"

        # File extension to look for
        file_ext = '.json'

        # Create visualizations and save as individual files
        create_trajectory_plots(data_directory, graphs_directory, file_ext)

        print(f"Visualizations saved to {graphs_directory}")
    except Exception as main_error:
        print(f"Error creating visualization: {main_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
