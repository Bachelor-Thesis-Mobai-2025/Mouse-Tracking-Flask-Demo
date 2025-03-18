import glob
import os
import random
import traceback

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use TkAgg backend for compatibility
matplotlib.use('TkAgg')


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


def load_and_preprocess_file(file_path, normalize_time=True, normalize_xy=True):
    """
    Load a mouse tracking CSV file and preprocess it.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    normalize_time : bool
        Whether to normalize time to 0-1 range
    normalize_xy : bool
        Whether to normalize x,y coordinates

    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    df = pd.read_csv(file_path)

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
    df['answer'] = 'yes' if '_yes.csv' in file_path.lower() else 'no'

    return df


def get_files_by_type(data_directory, truthful=True, max_files=None):
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

    Returns:
    --------
    list
        List of file paths
    """
    subfolder = 'truthful' if truthful else 'deceptive'
    folder_path = os.path.join(data_directory, subfolder)

    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory {folder_path} not found")

    files = glob.glob(os.path.join(folder_path, "*.csv"))

    return random.sample(files, max_files) if max_files and len(files) > max_files else files


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
        Average trajectory with normalized time, x, y, and velocity
    """
    if not files:
        return None

    # Common time grid for interpolation
    time_grid = np.linspace(0, 1, resolution)

    # Initialize lists to store truncated trajectories
    truncated_trajectories = []

    # Process each file
    for file_path in files:
        try:
            # Load and preprocess the file
            df = load_and_preprocess_file(file_path, normalize_time=True, normalize_xy=normalize)

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
                velocity_col: df_truncated[velocity_col]
            }
            truncated_trajectories.append(traj)

        except Exception as processing_error:
            print(f"Error processing {file_path}: {processing_error}")
            continue

    # If no valid trajectories, return None
    if not truncated_trajectories:
        return None

    # Interpolate each trajectory to the common time grid
    x_values = np.zeros((len(truncated_trajectories), resolution))
    y_values = np.zeros((len(truncated_trajectories), resolution))
    velocities = np.zeros((len(truncated_trajectories), resolution))

    for i, traj in enumerate(truncated_trajectories):
        x_values[i] = np.interp(time_grid, traj['time_normalized'], traj['x_normalized'])
        y_values[i] = np.interp(time_grid, traj['time_normalized'], traj['y_normalized'])
        velocities[i] = np.interp(time_grid, traj['time_normalized'], traj['velocity'])

    # Create a DataFrame for the average trajectory
    return pd.DataFrame({
        'time_normalized': time_grid,
        'x_normalized': np.mean(x_values, axis=0),
        'y_normalized': np.mean(y_values, axis=0),
        'velocity': np.mean(velocities, axis=0)
    })


def create_trajectory_plots(data_directory):
    """
    Create comprehensive visualizations of mouse tracking data.

    Parameters:
    -----------
    data_directory : str
        Path to the data directory

    Returns:
    --------
    matplotlib.figure.Figure
        Generated visualization figure
    """
    # Get sample files
    truthful_files = get_files_by_type(data_directory, truthful=True)
    deceptive_files = get_files_by_type(data_directory, truthful=False)

    print(f"Found {len(truthful_files)} truthful files and {len(deceptive_files)} deceptive files")

    if not truthful_files or not deceptive_files:
        print("Not enough data to create visualizations")
        return None

    # Create average trajectories
    truthful_avg = create_average_trajectory(truthful_files)
    deceptive_avg = create_average_trajectory(deceptive_files)

    # Get one sample file of each type for 3D visualization
    truthful_sample = load_and_preprocess_file(random.choice(truthful_files))
    deceptive_sample = load_and_preprocess_file(random.choice(deceptive_files))

    # Create figure with subplots
    result_figure = plt.figure(figsize=(20, 15))

    # 1. 2D Average Truthful Path with velocity heatmap
    ax1 = result_figure.add_subplot(231)
    plot_2d_trajectory(ax1, truthful_avg, "Average Truthful Trajectory")

    # 2. 2D Average Deceptive Path with velocity heatmap
    ax2 = result_figure.add_subplot(232)
    plot_2d_trajectory(ax2, deceptive_avg, "Average Deceptive Trajectory")

    # 3. 3D Sample Truthful Path
    ax3 = result_figure.add_subplot(233, projection='3d')
    plot_3d_trajectory(ax3, truthful_sample, "Sample Truthful Trajectory (3D)")

    # 4. 3D Sample Deceptive Path
    ax4 = result_figure.add_subplot(234, projection='3d')
    plot_3d_trajectory(ax4, deceptive_sample, "Sample Deceptive Trajectory (3D)")

    # 5. Velocity comparison
    ax5 = result_figure.add_subplot(235)
    plot_metric_comparison(ax5, truthful_avg, deceptive_avg, 'velocity', "Velocity Comparison")

    # 6. Path efficiency comparison
    ax6 = result_figure.add_subplot(236)
    plot_path_efficiency_metrics(ax6, truthful_files, deceptive_files, "Path Efficiency Metrics Comparison")

    # Add overall title
    result_figure.suptitle("Mouse Tracking Analysis: Truthful vs. Deceptive Responses", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return result_figure


def plot_3d_trajectory(ax, data_frame, title, color_by='velocity'):
    """
    Plot a 3D trajectory on the given axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data_frame : pandas.DataFrame
        The dataframe containing the trajectory data
    title : str
        Title for the plot
    color_by : str
        Which metric to use for coloring the trajectory
    """
    # Determine decision point
    decision_idx = find_decision_point(data_frame)
    decision_idx = len(data_frame) - 1 if decision_idx < 0 else decision_idx

    # Color by the specified metric
    color_values = data_frame[color_by] if color_by in data_frame.columns else np.arange(len(data_frame))
    norm = plt.Normalize(color_values.min(), color_values.max())
    colors = cm.viridis(norm(color_values))

    # Plot the 3D trajectory
    for i in range(decision_idx):
        ax.plot(
            data_frame['time_normalized'].iloc[i:i + 2],
            data_frame['x_normalized'].iloc[i:i + 2],
            data_frame['y_normalized'].iloc[i:i + 2],
            color=colors[i],
            linewidth=2
        )

    # Add start and decision points
    ax.scatter(data_frame['time_normalized'].iloc[0],
               data_frame['x_normalized'].iloc[0],
               data_frame['y_normalized'].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(data_frame['time_normalized'].iloc[decision_idx],
               data_frame['x_normalized'].iloc[decision_idx],
               data_frame['y_normalized'].iloc[decision_idx],
               color='purple', s=100, label='Decision Point')

    # Add optimal path line
    ax.plot(
        [data_frame['time_normalized'].iloc[0], data_frame['time_normalized'].iloc[decision_idx]],
        [data_frame['x_normalized'].iloc[0], data_frame['x_normalized'].iloc[decision_idx]],
        [data_frame['y_normalized'].iloc[0], data_frame['y_normalized'].iloc[decision_idx]],
        'g--', linewidth=2, alpha=0.7, label='Optimal Path'
    )

    # Set labels and title
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Normalized X Position')
    ax.set_zlabel('Normalized Y Position')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_2d_trajectory(ax, data_frame, title, color_by='velocity'):
    """
    Plot a 2D trajectory on the given axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data_frame : pandas.DataFrame
        The dataframe containing the trajectory data
    title : str
        Title for the plot
    color_by : str
        Which metric to use for coloring the trajectory
    """
    # Color by the specified metric
    scatter = ax.scatter(
        data_frame['x_normalized'],
        data_frame['y_normalized'],
        c=data_frame[color_by] if color_by in data_frame.columns else np.arange(len(data_frame)),
        cmap='viridis',
        s=15,
        alpha=0.7
    )

    # Plot the actual path line
    ax.plot(data_frame['x_normalized'], data_frame['y_normalized'], 'k-', alpha=0.3, linewidth=1, label='Actual Path')

    # Plot the optimal path
    ax.plot([data_frame['x_normalized'].iloc[0], data_frame['x_normalized'].iloc[-1]],
            [data_frame['y_normalized'].iloc[0], data_frame['y_normalized'].iloc[-1]],
            'g--', linewidth=2, alpha=0.7, label='Optimal Path')

    # Add start and decision points
    ax.scatter(data_frame['x_normalized'].iloc[0], data_frame['y_normalized'].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(data_frame['x_normalized'].iloc[-1], data_frame['y_normalized'].iloc[-1],
               color='purple', s=100, label='Decision Point')

    # Set labels and title
    ax.set_xlabel('Normalized X Position')
    ax.set_ylabel('Normalized Y Position')
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=color_by.replace('_', ' ').title())

    return ax


def plot_metric_comparison(ax, truthful_data, deceptive_data, metric, title):
    """
    Plot a comparison of a specific metric between truthful and deceptive trajectories.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_data : pandas.DataFrame
        Truthful trajectory data
    deceptive_data : pandas.DataFrame
        Deceptive trajectory data
    metric : str
        Metric to compare
    title : str
        Plot title
    """
    ax.plot(truthful_data['time_normalized'], truthful_data[metric], 'b-', label='Truthful', linewidth=2)
    ax.plot(deceptive_data['time_normalized'], deceptive_data[metric], 'r-', label='Deceptive', linewidth=2)

    ax.set_xlabel('Normalized Time')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()

    return ax


def plot_path_efficiency_metrics(ax, truthful_files, deceptive_files, title="Path Efficiency Comparison"):
    """
    Create a bar chart comparing path efficiency metrics between truthful and deceptive data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_files : list
        List of truthful trajectory files
    deceptive_files : list
        List of deceptive trajectory files
    title : str, optional
        Plot title
    """
    def _extract_metrics(files):
        """
        Extract metrics from given files.

        Parameters:
        -----------
        files : list
            List of file paths

        Returns:
        --------
        dict
            Dictionary of extracted metrics
        """
        metrics = {
            'path_efficiency': [],
            'decision_path_efficiency': [],
            'final_decision_path_efficiency': [],
            'changes_of_mind': []
        }

        for file in files[:10]:  # Limit to first 10 files for speed
            try:
                df = pd.read_csv(file)
                for metric_name in metrics:
                    if metric_name in df.columns:
                        metrics[metric_name].append(df[metric_name].iloc[0])
            except Exception as processing_error:
                print(f"Error reading metrics from {file}: {processing_error}")

        return metrics

    # Extract metrics for truthful and deceptive files
    truthful_metrics = _extract_metrics(truthful_files)
    deceptive_metrics = _extract_metrics(deceptive_files)

    # Calculate averages for metrics with data
    def _calculate_averages(metrics_dict):
        """
        Calculate averages for metrics.

        Parameters:
        -----------
        metrics_dict : dict
            Dictionary of metrics

        Returns:
        --------
        dict
            Dictionary of average metrics
        """
        return {metric: np.mean(values) for metric, values in metrics_dict.items() if values}

    truthful_averages = _calculate_averages(truthful_metrics)
    deceptive_averages = _calculate_averages(deceptive_metrics)

    # Determine metrics to plot
    metrics_to_plot = [
        metric for metric in ['path_efficiency', 'decision_path_efficiency', 'final_decision_path_efficiency']
        if metric in truthful_averages and metric in deceptive_averages
    ]

    # If no metrics to plot, show message
    if not metrics_to_plot:
        ax.text(0.5, 0.5, "No path efficiency metrics available",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax

    # Plot bars
    bar_width = 0.35
    x_indices = np.arange(len(metrics_to_plot))

    truthful_bars = ax.bar(
        x_indices - bar_width/2,
        [truthful_averages[metric] for metric in metrics_to_plot],
        bar_width, label='Truthful', color='blue'
    )

    deceptive_bars = ax.bar(
        x_indices + bar_width/2,
        [deceptive_averages[metric] for metric in metrics_to_plot],
        bar_width, label='Deceptive', color='red'
    )

    # Add value labels on bars
    for bars, averages in [(truthful_bars, truthful_averages), (deceptive_bars, deceptive_averages)]:
        for i, bar in enumerate(bars):
            metric = metrics_to_plot[i]
            height = averages[metric]
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Add changes of mind annotation
    if 'changes_of_mind' in truthful_averages and 'changes_of_mind' in deceptive_averages:
        t_changes = truthful_averages['changes_of_mind']
        d_changes = deceptive_averages['changes_of_mind']
        ax.text(0.5, 0.01,
                f"Avg. Changes of Mind: Truthful={t_changes:.2f}, Deceptive={d_changes:.2f}",
                ha='center', va='bottom', transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Customize plot
    ax.set_ylabel('Efficiency Ratio')
    ax.set_title(title)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot],
                       rotation=45, ha='right')
    ax.legend()

    return ax


def main():
    """
    Main function to run the visualization.
    """
    try:
        # Default data directory
        data_directory = "data"

        # Create visualization
        visualization = create_trajectory_plots(data_directory)

        # Display the plot
        if visualization:
            plt.show()
    except Exception as main_error:
        print(f"Error creating visualization: {main_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
