import glob
import os
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

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

    # Calculate min and max for each metric
    min_velocities = np.min(velocities, axis=0)
    max_velocities = np.max(velocities, axis=0)

    # Create a DataFrame for the average trajectory with additional columns for min/max
    return pd.DataFrame({
        'time_normalized': time_grid,
        'x_normalized': np.mean(x_values, axis=0),
        'y_normalized': np.mean(y_values, axis=0),
        'velocity': np.mean(velocities, axis=0),
        'velocity_min': min_velocities,
        'velocity_max': max_velocities
    })


def create_trajectory_plots(data_directory, save_directory=None):
    """
    Create comprehensive visualizations of mouse tracking data and save individual plots.

    Parameters:
    -----------
    data_directory : str
        Path to the data directory
    save_directory : str, optional
        Directory to save individual plots. If None, plots are not saved.

    Returns:
    --------
    None
    """
    # Get all files
    truthful_files = get_files_by_type(data_directory, truthful=True)
    deceptive_files = get_files_by_type(data_directory, truthful=False)

    print(f"Found {len(truthful_files)} truthful files and {len(deceptive_files)} deceptive files")

    if not truthful_files or not deceptive_files:
        print("Not enough data to create visualizations")
        return None

    # Create average trajectories
    truthful_avg = create_average_trajectory(truthful_files)
    deceptive_avg = create_average_trajectory(deceptive_files)

    # List of metrics that might be available for coloring
    possible_metrics = ['velocity', 'velocity_variability', 'curvature']

    # Filter to metrics that actually exist in the data
    available_metrics = [m for m in possible_metrics if m in truthful_avg.columns]

    # Create directory for saving plots if it doesn't exist
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Base plot functions (standard metrics)
    plot_functions = [
        # 2D Trajectories with velocity coloring
        (plot_2d_trajectory, [truthful_avg, True, 'velocity'], "2D_Truthful_Trajectory_Velocity"),
        (plot_2d_trajectory, [deceptive_avg, False, 'velocity'], "2D_Deceptive_Trajectory_Velocity"),

        # 3D Trajectories with standard configuration
        (plot_3d_average_trajectory, [truthful_avg, True, 'velocity', None], "3D_Truthful_Trajectory"),
        (plot_3d_average_trajectory, [deceptive_avg, False, 'velocity', None], "3D_Deceptive_Trajectory"),

        # Velocity visualizations
        (plot_normalized_velocity_comparison, [truthful_avg, deceptive_avg], "Normalized_Velocity_Comparison"),
        (plot_velocity_ranges, [truthful_avg, deceptive_avg], "Velocity_Ranges"),
        (plot_velocity_averages, [truthful_avg, deceptive_avg], "Velocity_Averages"),
        (plot_velocity_distribution, [truthful_avg, deceptive_avg], "Velocity_Distribution"),

        # Path efficiency metrics
        (plot_path_efficiency_metrics, [truthful_files, deceptive_files], "Path_Efficiency_Metrics")
    ]

    # Add additional plots for available alternative metrics
    for metric in available_metrics:
        if metric != 'velocity':  # Skip velocity as it's already included
            # Add 2D plots with this metric for coloring
            plot_functions.append(
                (plot_2d_trajectory, [truthful_avg, True, metric], f"2D_Truthful_Trajectory_{metric}")
            )
            plot_functions.append(
                (plot_2d_trajectory, [deceptive_avg, False, metric], f"2D_Deceptive_Trajectory_{metric}")
            )

            # Add 3D plots with this metric for coloring
            plot_functions.append(
                (plot_3d_average_trajectory, [truthful_avg, True, metric, None], f"3D_Truthful_Trajectory_{metric}")
            )
            plot_functions.append(
                (plot_3d_average_trajectory, [deceptive_avg, False, metric, None], f"3D_Deceptive_Trajectory_{metric}")
            )

            # Add 3D plots with alternative z-axis
            plot_functions.append(
                (plot_3d_average_trajectory, [truthful_avg, True, 'velocity', metric], f"3D_Truthful_Z_{metric}")
            )
            plot_functions.append(
                (plot_3d_average_trajectory, [deceptive_avg, False, 'velocity', metric], f"3D_Deceptive_Z_{metric}")
            )

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


def plot_2d_trajectory(ax, data, is_truthful=True, color_metric='velocity'):
    """
    Plot a single 2D trajectory with optimal path indicator and colored by a metric.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pandas.DataFrame
        Trajectory data
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
        # Create a colored line segment plot
        points = np.array([data['x_normalized'], data['y_normalized']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with the specified colormap
        from matplotlib.collections import LineCollection
        norm = plt.Normalize(data[color_metric].min(), data[color_metric].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=3, alpha=0.8)
        lc.set_array(data[color_metric][:-1])  # Set the colors
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax, label=color_metric.replace('_', ' ').title())
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

    # Set labels and title
    ax.set_xlabel('Normalized X Position')
    ax.set_ylabel('Normalized Y Position')
    ax.set_title(f'Average {title_prefix} 2D Trajectory (Colored by {color_metric.replace("_", " ").title()})')

    # Add legend
    ax.legend(loc='best')

    return ax


def plot_3d_average_trajectory(ax, data, is_truthful=True, color_metric='velocity', z_metric=None):
    """
    Plot a single 3D trajectory with optimal path indicator and custom metrics.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on (must be 3D)
    data : pandas.DataFrame
        Trajectory data
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
        norm = plt.Normalize(data[color_metric].min(), data[color_metric].max())
        colors = plt.cm.viridis(norm(data[color_metric]))

        # Plot the 3D trajectory with metric-based coloring
        for i in range(len(data) - 1):
            ax.plot(
                data['time_normalized'].iloc[i:i+2],
                data['x_normalized'].iloc[i:i+2],
                z_values.iloc[i:i+2],
                color=colors[i],
                linewidth=2
            )

        # Create a scalar mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_metric.replace('_', ' ').title())
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
    truthful_density = gaussian_kde(truthful_vel)
    deceptive_density = gaussian_kde(deceptive_vel)

    # Create x range for plotting
    x_range = np.linspace(min(truthful_vel.min(), deceptive_vel.min()),
                          max(truthful_vel.max(), deceptive_vel.max()),
                          1000)

    # Plot densities
    ax.plot(x_range, truthful_density(x_range), 'b-', linewidth=2, label='Truthful')
    ax.plot(x_range, deceptive_density(x_range), 'r-', linewidth=2, label='Deceptive')

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


def plot_path_efficiency_metrics(ax, truthful_files, deceptive_files):
    """
    Create a bar chart comparing path efficiency metrics with overlapping min, max, and average bars.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    truthful_files : list
        List of truthful trajectory files
    deceptive_files : list
        List of deceptive trajectory files
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

        for file in files:
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

    # Calculate statistics for metrics with data
    def _calculate_stats(metrics_dict):
        """
        Calculate min, max, and average for metrics.

        Parameters:
        -----------
        metrics_dict : dict
            Dictionary of metrics

        Returns:
        --------
        dict
            Dictionary of metric statistics
        """
        stats_dict = {}
        for metric, values in metrics_dict.items():
            if values:
                stats_dict[metric] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'avg': np.mean(values)
                }
        return stats_dict

    truthful_stats = _calculate_stats(truthful_metrics)
    deceptive_stats = _calculate_stats(deceptive_metrics)

    # Determine metrics to plot (exclude changes_of_mind as it will be handled separately)
    metrics_to_plot = [
        metric for metric in ['path_efficiency', 'decision_path_efficiency', 'final_decision_path_efficiency']
        if metric in truthful_stats and metric in deceptive_stats
    ]

    # If no metrics to plot, show message
    if not metrics_to_plot:
        ax.text(0.5, 0.5, "No path efficiency metrics available",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Path Efficiency Metrics Comparison")
        return ax

    # Set up bar positions
    bar_width = 0.35
    x_indices = np.arange(len(metrics_to_plot))

    # Define colors for different statistics
    colors = {
        'truthful': {
            'min': 'lightskyblue',
            'avg': 'royalblue',
            'max': 'darkblue'
        },
        'deceptive': {
            'min': 'lightcoral',
            'avg': 'firebrick',
            'max': 'darkred'
        }
    }

    # Create overlapping bars for truthful data
    for i, metric in enumerate(metrics_to_plot):
        # Truthful data - left side
        path_eff_stats = truthful_stats[metric]

        # Plot min, avg, max bars with descending width
        ax.bar(x_indices[i] - bar_width / 2, path_eff_stats['min'], width=bar_width,
               color=colors['truthful']['min'], alpha=0.7, label='Truthful Min' if i == 0 else "")

        ax.bar(x_indices[i] - bar_width / 2, path_eff_stats['avg'], width=bar_width * 0.8,
               color=colors['truthful']['avg'], alpha=0.8, label='Truthful Avg' if i == 0 else "")

        ax.bar(x_indices[i] - bar_width / 2, path_eff_stats['max'], width=bar_width * 0.6,
               color=colors['truthful']['max'], alpha=0.9, label='Truthful Max' if i == 0 else "")

        # Add text labels
        ax.text(x_indices[i] - bar_width / 2, path_eff_stats['min'] - 0.03,
                f"{path_eff_stats['min']:.2f}", ha='center', va='top', fontsize=8, rotation=90)

        ax.text(x_indices[i] - bar_width / 2, path_eff_stats['avg'] + 0.01,
                f"{path_eff_stats['avg']:.2f}", ha='center', va='bottom', fontsize=8, color='white', weight='bold')

        ax.text(x_indices[i] - bar_width / 2, path_eff_stats['max'] + 0.03,
                f"{path_eff_stats['max']:.2f}", ha='center', va='bottom', fontsize=8)

        # Deceptive data - right side
        path_eff_stats = deceptive_stats[metric]

        # Plot min, avg, max bars with descending width
        ax.bar(x_indices[i] + bar_width / 2, path_eff_stats['min'], width=bar_width,
               color=colors['deceptive']['min'], alpha=0.7, label='Deceptive Min' if i == 0 else "")

        ax.bar(x_indices[i] + bar_width / 2, path_eff_stats['avg'], width=bar_width * 0.8,
               color=colors['deceptive']['avg'], alpha=0.8, label='Deceptive Avg' if i == 0 else "")

        ax.bar(x_indices[i] + bar_width / 2, path_eff_stats['max'], width=bar_width * 0.6,
               color=colors['deceptive']['max'], alpha=0.9, label='Deceptive Max' if i == 0 else "")

        # Add text labels
        ax.text(x_indices[i] + bar_width / 2, path_eff_stats['min'] - 0.03,
                f"{path_eff_stats['min']:.2f}", ha='center', va='top', fontsize=8, rotation=90)

        ax.text(x_indices[i] + bar_width / 2, path_eff_stats['avg'] + 0.01,
                f"{path_eff_stats['avg']:.2f}", ha='center', va='bottom', fontsize=8, color='white', weight='bold')

        ax.text(x_indices[i] + bar_width / 2, path_eff_stats['max'] + 0.03,
                f"{path_eff_stats['max']:.2f}", ha='center', va='bottom', fontsize=8)

    # Add changes of mind annotation
    if 'changes_of_mind' in truthful_stats and 'changes_of_mind' in deceptive_stats:
        t_changes_avg = truthful_stats['changes_of_mind']['avg']
        t_changes_max = truthful_stats['changes_of_mind']['max']
        d_changes_avg = deceptive_stats['changes_of_mind']['avg']
        d_changes_max = deceptive_stats['changes_of_mind']['max']
        ax.text(0.5, 0.01,
                f"Avg. Changes of Mind: Truthful={t_changes_avg:.2f}, Deceptive={d_changes_avg:.2f}\n"
                f"Max. Changes of Mind: Truthful={t_changes_max:.2f}, Deceptive={d_changes_max:.2f}",
                ha='center', va='bottom', transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Customize plot
    ax.set_ylabel('Efficiency Ratio')
    ax.set_title("Path Efficiency Metrics Comparison")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot],
                       rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=2)

    return ax


def main():
    """
    Main function to run the visualization.
    """
    try:
        # Default data directory
        data_directory = "data"

        # Create output directory for individual plots
        graphs_directory = "graphs"

        # Create visualizations and save as individual files
        create_trajectory_plots(data_directory, graphs_directory)

        print(f"Visualizations saved to {graphs_directory}")
    except Exception as main_error:
        print(f"Error creating visualization: {main_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
