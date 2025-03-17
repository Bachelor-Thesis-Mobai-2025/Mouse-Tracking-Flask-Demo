import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import random
import traceback
import matplotlib.cm as cm

# Use TkAgg backend for compatibility
matplotlib.use('TkAgg')


def load_and_preprocess_file(file_path, normalize_time=True, normalize_xy=True):
    """
    Load a mouse tracking CSV file and preprocess it.
    """
    df = pd.read_csv(file_path)

    # Normalize time to 0-1 range if requested
    if normalize_time and 'timestamp' in df.columns:
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        if max_time > min_time:
            df['time_normalized'] = (df['timestamp'] - min_time) / (max_time - min_time)
        else:
            df['time_normalized'] = df['timestamp'] - min_time

    # Normalize x,y coordinates if requested
    if normalize_xy:
        min_x, max_x = df['x'].min(), df['x'].max()
        min_y, max_y = df['y'].min(), df['y'].max()

        if max_x > min_x:
            df['x_normalized'] = (df['x'] - min_x) / (max_x - min_x)
        else:
            df['x_normalized'] = 0

        if max_y > min_y:
            df['y_normalized'] = (df['y'] - min_y) / (max_y - min_y)
        else:
            df['y_normalized'] = 0

    # Extract truthfulness from the file path
    df['is_truthful'] = 'truthful' in file_path.lower()
    df['answer'] = 'yes' if '_yes.csv' in file_path.lower() else 'no'

    return df


def get_files_by_type(data_dir1, truthful=True, max_files=None):
    """
    Get mouse tracking files of a specified type.
    """
    subfolder = 'truthful' if truthful else 'deceptive'
    folder_path = os.path.join(data_dir1, subfolder)

    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory {folder_path} not found")

    files = glob.glob(os.path.join(folder_path, "*.csv"))

    if max_files and len(files) > max_files:
        return random.sample(files, max_files)
    return files


def create_average_trajectory(files, normalize=True, resolution=100):
    """
    Create an average trajectory from multiple files.
    """
    if not files:
        return None

    # Common time grid for interpolation
    time_grid = np.linspace(0, 1, resolution)

    # Initialize arrays for accumulating data
    x_values = np.zeros((len(files), resolution))
    y_values = np.zeros((len(files), resolution))
    velocities = np.zeros((len(files), resolution))

    # Process each file
    for i, file_path in enumerate(files):
        try:
            df = load_and_preprocess_file(file_path, normalize_time=True, normalize_xy=normalize)

            # Interpolate to common time grid
            x_values[i] = np.interp(time_grid, df['time_normalized'], df['x_normalized'] if normalize else df['x'])
            y_values[i] = np.interp(time_grid, df['time_normalized'], df['y_normalized'] if normalize else df['y'])
            velocities[i] = np.interp(time_grid, df['time_normalized'], df['velocity'])
        except Exception as e3:
            print(f"Error processing {file_path}: {e3}")

    # Calculate averages
    avg_x = np.mean(x_values, axis=0)
    avg_y = np.mean(y_values, axis=0)
    avg_velocity = np.mean(velocities, axis=0)

    # Create a DataFrame for the average trajectory
    avg_df = pd.DataFrame({
        'time_normalized': time_grid,
        'x_normalized': avg_x,
        'y_normalized': avg_y,
        'velocity': avg_velocity
    })

    return avg_df


def plot_3d_trajectory(ax, df, title, color_by='velocity'):
    """
    Plot a 3D trajectory on the given axes.
    """
    # First determine which time column to use
    time_col = 'time_normalized' if 'time_normalized' in df.columns else 'timestamp'
    if time_col not in df.columns:
        time_col = 'time'  # Fall back to 'time' if neither exists

    # Determine which coordinate columns to use
    x_col = 'x_normalized' if 'x_normalized' in df.columns else 'x'
    y_col = 'y_normalized' if 'y_normalized' in df.columns else 'y'

    # Color by the specified metric
    color_values = df[color_by] if color_by in df.columns else np.arange(len(df))
    norm = plt.Normalize(color_values.min(), color_values.max())
    colors = cm.plasma(norm(color_values))

    # Plot the 3D trajectory
    for i in range(len(df) - 1):
        ax.plot(
            df[time_col].iloc[i:i + 2],
            df[x_col].iloc[i:i + 2],
            df[y_col].iloc[i:i + 2],
            color=colors[i],
            linewidth=2
        )

    # Add start and end points
    ax.scatter(df[time_col].iloc[0], df[x_col].iloc[0], df[y_col].iloc[0],
               color='green', s=100, label='Start')
    ax.scatter(df[time_col].iloc[-1], df[x_col].iloc[-1], df[y_col].iloc[-1],
               color='red', s=100, label='End')

    # Set labels and title
    time_label = 'Normalized Time' if time_col == 'time_normalized' else 'Time'
    x_label = 'Normalized X Position' if x_col == 'x_normalized' else 'X Position'
    y_label = 'Normalized Y Position' if y_col == 'y_normalized' else 'Y Position'

    ax.set_xlabel(time_label)
    ax.set_ylabel(x_label)
    ax.set_zlabel(y_label)
    ax.set_title(title)

    return ax


def plot_2d_trajectory(ax, df, title, color_by='velocity'):
    """
    Plot a 2D trajectory on the given axes.
    """
    # Determine which coordinate columns to use
    x_col = 'x_normalized' if 'x_normalized' in df.columns else 'x'
    y_col = 'y_normalized' if 'y_normalized' in df.columns else 'y'

    # Scatter plot colored by the specified metric
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_by] if color_by in df.columns else np.arange(len(df)),
        cmap='plasma',
        s=15,
        alpha=0.7
    )

    # Plot the path line
    ax.plot(df[x_col], df[y_col], 'k-', alpha=0.3, linewidth=1)

    # Add start and end points
    ax.scatter(df[x_col].iloc[0], df[y_col].iloc[0], color='green', s=100, label='Start')
    ax.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], color='red', s=100, label='End')

    # Add direction arrows
    if len(df) >= 10:
        arrow_indices = np.linspace(0, len(df) - 2, 5, dtype=int)
        for idx in arrow_indices:
            ax.annotate('',
                        xy=(df[x_col].iloc[idx + 1], df[y_col].iloc[idx + 1]),
                        xytext=(df[x_col].iloc[idx], df[y_col].iloc[idx]),
                        arrowprops=dict(facecolor='white', edgecolor='black', width=1, headwidth=8, alpha=0.7)
                        )

    # Set labels and title
    x_label = 'Normalized X Position' if x_col == 'x_normalized' else 'X Position'
    y_label = 'Normalized Y Position' if y_col == 'y_normalized' else 'Y Position'

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=color_by.replace('_', ' ').title())

    return ax


def plot_metric_comparison(ax, truthful_df, deceptive_df, metric, title):
    """
    Plot a comparison of a specific metric between truthful and deceptive trajectories.
    """
    # Determine which time column to use
    time_col = 'time_normalized' if 'time_normalized' in truthful_df.columns else 'time'

    ax.plot(truthful_df[time_col], truthful_df[metric], 'b-', label='Truthful', linewidth=2)
    ax.plot(deceptive_df[time_col], deceptive_df[metric], 'r-', label='Deceptive', linewidth=2)

    time_label = 'Normalized Time' if time_col == 'time_normalized' else 'Time'
    ax.set_xlabel(time_label)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()

    return ax


def create_visualization(data_dir_vis):
    """
    Create comprehensive visualizations of mouse tracking data.
    """
    # Get sample files
    truthful_files = get_files_by_type(data_dir_vis, truthful=True)
    deceptive_files = get_files_by_type(data_dir_vis, truthful=False)

    print(f"Found {len(truthful_files)} truthful files and {len(deceptive_files)} deceptive files")

    if not truthful_files or not deceptive_files:
        print("Not enough data to create visualizations")
        return

    # Create average trajectories
    truthful_avg = create_average_trajectory(truthful_files, normalize=True)
    deceptive_avg = create_average_trajectory(deceptive_files, normalize=True)

    # Get one sample file of each type for 3D visualization
    truthful_sample = load_and_preprocess_file(random.choice(truthful_files), normalize_time=True, normalize_xy=True)
    deceptive_sample = load_and_preprocess_file(random.choice(deceptive_files), normalize_time=True, normalize_xy=True)

    # Create figure with subplots
    fig1 = plt.figure(figsize=(20, 15))

    # 1. 2D Average Truthful Path with velocity heatmap
    ax1 = fig1.add_subplot(231)
    plot_2d_trajectory(ax1, truthful_avg, "Average Truthful Trajectory", color_by='velocity')

    # 2. 2D Average Deceptive Path with velocity heatmap
    ax2 = fig1.add_subplot(232)
    plot_2d_trajectory(ax2, deceptive_avg, "Average Deceptive Trajectory", color_by='velocity')

    # 3. 3D Sample Truthful Path
    ax3 = fig1.add_subplot(233, projection='3d')
    plot_3d_trajectory(ax3, truthful_sample, "Sample Truthful Trajectory (3D)", color_by='velocity')

    # 4. 3D Sample Deceptive Path
    ax4 = fig1.add_subplot(234, projection='3d')
    plot_3d_trajectory(ax4, deceptive_sample, "Sample Deceptive Trajectory (3D)", color_by='velocity')

    # 5. Velocity comparison
    ax5 = fig1.add_subplot(235)
    plot_metric_comparison(ax5, truthful_avg, deceptive_avg, 'velocity', "Velocity Comparison")

    # 6. Path efficiency comparison (if available)
    try:
        # Try to get path efficiency metrics
        efficiency_data = []

        # Get path efficiency for truthful files
        for f in truthful_files[:10]:  # Limit to first 10 files
            try:
                df = load_and_preprocess_file(f)
                if 'path_efficiency' in df.columns:
                    efficiency_data.append(('Truthful', df['path_efficiency'].iloc[0]))
                elif 'decision_path_efficiency' in df.columns:
                    efficiency_data.append(('Truthful', df['decision_path_efficiency'].iloc[0]))
            except Exception as e1:
                print(f"Error getting efficiency from {f}: {e1}")

        # Get path efficiency for deceptive files
        for f in deceptive_files[:10]:  # Limit to first 10 files
            try:
                df = load_and_preprocess_file(f)
                if 'path_efficiency' in df.columns:
                    efficiency_data.append(('Deceptive', df['path_efficiency'].iloc[0]))
                elif 'decision_path_efficiency' in df.columns:
                    efficiency_data.append(('Deceptive', df['decision_path_efficiency'].iloc[0]))
            except Exception as e2:
                print(f"Error getting efficiency from {f}: {e2}")

        # If we have data, create a bar chart
        if efficiency_data:
            # Group by truthful/deceptive and calculate averages
            truthful_values = [value for label, value in efficiency_data if label == 'Truthful']
            deceptive_values = [value for label, value in efficiency_data if label == 'Deceptive']

            truthful_avg_efficiency = np.mean(truthful_values) if truthful_values else 0
            deceptive_avg_efficiency = np.mean(deceptive_values) if deceptive_values else 0

            # Create bar chart
            ax6 = fig1.add_subplot(236)
            bars = ax6.bar(['Truthful', 'Deceptive'],
                           [truthful_avg_efficiency, deceptive_avg_efficiency],
                           color=['blue', 'red'])

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            ax6.set_ylabel('Path Efficiency')
            ax6.set_title('Average Path Efficiency Comparison')
        else:
            # If no path efficiency data, show velocity variability instead
            ax6 = fig1.add_subplot(236)
            plot_metric_comparison(ax6, truthful_avg, deceptive_avg, 'velocity',
                                   "Velocity Variability Over Time")

    except Exception as e2:
        print(f"Could not create efficiency comparison: {e2}")
        # Fallback to another plot
        ax6 = fig1.add_subplot(236)
        ax6.text(0.5, 0.5, "Path efficiency data not available",
                 ha='center', va='center', transform=ax6.transAxes)

    # Add overall title
    fig1.suptitle("Mouse Tracking Analysis: Truthful vs. Deceptive Responses", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig1


if __name__ == "__main__":
    data_dir = "data"  # Change to your data directory

    try:
        fig = create_visualization(data_dir)
        plt.show()
    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
