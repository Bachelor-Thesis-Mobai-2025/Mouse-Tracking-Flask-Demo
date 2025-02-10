import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')


def slideshow_bot_trajectories_last_10(dir_name='data/bot'):
    """
    Displays each of the last 10 CSV plots in sequence (by file creation time),
    letting you press Enter to move to the next.
    """
    # Find all matching CSV files
    all_files = [
        f for f in os.listdir(dir_name)
        if f.startswith('tracking') and f.endswith('.csv')
    ]
    if not all_files:
        raise FileNotFoundError(f"No mouse tracking CSV files found in '{dir_name}'")

    # Sort by creation time (oldest first)
    all_files.sort(key=lambda f: os.path.getctime(os.path.join(dir_name, f)))

    # Take only the last 10 (most recent)
    mouse_files = all_files[:]

    _, ax = plt.subplots(figsize=(12, 8))

    for i, file in enumerate(mouse_files, start=1):
        csv_path = os.path.join(dir_name, file)

        # Read the CSV
        df = pd.read_csv(csv_path)

        # Convert 'click' to boolean if it's 0/1
        if 'click' in df.columns and df['click'].dtype != bool:
            df['click'] = df['click'] == 1

        # Optional: Flip the y-coordinates
        df['flipped_y'] = df['y'].max() - df['y']

        # Clear the axes for the new plot
        ax.clear()

        # Plot the trajectory
        ax.plot(df['x'], df['flipped_y'], color='blue', alpha=0.7)
        ax.set_title(f"File: {file} ({i}/{len(mouse_files)})")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate (Flipped)")

        # Mark clicks if present
        clicks_df = df[df['click'] == True]
        if not clicks_df.empty:
            ax.scatter(
                clicks_df['x'],
                clicks_df['flipped_y'],
                color='green',
                marker='x',
                s=100,
                label='Clicks'
            )
            ax.legend()

        plt.pause(1)
        plt.plot()


if __name__ == '__main__':
    # slideshow_bot_trajectories_last_10('data/human')
    slideshow_bot_trajectories_last_10('data/bot')
