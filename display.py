import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_enhanced_trajectory(mouse_csv):
    # Read mouse tracking data
    mouse_df = pd.read_csv(mouse_csv)
    
    # Flip y-coordinates
    mouse_df['flipped_y'] = mouse_df['y'].max() - mouse_df['y']
    
    # Create color gradient based on acceleration
    acceleration_normalized = (mouse_df['acceleration'] - mouse_df['acceleration'].min()) / \
                               (mouse_df['acceleration'].max() - mouse_df['acceleration'].min())
    
    plt.figure(figsize=(12, 8))

    plt.plot(mouse_df['x'], mouse_df['flipped_y'])
    
    # Plot trajectory with color gradient
    plt.scatter(mouse_df['x'], mouse_df['flipped_y'], 
                c=acceleration_normalized, 
                cmap='coolwarm', 
                s=20)
    
    plt.colorbar(label='Normalized Acceleration')
    plt.title('Mouse Movement Trajectory with Acceleration')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add click markers if click data provided
    clicks = mouse_df.index[mouse_df['click'] == True].tolist()

    for click in clicks:
        plt.scatter(mouse_df['x'][click], mouse_df['flipped_y'][click], 
                    color='green', 
                    marker='x', 
                    s=100, 
                    label='Clicks')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def get_latest_files(dir_name='data'):
    mouse_files = [f for f in os.listdir(dir_name) if f.startswith('tracking') and f.endswith('.csv')]
    
    if not mouse_files:
        raise FileNotFoundError("No mouse tracking CSV files found")
    
    latest_mouse = max([os.path.join(dir_name, f) for f in mouse_files], key=os.path.getctime)
    
    return latest_mouse

if __name__ == '__main__':
    mouse_csv = get_latest_files()
    plot_enhanced_trajectory(mouse_csv)