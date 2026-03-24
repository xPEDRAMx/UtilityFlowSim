import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
import os

# --- USER-CONFIGURABLE PARAMETERS ---
NETWORK_LENGTH_X_METERS = 101.0
NETWORK_LENGTH_Y_METERS = 188.0

class InteractiveSyncedPlot:
    """
    Manages synced 2D and 3D scatter plots.
    Right-click removes points near the click location.
    """
    def __init__(self, fig, ax_2d, ax_3d, time_data, density_data, flow_data, color, cmap, norm):
        self.fig = fig
        self.ax_2d = ax_2d
        self.ax_3d = ax_3d
        
        self.time_data = list(time_data)
        self.density_data = list(density_data)
        self.flow_data = list(flow_data)
        
        self.scatter_2d = ax_2d.scatter(self.density_data, self.flow_data, alpha=0.5, edgecolors='k', s=50, c=color)
        self.scatter_3d = ax_3d.scatter3D(self.time_data, self.density_data, self.flow_data,
                                          c=self.time_data, cmap=cmap, norm=norm, s=50, edgecolors='k')
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax_2d:
            return

        if event.button == 3:  # Right-click to remove a point
            if not self.density_data:
                return
            
            distances = np.hypot(np.array(self.density_data) - event.xdata,
                                 np.array(self.flow_data) - event.ydata)
            idx = np.argmin(distances)
            
            # Remove point only if close enough (within 5% of x-axis width)
            threshold = (self.ax_2d.get_xlim()[1] - self.ax_2d.get_xlim()[0]) * 0.05
            if distances[idx] < threshold:
                removed = (self.time_data.pop(idx), self.density_data.pop(idx), self.flow_data.pop(idx))
                print(f"Removed: (Time={removed[0]:.2f}, Density={removed[1]:.2f}, Flow={removed[2]:.2f})")

            self.update_plots()

    def update_plots(self):
        """Refresh 2D and 3D scatter plots."""
        self.scatter_2d.set_offsets(np.c_[self.density_data, self.flow_data])
        self.scatter_3d._offsets3d = (np.array(self.time_data), np.array(self.density_data), np.array(self.flow_data))
        self.scatter_3d.set_array(self.time_data)
        self.fig.canvas.draw_idle()

    def get_modified_data(self):
        """Return modified data as a DataFrame."""
        return pd.DataFrame({
            'time_window': self.time_data,
            'density': self.density_data,
            'flow': self.flow_data
        })

def run_interactive_plotting_session(file_path, time_window_seconds=1.0):
    METER_TO_MILE = 1 / 1609.344
    MS_TO_MPH = 2.23693629
    length_x_miles = NETWORK_LENGTH_X_METERS * METER_TO_MILE
    length_y_miles = NETWORK_LENGTH_Y_METERS * METER_TO_MILE
    area_miles2 = length_x_miles * length_y_miles

    print(f"Aggregating over {time_window_seconds}s window. Network lengths X: {NETWORK_LENGTH_X_METERS}m, Y: {NETWORK_LENGTH_Y_METERS}m.")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    ground_agents = ["vehicle", "pedestrian", "bike"]
    air_agents = ["drone"]
    df_xy = df[df['agent_type'].isin(ground_agents)].copy()
    df_z = df[df['agent_type'].isin(air_agents)].copy()
    
    final_data_xy = pd.DataFrame()
    if not df_xy.empty:
        df_xy['dt'] = df_xy.groupby('id')['time'].diff()
        df_xy['vx'] = (df_xy.groupby('id')['smoothed_x'].diff() / df_xy['dt']).abs()
        df_xy['vy'] = (df_xy.groupby('id')['smoothed_y'].diff() / df_xy['dt']).abs()
        fine_res_xy = df_xy.groupby('time').agg(density=('id', 'count'), avg_speed_x=('vx', 'mean'), avg_speed_y=('vy', 'mean')).reset_index()
        fine_res_xy['time_window'] = np.floor(fine_res_xy['time'] / time_window_seconds)
        final_data_xy = fine_res_xy.groupby('time_window').agg(density=('density', 'mean'), avg_speed_x=('avg_speed_x', 'mean'), avg_speed_y=('avg_speed_y', 'mean')).reset_index()
        final_data_xy['density_x_apm'] = final_data_xy['density'] / length_x_miles
        final_data_xy['density_y_apm'] = final_data_xy['density'] / length_y_miles
        final_data_xy['avg_speed_x_mph'] = final_data_xy['avg_speed_x'] * MS_TO_MPH
        final_data_xy['avg_speed_y_mph'] = final_data_xy['avg_speed_y'] * MS_TO_MPH
        final_data_xy['flow_x_aph'] = final_data_xy['density_x_apm'] * final_data_xy['avg_speed_x_mph']
        final_data_xy['flow_y_aph'] = final_data_xy['density_y_apm'] * final_data_xy['avg_speed_y_mph']

    final_data_z = pd.DataFrame()
    if not df_z.empty:
        df_z['dt'] = df_z.groupby('id')['time'].diff()
        df_z['vz'] = (df_z.groupby('id')['smoothed_z'].diff() / df_z['dt']).abs()
        fine_res_z = df_z.groupby('time').agg(density=('id', 'count'), avg_speed_z=('vz', 'mean')).reset_index()
        fine_res_z['time_window'] = np.floor(fine_res_z['time'] / time_window_seconds)
        final_data_z = fine_res_z.groupby('time_window').agg(density=('density', 'mean'), avg_speed_z=('avg_speed_z', 'mean')).reset_index()
        final_data_z['density_z_apm2'] = final_data_z['density'] / area_miles2
        final_data_z['avg_speed_z_mph'] = final_data_z['avg_speed_z'] * MS_TO_MPH
        final_data_z['flow_z_amph'] = final_data_z['density_z_apm2'] * final_data_z['avg_speed_z_mph']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle('Live Interactive Editor (Top Row) with Synced 3D View (Bottom Row)', fontsize=18)
    
    ax1_2d = fig.add_subplot(2, 3, 1)
    ax2_2d = fig.add_subplot(2, 3, 2)
    ax3_2d = fig.add_subplot(2, 3, 3)
    ax1_3d = fig.add_subplot(2, 3, 4, projection='3d')
    ax2_3d = fig.add_subplot(2, 3, 5, projection='3d')
    ax3_3d = fig.add_subplot(2, 3, 6, projection='3d')
    
    min_time = min(
        final_data_xy['time_window'].min() if not final_data_xy.empty else 0,
        final_data_z['time_window'].min() if not final_data_z.empty else 0
    )
    max_time = max(
        final_data_xy['time_window'].max() if not final_data_xy.empty else 0,
        final_data_z['time_window'].max() if not final_data_z.empty else 0
    )
    norm = mcolors.Normalize(vmin=min_time, vmax=max_time if max_time > min_time else min_time + 1)
    cmap = 'viridis'

    plotter_x = InteractiveSyncedPlot(fig, ax1_2d, ax1_3d,
                                     final_data_xy['time_window'], final_data_xy['density_x_apm'], final_data_xy['flow_x_aph'],
                                     'blue', cmap, norm)
    plotter_y = InteractiveSyncedPlot(fig, ax2_2d, ax2_3d,
                                     final_data_xy['time_window'], final_data_xy['density_y_apm'], final_data_xy['flow_y_aph'],
                                     'green', cmap, norm)
    plotter_z = InteractiveSyncedPlot(fig, ax3_2d, ax3_3d,
                                     final_data_z['time_window'], final_data_z['density_z_apm2'], final_data_z['flow_z_amph'],
                                     'red', cmap, norm)
    
    ax1_2d.set_title("X-Dim"); ax1_2d.set_xlabel("Density [agents/mile]"); ax1_2d.set_ylabel("Flow [agents/hour]")
    ax2_2d.set_title("Y-Dim"); ax2_2d.set_xlabel("Density [agents/mile]"); ax2_2d.set_ylabel("Flow [agents/hour]")
    ax3_2d.set_title("Z-Dim"); ax3_2d.set_xlabel("Density [agents/mile²]"); ax3_2d.set_ylabel("Flow [agents/(mile·hour)]")
    
    ax1_3d.set_title("3D View (X-Dim)"); ax1_3d.set_xlabel("Time (s)"); ax1_3d.set_ylabel("Density [agents/mile]"); ax1_3d.set_zlabel("Flow [agents/hour]", labelpad=10)
    ax2_3d.set_title("3D View (Y-Dim)"); ax2_3d.set_xlabel("Time (s)"); ax2_3d.set_ylabel("Density [agents/mile]"); ax2_3d.set_zlabel("Flow [agents/hour]", labelpad=10)
    ax3_3d.set_title("3D View (Z-Dim)"); ax3_3d.set_xlabel("Time (s)"); ax3_3d.set_ylabel("Density [agents/mile²]"); ax3_3d.set_zlabel("Flow [agents/(mile·hour)]", labelpad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("\nDisplaying interactive plot window. CLOSE the window to save the final data and plot image.")
    plt.show()  # pauses script until window closed

    # Save modified data after interaction
    print("\nInteractive session finished. Retrieving and saving modified data.")
    modified_x = plotter_x.get_modified_data().rename(columns={'density': 'density_x_apm', 'flow': 'flow_x_aph'})
    modified_y = plotter_y.get_modified_data().rename(columns={'density': 'density_y_apm', 'flow': 'flow_y_aph'})
    modified_z = plotter_z.get_modified_data().rename(columns={'density': 'density_z_apm2', 'flow': 'flow_z_amph'})
    
    merged_df = pd.merge(modified_x, modified_y, on='time_window', how='outer')
    final_merged_df = pd.merge(merged_df, modified_z, on='time_window', how='outer').sort_values(by='time_window').fillna(0)
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    try:
        output_csv_path = os.path.join(desktop_path, "final_edited_flow_density_data.csv")
        final_merged_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully saved modified data to:\n{output_csv_path}")
    except Exception as e:
        print(f"\nError saving CSV file: {e}")

    try:
        output_image_path = os.path.join(desktop_path, "final_interactive_plot_view.png")
        fig.savefig(output_image_path, dpi=300)
        print(f"Successfully saved final plot view to:\n{output_image_path}")
    except Exception as e:
        print(f"\nError saving plot image: {e}")

if __name__ == '__main__':
    csv_file_path = '/Users/pedrambeigi/Desktop/2025/ISTTT/Codes/simulation_code_v14 (Fixed Drone Stations)/intersection_plots (2)/full_simulation_trajectories.csv'
    aggregation_window = 1.0 
    run_interactive_plotting_session(csv_file_path, time_window_seconds=aggregation_window)
