# visualization.py

import os
import matplotlib
backend = os.environ.get("MPLBACKEND") or "TkAgg"
try:
    matplotlib.use(backend)
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from environment import plot_environment
import utils

# (All previous functions like update_live_plot, plot_ground_trajectories, etc., remain here)
# ...

def update_live_plot(ax, current_agents, config, current_time):
    ax.clear()
    plot_environment(config, ax=ax)
    
    for agent_obj in current_agents:
        if agent_obj.pos is None: continue
        
        pos = agent_obj.pos
        heading = agent_obj.heading_angle_rad
        
        patch = None
        if agent_obj.agent_type == 'vehicle':
            patch = Polygon(utils.get_agent_corners(pos, heading, agent_obj.length, agent_obj.width), 
                            closed=True, facecolor='royalblue', edgecolor='black', zorder=10)
        elif agent_obj.agent_type == 'bike':
            patch = Polygon(utils.get_agent_corners(pos, heading, agent_obj.length, agent_obj.width),
                            closed=True, facecolor='limegreen', edgecolor='black', zorder=11)
        elif agent_obj.agent_type == 'pedestrian':
            patch = Circle((pos[0], pos[1]), radius=config['pedestrian_radius'], 
                           facecolor='magenta', edgecolor='black', zorder=12)
        elif agent_obj.agent_type == 'drone':
            scale = max(0.3, 1 - 0.05 * pos[2])
            patch = Circle((pos[0], pos[1]), radius=config['drone_radius'] * 2 * scale,
                           facecolor='darkorange', edgecolor='black', zorder=14)
            ax.text(pos[0], pos[1] + 1, f"z:{pos[2]:.1f}", fontsize=8, ha='center', color='white',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'), zorder=20)

        if patch:
            ax.add_patch(patch)
            
    ax.set_title(f"Simulation Time: {current_time:.1f}s | Agents: {len(current_agents)}")

def plot_ground_trajectories(simulated_agents, config):
    print("\n--- Plotting ground agent trajectories ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_environment(config, ax=ax, show_street_names=False)
    
    vehicle_color, pedestrian_color, bike_color = 'deepskyblue', 'fuchsia', 'lime'
    
    for agent in simulated_agents:
        if agent.agent_type not in ['vehicle', 'pedestrian', 'bike'] or not agent.history_pos:
            continue
        
        x_coords = [pos[0] for pos in agent.history_pos]
        y_coords = [pos[1] for pos in agent.history_pos]
        
        color = vehicle_color
        if agent.agent_type == 'pedestrian':
            color = pedestrian_color
        elif agent.agent_type == 'bike':
            color = bike_color
        
        ax.plot(x_coords, y_coords, color=color, linewidth=1.5, alpha=0.8, zorder=8)
        
    ax.set_title("Simulation Layout")

    ax.set_title("Ground Agent Trajectories (Vehicles, Pedestrians, and Bikes)")
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=vehicle_color, lw=2, label='Vehicle Trajectory'),
                       Line2D([0], [0], color=pedestrian_color, lw=2, label='Pedestrian Trajectory'),
                       Line2D([0], [0], color=bike_color, lw=2, label='Bike Trajectory')]
    ax.legend(handles=legend_elements, loc='upper right')

    filepath = os.path.join(config.get("animation_output_path", "."), "ground_trajectories.png")
    plt.savefig(filepath)
    print(f"Saved ground trajectory plot to {filepath}")
    plt.close(fig)

def plot_ground_trajectories_with_ids(simulated_agents, config):
    print("\n--- Plotting ground agent trajectories with IDs ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_environment(config, ax=ax, show_street_names=False)
    
    vehicle_color, pedestrian_color, bike_color = 'deepskyblue', 'fuchsia', 'lime'
    
    for agent in simulated_agents:
        if agent.agent_type not in ['vehicle', 'pedestrian', 'bike'] or not agent.history_pos:
            continue
        
        x_coords = [pos[0] for pos in agent.history_pos]
        y_coords = [pos[1] for pos in agent.history_pos]
        
        color = vehicle_color
        if agent.agent_type == 'pedestrian':
            color = pedestrian_color
        elif agent.agent_type == 'bike':
            color = bike_color
        
        ax.plot(x_coords, y_coords, color=color, linewidth=1.5, alpha=0.7, zorder=8)
        
        start_pos = agent.history_pos[0]
        ax.text(start_pos[0], start_pos[1], str(agent.id), color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=15,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.2'))

    ax.set_title("Ground Agent Trajectories with Start-Point IDs")
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=vehicle_color, lw=2, label='Vehicle Trajectory'),
                       Line2D([0], [0], color=pedestrian_color, lw=2, label='Pedestrian Trajectory'),
                       Line2D([0], [0], color=bike_color, lw=2, label='Bike Trajectory')]
    ax.legend(handles=legend_elements, loc='upper right')

    filepath = os.path.join(config.get("animation_output_path", "."), "ground_trajectories_with_ids.png")
    plt.savefig(filepath)
    print(f"Saved ground trajectory plot with IDs to {filepath}")
    plt.close(fig)


def plot_drone_trajectories(simulated_agents, config):
    print("\n--- Plotting drone trajectories with altitude coloring ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_environment(config, ax=ax, show_street_names=False)
    
    drones = [agent for agent in simulated_agents if agent.agent_type == 'drone' and len(agent.history_pos) > 1]
    if not drones:
        print("No drone trajectories to plot.")
        plt.close(fig)
        return

    all_z = np.concatenate([np.array(d.history_pos)[:, 2] for d in drones])
    min_alt, max_alt = (all_z.min(), all_z.max()) if all_z.size > 0 else (0, 1)
    
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min_alt, vmax=max_alt)

    for drone in drones:
        points = np.array(drone.history_pos)[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        z_coords = np.array(drone.history_pos)[:, 2]
        segment_colors = (z_coords[:-1] + z_coords[1:]) / 2
        
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_colors)
        lc.set_linewidth(2)
        lc.set_alpha(0.8)
        lc.set_zorder(10)
        ax.add_collection(lc)

    ax.set_title("Drone Trajectories (Color Mapped to Altitude)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Altitude (m)')

    filepath = os.path.join(config.get("animation_output_path", "."), "drone_trajectories_altitude.png")
    plt.savefig(filepath)
    print(f"Saved drone trajectory plot to {filepath}")
    plt.close(fig)

def plot_smoothed_ground_trajectories(simulated_agents, smoothed_trajectories, config):
    print("\n--- Plotting Smoothed vs. Raw Ground Trajectories ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_environment(config, ax=ax, show_street_names=False)
    
    colors = {'vehicle': 'deepskyblue', 'pedestrian': 'fuchsia', 'bike': 'lime'}

    for agent in simulated_agents:
        if agent.agent_type not in ['vehicle', 'pedestrian', 'bike']:
            continue
        if not agent.history_pos or agent.id not in smoothed_trajectories:
            continue
        
        color = colors.get(agent.agent_type, 'gray')
        
        raw_x = [pos[0] for pos in agent.history_pos]
        raw_y = [pos[1] for pos in agent.history_pos]
        ax.plot(raw_x, raw_y, color=color, linestyle='--', linewidth=1.0, alpha=0.6, zorder=8)
        
        smooth_path = smoothed_trajectories[agent.id]
        smooth_x = [pos[0] for pos in smooth_path]
        smooth_y = [pos[1] for pos in smooth_path]
        ax.plot(smooth_x, smooth_y, color=color, linestyle='-', linewidth=2.0, alpha=1.0, zorder=9)

    ax.set_title("Smoothed Ground Trajectories (Solid) vs. Raw (Dashed)")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='deepskyblue', lw=2, label='Smoothed Vehicle Path'),
        Line2D([0], [0], color='fuchsia', lw=2, label='Smoothed Pedestrian Path'),
        Line2D([0], [0], color='lime', lw=2, label='Smoothed Bike Path'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Raw Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    filepath = os.path.join(config.get("animation_output_path", "."), "smoothed_ground_trajectories.png")
    plt.savefig(filepath)
    print(f"Saved smoothed ground trajectory plot to {filepath}")
    plt.close(fig)

def plot_smoothed_drone_trajectories(simulated_agents, smoothed_trajectories, config):
    print("\n--- Plotting Smoothed vs. Raw Drone Trajectories ---")
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_environment(config, ax=ax, show_street_names=False)
    
    drones = [agent for agent in simulated_agents if agent.agent_type == 'drone' and len(agent.history_pos) > 1]
    if not drones:
        print("No smoothed drone trajectories to plot.")
        plt.close(fig)
        return

    all_z = np.concatenate([np.array(smoothed_trajectories[d.id])[:, 2] for d in drones if d.id in smoothed_trajectories])
    min_alt, max_alt = (all_z.min(), all_z.max()) if all_z.size > 0 else (0, 1)
    
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min_alt, vmax=max_alt)

    for drone in drones:
        if drone.id not in smoothed_trajectories:
            continue

        raw_x = [pos[0] for pos in drone.history_pos]
        raw_y = [pos[1] for pos in drone.history_pos]
        ax.plot(raw_x, raw_y, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=8, label='Raw Path' if drone == drones[0] else "")

        smooth_path = np.array(smoothed_trajectories[drone.id])
        points = smooth_path[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        z_coords = smooth_path[:, 2] 
        segment_colors = (z_coords[:-1] + z_coords[1:]) / 2

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_colors)
        lc.set_linewidth(2.5)
        lc.set_alpha(1.0)
        lc.set_zorder(9)
        ax.add_collection(lc)

    ax.set_title("Smoothed Drone Trajectories (Colored by Altitude)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Altitude (m)')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='Raw Path'),
        Line2D([0], [0], color=cmap(0.5), lw=2.5, label='Smoothed Path (Colored by Alt.)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    filepath = os.path.join(config.get("animation_output_path", "."), "smoothed_drone_trajectories.png")
    plt.savefig(filepath)
    print(f"Saved smoothed drone trajectory plot to {filepath}")
    plt.close(fig)


def animate_simulation_results(simulated_agents_final_state, config):
    if config.get("show_live_simulation", False): return
    if not config.get("create_animation", False): return
    
    if not simulated_agents_final_state:
        print("No agents to animate.")
        return
    
    max_frames = 0
    for agent in simulated_agents_final_state:
        if agent.history_time:
            last_agent_time = agent.history_time[-1]
            last_frame_index = int(round(last_agent_time / config['dt']))
            if last_frame_index > max_frames:
                max_frames = last_frame_index
    max_frames += 1
            
    if max_frames == 0: 
        print("No history to animate.")
        return
        
    fig, ax = plt.subplots(figsize=(14, 14))
    
    def update_frame(frame_num):
        current_anim_time = frame_num * config['dt']
        agents_at_frame = []
        
        for agent in simulated_agents_final_state:
            if not agent.history_time: continue
            spawn_time = agent.history_time[0]
            
            if current_anim_time >= spawn_time:
                history_index = int(round((current_anim_time - spawn_time) / config['dt']))

                if history_index < len(agent.history_pos):
                    agent.pos = agent.history_pos[history_index]
                    agent.heading_angle_rad = agent.history_heading[history_index]
                    agents_at_frame.append(agent)

        update_live_plot(ax, agents_at_frame, config, current_anim_time)

    ani = animation.FuncAnimation(fig, update_frame, frames=max_frames,
                                  interval=max(50, int(config["dt"]*1000)), repeat=False)

    output_path = config.get("animation_output_path", ".")
    if not os.path.exists(output_path): os.makedirs(output_path)
    animation_filename = os.path.join(output_path, "intersection_full_simulation.mp4")
    try:
        print(f"\nSaving full simulation animation to: {animation_filename}")
        writer = animation.FFMpegWriter(fps=max(1, int(1.0/config["dt"])), bitrate=5000)
        ani.save(animation_filename, writer=writer)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure that ffmpeg is installed and accessible in your system's PATH.")
            
    plt.close(fig)

# --- MODIFIED FUNCTION ---
def plot_individual_agent_trajectories(simulated_agents, config):
    """
    Generates and saves a separate plot for each agent's trajectory,
    including its target location and a heatmap of its lane-keeping utility.
    """
    if not config.get("plot_individual_trajectories", False):
        return

    print("\n--- Plotting individual agent trajectories with utility heatmaps ---")
    output_path = config.get("animation_output_path", ".")
    individual_plots_path = os.path.join(output_path, "individual_trajectories")
    
    if not os.path.exists(individual_plots_path):
        os.makedirs(individual_plots_path)

    colors = {'vehicle': 'deepskyblue', 'pedestrian': 'fuchsia', 'drone': 'darkorange', 'bike': 'lime'}

    for agent in simulated_agents:
        if not agent.history_pos:
            continue

        fig, ax = plt.subplots(figsize=(10, 12))
        plot_environment(config, ax=ax, show_street_names=False)

        # --- HEATMAP GENERATION ---
        # 1. Define the grid for the heatmap
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        grid_res = 150 # Resolution of the heatmap grid
        x_coords_grid = np.linspace(x_lims[0], x_lims[1], grid_res)
        y_coords_grid = np.linspace(y_lims[0], y_lims[1], grid_res)
        xx, yy = np.meshgrid(x_coords_grid, y_coords_grid)
        utility_values = np.zeros_like(xx)

        # 2. Calculate utility for each point on the grid
        for i in range(grid_res):
            for j in range(grid_res):
                # For drones, we calculate utility at their cruise altitude.
                # For ground agents, z-coordinate is 0.
                z_coord = config["drone_cruise_altitude"] if agent.agent_type == 'drone' else 0.0
                candidate_pos = np.array([xx[i, j], yy[i, j], z_coord])
                
                utility_values[i, j] = utils.calculate_lane_keeping_reward(
                    agent, candidate_pos, agent.params
                )
        
        # Replace zero values with NaN so they don't dominate the colormap
        if np.any(utility_values):
            utility_values[utility_values == 0] = np.nan

        # 3. Plot the heatmap
        heatmap = ax.imshow(utility_values, origin='lower',
                            extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
                            aspect='auto', cmap='viridis', alpha=0.6, zorder=0.8)

        # Add a color bar
        cbar = fig.colorbar(heatmap, ax=ax, shrink=0.7)
        cbar.set_label('Lane Keeping Utility')
        # --- END HEATMAP ---

        x_coords = [pos[0] for pos in agent.history_pos]
        y_coords = [pos[1] for pos in agent.history_pos]
        color = colors.get(agent.agent_type, 'gray')

        # Add a white outline to the trajectory for better visibility over the heatmap
        ax.plot(x_coords, y_coords, color='white', linewidth=3.5, alpha=1.0, zorder=9)
        ax.plot(x_coords, y_coords, color=color, linewidth=2.0, alpha=1.0, zorder=10)
        
        start_pos = agent.history_pos[0]
        end_pos = agent.history_pos[-1]
        ax.scatter(start_pos[0], start_pos[1], color='green', s=150, zorder=15, label='Start', ec='black')
        ax.scatter(end_pos[0], end_pos[1], color='red', s=150, zorder=15, label='End', ec='black')
        
        # --- ADDED: Plot the agent's target destination ---
        ax.scatter(agent.destination_pos[0], agent.destination_pos[1], color='cyan', marker='X', s=200, 
                   zorder=16, label='Target', ec='black', lw=2)
        
        ax.text(start_pos[0], start_pos[1] + 2.5, f"ID: {agent.id} (Start)", color='black', fontsize=10, fontweight='bold', ha='center')

        origin_str = agent.origin_key if agent.origin_key is not None else 'N/A'
        dest_str = agent.destination_key if agent.destination_key is not None else 'N/A'
        title = f"Agent ID: {agent.id} ({agent.agent_type})\nOrigin: {origin_str}   |   Destination: {dest_str}"
        ax.set_title(title, fontsize=14)
        
        ax.legend()
        # Reset limits which might be affected by imshow
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)

        filepath = os.path.join(individual_plots_path, f"agent_{agent.id}_trajectory.png")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  - Saved plot for Agent {agent.id}")
        plt.close(fig)
        
    print("Finished plotting individual trajectories.")