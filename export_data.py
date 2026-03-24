# export_data.py

import csv
import os
from environment import OD_LANE_KEEPING_ROADS

def export_trajectories_to_csv(simulated_agents, smoothed_trajectories, config):
    """
    Exports the raw and smoothed trajectory data for all agents to a CSV file.
    """
    print("\n--- Exporting all trajectory data to CSV file ---")
    
    output_path = config.get("animation_output_path", ".")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filepath = os.path.join(output_path, "full_simulation_trajectories.csv")

    # --- MODIFICATION: Add the new 'lane_segment' column to the header ---
    header = [
        'time', 'id', 'agent_type', 
        'raw_x', 'raw_y', 'raw_z', 
        'smoothed_x', 'smoothed_y', 'smoothed_z', 
        'origin', 'destination', 'path', 'lane_segment'
    ]

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        # Iterate through each agent and its history
        for agent in simulated_agents:
            agent_id = agent.id
            agent_type = agent.agent_type
            origin_key = agent.origin_key if agent.origin_key is not None else 'NA'
            dest_key = agent.destination_key if agent.destination_key is not None else 'NA'
            
            path_str = "NA"
            if agent_type == 'vehicle':
                od_pair = (origin_key, dest_key)
                path_list = OD_LANE_KEEPING_ROADS.get(od_pair)
                if path_list:
                    path_str = " -> ".join(path_list)

            agent_smoothed_path = smoothed_trajectories.get(agent_id)

            for i in range(len(agent.history_pos)):
                time = agent.history_time[i]
                raw_pos = agent.history_pos[i]
                
                # --- MODIFICATION: Get the lane segment from history ---
                # Safely get the lane info for the current time step
                lane_segment = agent.history_lane_info[i] if i < len(agent.history_lane_info) else 'NA'

                if agent_smoothed_path and i < len(agent_smoothed_path):
                    smooth_pos = agent_smoothed_path[i]
                    smoothed_x, smoothed_y, smoothed_z = smooth_pos[0], smooth_pos[1], smooth_pos[2]
                else:
                    smoothed_x, smoothed_y, smoothed_z = 'NA', 'NA', 'NA'

                # --- MODIFICATION: Add the lane_segment to the row ---
                row = [
                    f"{time:.2f}",
                    agent_id,
                    agent_type,
                    raw_pos[0], raw_pos[1], raw_pos[2],
                    smoothed_x, smoothed_y, smoothed_z,
                    origin_key,
                    dest_key,
                    path_str,
                    lane_segment
                ]
                writer.writerow(row)

    print(f"Successfully exported data to {filepath}")