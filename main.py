# main.py

import os
import random
import numpy as np
from config import CONFIG
from simulation import create_agents, run_simulation_loop
# MODIFIED: Import the new plotting function
from visualization import (
    animate_simulation_results, plot_ground_trajectories, plot_drone_trajectories,
    plot_smoothed_ground_trajectories, plot_smoothed_drone_trajectories,
    plot_ground_trajectories_with_ids, plot_individual_agent_trajectories
)
from smoothing import smooth_trajectories
from export_data import export_trajectories_to_csv

def main():
    """
    Main function to run the agent-based simulation.
    """
    random_seed = CONFIG.get("random_seed", None)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    initial_agents = create_agents(CONFIG)
    if not initial_agents:
        print("No agents were created. Exiting."); return

    print(f"\n--- Created {len(initial_agents)} initial agents. ---")

    simulated_agents_run = run_simulation_loop(initial_agents, CONFIG)

    print("\n--- Final Agent Status ---")
    for agent in simulated_agents_run:
        status = "Reached Dest." if agent.reached_destination else "In-Progress"
        
        agent_id_str = f"Agent {agent.id} ({agent.agent_type})"
        
        # --- MODIFIED: More descriptive route string for bikes ---
        if agent.agent_type == 'vehicle':
             route_str = f"Route: {agent.origin_key}->{agent.destination_key}"
        elif agent.agent_type == 'bike':
            if agent.origin_key and agent.destination_key:
                route_str = f"Route: {agent.origin_key}->{agent.destination_key}"
            else:
                route_str = "Path: Bike (Sidewalk)"
        elif agent.agent_type == 'drone':
             origin_str = agent.params.get('origin_key', 'Sidewalk Start')
             route_str = f"Route: {origin_str}->{agent.destination_key}"
        else: # Pedestrian
            route_str = 'Path: Pedestrian'
        print(f"{agent_id_str:<25} | {status:<15} | {route_str}")


    if not CONFIG.get("show_live_simulation", False):
        print("\n--- Save/Analysis mode selected. Generating all plots and data. ---")
        
        output_path = CONFIG["animation_output_path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if CONFIG.get("plot_trajectories", False):
            plot_ground_trajectories(simulated_agents_run, CONFIG)
            plot_drone_trajectories(simulated_agents_run, CONFIG)
        
        if CONFIG.get("plot_trajectories_with_ids", False):
            plot_ground_trajectories_with_ids(simulated_agents_run, CONFIG)
        
        # --- NEW: Call the individual trajectory plotting function ---
        if CONFIG.get("plot_individual_trajectories", False):
            plot_individual_agent_trajectories(simulated_agents_run, CONFIG)

        if CONFIG.get("enable_trajectory_smoothing", False) or CONFIG.get("export_to_csv", False):
            smoothed_paths = smooth_trajectories(simulated_agents_run, CONFIG)
            
            if CONFIG.get("enable_trajectory_smoothing", False):
                plot_smoothed_ground_trajectories(simulated_agents_run, smoothed_paths, CONFIG)
                plot_smoothed_drone_trajectories(simulated_agents_run, smoothed_paths, CONFIG)
            
            if CONFIG.get("export_to_csv", False):
                export_trajectories_to_csv(simulated_agents_run, smoothed_paths, CONFIG)

    if CONFIG.get("create_animation", False):
        animate_simulation_results(simulated_agents_run, CONFIG)

if __name__ == "__main__":
    main()