# simulation.py

import random
import copy
import numpy as np
from config import CONFIG
from environment import ROADS, OD_LANE_KEEPING_ROADS, SIDEWALKS, DRONE_STATIONS, BIKE_LANES
from agents import Agent
import matplotlib.pyplot as plt
from visualization import update_live_plot


SPAWN_DATA = {
    "VEHICLE_SPAWN_POINTS": {
        'I1_NBO': lambda: (random.uniform(ROADS['I1_NBO']['x_min'], ROADS['I1_NBO']['x_max']), ROADS['I1_NBO']['y_min'] + 5, np.pi/2),
        'I1_SBO': lambda: (random.uniform(ROADS['I1_SBO']['x_min'], ROADS['I1_SBO']['x_max']), ROADS['I1_SBO']['y_max'] - 5, -np.pi/2),
        'I1_EBO': lambda: (ROADS['I1_EBO']['x_min'] + 5, random.uniform(ROADS['I1_EBO']['y_min'], ROADS['I1_EBO']['y_max']), 0),
        'I1_WBO': lambda: (ROADS['I1_WBO']['x_max'] - 5, random.uniform(ROADS['I1_WBO']['y_min'], ROADS['I1_WBO']['y_max']), np.pi),
        'I2_NBO': lambda: (random.uniform(ROADS['I2_NBO']['x_min'], ROADS['I2_NBO']['x_max']), ROADS['I2_NBO']['y_min'] + 5, np.pi/2),
        'I2_SBO': lambda: (random.uniform(ROADS['I2_SBO']['x_min'], ROADS['I2_SBO']['x_max']), ROADS['I2_SBO']['y_max'] - 5, -np.pi/2),
        'I2_WBO': lambda: (ROADS['I2_WBO']['x_max'] - 5, random.uniform(ROADS['I2_WBO']['y_min'], ROADS['I2_WBO']['y_max']), np.pi),
    },
    "VEHICLE_DEST_POINTS": {
        'I1_NBD': lambda: (random.uniform(ROADS['I1_NBD']['x_min'], ROADS['I1_NBD']['x_max']), ROADS['I1_NBD']['y_max']),
        'I1_SBD': lambda: (random.uniform(ROADS['I1_SBD']['x_min'], ROADS['I1_SBD']['x_max']), ROADS['I1_SBD']['y_min']),
        'I1_EBD': lambda: (ROADS['I1_EBD']['x_max'], random.uniform(ROADS['I1_EBD']['y_min'], ROADS['I1_EBD']['y_max'])),
        'I1_WBD': lambda: (ROADS['I1_WBD']['x_min'], random.uniform(ROADS['I1_WBD']['y_min'], ROADS['I1_WBD']['y_max'])),
        'I2_NBD': lambda: (random.uniform(ROADS['I2_NBD']['x_min'], ROADS['I2_NBD']['x_max']), ROADS['I2_NBD']['y_max']),
        'I2_SBD': lambda: (random.uniform(ROADS['I2_SBD']['x_min'], ROADS['I2_SBD']['x_max']), ROADS['I2_SBD']['y_min']),
        'I2_EBD': lambda: (ROADS['I2_EBD']['x_max'], random.uniform(ROADS['I2_EBD']['y_min'], ROADS['I2_EBD']['y_max'])),
    },
    "VALID_OD_PAIRS": list(OD_LANE_KEEPING_ROADS.keys()),
    "SIDEWALK_KEYS": list(SIDEWALKS.keys()),
    "BIKE_LANE_KEYS": list(BIKE_LANES.keys()), # ADDED
    "STATION_KEYS": list(DRONE_STATIONS.keys())
}

def _create_single_agent(agent_type, agent_id, current_agents, config, spawn_data, creation_time):
    for _ in range(50):
        if agent_type == 'vehicle':
            if not spawn_data["VALID_OD_PAIRS"]: return None
            origin_key, dest_key = random.choice(spawn_data["VALID_OD_PAIRS"])
            initial_pos_x, initial_pos_y, initial_heading = spawn_data["VEHICLE_SPAWN_POINTS"][origin_key]()
            initial_pos = np.array([initial_pos_x, initial_pos_y, 0.0])
            if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config["vehicle_length"] for a in current_agents):
                continue
            dest_pos_x, dest_pos_y = spawn_data["VEHICLE_DEST_POINTS"][dest_key]()
            destination_pos = [dest_pos_x, dest_pos_y, 0.0]
            v_min, v_max = config["vehicle_speed_range"]
            agent_params = config.copy()
            agent_params["desired_speed_agent"] = random.uniform(v_min, v_max)
            agent_params["origin_key"], agent_params["destination_key"] = origin_key, dest_key
            return Agent(id=agent_id, agent_type='vehicle', initial_pos=initial_pos, initial_heading=initial_heading, initial_speed=random.uniform(v_min, v_max * 0.8), destination_pos=destination_pos, params=agent_params, creation_time=creation_time)

        elif agent_type == 'bike':
            agent_params = config.copy()
            bike_mode = config.get("bike_behavior_mode", "road")
            is_setting_2 = config.get("environment_setting", 1) == 2

            # In Setting 2, "sidewalk" mode is re-purposed to be "bike_lane" mode.
            if bike_mode == "sidewalk" and is_setting_2:
                if len(spawn_data["BIKE_LANE_KEYS"]) < 2: return None
                origin_bl_key, dest_bl_key = random.sample(spawn_data["BIKE_LANE_KEYS"], 2)
                origin_bounds, dest_bounds = BIKE_LANES[origin_bl_key], BIKE_LANES[dest_bl_key]
                initial_pos = np.array([random.uniform(origin_bounds['x_min'], origin_bounds['x_max']), random.uniform(origin_bounds['y_min'], origin_bounds['y_max']), 0.0])
                if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config['bb_safety_distance'] for a in current_agents):
                    continue
                destination_pos = [random.uniform(dest_bounds['x_min'], dest_bounds['x_max']), random.uniform(dest_bounds['y_min'], dest_bounds['y_max']), 0.0]
                initial_heading = np.arctan2(destination_pos[1] - initial_pos[1], destination_pos[0] - initial_pos[0])
                agent_params["origin_key"], agent_params["destination_key"] = None, None
            
            elif bike_mode == "road":
                if not spawn_data["VALID_OD_PAIRS"]: return None
                origin_key, dest_key = random.choice(spawn_data["VALID_OD_PAIRS"])
                initial_pos_x, initial_pos_y, initial_heading = spawn_data["VEHICLE_SPAWN_POINTS"][origin_key]()
                initial_pos = np.array([initial_pos_x, initial_pos_y, 0.0])
                if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config["bike_length"] for a in current_agents):
                    continue
                dest_pos_x, dest_pos_y = spawn_data["VEHICLE_DEST_POINTS"][dest_key]()
                destination_pos = [dest_pos_x, dest_pos_y, 0.0]
                agent_params["origin_key"], agent_params["destination_key"] = origin_key, dest_key
            
            else: # Sidewalk mode in Setting 1 (or if bike lanes don't exist)
                if len(spawn_data["SIDEWALK_KEYS"]) < 2: return None
                origin_sw_key, dest_sw_key = random.sample(spawn_data["SIDEWALK_KEYS"], 2)
                origin_bounds, dest_bounds = SIDEWALKS[origin_sw_key], SIDEWALKS[dest_sw_key]
                initial_pos = np.array([random.uniform(origin_bounds['x_min'], origin_bounds['x_max']), random.uniform(origin_bounds['y_min'], origin_bounds['y_max']), 0.0])
                if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config['pb_safety_distance'] for a in current_agents):
                    continue
                destination_pos = [random.uniform(dest_bounds['x_min'], dest_bounds['x_max']), random.uniform(dest_bounds['y_min'], dest_bounds['y_max']), 0.0]
                initial_heading = np.arctan2(destination_pos[1] - initial_pos[1], destination_pos[0] - initial_pos[0])
                agent_params["origin_key"], agent_params["destination_key"] = None, None

            v_min, v_max = config["bike_speed_range"]
            agent_params["desired_speed_agent"] = random.uniform(v_min, v_max)
            return Agent(id=agent_id, agent_type='bike', initial_pos=initial_pos, initial_heading=initial_heading, initial_speed=random.uniform(v_min, v_max * 0.8), destination_pos=destination_pos, params=agent_params, creation_time=creation_time)

        elif agent_type == 'pedestrian':
            if len(spawn_data["SIDEWALK_KEYS"]) < 2: return None
            origin_sw_key, dest_sw_key = random.sample(spawn_data["SIDEWALK_KEYS"], 2)
            origin_bounds, dest_bounds = SIDEWALKS[origin_sw_key], SIDEWALKS[dest_sw_key]
            initial_pos = np.array([random.uniform(origin_bounds['x_min'], origin_bounds['x_max']), random.uniform(origin_bounds['y_min'], origin_bounds['y_max']), 0.0])
            if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config['pp_safety_distance'] for a in current_agents):
                continue
            destination_pos = [random.uniform(dest_bounds['x_min'], dest_bounds['x_max']), random.uniform(dest_bounds['y_min'], dest_bounds['y_max']), 0.0]
            initial_heading = np.arctan2(destination_pos[1] - initial_pos[1], destination_pos[0] - initial_pos[0])
            p_min, p_max = config["pedestrian_speed_range"]
            return Agent(id=agent_id, agent_type='pedestrian', initial_pos=initial_pos, initial_heading=initial_heading, initial_speed=random.uniform(p_min, p_max), destination_pos=destination_pos, params=config.copy(), creation_time=creation_time)

        elif agent_type == 'drone':
            if not spawn_data["SIDEWALK_KEYS"] or not spawn_data["STATION_KEYS"]: return None
            origin_sw_key = random.choice(spawn_data["SIDEWALK_KEYS"])
            origin_bounds = SIDEWALKS[origin_sw_key]
            initial_pos = np.array([random.uniform(origin_bounds['x_min'], origin_bounds['x_max']), random.uniform(origin_bounds['y_min'], origin_bounds['y_max']), 0.0])
            if any(np.linalg.norm(initial_pos[:2] - a.pos[:2]) < config['dd_safety_distance'] for a in current_agents):
                continue
            dest_key = random.choice(spawn_data["STATION_KEYS"])
            destination_pos = DRONE_STATIONS[dest_key]['pos']
            initial_heading = np.arctan2(destination_pos[1] - initial_pos[1], destination_pos[0] - initial_pos[0])
            d_min, d_max = config["drone_speed_range"]
            agent_params = config.copy()
            agent_params["origin_key"], agent_params["destination_key"] = origin_sw_key, dest_key
            return Agent(id=agent_id, agent_type='drone', initial_pos=initial_pos, initial_heading=initial_heading, initial_speed=random.uniform(d_min, d_max), destination_pos=destination_pos, params=agent_params, creation_time=creation_time)
    
    print(f"Warning: Could not place new {agent_type} without overlap after 50 attempts.")
    return None

def create_agents(config):
    agents = []
    agent_id_counter = 0
    initial_counts = config.get("initial_agent_counts", {})
    for agent_type, count in initial_counts.items():
        for _ in range(count):
            new_agent = _create_single_agent(agent_type, agent_id_counter, agents, config, SPAWN_DATA, creation_time=0.0)
            if new_agent:
                agents.append(new_agent)
                agent_id_counter += 1
    return agents

def run_simulation_loop(initial_agents, config):
    num_time_steps = int(config["total_simulation_time"] / config["dt"])
    all_agents_for_history = list(initial_agents)
    active_agents = list(initial_agents)
    agent_id_counter = len(initial_agents)
    time_since_last_spawn = 0.0
    dynamic_generation_enabled = config.get("enable_dynamic_generation", False)
    generation_interval = config.get("agent_generation_interval", 10.0)
    fig, ax = None, None
    if config.get("show_live_simulation", False):
        fig, ax = plt.subplots(figsize=(14, 14)); plt.ion(); plt.show(block=False)
    print(f"\nStarting simulation for {config['total_simulation_time']}s ({num_time_steps} steps).")
    progress_stride = max(1, num_time_steps // 10)
    for t_step in range(num_time_steps):
        current_time = t_step * config["dt"]
        # Keep active set clean: completed agents are removed from simulation updates.
        active_agents = [agent for agent in active_agents if not agent.reached_destination]
        if dynamic_generation_enabled:
            time_since_last_spawn += config["dt"]
            if time_since_last_spawn >= generation_interval:
                time_since_last_spawn -= generation_interval
                generation_counts = config.get("dynamic_generation_counts", {})
                for agent_type, count in generation_counts.items():
                    for _ in range(count):
                        new_agent = _create_single_agent(agent_type, agent_id_counter, active_agents, config, SPAWN_DATA, creation_time=current_time)
                        if new_agent:
                            all_agents_for_history.append(new_agent)
                            active_agents.append(new_agent)
                            agent_id_counter += 1
        if (t_step + 1) % progress_stride == 0 or t_step == 0:
            print(f"Simulating step {t_step+1}/{num_time_steps} (Time: {current_time:.1f}s) - Total Agents: {len(all_agents_for_history)}, Active: {len(active_agents)}")
        if not active_agents and not dynamic_generation_enabled: 
            print("All agents reached destination."); break
        # Synchronous decisions: all agents react to the same time-t snapshot.
        agent_state_snapshot = copy.deepcopy(active_agents)
        for agent_obj in active_agents:
            agent_obj.decide_and_move(agent_state_snapshot, current_time)
        active_agents = [agent for agent in active_agents if not agent.reached_destination]
        if fig and ax:
            update_live_plot(ax, active_agents, config, current_time); plt.pause(0.001)
    if fig and ax:
        plt.ioff(); plt.show()
    print("Simulation finished.")
    return all_agents_for_history