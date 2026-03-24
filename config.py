# config.py

import numpy as np

# --- Simulation Configuration for Intersection ---
CONFIG = {
    "environment_setting": 1, # Can be 1 (original) or 2 (with bike lane)
    "bike_behavior_mode": "sidewalk", # "road" or "sidewalk". Sidewalk mode will use bike lanes.
    
    "drone_station_placement_offset": 2.5,
    "num_drone_stations": 8,
    "num_lanes_per_direction_vertical": 2,

    "initial_agent_counts": {
        "vehicle": 2,
        "pedestrian": 0,
        "drone": 0,
        "bike": 0
    },
    "dynamic_generation_counts": {
        "vehicle": 1,
        "pedestrian": 0,
        "drone": 0,
        "bike": 0
    },
    
    "enable_dynamic_generation": True,
    "agent_generation_interval": 5.0,

    "plot_trajectories_with_ids": True,
    "plot_individual_trajectories": True,
    "show_live_simulation": False,
    "create_animation": True,
    "plot_trajectories": True,
    "enable_trajectory_smoothing": True,
    "export_to_csv": True,

    # Environment properties
    "num_lanes_per_direction_horizontal": 1,
    "lane_width": 3.5,
    "crosswalk_width": 3.5,
    "sidewalk_width": 3.0,
    "bike_lane_width": 3.5,
    "connecting_road_length": 0.0,
    "dt": 0.1,
    "total_simulation_time": 40.0,
    "grid_cell_length": 1.5,
    "grid_cell_width": 1.5,
    "random_seed": 42,
    
    # Agent Properties
    "vehicle_length": 4.5,
    "vehicle_width": 1.8,
    "vehicle_speed_range": [4.0, 10.0],
    "vehicle_destination_threshold": 2,
    
    "bike_length": 2.0,
    "bike_width": 0.9,
    "bike_speed_range": [3.0, 6.0],
    "bike_destination_threshold": 2,
    
    "pedestrian_radius": 0.4,
    "pedestrian_speed_range": [1.0, 2.5],
    "pedestrian_destination_threshold": 1,

    "drone_radius": 0.5,
    "drone_speed_range": [6.0, 12.0],
    "drone_destination_threshold": 2,
    "drone_cruise_altitude": 4.0,
    "drone_glide_path_radius": 10.0,

    "max_acceleration": 4.0,
    "max_deceleration": 6.0,
    "max_steering_angle_rad": np.deg2rad(6),
    
    # Utility function parameters
    # Total utility (per candidate grid) uses:
    # U = U_dir + U_speed + U_dist + U_lane - w_c * P_collision - ped_on_road_penalty_term
    # NOTE: lane utility parameter = w_ell (for ground agents) and w_ell_drone (for drones).
    "w_c": 30.0,                  # Weight of collision-risk penalty term.
    "w_ell": 60.0,                # Lane/path adherence reward weight (vehicles/bikes/pedestrians).
    "w_ell_drone": 2.0,           # Altitude-path adherence reward weight (drones).
    "ped_on_road_penalty": 1000.0,# Extra penalty when a pedestrian candidate is on road outside crosswalk.
    "S_theta": 0.75,              # Directional-alignment term scale.
    "eta_i": 1.5,                 # Directional sensitivity (larger = stronger heading preference).
    "S_v": 0.8,                   # Speed-alignment term scale.
    "xi_i": 1.5,                  # Speed-shape parameter in speed utility.
    "S_d": 0.9,                   # Distance/proximity reward scale.
    "gamma": 1.5,                 # Distance reward steepness vs perception horizon.
    "w_x": 1.0, "w_y": 1.0, "w_z": 1.0,  # Axis weights used in effective distance d_eff.
    "beta": 3.2,                  # Nonlinearity for lane/altitude adherence reward decay.
    "epsilon": 1e-6,              # Small numerical constant to avoid divide-by-zero.
    
    # Local speed field around agents (distance-decayed grid influence)
    "speed_influence_radius": 10.0,         # [m] Radius where nearby agents influence grid target speed.
    "speed_influence_decay_factor": 0.6,    # Exponential decay factor (smaller = faster decay).
    "speed_influence_close_cap_radius": 3.0,# [m] Strong cap zone near other agents.
    "speed_influence_base_weight": 1.0,     # Blend weight of agent desired speed vs neighbor-influenced speed.
    # Collision probability uncertainty (Gaussian prediction, paper-style)
    "collision_sigma_x": 1.0,               # [m] Uncertainty in predicted x position.
    "collision_sigma_y": 1.0,               # [m] Uncertainty in predicted y position.
    "collision_sigma_z": 0.5,               # [m] Uncertainty in predicted z position.

    # Perception and Safety Distances
    # MODIFIED: Increased from 2.0 to 5.0 for better forward-looking behavior
    "kappa_perception_horizon_factor": 4.0, 
    "min_perception_horizon_abs": 5.0,
    "vv_safety_distance": 1.2, "vp_safety_distance": 0.6, "pp_safety_distance": 0.3,
    "dd_safety_distance": 0.5, "vd_safety_distance": 1.0, "pd_safety_distance": 2.0,
    "bb_safety_distance": 0.6, "vb_safety_distance": 1.0, "pb_safety_distance": 0.5,
    "db_safety_distance": 1.0,
    
    "animation_output_path": "./intersection_plots/",
}

CONFIG["one_way_road_width_vertical"] = CONFIG["num_lanes_per_direction_vertical"] * CONFIG["lane_width"]
CONFIG["one_way_road_width_horizontal"] = CONFIG["num_lanes_per_direction_horizontal"] * CONFIG["lane_width"]
CONFIG["intersection_size"] = 2 * CONFIG["one_way_road_width_vertical"]
CONFIG["road_segment_length"] = 50.0