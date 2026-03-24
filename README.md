# UtilityFlowSim

UtilityFlowSim is a stochastic, utility-based multi-agent traffic simulation for an urban intersection.
It simulates mixed traffic participants:

- vehicles
- pedestrians
- bikes
- drones

Agents choose their next movement by maximizing a utility objective that balances route/heading alignment, speed preference, path adherence (lane/sidewalk/altitude behavior), and collision-risk penalties.

## What the simulation does

- Builds an intersection environment (including roads, sidewalks, and optional bike-lane setting).
- Spawns initial agents by type based on configuration.
- Optionally generates additional agents during runtime at fixed intervals.
- Advances the world in discrete time steps (`dt`) until `total_simulation_time`.
- Runs synchronous decision-making each step so all agents react to the same snapshot.
- Stops updating agents once they reach their destination.
- Produces visualizations, optional animation, and optional CSV trajectory export.

## Inputs

The main inputs are defined in `config.py` inside the `CONFIG` dictionary.

### 1) Scenario and environment

- `environment_setting`: selects layout variant (`1` or `2`).
- `bike_behavior_mode`: bike routing mode (`"road"` or `"sidewalk"`; in setting 2 this maps to bike-lane behavior).
- Geometry inputs such as `lane_width`, `sidewalk_width`, `bike_lane_width`, and lane counts.
- Drone infrastructure inputs such as `num_drone_stations` and `drone_station_placement_offset`.

### 2) Agent demand (how many agents enter)

- `initial_agent_counts`: number of each type spawned at time 0.
- `enable_dynamic_generation`: enables ongoing spawning during simulation.
- `dynamic_generation_counts`: number of each type spawned per generation event.
- `agent_generation_interval`: seconds between spawn events.

### 3) Timing and reproducibility

- `dt`: simulation step size in seconds.
- `total_simulation_time`: total duration in seconds.
- `random_seed`: fixed seed for reproducible runs.

### 4) Agent dynamics and safety

- Type-specific size/speed/destination-threshold inputs:
  - `vehicle_*`, `bike_*`, `pedestrian_*`, `drone_*`
- Motion limits:
  - `max_acceleration`, `max_deceleration`, `max_steering_angle_rad`
- Interaction distances:
  - safety-distance parameters like `vv_safety_distance`, `vp_safety_distance`, `dd_safety_distance`, etc.

### 5) Utility-model parameters

Utility scoring is controlled by weights and shaping parameters such as:

- `w_c`, `w_ell`, `w_ell_drone`
- `S_theta`, `eta_i`, `S_v`, `xi_i`, `S_d`, `gamma`
- `beta`, `epsilon`
- collision uncertainty terms (`collision_sigma_x/y/z`)
- local speed-field terms (`speed_influence_*`)

These parameters control tradeoffs between efficiency, comfort/path-following, and collision avoidance.

### 6) Output controls

- `show_live_simulation`: live plot during simulation.
- `create_animation`: generate animated output after simulation.
- `plot_trajectories`, `plot_trajectories_with_ids`, `plot_individual_trajectories`
- `enable_trajectory_smoothing`
- `export_to_csv`
- `animation_output_path`: output folder for generated files.

## How to run

From the project root:

```bash
python main.py
```

Before running, edit `config.py` to choose:

- environment mode
- number/type of agents
- total duration and step size
- whether to export plots/animation/CSV

## Outputs

Depending on configuration, files are written to `animation_output_path` (default: `./intersection_plots/`), including:

- trajectory plots for ground agents and drones
- optional ID-labeled and individual-agent plots
- optional smoothed trajectory plots
- optional animation
- optional CSV export of trajectories (`full_simulation_trajectories.csv`)

## Notes

- If `show_live_simulation` is `False`, the project runs in save/analysis mode and generates configured artifacts after the simulation loop.
- Dynamic generation can substantially increase total simulated agent count during long runs.
