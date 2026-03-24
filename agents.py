# agents.py

import numpy as np
import utils
from environment import (ROADS, SIDEWALKS, CROSSWALKS, BIKE_LANES, 
                         get_road_segment_for_point, get_crosswalk_for_point, 
                         get_sidewalk_for_point, get_bike_lane_for_point)

class Agent:
    def __init__(self, id, agent_type, initial_pos, initial_heading, initial_speed, destination_pos, params, creation_time):
        self.id = id
        self.agent_type = agent_type
        self.params = params
        if self.agent_type == 'vehicle':
            self.length, self.width = params['vehicle_length'], params['vehicle_width']
            self.max_speed = params['vehicle_speed_range'][1]
            self.destination_threshold = params['vehicle_destination_threshold']
            self.origin_key, self.destination_key = params.get('origin_key'), params.get('destination_key')
        elif self.agent_type == 'bike':
            self.length, self.width = params['bike_length'], params['bike_width']
            self.max_speed = params['bike_speed_range'][1]
            self.destination_threshold = params['bike_destination_threshold']
            self.origin_key, self.destination_key = params.get('origin_key'), params.get('destination_key')
        elif self.agent_type == 'pedestrian':
            self.length, self.width = params['pedestrian_radius'] * 2, params['pedestrian_radius'] * 2
            self.max_speed = params['pedestrian_speed_range'][1]
            self.destination_threshold = params['pedestrian_destination_threshold']
            self.origin_key, self.destination_key = None, None
        elif self.agent_type == 'drone':
            self.length, self.width = params['drone_radius'] * 2, params['drone_radius'] * 2
            self.max_speed = params['drone_speed_range'][1]
            self.destination_threshold = params['drone_destination_threshold']
            self.origin_key, self.destination_key = params.get('origin_key'), params.get('destination_key')

        self.pos, self.speed, self.heading_angle_rad = np.array(initial_pos, dtype=float), initial_speed, initial_heading
        self.velocity_vector = np.array([self.speed * np.cos(self.heading_angle_rad), self.speed * np.sin(self.heading_angle_rad), 0.0])
        self.destination_pos = np.array(destination_pos, dtype=float)
        
        self.reached_destination = False
        self.current_time = creation_time
        self.history_pos = [self.pos.copy()]
        self.history_heading = [self.heading_angle_rad]
        self.history_speed = [self.speed]
        self.history_time = [creation_time]
        
        # --- MODIFICATION: Initialize lane history ---
        initial_lane = self._get_current_segment_name()
        self.history_lane_info = [initial_lane]

    def _get_current_segment_name(self):
        """Helper function to determine the name of the current segment."""
        # Drones are not on a ground segment
        if self.agent_type == 'drone':
            return 'In-Flight'
            
        x, y = self.pos[0], self.pos[1]
        
        segment_name = get_road_segment_for_point(x, y)
        if not segment_name:
            segment_name = get_crosswalk_for_point(x, y)
        if not segment_name:
            segment_name = get_sidewalk_for_point(x, y)
        if not segment_name:
            # Check for bike lanes last, as they might overlap with other areas
            segment_name = get_bike_lane_for_point(x, y)
        
        return segment_name if segment_name is not None else 'Off-Grid'

    def get_explorable_grid_cells_from_state(self, current_pos_state):
        explorable_cells, unique_grid_coords = [], set()
        perception_dist = max(self.params["min_perception_horizon_abs"], self.speed * 4.0)

        if self.agent_type == 'drone':
            num_radial_steps, num_azimuth_steps, num_inclination_steps = 10, 10, 10
            for i in range(1, num_radial_steps + 1):
                radius = (i / num_radial_steps) * perception_dist
                for j in range(num_azimuth_steps):
                    azimuth = (j / num_azimuth_steps) * 2 * np.pi
                    for k in range(num_inclination_steps):
                        inclination = (k / (num_inclination_steps - 1)) * np.pi
                        dx = radius * np.sin(inclination) * np.cos(azimuth)
                        dy = radius * np.sin(inclination) * np.sin(azimuth)
                        dz = radius * np.cos(inclination)
                        cell_center = current_pos_state + np.array([dx, dy, dz])
                        if cell_center[2] < 0.5: continue
                        g_coord = tuple(int(c / self.params["grid_cell_length"]) for c in cell_center)
                        if g_coord not in unique_grid_coords:
                            explorable_cells.append((g_coord, cell_center))
                            unique_grid_coords.add(g_coord)
            return explorable_cells
        else:
            num_radial_steps, num_angular_steps = 15, 15
            for i in range(1, num_radial_steps + 1):
                radius = (i / num_radial_steps) * perception_dist
                for j in range(num_angular_steps):
                    angle = self.heading_angle_rad - np.pi/2 + (j / (num_angular_steps -1)) * np.pi
                    cell_center_x, cell_center_y = current_pos_state[0] + radius * np.cos(angle), current_pos_state[1] + radius * np.sin(angle)
                    
                    is_valid_ground = any([
                        get_road_segment_for_point(cell_center_x, cell_center_y),
                        get_crosswalk_for_point(cell_center_x, cell_center_y),
                        get_sidewalk_for_point(cell_center_x, cell_center_y),
                        get_bike_lane_for_point(cell_center_x, cell_center_y)
                    ])
                    if not is_valid_ground:
                        continue
                    if not utils.is_position_allowed_for_agent(self, cell_center_x, cell_center_y, self.params):
                        continue
                    gx, gy = int(cell_center_x / self.params["grid_cell_length"]), int(cell_center_y / self.params["grid_cell_width"])
                    if (gx, gy) not in unique_grid_coords:
                        final_cell_center = np.array([(gx + 0.5) * self.params["grid_cell_length"], (gy + 0.5) * self.params["grid_cell_width"], 0.0])
                        explorable_cells.append(((gx, gy), final_cell_center))
                        unique_grid_coords.add((gx, gy))
            return explorable_cells

    def decide_and_move(self, all_agents, current_sim_time):
        self.current_time = current_sim_time
        if self.reached_destination:
            self.velocity_vector, self.speed = np.zeros(3), 0
        else:
            effective_destination = self.destination_pos
            if self.agent_type == 'drone':
                xy_dist_to_dest = np.linalg.norm(self.pos[:2] - self.destination_pos[:2])
                cruise_alt = self.params["drone_cruise_altitude"]
                landing_radius = 10.0
                station_radius = 1.0
                target_alt = 0.0
                if xy_dist_to_dest <= station_radius: target_alt = 0.0
                elif xy_dist_to_dest <= landing_radius:
                    t = (landing_radius - xy_dist_to_dest) / (landing_radius - station_radius)
                    target_alt = cruise_alt * np.cos(t * (np.pi / 2))
                else: target_alt = cruise_alt
                effective_destination = np.array([self.destination_pos[0], self.destination_pos[1], target_alt])

            candidate_grid_cells = self.get_explorable_grid_cells_from_state(self.pos)
            if not candidate_grid_cells:
                self.speed = max(0, self.speed - self.params["max_deceleration"] * 0.5 * self.params["dt"])
            else:
                best_utility, best_target_grid_center = -float('inf'), self.pos
                best_target_ref_speed = self.params.get("desired_speed_agent", self.max_speed)
                for _, grid_center_pos in candidate_grid_cells:
                    dist_to_grid = np.linalg.norm(grid_center_pos - self.pos)
                    time_to_reach = max(dist_to_grid / (self.speed + self.params["epsilon"]), self.params["dt"])
                    p_i_g = utils.calculate_collision_penalty(self, grid_center_pos, all_agents, self.params, time_to_reach)
                    penalty_coll = self.params["w_c"] * p_i_g
                    penalty_off_road, penalty_ped_on_road = 0.0, 0.0
                    if self.agent_type != 'drone':
                        penalty_off_road = utils.calculate_off_road_penalty(self, self.pos, grid_center_pos, self.params)
                        penalty_ped_on_road = utils.calculate_ped_on_road_penalty(self, grid_center_pos, self.params)

                    if penalty_off_road == float('inf'):
                        total_utility = -float('inf')
                    else:
                        v_ref_candidate = utils.calculate_grid_reference_speed(self, grid_center_pos, all_agents, self.params)
                        util_dir = utils.calculate_directional_alignment(self, grid_center_pos, effective_destination, self.params)
                        util_dist = utils.calculate_distance_reward(grid_center_pos, effective_destination, self.speed, self.params)
                        util_speed = utils.calculate_speed_alignment(self, grid_center_pos, all_agents, self.params, v_ref_candidate)
                        lane_reward = utils.calculate_lane_keeping_reward(self, grid_center_pos, self.params)
                        total_utility = util_dir + util_dist + util_speed + lane_reward - penalty_coll - penalty_ped_on_road
                        
                    if total_utility > best_utility:
                        best_utility, best_target_grid_center = total_utility, grid_center_pos
                        best_target_ref_speed = v_ref_candidate if penalty_off_road != float('inf') else best_target_ref_speed

                target_vector = best_target_grid_center - self.pos
                target_distance = np.linalg.norm(target_vector)
                if target_distance < self.params["epsilon"]:
                    if self.speed > 0: self.speed = max(0, self.speed - self.params["max_deceleration"] * 0.2 * self.params["dt"])
                else:
                    unit_target_vector = target_vector / target_distance
                    desired_heading = np.arctan2(unit_target_vector[1], unit_target_vector[0])
                    angle_diff = desired_heading - self.heading_angle_rad
                    while angle_diff > np.pi: angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi: angle_diff += 2 * np.pi
                    steering_change = np.clip(angle_diff, -self.params["max_steering_angle_rad"], self.params["max_steering_angle_rad"])
                    self.heading_angle_rad += steering_change
                    desired_speed = self.params.get("desired_speed_agent", self.max_speed)
                    effective_target_speed = min(desired_speed, best_target_ref_speed, self.max_speed, target_distance / self.params["dt"])
                    speed_error = effective_target_speed - self.speed
                    accel = np.clip(speed_error / self.params["dt"], -self.params["max_deceleration"], self.params["max_acceleration"])
                    self.speed = np.clip(self.speed + accel * self.params["dt"], 0, self.max_speed)
            
            if self.agent_type == 'drone':
                if np.linalg.norm(best_target_grid_center - self.pos) > self.params['epsilon']:
                    self.velocity_vector = (best_target_grid_center - self.pos) / np.linalg.norm(best_target_grid_center - self.pos) * self.speed
                else: self.velocity_vector = np.zeros(3)
            else:
                self.velocity_vector = np.array([self.speed * np.cos(self.heading_angle_rad), self.speed * np.sin(self.heading_angle_rad), 0.0])
        
        self.pos += self.velocity_vector * self.params["dt"]
        if np.linalg.norm(self.pos - self.destination_pos) < self.destination_threshold:
            self.reached_destination = True
        
        # --- MODIFICATION: Update history with current segment name ---
        self.history_pos.append(self.pos.copy())
        self.history_heading.append(self.heading_angle_rad)
        self.history_speed.append(self.speed)
        self.history_time.append(self.current_time)
        self.history_lane_info.append(self._get_current_segment_name())