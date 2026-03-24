# utils.py

import numpy as np
from environment import (ROADS, CROSSWALKS, SIDEWALKS, BIKE_LANES, OD_LANE_KEEPING_ROADS, INTERSECTIONS,
                         get_road_segment_for_point, get_crosswalk_for_point, 
                         get_sidewalk_for_point, get_bike_lane_for_point)

def _get_road_info(road_name):
    bounds = ROADS[road_name]
    cx = (bounds['x_min'] + bounds['x_max']) / 2
    cy = (bounds['y_min'] + bounds['y_max']) / 2
    is_vertical = (bounds['y_max'] - bounds['y_min']) > (bounds['x_max'] - bounds['x_min'])
    if is_vertical:
        direction = np.array([0.0, 1.0 if 'NB' in road_name else -1.0, 0.0])
        boundary_y = bounds['y_max'] if 'NBO' in road_name else bounds['y_min']
        boundary_point = np.array([cx, boundary_y, 0.0])
    else:
        direction = np.array([1.0 if 'EB' in road_name else -1.0, 0.0, 0.0])
        boundary_x = bounds['x_max'] if 'EBO' in road_name else bounds['x_min']
        boundary_point = np.array([boundary_x, cy, 0.0])
    return np.array([cx, cy, 0.0]), direction, boundary_point

def _solve_line_intersect(p1, v1, p2, v2):
    A = np.array([v1[:2], -v2[:2]]).T
    b = p2[:2] - p1[:2]
    try:
        t, _ = np.linalg.solve(A, b)
        intersect_2d = p1[:2] + t * v1[:2]
        return np.array([intersect_2d[0], intersect_2d[1], 0.0])
    except np.linalg.LinAlgError: return None

def _generate_intersection_bezier_paths():
    paths = {}
    full_paths = OD_LANE_KEEPING_ROADS.copy()
    for od_pair, path_list in full_paths.items():
        for i in range(len(path_list) - 2):
            entry_road, ix_name, exit_road = path_list[i], path_list[i+1], path_list[i+2]
            if 'INTERSECTION' not in ix_name: continue
            if not all(k in ROADS for k in [entry_road, exit_road]): continue
            p_entry, v_entry, p0 = _get_road_info(entry_road)
            p_exit, v_exit, p2 = _get_road_info(exit_road)
            p1 = _solve_line_intersect(p_entry, v_entry, p_exit, v_exit)
            if p1 is None: p1 = (p0 + p2) / 2.0
            paths[(od_pair, ix_name)] = (p0, p1, p2)
    return paths

_INTERSECTION_BEZIER_CONTROLS = _generate_intersection_bezier_paths()

def _quadratic_bezier(t, p0, p1, p2):
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

def _calculate_min_dist_to_bezier(point, control_points, num_samples=20):
    p0, p1, p2 = control_points
    min_dist_sq = float('inf')
    for t in np.linspace(0, 1, num_samples):
        curve_point = _quadratic_bezier(t, p0, p1, p2)
        dist_sq = np.sum((point[:2] - curve_point[:2])**2)
        min_dist_sq = min(min_dist_sq, dist_sq)
    return np.sqrt(min_dist_sq)

def get_agent_corners(pos, heading_rad, length, width):
    hl, hw = length / 2.0, width / 2.0
    local_corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot_matrix @ local_corners.T).T + pos[:2]

def is_position_allowed_for_agent(agent, x, y, params):
    if agent.agent_type == 'drone':
        return True
    on_road = get_road_segment_for_point(x, y) is not None
    on_crosswalk = get_crosswalk_for_point(x, y) is not None
    on_sidewalk = get_sidewalk_for_point(x, y) is not None
    on_bike_lane = get_bike_lane_for_point(x, y) is not None

    if agent.agent_type == 'vehicle':
        return on_road
    if agent.agent_type == 'pedestrian':
        return on_crosswalk or on_sidewalk
    if agent.agent_type == 'bike':
        bike_mode = params.get("bike_behavior_mode", "road")
        is_setting_2 = params.get("environment_setting", 1) == 2
        if bike_mode == "road":
            return on_road
        if is_setting_2:
            return on_bike_lane or on_crosswalk
        return on_sidewalk or on_crosswalk
    return False

def calculate_off_road_penalty(agent, current_pos, candidate_pos, params):
    # Sample along the full segment to avoid midpoint-only tunneling.
    for alpha in np.linspace(0.0, 1.0, 5):
        sampled_point = current_pos + alpha * (candidate_pos - current_pos)
        if not is_position_allowed_for_agent(agent, sampled_point[0], sampled_point[1], params):
            return float('inf')
    return 0.0

def calculate_ped_on_road_penalty(agent, candidate_pos, params):
    if agent.agent_type != 'pedestrian': return 0.0
    on_road = get_road_segment_for_point(candidate_pos[0], candidate_pos[1]) is not None
    on_crosswalk = get_crosswalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
    if on_road and not on_crosswalk: return params["ped_on_road_penalty"]
    return 0.0

def get_speed_of_lane(grid_pos, ego_agent, all_agents, perception_range, desired_speed, params):
    road_segment = get_road_segment_for_point(grid_pos[0], grid_pos[1])
    if road_segment is None: return desired_speed
    ego_pos = ego_agent.pos
    if np.linalg.norm(ego_agent.velocity_vector) > 0.1:
        ego_heading_vector = ego_agent.velocity_vector / np.linalg.norm(ego_agent.velocity_vector)
    else: ego_heading_vector = np.array([np.cos(ego_agent.heading_angle_rad), np.sin(ego_agent.heading_angle_rad), 0])
    closest_dist = float('inf')
    speed_of_closest = desired_speed
    for other_agent in all_agents:
        if other_agent.id == ego_agent.id or other_agent.agent_type == 'drone': continue
        if other_agent.agent_type == 'pedestrian':
            on_road = get_road_segment_for_point(other_agent.pos[0], other_agent.pos[1]) is not None
            on_crosswalk = get_crosswalk_for_point(other_agent.pos[0], other_agent.pos[1]) is not None
            if not on_road and not on_crosswalk: continue
        vec_to_other = other_agent.pos - ego_pos
        dist_to_other = np.linalg.norm(vec_to_other)
        if dist_to_other > perception_range: continue
        if np.dot(vec_to_other, ego_heading_vector) <= 0: continue
        lateral_vector = np.array([-ego_heading_vector[1], ego_heading_vector[0], 0])
        if abs(np.dot(vec_to_other, lateral_vector)) > params['lane_width'] / 2.0: continue
        if dist_to_other < closest_dist:
            closest_dist = dist_to_other
            speed_of_closest = max(0, np.dot(other_agent.velocity_vector, ego_heading_vector))
    return speed_of_closest

def calculate_grid_reference_speed(agent, candidate_grid_center_pos, all_agents, params):
    """
    Assign a local speed to a candidate grid with distance-decayed influence
    from nearby agents. Influence is stronger when closer and mostly from
    agents ahead of ego's heading.
    """
    base_speed = agent.params.get("desired_speed_agent", agent.max_speed)
    influence_radius = max(params.get("speed_influence_radius", 10.0), params["epsilon"])
    close_cap_radius = max(params.get("speed_influence_close_cap_radius", 3.0), params["epsilon"])
    base_weight = max(params.get("speed_influence_base_weight", 1.0), params["epsilon"])
    decay_factor = max(params.get("speed_influence_decay_factor", 0.6), params["epsilon"])
    decay_den = max(influence_radius * decay_factor, params["epsilon"])

    weighted_speed_sum = 0.0
    weight_sum = 0.0
    close_caps = []
    heading_vec = np.array([np.cos(agent.heading_angle_rad), np.sin(agent.heading_angle_rad), 0.0])

    for other_agent in all_agents:
        if other_agent.id == agent.id or other_agent.reached_destination:
            continue
        if agent.agent_type != "drone" and other_agent.agent_type == "drone":
            continue

        dist = np.linalg.norm((other_agent.pos - candidate_grid_center_pos)[:2])
        if dist > influence_radius:
            continue

        vec_ego_to_other = other_agent.pos - agent.pos
        ego_to_other_norm = np.linalg.norm(vec_ego_to_other[:2])
        if ego_to_other_norm > params["epsilon"]:
            forwardness = np.dot(vec_ego_to_other[:2] / ego_to_other_norm, heading_vec[:2])
            directional_factor = 0.25 + 0.75 * max(0.0, forwardness)
        else:
            directional_factor = 1.0

        weight = np.exp(-dist / decay_den) * directional_factor
        weighted_speed_sum += weight * other_agent.speed
        weight_sum += weight

        if dist <= close_cap_radius:
            close_caps.append(other_agent.speed)

    if weight_sum > params["epsilon"]:
        neighbor_speed = weighted_speed_sum / weight_sum
        v_ref = (base_weight * base_speed + weight_sum * neighbor_speed) / (base_weight + weight_sum)
    else:
        v_ref = base_speed

    if close_caps:
        v_ref = min(v_ref, min(close_caps))

    return float(np.clip(v_ref, 0.0, agent.max_speed))

def calculate_directional_alignment(agent, candidate_pos, final_dest, params):
    vec_to_candidate = candidate_pos - agent.pos
    dist_to_candidate = np.linalg.norm(vec_to_candidate)
    if dist_to_candidate < params["epsilon"]:
        vec_to_dest_overall = final_dest - agent.pos
        if np.linalg.norm(vec_to_dest_overall) < params["epsilon"]: return params["S_theta"] * params["eta_i"]
        dir_to_dest_overall = vec_to_dest_overall / (np.linalg.norm(vec_to_dest_overall) + params["epsilon"])
        current_vel_norm = np.linalg.norm(agent.velocity_vector)
        if current_vel_norm > 0: current_heading_vector = agent.velocity_vector / current_vel_norm
        else: current_heading_vector = np.array([np.cos(agent.heading_angle_rad), np.sin(agent.heading_angle_rad), 0])
        dot_prod_overall = np.dot(current_heading_vector, dir_to_dest_overall)
        return params["S_theta"] * params["eta_i"] * max(0, dot_prod_overall)
    proposed_heading_vector = vec_to_candidate / dist_to_candidate
    vec_candidate_to_final_dest = final_dest - candidate_pos
    dist_candidate_to_final_dest = np.linalg.norm(vec_candidate_to_final_dest)
    if dist_candidate_to_final_dest < params["epsilon"]: return params["S_theta"] * params["eta_i"]
    dir_candidate_to_final_dest = vec_candidate_to_final_dest / dist_candidate_to_final_dest
    cos_theta = np.clip(np.dot(proposed_heading_vector, dir_candidate_to_final_dest), -1.0, 1.0)
    return params["S_theta"] * params["eta_i"] * max(0, cos_theta)

def calculate_speed_alignment(agent, candidate_grid_center_pos, all_agents, params, v_ref_g=None):
    dist_to_candidate_grid = np.linalg.norm(candidate_grid_center_pos - agent.pos)
    candidate_speed_to_reach_grid = np.clip(dist_to_candidate_grid / params["dt"], 0, agent.max_speed)
    if v_ref_g is None:
        if agent.agent_type in ['pedestrian', 'drone']:
            v_ref_g = agent.max_speed
        else:
            v_ref_g = get_speed_of_lane(
                candidate_grid_center_pos,
                agent,
                all_agents,
                params["kappa_perception_horizon_factor"] * agent.max_speed,
                agent.params.get("desired_speed_agent", agent.max_speed),
                params,
            )
    v_agent_at_candidate = candidate_speed_to_reach_grid
    if abs(v_agent_at_candidate) < params["epsilon"]: rho_g = 1000.0 if abs(v_ref_g) > params["epsilon"] else 1.0
    else: rho_g = v_ref_g / v_agent_at_candidate
    rho_g = max(params["epsilon"], rho_g)
    exponent = (params["xi_i"] - 1.0) / 2.0
    utility = 0
    if exponent >= 0 or rho_g > 0:
        try:
            if rho_g <= 0 and exponent < 0: pass
            else: utility = params["S_v"] * (rho_g / (1.0 + rho_g**exponent))
        except (ValueError, OverflowError, ZeroDivisionError): pass
    return utility

def calculate_distance_reward(candidate_grid_center_pos, final_destination_pos, current_agent_speed, params):
    diff = np.abs(candidate_grid_center_pos - final_destination_pos)
    d_eff = params["w_x"] * diff[0] + params["w_y"] * diff[1] + params["w_z"] * diff[2]
    H_p = max(params["kappa_perception_horizon_factor"] * current_agent_speed, params["min_perception_horizon_abs"], params["epsilon"])
    ratio_d_eff_H_p = d_eff / H_p
    return params["S_d"] / (1.0 + ratio_d_eff_H_p**params["gamma"])

def calculate_lane_keeping_reward(agent, candidate_pos, params):
    if agent.agent_type == 'vehicle' or (agent.agent_type == 'bike' and params.get("bike_behavior_mode") == "road"):
        road_segment = get_road_segment_for_point(candidate_pos[0], candidate_pos[1])
        if not road_segment: return 0.0
        od_pair = (agent.origin_key, agent.destination_key)
        if 'INTERSECTION' in road_segment:
            path_key = (od_pair, road_segment)
            if path_key in _INTERSECTION_BEZIER_CONTROLS:
                control_points = _INTERSECTION_BEZIER_CONTROLS[path_key]
                lateral_deviation = _calculate_min_dist_to_bezier(candidate_pos, control_points)
                normalized_deviation = lateral_deviation / (params["lane_width"] / 2.0)
                return params["w_ell"] / (1.0 + normalized_deviation ** params["beta"])
            return 0.0
        if road_segment not in OD_LANE_KEEPING_ROADS.get(od_pair, []): return 0.0
        is_vertical = 'N' in road_segment or 'S' in road_segment or 'CONN' in road_segment
        axis_coord = candidate_pos[0] if is_vertical else candidate_pos[1]
        offset = ROADS[road_segment]['x_min'] if is_vertical else ROADS[road_segment]['y_min']
        local_coord = axis_coord - offset
        lane_idx = int(local_coord // params["lane_width"])
        local_lane_center = (lane_idx + 0.5) * params["lane_width"]
        global_lane_center = local_lane_center + offset
        lateral_deviation = abs(axis_coord - global_lane_center)
        normalized_deviation = lateral_deviation / (params["lane_width"] / 2.0)
        return params["w_ell"] / (1.0 + normalized_deviation ** params["beta"])

    elif agent.agent_type == 'pedestrian':
        on_crosswalk = get_crosswalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
        on_sidewalk = get_sidewalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
        return params["w_ell"] if on_crosswalk or on_sidewalk else 0.0

    elif agent.agent_type == 'bike' and params.get("bike_behavior_mode") == "sidewalk":
        is_setting_2 = params.get("environment_setting", 1) == 2
        if is_setting_2:
            # Setting 2: Reward bike lanes AND crosswalks (for crossing intersections)
            on_bike_lane = get_bike_lane_for_point(candidate_pos[0], candidate_pos[1]) is not None
            on_crosswalk = get_crosswalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
            return params["w_ell"] if on_bike_lane or on_crosswalk else 0.0
        else:
            # Setting 1 (no bike lanes): Reward sidewalks/crosswalks
            on_crosswalk = get_crosswalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
            on_sidewalk = get_sidewalk_for_point(candidate_pos[0], candidate_pos[1]) is not None
            return params["w_ell"] if on_crosswalk or on_sidewalk else 0.0

    elif agent.agent_type == 'drone':
        xy_dist_to_dest = np.linalg.norm(candidate_pos[:2] - agent.destination_pos[:2])
        cruise_alt = params["drone_cruise_altitude"]
        landing_radius = params.get("drone_glide_path_radius", 10.0)
        station_radius = 1.0
        target_alt = 0.0
        if xy_dist_to_dest <= station_radius: target_alt = 0.0
        elif xy_dist_to_dest <= landing_radius:
            t = (landing_radius - xy_dist_to_dest) / (landing_radius - station_radius)
            target_alt = cruise_alt * np.cos(t * (np.pi / 2))
        else: target_alt = cruise_alt
        z_candidate = candidate_pos[2]
        deviation = abs(z_candidate - target_alt)
        normalized_deviation = deviation / 1.0 
        return params["w_ell_drone"] / (1.0 + normalized_deviation ** params["beta"])
        
    return 0.0

def calculate_collision_penalty(ego_agent, candidate_grid_center_pos, all_agents, params, time_to_reach_candidate):
    """Bounded collision occupancy probability around candidate point g."""
    delta_t = max(time_to_reach_candidate, params["dt"])
    sigma_x = max(params.get("collision_sigma_x", 1.0), params["epsilon"])
    sigma_y = max(params.get("collision_sigma_y", 1.0), params["epsilon"])
    sigma_z = max(params.get("collision_sigma_z", 1.0), params["epsilon"])

    h_p = max(
        params["kappa_perception_horizon_factor"] * max(ego_agent.speed, params["epsilon"]),
        params["min_perception_horizon_abs"],
    )
    p_i_g = 0.0

    for other_agent in all_agents:
        if other_agent.id == ego_agent.id:
            continue

        if other_agent.reached_destination:
            mu = other_agent.pos.copy()
        else:
            mu = other_agent.pos + other_agent.velocity_vector * delta_t

        if np.linalg.norm(other_agent.pos - ego_agent.pos) > h_p and np.linalg.norm(mu - candidate_grid_center_pos) > h_p:
            continue

        types = frozenset({ego_agent.agent_type, other_agent.agent_type})
        if types == {'vehicle'}: safety_dist = params['vv_safety_distance']
        elif types == {'pedestrian'}: safety_dist = params['pp_safety_distance']
        elif types == {'drone'}: safety_dist = params['dd_safety_distance']
        elif types == {'bike'}: safety_dist = params['bb_safety_distance']
        elif types == {'vehicle', 'pedestrian'}: safety_dist = params['vp_safety_distance']
        elif types == {'vehicle', 'drone'}: safety_dist = params['vd_safety_distance']
        elif types == {'vehicle', 'bike'}: safety_dist = params['vb_safety_distance']
        elif types == {'pedestrian', 'drone'}: safety_dist = params['pd_safety_distance']
        elif types == {'pedestrian', 'bike'}: safety_dist = params['pb_safety_distance']
        elif types == {'drone', 'bike'}: safety_dist = params['db_safety_distance']
        else: safety_dist = 2.0

        collision_buffer = (ego_agent.length / 2.0) + (other_agent.length / 2.0) + safety_dist
        sigma_x_eff = sigma_x + collision_buffer
        sigma_y_eff = sigma_y + collision_buffer
        sigma_z_eff = sigma_z + collision_buffer

        diff = candidate_grid_center_pos - mu
        mahalanobis = (
            (diff[0] ** 2) / (sigma_x_eff ** 2)
            + (diff[1] ** 2) / (sigma_y_eff ** 2)
            + (diff[2] ** 2) / (sigma_z_eff ** 2)
        )
        p_j_g = np.exp(-0.5 * mahalanobis)
        # Union probability update: keeps p_i_g in [0, 1].
        p_i_g = 1.0 - (1.0 - p_i_g) * (1.0 - p_j_g)

    return float(np.clip(p_i_g, 0.0, 1.0))