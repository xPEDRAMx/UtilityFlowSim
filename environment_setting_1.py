# environment_setting_1.py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random
from config import CONFIG

def define_geometry(config):
    all_roads, all_crosswalks, all_sidewalks, all_bike_lanes = {}, {}, {}, {}
    intersections_data = []

    i1_center_y = config['road_segment_length'] + config['crosswalk_width'] + config['one_way_road_width_horizontal']
    r1, c1, s1, b1 = generate_single_intersection_geometry(config, 'I1', i1_center_y, '4_leg')
    all_roads.update(r1); all_crosswalks.update(c1); all_sidewalks.update(s1); all_bike_lanes.update(b1)
    intersections_data.append({'id': 'I1', 'center_y': i1_center_y, 'type': '4_leg'})

    i2_center_y = r1['I1_NBD']['y_max'] + config["connecting_road_length"] + config['crosswalk_width'] + config['one_way_road_width_horizontal']
    r2, c2, s2, b2 = generate_single_intersection_geometry(config, 'I2', i2_center_y, '3_leg_t_no_west')
    all_roads.update(r2); all_crosswalks.update(c2); all_sidewalks.update(s2); all_bike_lanes.update(b2)
    intersections_data.append({'id': 'I2', 'center_y': i2_center_y, 'type': '3_leg_t_no_west'})

    conn_y_min = all_roads['I1_NBD']['y_max']
    conn_y_max = all_roads['I2_SBD']['y_min']
    ix_size_vert_half = config['intersection_size'] / 2
    sw_width = config['sidewalk_width']
    
    all_roads['CONN_I2S_I1N'] = {'x_min': 0, 'x_max': ix_size_vert_half, 'y_min': conn_y_min, 'y_max': conn_y_max}
    all_roads['CONN_I1N_I2S'] = {'x_min': ix_size_vert_half, 'x_max': 2 * ix_size_vert_half, 'y_min': conn_y_min, 'y_max': conn_y_max}
    all_sidewalks['CONN_SW_W'] = {'x_min': -sw_width, 'x_max': 0, 'y_min': conn_y_min, 'y_max': conn_y_max}
    all_sidewalks['CONN_SW_E'] = {'x_min': 2 * ix_size_vert_half, 'x_max': 2 * ix_size_vert_half + sw_width, 'y_min': conn_y_min, 'y_max': conn_y_max}

    return all_roads, all_crosswalks, all_sidewalks, intersections_data, all_bike_lanes

def generate_single_intersection_geometry(config, ix_id, center_y, ix_type):
    roads, crosswalks, sidewalks, bike_lanes = {}, {}, {}, {}
    road_len, cw_width, sw_width = config["road_segment_length"], config["crosswalk_width"], config["sidewalk_width"]
    one_way_w_vert, one_way_w_horiz = config["one_way_road_width_vertical"], config["one_way_road_width_horizontal"]
    ix_size_x = 2 * one_way_w_vert
    x_w, x_e = 0, ix_size_x
    y_s_horiz, y_n_horiz = center_y - one_way_w_horiz, center_y + one_way_w_horiz
    
    roads[f'{ix_id}_INTERSECTION'] = {'x_min': x_w, 'x_max': x_e, 'y_min': y_s_horiz, 'y_max': y_n_horiz}
    crosswalks[f'{ix_id}_CW_S'] = {'x_min': x_w, 'x_max': x_e, 'y_min': y_s_horiz - cw_width, 'y_max': y_s_horiz}
    crosswalks[f'{ix_id}_CW_N'] = {'x_min': x_w, 'x_max': x_e, 'y_min': y_n_horiz, 'y_max': y_n_horiz + cw_width}
    crosswalks[f'{ix_id}_CW_E'] = {'x_min': x_e, 'x_max': x_e + cw_width, 'y_min': y_s_horiz, 'y_max': y_n_horiz}
    if ix_type != '3_leg_t_no_west':
        crosswalks[f'{ix_id}_CW_W'] = {'x_min': x_w - cw_width, 'x_max': x_w, 'y_min': y_s_horiz, 'y_max': y_n_horiz}
    roads[f'{ix_id}_SBO'] = {'x_min': x_w, 'x_max': one_way_w_vert, 'y_min': crosswalks[f'{ix_id}_CW_N']['y_max'], 'y_max': crosswalks[f'{ix_id}_CW_N']['y_max'] + road_len}
    roads[f'{ix_id}_NBD'] = {'x_min': one_way_w_vert, 'x_max': x_e, 'y_min': crosswalks[f'{ix_id}_CW_N']['y_max'], 'y_max': crosswalks[f'{ix_id}_CW_N']['y_max'] + road_len}
    roads[f'{ix_id}_NBO'] = {'x_min': one_way_w_vert, 'x_max': x_e, 'y_min': crosswalks[f'{ix_id}_CW_S']['y_min'] - road_len, 'y_max': crosswalks[f'{ix_id}_CW_S']['y_min']}
    roads[f'{ix_id}_SBD'] = {'x_min': x_w, 'x_max': one_way_w_vert, 'y_min': crosswalks[f'{ix_id}_CW_S']['y_min'] - road_len, 'y_max': crosswalks[f'{ix_id}_CW_S']['y_min']}
    roads[f'{ix_id}_EBD'] = {'x_min': crosswalks[f'{ix_id}_CW_E']['x_max'], 'x_max': crosswalks[f'{ix_id}_CW_E']['x_max'] + road_len, 'y_min': center_y - one_way_w_horiz, 'y_max': center_y}
    roads[f'{ix_id}_WBO'] = {'x_min': crosswalks[f'{ix_id}_CW_E']['x_max'], 'x_max': crosswalks[f'{ix_id}_CW_E']['x_max'] + road_len, 'y_min': center_y, 'y_max': center_y + one_way_w_horiz}
    if ix_type != '3_leg_t_no_west':
        roads[f'{ix_id}_EBO'] = {'x_min': crosswalks[f'{ix_id}_CW_W']['x_min'] - road_len, 'x_max': crosswalks[f'{ix_id}_CW_W']['x_min'], 'y_min': center_y - one_way_w_horiz, 'y_max': center_y}
        roads[f'{ix_id}_WBD'] = {'x_min': crosswalks[f'{ix_id}_CW_W']['x_min'] - road_len, 'x_max': crosswalks[f'{ix_id}_CW_W']['x_min'], 'y_min': center_y, 'y_max': center_y + one_way_w_horiz}
    
    sidewalks[f'{ix_id}_SW_N_W'] = {'x_min': -sw_width, 'x_max': x_w, 'y_min': crosswalks[f'{ix_id}_CW_N']['y_max'], 'y_max': crosswalks[f'{ix_id}_CW_N']['y_max'] + road_len}
    sidewalks[f'{ix_id}_SW_N_E'] = {'x_min': x_e, 'x_max': x_e + sw_width, 'y_min': crosswalks[f'{ix_id}_CW_N']['y_max'], 'y_max': crosswalks[f'{ix_id}_CW_N']['y_max'] + road_len}
    sidewalks[f'{ix_id}_SW_S_W'] = {'x_min': -sw_width, 'x_max': x_w, 'y_min': crosswalks[f'{ix_id}_CW_S']['y_min'] - road_len, 'y_max': crosswalks[f'{ix_id}_CW_S']['y_min']}
    sidewalks[f'{ix_id}_SW_S_E'] = {'x_min': x_e, 'x_max': x_e + sw_width, 'y_min': crosswalks[f'{ix_id}_CW_S']['y_min'] - road_len, 'y_max': crosswalks[f'{ix_id}_CW_S']['y_min']}
    sidewalks[f'{ix_id}_SW_E_N'] = {'x_min': crosswalks[f'{ix_id}_CW_E']['x_max'], 'x_max': crosswalks[f'{ix_id}_CW_E']['x_max'] + road_len, 'y_min': y_n_horiz, 'y_max': y_n_horiz + sw_width}
    sidewalks[f'{ix_id}_SW_E_S'] = {'x_min': crosswalks[f'{ix_id}_CW_E']['x_max'], 'x_max': crosswalks[f'{ix_id}_CW_E']['x_max'] + road_len, 'y_min': y_s_horiz - sw_width, 'y_max': y_s_horiz}
    
    sidewalks[f'{ix_id}_CORNER_SE'] = {'x_min': x_e, 'x_max': x_e + cw_width, 'y_min': y_s_horiz - sw_width, 'y_max': y_s_horiz}
    sidewalks[f'{ix_id}_CORNER_NE'] = {'x_min': x_e, 'x_max': x_e + cw_width, 'y_min': y_n_horiz, 'y_max': y_n_horiz + sw_width}

    if ix_type != '3_leg_t_no_west':
        sidewalks[f'{ix_id}_SW_W_N'] = {'x_min': crosswalks[f'{ix_id}_CW_W']['x_min'] - road_len, 'x_max': crosswalks[f'{ix_id}_CW_W']['x_min'], 'y_min': y_n_horiz, 'y_max': y_n_horiz + sw_width}
        sidewalks[f'{ix_id}_SW_W_S'] = {'x_min': crosswalks[f'{ix_id}_CW_W']['x_min'] - road_len, 'x_max': crosswalks[f'{ix_id}_CW_W']['x_min'], 'y_min': y_s_horiz - sw_width, 'y_max': y_s_horiz}
        sidewalks[f'{ix_id}_CORNER_SW'] = {'x_min': x_w - cw_width, 'x_max': x_w, 'y_min': y_s_horiz - sw_width, 'y_max': y_s_horiz}
        sidewalks[f'{ix_id}_CORNER_NW'] = {'x_min': x_w - cw_width, 'x_max': x_w, 'y_min': y_n_horiz, 'y_max': y_n_horiz + sw_width}
    else: # FIX: Add continuous vertical sidewalk for T-junction
        sidewalks[f'{ix_id}_SW_W_MID'] = {'x_min': -sw_width, 'x_max': 0, 'y_min': y_s_horiz, 'y_max': y_n_horiz}

    return roads, crosswalks, sidewalks, bike_lanes


def _create_corner_drone_stations(roads_geom, intersections_geom, config):
    corners = []
    num_stations = config.get("num_drone_stations", 0)
    cw_width = config["crosswalk_width"]
    # Get the new offset from the config, with a fallback default of 2.0
    offset = config.get("drone_station_placement_offset", 2.0)

    for ix in intersections_geom:
        ix_bounds = roads_geom[f'{ix["id"]}_INTERSECTION']
        # Apply the offset to push stations further from the intersection corners
        ne_corner = [ix_bounds['x_max'] + cw_width + offset, ix_bounds['y_max'] + cw_width + offset]
        se_corner = [ix_bounds['x_max'] + cw_width + offset, ix_bounds['y_min'] - cw_width - offset]
        nw_corner = [ix_bounds['x_min'] - cw_width - offset, ix_bounds['y_max'] + cw_width + offset]
        sw_corner = [ix_bounds['x_min'] - cw_width - offset, ix_bounds['y_min'] - cw_width - offset]

        # MODIFIED: Consider all four corners for any intersection type.
        # This fixes the missing station at the 3-leg T-junction.
        corners.extend([sw_corner, nw_corner, ne_corner, se_corner])

    if not corners or num_stations == 0: return {}
    selected_corners = random.sample(corners, min(len(corners), num_stations))
    return {f'DS_{i+1}': {'pos': np.array([c[0], c[1], 0.0])} for i, c in enumerate(selected_corners)}


VALID_ORIGINS = ['I1_NBO', 'I1_EBO', 'I1_WBO', 'I2_SBO', 'I2_WBO']
VALID_DESTINATIONS = ['I1_SBD', 'I1_EBD', 'I1_WBD', 'I2_NBD', 'I2_EBD']
OD_LANE_KEEPING_ROADS = {
    ('I1_NBO', 'I1_EBD'): ['I1_NBO', 'I1_EBD'], ('I1_NBO', 'I1_WBD'): ['I1_NBO', 'I1_WBD'],
    ('I1_NBO', 'I2_NBD'): ['I1_NBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_NBD'],
    ('I1_NBO', 'I2_EBD'): ['I1_NBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_EBD'],
    ('I2_SBO', 'I2_EBD'): ['I2_SBO', 'I2_EBD'],
    ('I2_SBO', 'I1_SBD'): ['I2_SBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_SBD'],
    ('I2_SBO', 'I1_EBD'): ['I2_SBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_EBD'],
    ('I2_SBO', 'I1_WBD'): ['I2_SBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_WBD'],
    ('I1_EBO', 'I1_SBD'): ['I1_EBO', 'I1_SBD'], ('I1_EBO', 'I1_EBD'): ['I1_EBO', 'I1_EBD'],
    ('I1_EBO', 'I2_NBD'): ['I1_EBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_NBD'],
    ('I1_EBO', 'I2_EBD'): ['I1_EBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_EBD'],
    ('I1_WBO', 'I1_SBD'): ['I1_WBO', 'I1_SBD'], ('I1_WBO', 'I1_WBD'): ['I1_WBO', 'I1_WBD'],
    ('I1_WBO', 'I2_NBD'): ['I1_WBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_NBD'],
    ('I1_WBO', 'I2_EBD'): ['I1_WBO', 'I1_NBD', 'CONN_I1N_I2S', 'I2_NBO', 'I2_EBD'],
    ('I2_WBO', 'I2_NBD'): ['I2_WBO', 'I2_NBD'],
    ('I2_WBO', 'I1_SBD'): ['I2_WBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_SBD'],
    ('I2_WBO', 'I1_EBD'): ['I2_WBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_EBD'],
    ('I2_WBO', 'I1_WBD'): ['I2_WBO', 'I2_SBD', 'CONN_I2S_I1N', 'I1_SBO', 'I1_WBD'],
}

ROADS, CROSSWALKS, SIDEWALKS, INTERSECTIONS, BIKE_LANES = define_geometry(CONFIG)
DRONE_STATIONS = _create_corner_drone_stations(ROADS, INTERSECTIONS, CONFIG)

def get_road_segment_for_point(x, y):
    for name, bounds in ROADS.items():
        if bounds['x_min'] <= x < bounds['x_max'] and bounds['y_min'] <= y < bounds['y_max']: return name
    return None
def get_crosswalk_for_point(x, y):
    for name, bounds in CROSSWALKS.items():
        if bounds['x_min'] <= x < bounds['x_max'] and bounds['y_min'] <= y < bounds['y_max']: return name
    return None
def get_sidewalk_for_point(x, y):
    for name, bounds in SIDEWALKS.items():
        if bounds['x_min'] <= x < bounds['x_max'] and bounds['y_min'] <= y < bounds['y_max']: return name
    return None
def get_bike_lane_for_point(x, y):
    return None

def plot_environment(config, ax=None, show_street_names=False, show_road_names=False):
    if ax is None: fig, ax = plt.subplots(figsize=(14, 14))
    road_color, sidewalk_color = '#666666', '#B0B0B0'
    all_x, all_y = [], []
    for collection in [ROADS, SIDEWALKS, CROSSWALKS]:
        for _, bounds in collection.items():
            all_x.extend([bounds['x_min'], bounds['x_max']])
            all_y.extend([bounds['y_min'], bounds['y_max']])
    
    for _, bounds in ROADS.items(): ax.add_patch(Rectangle((bounds['x_min'], bounds['y_min']), bounds['x_max'] - bounds['x_min'], bounds['y_max'] - bounds['y_min'], facecolor=road_color, alpha=0.8, zorder=0))
    for _, bounds in SIDEWALKS.items(): ax.add_patch(Rectangle((bounds['x_min'], bounds['y_min']), bounds['x_max'] - bounds['x_min'], bounds['y_max'] - bounds['y_min'], facecolor=sidewalk_color, zorder=0.4))
    for name, bounds in CROSSWALKS.items():
        ax.add_patch(Rectangle((bounds['x_min'], bounds['y_min']), bounds['x_max'] - bounds['x_min'], bounds['y_max'] - bounds['y_min'], facecolor='#BBBBBB', alpha=1, zorder=0.5))
        stripe_spacing = 2.0
        if 'S' in name or 'N' in name:
            width = bounds['x_max'] - bounds['x_min']
            num_stripes = int(width / stripe_spacing)
            for i in range(num_stripes):
                x_pos = bounds['x_min'] + (i + 0.5) * width / num_stripes
                ax.plot([x_pos, x_pos], [bounds['y_min'], bounds['y_max']], color='white', linewidth=2.5, zorder=0.6)
        else:
            height = bounds['y_max'] - bounds['y_min']
            num_stripes = int(height / stripe_spacing)
            for i in range(num_stripes):
                y_pos = bounds['y_min'] + (i + 0.5) * height / num_stripes
                ax.plot([bounds['x_min'], bounds['x_max']], [y_pos, y_pos], color='white', linewidth=2.5, zorder=0.6)
    
    for _, station in DRONE_STATIONS.items():
        pos, size = station['pos'], 4.0
        ax.add_patch(Rectangle((pos[0] - size / 2, pos[1] - size / 2), size, size, facecolor='red', zorder=0.7))
            
    lane_w, num_lanes_vert = config["lane_width"], config["num_lanes_per_direction_vertical"]
    one_way_w_vert = config["one_way_road_width_vertical"]
    
    for i in range(1, 2 * num_lanes_vert):
        if i == num_lanes_vert: continue
        ax.axvline(x=i * lane_w, color='white', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
    
    plot_min_y, plot_max_y = (min(all_y) - 10, max(all_y) + 10) if all_y else (-50, 250)
    y_gaps = [[CROSSWALKS[f'{ix["id"]}_CW_S']['y_min'], CROSSWALKS[f'{ix["id"]}_CW_N']['y_max']] for ix in INTERSECTIONS]
    ax.plot([one_way_w_vert, one_way_w_vert], [plot_min_y, y_gaps[0][0]], color='yellow', linestyle='-', linewidth=1, zorder=1)
    ax.plot([one_way_w_vert, one_way_w_vert], [y_gaps[0][1], y_gaps[1][0]], color='yellow', linestyle='-', linewidth=1, zorder=1)
    ax.plot([one_way_w_vert, one_way_w_vert], [y_gaps[1][1], plot_max_y], color='yellow', linestyle='-', linewidth=1, zorder=1)
    
    for spec in INTERSECTIONS:
        center_y, ix_id = spec['center_y'], spec['id']
        start_x_east, end_x_east = CROSSWALKS[f'{ix_id}_CW_E']['x_max'], CROSSWALKS[f'{ix_id}_CW_E']['x_max'] + config['road_segment_length']
        ax.plot([start_x_east, end_x_east], [center_y, center_y], color='yellow', linestyle='-', linewidth=1, zorder=1)
        if spec['type'] != '3_leg_t_no_west':
            start_x_west, end_x_west = CROSSWALKS[f'{ix_id}_CW_W']['x_min'], CROSSWALKS[f'{ix_id}_CW_W']['x_min'] - config['road_segment_length']
            ax.plot([start_x_west, end_x_west], [center_y, center_y], color='yellow', linestyle='-', linewidth=1, zorder=1)

    ax.set_xlabel("X Position (m)"); ax.set_ylabel("Y Position (m)")
    if all_x and all_y: ax.set_xlim(min(all_x) - 10, max(all_x) + 10); ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    ax.set_aspect('equal', adjustable='box'); ax.set_facecolor('#DDDDDD')
    return ax