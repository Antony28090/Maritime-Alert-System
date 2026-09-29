import numpy as np
import pandas as pd
import random
from shapely.geometry import Point, LineString
from src.config import *

from src.geometry import distance_from_polyline, is_sri_lankan_side


# Global to store real data points for seeding
REAL_DATA_POINTS = []

def load_real_seed_data():
    """Loads processed CSVs to build a list of valid start locations."""
    global REAL_DATA_POINTS
    if REAL_DATA_POINTS: return # Already loaded

    import glob
    import os
    print("Loading real data for seeding...")
    files = glob.glob("data/Processed/*.csv")
    points = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Sample a fraction to avoid huge memory usage if necessary,
            # but usually fine to keep all for density estimation.
            # We just need [lat, lon] pairs.
            if 'lat' in df.columns:
                points.extend(df[['lat', 'lon']].values.tolist())
            elif 'cell_ll_lat' in df.columns:
                points.extend(df[['cell_ll_lat', 'cell_ll_lon']].values.tolist())
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if points:
        REAL_DATA_POINTS = points
        print(f"Loaded {len(REAL_DATA_POINTS)} real seed points.")
    else:
        print("Warning: No real data found. Using default locations.")

def get_zone(dist_km):
    if dist_km < DANGER_DIST_KM:
        return "DANGER"
    elif dist_km < CAUTION_DIST_KM:
        return "CAUTION"
    else:
        return "SAFE"


# Near-miss trips loiter this far (km) on the Indian side of the line.  The
# standoff is sampled LOG-UNIFORMLY over the range so the 2-4 km decade is not
# starved: the false-alert experiment must test the "hold 3.5 km off" rule on
# trips that were never designed around it.  Trips that stray into the 2 km
# DANGER buffer are rejected by generate_near_miss_trajectory(); the standoff
# is drawn once per trip and re-used across retries so rejection does not
# reshape the standoff distribution.
NEAR_MISS_OFFSET_KM = (2.2, 12.0)
NEAR_MISS_LOITER_DEG = (0.08, 0.15)   # length of the along-boundary trawl leg

# Default step lengths (deg/tick) when the caller does not pin one.  These are
# the legacy values: forced trips ran faster than the rest.  The evaluation
# harness always passes base_speed explicitly so every mode in a sweep
# configuration shares one step length.
DEFAULT_SPEED = {"normal": 0.008, "forced": 0.012, "near_miss": 0.008}
SPEED_PROFILES = [0.008, 0.012, 0.018]

MOTION_PROFILES = ("smooth", "trawl")
# 'trawl' motion parameters (all per tick):
TRAWL = {
    "pause_p": 0.02,            # probability of starting an on-station pause
    "pause_len": (5, 15),       # pause duration, ticks (uniform integer)
    "pause_factor": 0.1,        # speed multiplier while paused
    "p_track_to_circle": 0.03,  # heading regime: bearing-tracking -> slow circling
    "p_circle_to_track": 0.10,  # heading regime: circling -> bearing-tracking
    "circle_rate": 0.05,        # rad/tick constant turn while circling (sign random)
    "jump_p": 0.01,             # probability of an abrupt heading change
    "jump_mag": (0.5, 1.0),     # magnitude of the abrupt change, rad (sign random)
}


def sample_near_miss_offset_km(rng=random):
    lo, hi = NEAR_MISS_OFFSET_KM
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _land_avoid_waypoints(start_lat, start_lon, target_lat):
    """Waypoints that keep generated tracks off Point Calimere / Pamban Island."""
    wps = []
    # Point Calimere (Vedaranyam) is approx 10.29N, 79.85E.
    if start_lat > 10.3 and target_lat < 10.3:
        wps.append([10.30, 80.0])
    # Avoid Pamban Island / Dhanushkodi landmass for all Palk Bay ports
    if 9.2 < start_lat < 10.3 and start_lon < 79.4:
        if target_lat < 9.3:
            # Route strictly North-East through Palk Bay to safely clear the island tip
            wps.append([9.30, 79.45])
    return wps


def generate_trajectory(trip_id, n_points=100, force_crossing=False, mode=None,
                        base_speed=None, motion_profile="smooth",
                        near_miss_offset_km=None):
    """
    Generates a realistic vessel path with drift, steering noise, and sensor noise.

    base_speed      - mean step length in deg/tick.  None -> DEFAULT_SPEED[mode].
    motion_profile  - 'smooth': waypoint tracking with steering wiggle only.
                      'trawl' : adds on-station pauses (speed x 0.1 for 5-15
                                ticks, p = 0.02/tick), a two-state Markov heading
                                regime that alternates bearing-tracking with a
                                slow constant-rate circling turn (+-0.05 rad/tick),
                                and abrupt heading changes of 0.5-1.0 rad
                                (p = 0.01/tick).  See TRAWL.
    near_miss_offset_km - standoff of the near-miss loiter point from the line;
                      None -> log-uniform draw over NEAR_MISS_OFFSET_KM.

    mode:
      'normal'    - head to a destination near the boundary (seeded from real AIS
                    anchors), no crossing intended.  (default)
      'forced'    - destination pushed 5.5-8.9 km onto the Sri Lankan side, so the
                    trip crosses.  Equivalent to force_crossing=True.
      'near_miss' - approach to 3-4.5 km from the line on the Indian side, trawl
                    parallel to it, then return to the start port.  Never intended
                    to cross; used to measure false-alert rate.  Callers should use
                    generate_near_miss_trajectory(), which enforces the no-DANGER
                    invariant by accept/reject.
    """
    if mode is None:
        mode = 'forced' if force_crossing else 'normal'
    force_crossing = (mode == 'forced')
    near_miss = (mode == 'near_miss')
    if motion_profile not in MOTION_PROFILES:
        raise ValueError(f"motion_profile must be one of {MOTION_PROFILES}")
    trawl = (motion_profile == 'trawl')
    if base_speed is None:
        base_speed = DEFAULT_SPEED[mode]

    if not REAL_DATA_POINTS:
        load_real_seed_data()

    # Start from any Coastal Port (Tamil Nadu) randomly
    loc_name = random.choice(list(START_LOCATIONS.keys()))

    base_loc = START_LOCATIONS[loc_name]
    start_lat = base_loc["lat"]
    start_lon = base_loc["lon"]

    # Find the IMBL segment closest to the start port
    best_seg_idx = 0
    best_dist = float('inf')
    for si in range(len(IMBL_POINTS) - 1):
        mid_lat = (IMBL_POINTS[si][0] + IMBL_POINTS[si+1][0]) / 2
        mid_lon = (IMBL_POINTS[si][1] + IMBL_POINTS[si+1][1]) / 2
        d = np.sqrt((mid_lat - start_lat)**2 + (mid_lon - start_lon)**2)
        if d < best_dist:
            best_dist = d
            best_seg_idx = si

    p1 = IMBL_POINTS[best_seg_idx]
    p2 = IMBL_POINTS[best_seg_idx + 1]

    # Random point on this nearest segment
    t = random.random()
    imbl_target_lat = p1[0] + t * (p2[0] - p1[0])
    imbl_target_lon = p1[1] + t * (p2[1] - p1[1])

    # Unit normal to the segment (lat, lon components) and unit along-segment vector
    dx = p2[1] - p1[1]
    dy = p2[0] - p1[0]
    perp_lon, perp_lat = dy, -dx
    length = np.sqrt(perp_lon**2 + perp_lat**2)
    if length > 0:
        perp_lon /= length
        perp_lat /= length
        along_lat, along_lon = dy / length, dx / length
    else:
        along_lat, along_lon = 0.0, 0.0

    # Orient the normal: does +perp point to the Sri Lankan side?
    test_lat = imbl_target_lat + (perp_lat * 0.01)
    test_lon = imbl_target_lon + (perp_lon * 0.01)
    perp_is_sl = is_sri_lankan_side([test_lat, test_lon], IMBL_POINTS)

    pre_waypoints = []     # inserted before the primary target (near-miss only)
    extra_waypoints = []   # appended after the primary target (near-miss only)

    if force_crossing:
        # Push the target BEYOND the IMBL into Sri Lankan waters, 5-8 km
        if not perp_is_sl:
            perp_lon, perp_lat = -perp_lon, -perp_lat
        push_dist = random.uniform(0.05, 0.08)
        target_lat = imbl_target_lat + (perp_lat * push_dist)
        target_lon = imbl_target_lon + (perp_lon * push_dist)

    elif near_miss:
        # Loiter point on the INDIAN side, inside the caution band
        if perp_is_sl:
            perp_lon, perp_lat = -perp_lon, -perp_lat
        if near_miss_offset_km is None:
            near_miss_offset_km = sample_near_miss_offset_km()
        offset_deg = near_miss_offset_km / 111.32
        target_lat = imbl_target_lat + (perp_lat * offset_deg)
        target_lon = imbl_target_lon + (perp_lon * offset_deg)

        # Trawl leg parallel to the boundary, then go home.  A turn-in waypoint
        # ~4 km further out and ~6.7 km "upstream" along the boundary makes the
        # final approach diagonal rather than head-on, so the slow steering
        # controller does not carry the vessel through the loiter point and
        # into the DANGER buffer.
        sign = random.choice([-1.0, 1.0])
        loiter_len = random.uniform(*NEAR_MISS_LOITER_DEG)
        turn_in_lat = target_lat + perp_lat * (4.0 / 111.32) - sign * along_lat * 0.06
        turn_in_lon = target_lon + perp_lon * (4.0 / 111.32) - sign * along_lon * 0.06
        loiter_end_lat = target_lat + sign * along_lat * loiter_len
        loiter_end_lon = target_lon + sign * along_lon * loiter_len
        pre_waypoints = [[turn_in_lat, turn_in_lon]]
        extra_waypoints = [[loiter_end_lat, loiter_end_lon], [start_lat, start_lon]]

    else:
        # Normal trip: target near IMBL with some variation
        if REAL_DATA_POINTS and random.random() < 0.9:
            seed_pt = random.choice(REAL_DATA_POINTS)
            target_lat = seed_pt[0]
            target_lon = seed_pt[1]
            # If too close to start, use IMBL point instead
            if abs(target_lat - start_lat) < 0.05 and abs(target_lon - start_lon) < 0.05:
                target_lat = imbl_target_lat + random.uniform(-0.05, 0.05)
                target_lon = imbl_target_lon + random.uniform(-0.05, 0.05)
        else:
            target_lat = imbl_target_lat + random.uniform(-0.05, 0.05)
            target_lon = imbl_target_lon + random.uniform(-0.05, 0.05)

    # Navigation Checkpoints (Avoid Land)
    waypoints = _land_avoid_waypoints(start_lat, start_lon, target_lat)
    waypoints.extend(pre_waypoints)
    waypoints.append([target_lat, target_lon])
    waypoints.extend(extra_waypoints)

    # Near-miss trips need a tighter waypoint-reach radius and quicker steering:
    # with the default 0.02 deg (~2.2 km) radius the vessel switches waypoint
    # while still 2 km short of the loiter point and coasts into the buffer.
    reach_deg = 0.01 if near_miss else 0.02
    steer_factor = 0.2 if near_miss else (0.1 if force_crossing else 0.05)
    # Near-miss only: also advance when the vessel has clearly passed its
    # closest approach to the waypoint, so a near miss does not turn into a
    # loop back across the boundary.
    min_dist_to_wp = None

    current_waypoint_idx = 0
    sub_target_lat, sub_target_lon = waypoints[0]


    # 2. Physics / Movement Parameters
    current_lat, current_lon = start_lat, start_lon

    # Calculate initial bearing to target
    d_lat = target_lat - start_lat
    d_lon = target_lon - start_lon
    target_heading = np.arctan2(d_lat, d_lon) # Radians

    current_heading = target_heading

    # Speed setup: base_speed is fixed above (explicit argument or DEFAULT_SPEED)

    # 'trawl' motion state
    pause_left = 0
    regime = 'track'          # 'track' (bearing-tracking) or 'circle'
    circle_sign = 1.0

    # Environmental Drift (Currents/Wind) - Constant for the trip
    drift_lat = np.random.normal(0, 0.0002)
    drift_lon = np.random.normal(0, 0.0002)

    data = []

    for i in range(n_points):
        # A. Update Heading (Steering)
        # Check if we reached current waypoint (within ~2km to avoid overshoot)
        dist_to_wp = np.sqrt((sub_target_lat - current_lat)**2 + (sub_target_lon - current_lon)**2)
        if min_dist_to_wp is None or dist_to_wp < min_dist_to_wp:
            min_dist_to_wp = dist_to_wp
        passed_closest = (near_miss and min_dist_to_wp < 0.02
                          and dist_to_wp > min_dist_to_wp + 0.004)
        if dist_to_wp < reach_deg or passed_closest:
            current_waypoint_idx += 1
            min_dist_to_wp = None
            if current_waypoint_idx < len(waypoints):
                sub_target_lat, sub_target_lon = waypoints[current_waypoint_idx]
            else:
                pass # Just keep coasting / steering towards the end point

        # Calculate bearing to CURRENT waypoint
        d_lat = sub_target_lat - current_lat
        d_lon = sub_target_lon - current_lon
        desired_heading = np.arctan2(d_lat, d_lon)

        # Smooth steering with angle wrap-around handling (steer_factor set above)
        diff_angle = desired_heading - current_heading
        # Normalize diff_angle to [-pi, pi] to take shortest turn
        diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi

        if trawl:
            # (b) two-state Markov heading regime
            if regime == 'track' and random.random() < TRAWL["p_track_to_circle"]:
                regime, circle_sign = 'circle', random.choice([-1.0, 1.0])
            elif regime == 'circle' and random.random() < TRAWL["p_circle_to_track"]:
                regime = 'track'
        if regime == 'track':
            current_heading = current_heading + steer_factor * diff_angle
        else:
            current_heading = current_heading + circle_sign * TRAWL["circle_rate"]
        current_heading += np.random.normal(0, 0.03) # Random steering wiggle (radians)
        if trawl and random.random() < TRAWL["jump_p"]:
            # (c) abrupt heading change
            current_heading += random.choice([-1.0, 1.0]) * random.uniform(*TRAWL["jump_mag"])

        # B. Update Speed
        # If reached final waypoint completely, slow down to drift speed
        if current_waypoint_idx >= len(waypoints) and dist_to_wp < 0.01:
            step_speed = base_speed * 0.1 * np.random.normal(1.0, 0.2)
        else:
            step_speed = base_speed * np.random.normal(1.0, 0.1)
        if trawl:
            # (a) on-station pauses
            if pause_left == 0 and random.random() < TRAWL["pause_p"]:
                pause_left = random.randint(*TRAWL["pause_len"])
            if pause_left > 0:
                step_speed *= TRAWL["pause_factor"]
                pause_left -= 1

        # C. Update Position (Physics)
        # Move in direction of heading
        delta_lat = step_speed * np.sin(current_heading)
        delta_lon = step_speed * np.cos(current_heading)

        # Add drift
        current_lat += delta_lat + drift_lat
        current_lon += delta_lon + drift_lon

        # D. Add Sensor Noise (GPS Error)
        # GPS error is usually few meters, so much smaller than movement
        # 0.00005 deg ~= 5 meters
        recorded_lat = current_lat + np.random.normal(0, 0.00005)
        recorded_lon = current_lon + np.random.normal(0, 0.00005)

        # E. Calculate Zone
        # Pass [lat, lon] to distance_from_polyline
        dist, _ = distance_from_polyline([recorded_lat, recorded_lon], IMBL_POINTS)
        is_sl_side = is_sri_lankan_side([recorded_lat, recorded_lon], IMBL_POINTS)

        if is_sl_side:
            zone = "DANGER" # Always danger if on the wrong side
        else:
            zone = get_zone(dist)

        data.append({
            'trip_id': trip_id,
            'timestamp': i,
            'lat': recorded_lat,
            'lon': recorded_lon,
            'distance_to_imbl': dist,
            'zone': zone
        })

    return data


def generate_near_miss_trajectory(trip_id, n_points=220, max_tries=60, base_speed=None,
                                  motion_profile="smooth", offset_km=None):
    """
    A non-crossing trip that approaches the boundary, trawls along it at a
    standoff drawn log-uniformly from NEAR_MISS_OFFSET_KM, and returns to port.
    Enforces the invariant that no fix is in the DANGER zone (neither on the
    Sri Lankan side nor inside the 2 km buffer); a trip that violates it is
    discarded and regenerated WITH THE SAME STANDOFF, so rejection changes only
    the realisation, not the standoff distribution.

    Returns (trajectory, tries) or (None, tries) if max_tries is exhausted.
    """
    if offset_km is None:
        offset_km = sample_near_miss_offset_km()
    for attempt in range(1, max_tries + 1):
        traj = generate_trajectory(trip_id=trip_id * 100 + attempt,
                                   n_points=n_points, mode='near_miss',
                                   base_speed=base_speed, motion_profile=motion_profile,
                                   near_miss_offset_km=offset_km)
        if all(step['zone'] != 'DANGER' for step in traj):
            for step in traj:
                step['trip_id'] = trip_id
                step['standoff_km'] = offset_km
            return traj, attempt
    return None, max_tries


def main():
    print("Generating synthetic data...")
    all_data = []

    # Generate 500 trips total
    # 50% Normal, 50% Forced Crossing (to learn the aggressive behavior)
    n_trips = 500

    for trip in range(n_trips):
        # Alternate between normal and forced
        is_forced = (trip % 2 == 0)
        all_data.extend(generate_trajectory(trip_id=trip, force_crossing=is_forced))

    df = pd.DataFrame(all_data)
    df.to_csv('data/vessel_data.csv', index=False)
    print(f"Data saved to data/vessel_data.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()
