from matplotlib import pyplot as plt
from controller import Robot
import math
import numpy as np
import cv2
import os
from collections import defaultdict

# ------------------------
# Utilities & constants
# ------------------------
CLAMP = lambda val, low, high: max(min(val, high), low)
distance = lambda x1, y1, x2, y2: math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

SCAN_ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
PAUSE_DURATION_S = 0.5
HEADING_TOLERANCE_DEG = 3.0
ALTITUDE_TOLERANCE = 0.1

K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
K_YAW_P = 0.8
WAYPOINT_TOLERANCE = 0.3
NAV_HEADING_TOLERANCE_DEG = 15.0
MAX_FORWARD_COMMAND = 1
MAX_YAW_DISTURBANCE = 0.7


QR_WAYPOINTS = [
    (-6.5, -4.35),
    (-5.36, -7.55),
    (-1.8, -7.55),
]

MAPPING_WAYPOINTS = [
    (-7.6, -10.3),
    (-4.5, -11.5),
    (-1.65, -11.5),
    (-1.79, -5.82),
    (-1.79, -2.62),
    (-6.58, -2.62),
    (-8.4, -7.63),
    (-3.71, -7.63),
]

# -------------------------------------------------------------------
# --- MAPPING CODE: NOW ACTIVE FOR PHASE 2 ---
# -------------------------------------------------------------------
OG_CELL_SIZE = 0.1
OG_L_FREE = -0.4
OG_L_OCC  = +0.85
OG_L_CLIP = 4.0
OUTPUT_MAP_PATH = "mnt/occupancy_gird.csv"
MAPPING_ROTATION_RATE_DEG_S = 45.0

OG_X_MIN, OG_X_MAX, OG_Y_MIN, OG_Y_MAX = None, None, None, None
og_w, og_h = 0, 0
og_log_odds = None

mission_phase = "QR_SCANNING"
state = "NAVIGATING"
qr_codes_found = defaultdict(set)
current_qr_wp_idx = 0
current_mapping_wp_idx = 0
current_scan_angle_idx = 0
pause_start_time = 0.0
scan_start_time = 0.0
scan_accum_points = []

# ------------------------
# Initialize Webots devices
# ------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
camera = robot.getDevice("camera")
camera.enable(timestep)
camera_width, camera_height = camera.getWidth(), camera.getHeight()
imu, gps, gyro, compass = robot.getDevice("inertial unit"), robot.getDevice("gps"), robot.getDevice("gyro"), robot.getDevice("compass")
imu.enable(timestep); gps.enable(timestep); gyro.enable(timestep); compass.enable(timestep)
lidar = robot.getDevice("lidar")
lidar.enable(timestep); lidar.enablePointCloud()
front_left_led, front_right_led = robot.getDevice("front left led"), robot.getDevice("front right led")
camera_roll_motor, camera_pitch_motor = robot.getDevice("camera roll"), robot.getDevice("camera pitch")
motors = [robot.getDevice(p) for p in ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]]
for m in motors: m.setPosition(float("inf")); m.setVelocity(1.0)
qr_detector = cv2.QRCodeDetector()

# --- MAPPING & QR DETECTION FUNCTIONS ---

def perform_lidar_scan_world(pos, heading_deg):
    raw = lidar.getRangeImage()
    if not raw: return []
    fov, res, max_range = lidar.getFov(), lidar.getHorizontalResolution(), lidar.getMaxRange()
    if res == 0: return []
    step = fov / res
    heading_rad = math.radians(heading_deg)
    pts = []
    max_range_sq = (max_range * 0.99)**2
    for i, d in enumerate(raw):
        if not np.isfinite(d): continue
        dist_sq = d * d
        beam_ang = -fov / 2.0 + i * step
        global_ang = heading_rad + beam_ang
        gx = pos[0] + d * math.cos(global_ang)
        gy = pos[1] + d * math.sin(global_ang)
        is_max_range = dist_sq >= max_range_sq
        pts.append(((gx, gy), is_max_range))
    return pts

def bresenham(i0, j0, i1, j1):
    points = []; dx = abs(i1 - i0); dy = abs(j1 - j0)
    x, y = i0, j0; sx = 1 if i0 < i1 else -1; sy = 1 if j0 < j1 else -1
    if dx >= dy:
        err = dx // 2
        while True:
            points.append((x, y));
            if x == i1 and y == j1: break
            err -= dy
            if err < 0: y += sy; err += dx
            x += sx
    else:
        err = dy // 2
        while True:
            points.append((x, y));
            if x == i1 and y == j1: break
            err -= dx
            if err < 0: x += sx; err += dy
            y += sy
    return points

def init_occupancy_grid(all_waypoints, max_range):
    global OG_X_MIN, OG_X_MAX, OG_Y_MIN, OG_Y_MAX, og_w, og_h, og_log_odds
    xs = [p[0] for p in all_waypoints]; ys = [p[1] for p in all_waypoints]
    pad = float(max_range) + 2.0
    OG_X_MIN, OG_X_MAX = min(xs) - pad, max(xs) + pad
    OG_Y_MIN, OG_Y_MAX = min(ys) - pad, max(ys) + pad
    og_w = max(10, int(math.ceil((OG_X_MAX - OG_X_MIN) / OG_CELL_SIZE)))
    og_h = max(10, int(math.ceil((OG_Y_MAX - OG_Y_MIN) / OG_CELL_SIZE)))
    og_log_odds = np.zeros((og_h, og_w), dtype=np.float32)
    og_w = 28
    og_h = 28

def world_to_grid(x, y):
    if OG_X_MIN is None: return None
    i = int((x - OG_X_MIN) / OG_CELL_SIZE); j = int((y - OG_Y_MIN) / OG_CELL_SIZE)
    if 0 <= i < og_w and 0 <= j < og_h: return i, j
    return None

def update_occupancy_from_scan(robot_pos, scan_points):
    origin_idx = world_to_grid(robot_pos[0], robot_pos[1])
    if origin_idx is None: return
    i0, j0 = origin_idx
    for (xg, yg), is_max_range in scan_points:
        end_idx = world_to_grid(xg, yg)
        if end_idx is None: continue
        i1, j1 = end_idx
        ray_cells = bresenham(i0, j0, i1, j1)
        if not ray_cells: continue
        for (ci, cj) in ray_cells[:-1]: og_log_odds[cj, ci] = np.clip(og_log_odds[cj, ci] + OG_L_FREE, -OG_L_CLIP, OG_L_CLIP)
        if not is_max_range:
            ci, cj = ray_cells[-1]
            og_log_odds[cj, ci] = np.clip(og_log_odds[cj, ci] + OG_L_OCC, -OG_L_CLIP, OG_L_CLIP)

def detect_qr_from_camera(camera):
    image_data = camera.getImage()
    if image_data is None: return None
    img = np.frombuffer(image_data, np.uint8).reshape((camera_height, camera_width, 4))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    retval, points = qr_detector.detect(img_rgb)
    if retval:
        data, _ = qr_detector.decode(img_rgb, points)
        if data:
            timestamp = int(robot.getTime() * 1000)
            output_dir = "/mnt/data"; os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/qr_detected_{timestamp}.png"
            cv2.imwrite(filename, img_rgb)
            return data
    return None

# ------------------------
# Main Logic
# ------------------------
print("Waiting for sensors...")
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0: break

init_occupancy_grid(QR_WAYPOINTS + MAPPING_WAYPOINTS, lidar.getMaxRange())
print("Starting autonomous mission...")
target_altitude = 1.2
scan_duration_s = 360.0 / MAPPING_ROTATION_RATE_DEG_S

while robot.step(timestep) != -1:
    time = robot.getTime()
    roll, pitch, yaw = imu.getRollPitchYaw()
    roll_vel, pitch_vel, _ = gyro.getValues()
    gx, gy, gz = gps.getValues()
    current_heading_deg = (math.degrees(yaw) + 360) % 360
    led_state = int(time) % 2; front_left_led.set(led_state); front_right_led.set(1 - led_state)
    camera_roll_motor.setPosition(-0.115 * roll_vel); camera_pitch_motor.setPosition(-0.1 * pitch_vel)

    roll_disturbance, pitch_disturbance, yaw_disturbance = 0.0, 0.0, 0.0

    # --- PHASE 1: QR SCANNING ---
    if mission_phase == "QR_SCANNING":
        target_altitude = 1.2
        waypoints = QR_WAYPOINTS
        if current_qr_wp_idx >= len(waypoints):
            print("\n--- QR SCANNING PHASE COMPLETE ---\n--- BEGINNING MAPPING PHASE ---\n")
            mission_phase = "MAPPING"
            state = "MAPPING_ASCENDING"
            continue

        if state == "NAVIGATING":
            tx, ty = waypoints[current_qr_wp_idx]
            dist_to_wp = distance(gx, gy, tx, ty)
            if dist_to_wp < WAYPOINT_TOLERANCE:
                print(f"[STATE CHANGE] Reached QR waypoint {current_qr_wp_idx+1}. Beginning scan.")
                state = "AIMING"; current_scan_angle_idx = 0
            else:
                target_heading_rad = math.atan2(ty - gy, tx - gx)
                heading_error_rad = target_heading_rad - yaw
                if heading_error_rad > math.pi: heading_error_rad -= 2 * math.pi
                elif heading_error_rad < -math.pi: heading_error_rad += 2 * math.pi
                if abs(math.degrees(heading_error_rad)) > NAV_HEADING_TOLERANCE_DEG:
                    yaw_command = K_YAW_P * heading_error_rad
                    yaw_disturbance = CLAMP(yaw_command, -MAX_YAW_DISTURBANCE, MAX_YAW_DISTURBANCE)
                else:
                    forward_speed = CLAMP(dist_to_wp, 0.0, MAX_FORWARD_COMMAND)
                    pitch_disturbance = -forward_speed * math.cos(heading_error_rad)
                    roll_disturbance = 2.0 * math.sin(heading_error_rad)
        elif state == "AIMING":
            target_heading_deg = SCAN_ANGLES_DEG[current_scan_angle_idx]
            heading_error_deg = target_heading_deg - current_heading_deg
            if heading_error_deg > 180: heading_error_deg -= 360
            elif heading_error_deg < -180: heading_error_deg += 360
            if abs(heading_error_deg) < HEADING_TOLERANCE_DEG:
                state = "SCANNING_PAUSED"; pause_start_time = time
            else:
                yaw_command = K_YAW_P * math.radians(heading_error_deg)
                yaw_disturbance = CLAMP(yaw_command, -MAX_YAW_DISTURBANCE, MAX_YAW_DISTURBANCE)
        elif state == "SCANNING_PAUSED":
            qr_data = detect_qr_from_camera(camera)
            if qr_data and qr_data not in qr_codes_found[current_qr_wp_idx]:
                print(f"  [QR DETECTED] Found '{qr_data}' at waypoint {current_qr_wp_idx+1}.")
                qr_codes_found[current_qr_wp_idx].add(qr_data)
            if time - pause_start_time > PAUSE_DURATION_S:
                current_scan_angle_idx += 1
                if current_scan_angle_idx >= len(SCAN_ANGLES_DEG):
                    current_qr_wp_idx += 1; state = "NAVIGATING"
                else:
                    state = "AIMING"

    # --- PHASE 2: MAPPING ---
    elif mission_phase == "MAPPING":
        waypoints = MAPPING_WAYPOINTS
        if current_mapping_wp_idx >= len(waypoints):
            print("--- MAPPING PHASE COMPLETE ---")
            break
        
        if state == "MAPPING_ASCENDING":
            target_altitude = 3.0
            if abs(gz - target_altitude) < ALTITUDE_TOLERANCE:
                state = "MAPPING_NAVIGATING_HIGH"
        elif state == "MAPPING_NAVIGATING_HIGH":
            target_altitude = 3.0
            tx, ty = waypoints[current_mapping_wp_idx]
            dist_to_wp = distance(gx, gy, tx, ty)
            if dist_to_wp < WAYPOINT_TOLERANCE:
                state = "MAPPING_DESCENDING"
            else:
                target_heading_rad = math.atan2(ty - gy, tx - gx)
                heading_error_rad = target_heading_rad - yaw
                if heading_error_rad > math.pi: heading_error_rad -= 2 * math.pi
                elif heading_error_rad < -math.pi: heading_error_rad += 2 * math.pi
                
                if abs(math.degrees(heading_error_rad)) > NAV_HEADING_TOLERANCE_DEG:
                    pitch_disturbance = 0.0
                    roll_disturbance = 0.0
                    yaw_command = K_YAW_P * heading_error_rad
                    yaw_disturbance = CLAMP(yaw_command, -MAX_YAW_DISTURBANCE, MAX_YAW_DISTURBANCE)
                else:
                    forward_speed = CLAMP(dist_to_wp, 0.0, MAX_FORWARD_COMMAND)
                    pitch_disturbance = -forward_speed * math.cos(heading_error_rad)
                    roll_disturbance = 2.0 * math.sin(heading_error_rad)
        
        elif state == "MAPPING_DESCENDING":
            target_altitude = 1.0
            if abs(gz - target_altitude) < ALTITUDE_TOLERANCE:
                state = "MAPPING_SCANNING"; scan_start_time = time; scan_accum_points = []
        
        elif state == "MAPPING_SCANNING":
            target_altitude = 1.0
            yaw_disturbance = math.radians(MAPPING_ROTATION_RATE_DEG_S)
            scan_points = perform_lidar_scan_world([gx, gy], current_heading_deg)
            if scan_points: scan_accum_points.extend(scan_points)
            if time - scan_start_time > scan_duration_s:
                if scan_accum_points:
                    update_occupancy_from_scan([gx, gy], scan_accum_points)
                current_mapping_wp_idx += 1
                state = "MAPPING_ASCENDING"

    clamped_diff_alt = CLAMP(target_altitude - gz + K_VERTICAL_OFFSET, -1.0, 1.0)
    vertical_input = K_VERTICAL_P * pow(clamped_diff_alt, 3)
    roll_input = K_ROLL_P * CLAMP(roll, -1.0, 1.0) + roll_vel + roll_disturbance
    pitch_input = K_PITCH_P * CLAMP(pitch, -1.0, 1.0) + pitch_vel + pitch_disturbance
    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance
    motors[0].setVelocity(fl); motors[1].setVelocity(-fr); motors[2].setVelocity(-rl); motors[3].setVelocity(rr)


print("\n-------------------------------------------")
print("           MISSION COMPLETE")
print("-------------------------------------------")

# --- QR Code Results ---
if not qr_codes_found:
    print("No QR codes were detected.")
else:
    print("Detected Door Numbers:")
    for i in range(len(QR_WAYPOINTS)):
        wp_coords = QR_WAYPOINTS[i]
        results = qr_codes_found.get(i)
        if results:
            print(f"  - Waypoint {i+1} at ({wp_coords[0]:.2f}, {wp_coords[1]:.2f}): {', '.join(sorted(list(results)))}")
        else:
            print(f"  - Waypoint {i+1} at ({wp_coords[0]:.2f}, {wp_coords[1]:.2f}): NOT_FOUND")

# --- Occupancy Map Saving ---
if og_log_odds is not None:
    prob = 1.0 / (1.0 + np.exp(-og_log_odds))
    plt.figure(figsize=(10, 10))
    extent = [OG_X_MIN, OG_X_MAX, OG_Y_MIN, OG_Y_MAX]
    plt.imshow(prob.T, origin='lower', extent=extent, vmin=0, vmax=1, cmap='gray_r')
    plt.colorbar(label="Occupancy Probability"); plt.title("Final Occupancy Grid Map")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.grid(True, linestyle='--', alpha=0.2)
    try:
        plt.savefig(OUTPUT_MAP_PATH, bbox_inches='tight', dpi=250)
    except Exception as e:
        pass
else:
    print("\n[MAP] Occupancy grid not generated.")

print("\nController finished.")