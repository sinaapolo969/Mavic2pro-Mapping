# turtlebot_controller_optimised.py

import numpy as np
import math
import random
from controller import Robot, Compass, Motor, Lidar

#======================================================================
# 1. Configuration
#======================================================================

# --- World and Map Configuration ---
GOAL_POS_WORLD = (-8.47, -5.4)    # (x, z) in meters
MAP_RESOLUTION = 0.5
MAP_ORIGIN = (-0.7, -1.45)
MAP_STRING = """
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1
0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1
0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1
0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,0,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1
0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1
0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1
0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,1,1,1,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1
0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,1
0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,1
0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1
0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1
0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
"""
# --- Robot and Simulation Parameters ---
MAX_SPEED = 6.28
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160
DISTANCE_TOLERANCE = 0.1

# --- Particle Filter (Localization) Parameters ---
NUM_PARTICLES = 1500             # Reduced particles, as lookup table is more efficient
LIDAR_SAMPLE_POINTS = 20         # Fewer samples are fine with this method
ODOMETRY_NOISE = np.array([0.02, 0.02, 0.05]) # Noise for [x, z, theta]
PARTICLE_STD_DEV_THRESHOLD = 0.1 # Convergence threshold in meters

#======================================================================
# 2. Helper Classes
#======================================================================

class OccupancyMap:
    def __init__(self, map_str, resolution, origin):
        rows = map_str.strip().split('\n')
        self.grid = np.array([list(map(int, row.split(','))) for row in rows], dtype=np.int8)
        self.height, self.width = self.grid.shape
        self.resolution = resolution
        self.origin_x, self.origin_z = origin

        self.free_spaces = np.argwhere(self.grid == 0)
        self.obstacle_lookup = self._precompute_obstacle_distances()
        print("Obstacle distance lookup table generated.")

    def _precompute_obstacle_distances(self):
        """
        OPTIMIZATION: Creates a lookup table for the distance to the nearest obstacle
        from every cell. This avoids repeated, slow ray-casting.
        """
        from scipy.ndimage import distance_transform_edt
        # Invert map so obstacles are 0 and free space is 1
        inverted_grid = 1 - self.grid
        # distance_transform_edt calculates distance from each 1 to the nearest 0
        dist_transform = distance_transform_edt(inverted_grid) * self.resolution
        return dist_transform

    def get_dist_to_obstacle(self, x, z):
        """Fast lookup of distance to the nearest wall."""
        r, c = self.world_to_grid(x, z)
        if not (0 <= r < self.height and 0 <= c < self.width):
            return 0.0
        return self.obstacle_lookup[r, c]

    def world_to_grid(self, x, z):
        c = int((x - self.origin_x) / self.resolution)
        r = int((z - self.origin_z) / self.resolution)
        return r, c

    def grid_to_world(self, r, c):
        x = self.origin_x + (c + 0.5) * self.resolution
        z = self.origin_z + (r + 0.5) * self.resolution
        return x, z

class ParticleFilter:
    def __init__(self, occupancy_map, num_particles):
        self.map = occupancy_map
        self.num_particles = num_particles
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        # Choose random free spaces and convert to world coordinates
        indices = np.random.choice(len(self.map.free_spaces), self.num_particles)
        grid_coords = self.map.free_spaces[indices]
        
        particles = np.zeros((self.num_particles, 4))
        for i, (r, c) in enumerate(grid_coords):
            particles[i, 0], particles[i, 1] = self.map.grid_to_world(r, c)
        
        particles[:, 2] = np.random.uniform(0, 2 * math.pi, self.num_particles)
        particles[:, 3] = 1.0 / self.num_particles
        return particles

    def predict(self, delta_dist, delta_theta):
        noise = (np.random.randn(self.num_particles, 3) * ODOMETRY_NOISE)
        self.particles[:, 0] += (delta_dist * np.cos(self.particles[:, 2])) + noise[:,0]
        self.particles[:, 1] += (delta_dist * np.sin(self.particles[:, 2])) + noise[:,1]
        self.particles[:, 2] = np.mod(self.particles[:, 2] + delta_theta + noise[:,2], 2 * math.pi)

    def weight(self, lidar_ranges):
        """Vectorized weighting using the pre-computed distance lookup."""
        total_error = np.ones(self.num_particles)
        
        for angle, dist in lidar_ranges.items():
            # Calculate where each particle thinks the Lidar beam ends
            beam_end_x = self.particles[:, 0] + dist * np.cos(self.particles[:, 2] + angle)
            beam_end_z = self.particles[:, 1] + dist * np.sin(self.particles[:, 2] + angle)
            
            # Lookup distance to nearest obstacle from that point
            # Note: This is a fast approximation. It measures how far the detected point
            # is from a known wall, penalizing particles that "see" through walls.
            dist_to_wall = np.array([self.map.get_dist_to_obstacle(x, z) for x, z in zip(beam_end_x, beam_end_z)])
            
            # Update error using a Gaussian model
            error = np.exp(-0.5 * (dist_to_wall**2) / (0.2**2))
            total_error *= error
        
        self.particles[:, 3] = total_error
        total_weight = np.sum(self.particles[:, 3])
        self.particles[:, 3] = self.particles[:, 3] / total_weight if total_weight > 0 else 1.0 / self.num_particles

    def resample(self):
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.particles[:, 3])
        self.particles = self.particles[indices]
        self.particles[:, 3] = 1.0 / self.num_particles

    def get_estimated_pose(self):
        mean_x = np.mean(self.particles[:, 0])
        mean_z = np.mean(self.particles[:, 1])
        mean_theta = math.atan2(np.mean(np.sin(self.particles[:, 2])), np.mean(np.cos(self.particles[:, 2])))
        return mean_x, mean_z, mean_theta

    def get_convergence(self):
        return np.max(np.std(self.particles[:, :2], axis=0))

class Bug2Planner:
    def __init__(self, occupancy_map):
        self.map = occupancy_map
        self.state = 'GOAL_SEEK'
        self.m_line_start_grid = None
        self.m_line_goal_grid = None
        self.hit_point_grid = None

    def plan(self, start_grid, goal_grid):
        print(f"Bug2 Planner Initialized. From {start_grid} to {goal_grid}")
        self.state = 'GOAL_SEEK'
        self.m_line_start_grid = start_grid
        self.m_line_goal_grid = goal_grid
        self.m_line_vector = np.array(goal_grid) - np.array(start_grid)
        self.m_line_length_sq = np.sum(self.m_line_vector**2)

    def _is_on_m_line(self, pos_grid):
        if self.m_line_length_sq == 0: return False
        pos_vector = np.array(pos_grid) - np.array(self.m_line_start_grid)
        t = np.dot(pos_vector, self.m_line_vector) / self.m_line_length_sq
        t = np.clip(t, 0, 1)
        closest_point = np.array(self.m_line_start_grid) + t * self.m_line_vector
        dist_sq = np.sum((np.array(pos_grid) - closest_point)**2)
        return dist_sq < 1.5

    def get_next_target(self, current_grid, goal_world, front_obstacle):
        if self.state == 'GOAL_SEEK':
            if front_obstacle:
                print("Obstacle detected! Switching to WALL_FOLLOW.")
                self.state = 'WALL_FOLLOW'
                self.hit_point_grid = current_grid
                return None
            return goal_world
        
        elif self.state == 'WALL_FOLLOW':
            dist_hit_to_goal_sq = np.sum((np.array(self.hit_point_grid) - np.array(self.m_line_goal_grid))**2)
            dist_to_goal_sq = np.sum((np.array(current_grid) - np.array(self.m_line_goal_grid))**2)
            
            if self._is_on_m_line(current_grid) and dist_to_goal_sq < dist_hit_to_goal_sq:
                print("M-line re-encountered closer to goal. Switching to GOAL_SEEK.")
                self.state = 'GOAL_SEEK'
                return goal_world
            return None
        return goal_world

#======================================================================
# 3. Main Robot Controller
#======================================================================

class TurtleBotController(Robot):
    def __init__(self):
        super(TurtleBotController, self).__init__()
        self.timeStep = int(self.getBasicTimeStep())
        
        # Devices
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)
        self.lidar = self.getDevice("LDS-01") # Make sure your lidar is named LDS-01
        self.lidar.enable(self.timeStep)
        
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.left_ps = self.getDevice("left wheel sensor")
        self.right_ps = self.getDevice("right wheel sensor")
        self.left_ps.enable(self.timeStep)
        self.right_ps.enable(self.timeStep)
        self.prev_left_rad = 0.0
        self.prev_right_rad = 0.0

        # State and Planner
        self.phase = 'LOCALIZING'
        self.map = OccupancyMap(MAP_STRING, MAP_RESOLUTION, MAP_ORIGIN)
        self.particle_filter = ParticleFilter(self.map, NUM_PARTICLES)
        self.planner = Bug2Planner(self.map)
        
        self.goal_world = GOAL_POS_WORLD
        self.goal_grid = self.map.world_to_grid(*self.goal_world)
        self.estimated_pose = (0, 0, 0)
        self.lidar_data = {}

    def update_sensors(self):
        # Update Lidar
        ranges = self.lidar.getRangeImage()
        num_points = self.lidar.getHorizontalResolution()
        step = num_points // LIDAR_SAMPLE_POINTS
        
        self.lidar_data.clear()
        for i in range(0, num_points, step):
            angle = (i / num_points) * 2 * math.pi - math.pi
            dist = ranges[i]
            if not math.isinf(dist) and dist > 0.01:
                self.lidar_data[angle] = dist
                
        # Update Odometry
        left_rad = self.left_ps.getValue()
        right_rad = self.right_ps.getValue()
        delta_left = (left_rad - self.prev_left_rad) * WHEEL_RADIUS
        delta_right = (right_rad - self.prev_right_rad) * WHEEL_RADIUS
        self.prev_left_rad, self.prev_right_rad = left_rad, right_rad
        
        delta_dist = (delta_left + delta_right) / 2.0
        delta_theta = (delta_right - delta_left) / WHEEL_BASE
        return delta_dist, delta_theta

    def run(self):
        print("Robot starting. Phase: LOCALIZING")
        
        while self.step(self.timeStep) != -1:
            delta_dist, delta_theta = self.update_sensors()

            # --- 1. Localize ---
            self.particle_filter.predict(delta_dist, delta_theta)
            if self.lidar_data:
                self.particle_filter.weight(self.lidar_data)
                self.particle_filter.resample()
            
            pos_x, pos_z, theta = self.particle_filter.get_estimated_pose()
            
            # --- 2. Decide and Act ---
            if self.phase == 'LOCALIZING':
                convergence = self.particle_filter.get_convergence()
                print(f"Localizing... Convergence: {convergence:.4f} m", end='\r')
                if convergence < PARTICLE_STD_DEV_THRESHOLD:
                    self.phase = 'NAVIGATING'
                    start_grid = self.map.world_to_grid(pos_x, pos_z)
                    self.planner.plan(start_grid, self.goal_grid)
                    print(f"\n*** ROBOT LOCALIZED at ({pos_x:.2f}, {pos_z:.2f}) ***")
                    print("--> Switching to NAVIGATING phase.")

                front_obstacle = any(dist < 0.25 for angle, dist in self.lidar_data.items() if abs(angle) < 0.3)
                left_vel, right_vel = (-0.5 * MAX_SPEED, 0.5 * MAX_SPEED) if front_obstacle else (MAX_SPEED, MAX_SPEED)
            
            elif self.phase == 'NAVIGATING':
                if math.hypot(pos_x - self.goal_world[0], pos_z - self.goal_world[1]) < DISTANCE_TOLERANCE:
                    print("\nGoal reached!")
                    left_vel, right_vel = 0, 0
                    self.left_motor.setVelocity(left_vel)
                    self.right_motor.setVelocity(right_vel)
                    break

                current_grid = self.map.world_to_grid(pos_x, pos_z)
                front_obs = any(dist < 0.25 for angle, dist in self.lidar_data.items() if abs(angle) < 0.3)
                right_obs = any(dist < 0.4 for angle, dist in self.lidar_data.items() if -1.8 < angle < -1.3)

                target = self.planner.get_next_target(current_grid, self.goal_world, front_obs)
                
                if target is None: # Wall following
                    if front_obs:
                        left_vel, right_vel = -0.6 * MAX_SPEED, 0.6 * MAX_SPEED
                    elif not right_obs:
                        left_vel, right_vel = 0.8 * MAX_SPEED, 0.3 * MAX_SPEED
                    else:
                        left_vel, right_vel = MAX_SPEED, MAX_SPEED
                else: # Goal seeking
                    angle_to_target = math.atan2(target[1] - pos_z, target[0] - pos_x)
                    angle_diff = (angle_to_target - theta + math.pi) % (2 * math.pi) - math.pi
                    
                    turn_speed = 4.0 * angle_diff
                    forward_speed = MAX_SPEED * max(0, 1 - 2.0 * abs(angle_diff))
                    left_vel = forward_speed - turn_speed
                    right_vel = forward_speed + turn_speed

            # --- 3. Command Motors ---
            self.left_motor.setVelocity(np.clip(left_vel, -MAX_SPEED, MAX_SPEED))
            self.right_motor.setVelocity(np.clip(right_vel, -MAX_SPEED, MAX_SPEED))

if __name__ == "__main__":
    # This check is crucial for the controller to run.
    # It also handles the installation of scipy if not found.
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        print("Scipy not found. Please install it for this controller to work:")
        print("pip install scipy")
        exit()
        
    controller = TurtleBotController()
    controller.run()