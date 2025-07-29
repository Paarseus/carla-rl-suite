import cv2
import carla
from gymnasium import spaces
import math
import numpy as np


class ObservationSpace:
        def __init__(self, hero_manager, sensor_interface):
                self.hero_manager = hero_manager
                self.sensor_interface = sensor_interface


                # ADD THESE NORMALIZATION PARAMETERS:
                self.max_speed = 100.0  # km/h - reasonable max speed
                self.max_position_offset = 1000.0  # meters from start
                self.max_velocity = 30.0  # m/s - about 108 km/h
                self.max_lane_deviation = 5.0  # meters from lane center
                self.start_position = None  # Track starting position

                self.control = None

                print("[OBSERVATION SPACE] Initialized with normalization")


        def hero_control(self):
                self.control = self.hero_manager.hero.get_control()

                steering = self.control.steer
                throttle = self.control.throttle
                brake = self.control.brake
                speed = self.get_speed()

                # NORMALIZE SPEED: 0-100 km/h → 0-1
                speed = np.clip(speed / self.max_speed, 0.0, 1.0)
                
                return np.array([steering, throttle, brake, speed], dtype=np.float32)

        def hero_position(self):
                transform = self.hero.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = rotation.yaw      # Heading direction (degrees)
                pitch = rotation.pitch  # Up/down tilt (degrees)
                roll = rotation.roll
                yaw_rad = np.radians(yaw)

                # ADD THIS NORMALIZATION:
                if self.start_position is None:
                        self.start_position = np.array([location.x, location.y])
                        relative_x, relative_y = 0.0, 0.0
                else:
                        relative_pos = np.array([location.x, location.y]) - self.start_position
                        relative_x = relative_pos[0]
                        relative_y = relative_pos[1]
                
                # Normalize: ±1000m → ±1
                normalized_x = np.clip(relative_x / self.max_position_offset, -1.0, 1.0)
                normalized_y = np.clip(relative_y / self.max_position_offset, -1.0, 1.0)
                normalized_yaw = yaw_rad / np.pi
                
                return np.array([normalized_x, normalized_y, normalized_yaw], dtype=np.float32)

                #return np.array([location.x, location.y, yaw_rad], dtype=np.float32)
        
        def get_acceleration(self):
                acceleration = self.hero.get_acceleration()
                accel_x = acceleration.x
                accel_y = acceleration.y
                accel_z = acceleration.z
                accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

                # Angular velocity (how fast it's rotating)
                angular_velocity = self.hero.get_angular_velocity()
                angular_vel_z = angular_velocity.z  # Yaw rate (how fast turning)

                return np.array([accel_magnitude, angular_vel_z], dtype=np.float32)


        def get_navigation(self):
                # Get current waypoint on the road
                world = self.hero.get_world()
                map = world.get_map()
                current_waypoint = map.get_waypoint(
                self.hero.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
                )

                # Distance from lane center
                waypoint_location = current_waypoint.transform.location
                vehicle_location = self.hero.get_location()
                distance_from_center = np.sqrt(
                (vehicle_location.x - waypoint_location.x)**2 + 
                (vehicle_location.y - waypoint_location.y)**2
                )

                hero_heading = self.hero.get_transform().get_forward_vector()
                hero_heading_vec = np.array([hero_heading.x, hero_heading.y])
                
                if current_waypoint:
                        wp_heading = current_waypoint.transform.get_forward_vector()
                        wp_heading_vec = np.array([wp_heading.x, wp_heading.y])
                        
                        # Calculate angle difference
                        dot_product = np.dot(hero_heading_vec, wp_heading_vec)
                        cross_product = hero_heading_vec[0] * wp_heading_vec[1] - hero_heading_vec[1] * wp_heading_vec[0]
                        angle_to_road = np.arctan2(cross_product, dot_product)
                else:
                        angle_to_road = 0.0


                # NORMALIZE LANE DEVIATION: 0-5m → 0-1
                distance_from_center = np.clip(distance_from_center / self.max_lane_deviation, 0.0, 1.0)

                # NORMALIZE ANGLE: [-π, π] → [-1, 1]  
                angle_to_road = angle_to_road / np.pi

                return np.array([distance_from_center, angle_to_road], dtype=np.float32)

        def get_speed(self):
                vel = self.hero.get_velocity()
                speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                return speed
        
        def get_velocity_vector(self):
                velocity = self.hero.get_velocity()
                # NORMALIZE VELOCITY: ±30 m/s → ±1
                normalized_vel_x = np.clip(velocity.x / self.max_velocity, -1.0, 1.0)
                normalized_vel_y = np.clip(velocity.y / self.max_velocity, -1.0, 1.0)
                return np.array([normalized_vel_x, normalized_vel_y], dtype=np.float32)
        
        def get_camera_data(self):
                sensor_data = self.sensor_interface.get_data()
                camera_data = sensor_data.get('camera_front')
                if camera_data is None:
                        print("No image recieved from cameras")
                        return np.zeros((300, 400, 3), dtype=np.uint8)
                frame_id, image_data = camera_data
                return image_data
        
        def get_collision_status(self):
                """Get collision detection status from collision sensor"""
                try:
                        sensor_data = self.sensor_interface.get_data()
                        collision_data = sensor_data.get('collision')
                        
                        if collision_data is not None:
                                # Collision detected!
                                frame_id, collision_event = collision_data
                                print(f"[COLLISION DETECTED] Frame: {frame_id}")
                                return 1.0
                        else:
                                # No collision
                                print("7")
                                return 0.0
                                
                except Exception as e:
                        print(f"[COLLISION ERROR] {e}")
                        return 0.0

        def _get_empty_obs(self):
                """Return empty observation when hero is None"""
                return {
                        'hero_state': np.zeros(10, dtype=np.float32),
                        'navigation': np.zeros(2, dtype=np.float32),
                        'camera': np.zeros((300, 400, 3), dtype=np.uint8)
                }

        def reset(self):
                """Reset the observation space state (call this on environment reset)"""
                self.start_position = None
                print("[OBSERVATION SPACE] Reset - start position cleared")

        def obs(self):
                self.hero = self.hero_manager.hero
                if self.hero is None:
                        print("[ERROR] No hero recieved by observation space, sending empty obs")
                        return self._get_empty_obs() 
                
                self.control = self.hero.get_control()

                # Get all observation components
                control_data = self.hero_control()          # [steering, throttle, brake, speed]
                position_data = self.hero_position()        # [x, y, yaw_rad]  
                velocity_data = self.get_velocity_vector()  # [vel_x, vel_y]
                navigation_data = self.get_navigation()     # [distance_from_center, angle_to_road]
                
                
                sensor_data = self.sensor_interface.get_data()
                camera_data = sensor_data.get('camera_front')
                collision_data = sensor_data.get('collision')


                if camera_data is None:
                        print("No image received from cameras")
                        camera_data = np.zeros((300, 400, 3), dtype=np.uint8)
                else:
                        frame_id, camera_data = camera_data


                if collision_data is not None:
                        frame_id, collision_event = collision_data
                        print(f"[COLLISION DETECTED] Frame: {frame_id}")
                        collision_status = 1.0
                else:
                        collision_status = 0.0

                
                # Format as dictionary (recommended for research)
                observation = {
                        'hero_state': np.concatenate([
                        control_data,           # 4 values: [steer, throttle, brake, speed]
                        position_data,          # 3 values: [x, y, yaw]
                        velocity_data,          # 2 values: [vel_x, vel_y]
                        [collision_status]      # 1 value: collision
                        ]).astype(np.float32),      # Total: 10 values
        
                        'navigation': navigation_data.astype(np.float32),  # 2 values
                        'camera': camera_data.astype(np.uint8)             # Image array
                }
                
                return observation